"""ACT policy wrapper: loss computation (training) + inference helper (rollout).

Adds on top of ACTCVAE:
- Dataset normalization stats registered as buffers, so a saved checkpoint
  is self-contained and does not need a separate stats file at load time.
- Training-time loss: masked L1 on the action chunk + KL(N(mu, sigma^2) || N(0, I)).
- Numpy-in / numpy-out `.inference()` for the ROS2 rollout loop.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import config as C
from model.cvae import ACTCVAE


class ACTPolicy(nn.Module):
    def __init__(
        self,
        stats: dict,
        kl_weight: float = C.KL_WEIGHT,
        **cvae_kwargs,
    ):
        super().__init__()
        self.model = ACTCVAE(**cvae_kwargs)
        self.kl_weight = kl_weight

        self.register_buffer("qpos_mean",   torch.from_numpy(stats["qpos_mean"]))
        self.register_buffer("qpos_std",    torch.from_numpy(stats["qpos_std"]))
        self.register_buffer("action_mean", torch.from_numpy(stats["action_mean"]))
        self.register_buffer("action_std",  torch.from_numpy(stats["action_std"]))
        self.register_buffer(
            "image_mean", torch.from_numpy(stats["image_mean"]).view(3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.from_numpy(stats["image_std"]).view(3, 1, 1)
        )

    # -----------------------------------------------------------------------
    # Training / eval forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,                     # pre-normalized by the DataLoader
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ):
        """Training (actions given) -> loss dict. Eval (no actions) -> a_hat tensor."""
        if actions is not None:
            assert is_pad is not None, "is_pad required when actions is given"
            a_hat, (mu, logvar) = self.model(image, qpos, actions, is_pad)
            return self._compute_loss(a_hat, actions, is_pad, mu, logvar)
        a_hat, _ = self.model(image, qpos)
        return a_hat

    def _compute_loss(
        self,
        a_hat: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        l1 = F.l1_loss(a_hat, actions, reduction="none")           # (B, k, action_dim)
        valid = (~is_pad).unsqueeze(-1).float()                    # (B, k, 1)
        l1 = (l1 * valid).sum() / (valid.sum().clamp_min(1.0) * actions.size(-1))

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        total = l1 + self.kl_weight * kl
        return {"loss": total, "l1": l1.detach(), "kl": kl.detach()}

    # -----------------------------------------------------------------------
    # Inference (single-sample, raw units)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        image_np: np.ndarray,   # (H, W, 3) uint8 RGB
        qpos_np: np.ndarray,    # (state_dim,) float, raw Sensapex counts
    ) -> np.ndarray:
        """Rollout entry point. Returns (chunk_size, action_dim) in absolute units.

        Callers feeding cv2-decoded frames must convert BGR -> RGB first; the
        training dataset loads with PIL which is RGB.
        """
        self.eval()
        device = self.qpos_mean.device

        img = self._preprocess_image(image_np).to(device).unsqueeze(0)    # (1, num_cam, 3, H, W)
        qpos = torch.from_numpy(qpos_np.astype(np.float32)).to(device)
        qpos = ((qpos - self.qpos_mean) / self.qpos_std).unsqueeze(0)     # (1, state_dim)

        a_hat, _ = self.model(img, qpos)                                  # (1, k, action_dim)
        a = a_hat[0] * self.action_std + self.action_mean
        return a.cpu().numpy().astype(np.float32)

    def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
        h, w = img_np.shape[:2]
        if (h, w) != (C.IMAGE_HEIGHT, C.IMAGE_WIDTH):
            pil = Image.fromarray(img_np).resize(
                (C.IMAGE_WIDTH, C.IMAGE_HEIGHT), Image.BILINEAR
            )
            img_np = np.array(pil)                          # copy so torch can own it
        x = torch.from_numpy(img_np).float() / 255.0        # (H, W, 3)
        x = x.permute(2, 0, 1).to(self.image_mean.device)  # (3, H, W), match buffer device
        x = (x - self.image_mean) / self.image_std
        return x.unsqueeze(0)                               # (num_cam=1, 3, H, W)


# ---------------------------------------------------------------------------

def build_policy(
    stats: Optional[dict] = None,
    stats_path: Path = C.STATS_PATH,
    kl_weight: float = C.KL_WEIGHT,
    **cvae_kwargs,
) -> ACTPolicy:
    if stats is None:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
    return ACTPolicy(stats=stats, kl_weight=kl_weight, **cvae_kwargs)
