"""Conditional VAE action-chunking model for MicroACT.

Two transformer stacks:

- **Style encoder**: encoder-only over `[CLS, qpos_token, action_token_1..k]`.
  The CLS output is projected to `(mu, logvar)`; a reparameterized latent
  `z` is sampled. Only used during training.

- **Main encoder-decoder**: encoder over `[latent_tok, qpos_tok, img_tokens]`,
  decoder over `chunk_size` learned query embeddings, with a linear action
  head mapping decoder outputs to `action_dim`.

At inference the style encoder is skipped and `z = 0` (the prior mean).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import config as C
from model.backbone import build_backbone
from model.transformer import build_encoder, build_transformer


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    return mu + torch.randn_like(std) * std


class ACTCVAE(nn.Module):
    def __init__(
        self,
        state_dim: int = C.STATE_DIM,
        action_dim: int = C.ACTION_DIM,
        hidden_dim: int = C.HIDDEN_DIM,
        latent_dim: int = C.LATENT_DIM,
        chunk_size: int = C.CHUNK_SIZE,
        num_cameras: int = C.NUM_CAMERAS,
        pretrained_backbone: bool = C.BACKBONE_PRETRAINED,
        backbone_name: str = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras

        # ---- Backbones ----
        self.backbone = build_backbone(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone,
            backbone_name=backbone_name,
            freeze=freeze_backbone,
        )
        self.transformer = build_transformer(d_model=hidden_dim)
        self.style_encoder = build_encoder(d_model=hidden_dim)

        # ---- Style-encoder IO ----
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.style_qpos_proj = nn.Linear(state_dim, hidden_dim)
        self.style_action_proj = nn.Linear(action_dim, hidden_dim)
        self.style_pos_embed = nn.Embedding(1 + 1 + chunk_size, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)

        # ---- Main-encoder non-image tokens ----
        self.latent_to_src = nn.Linear(latent_dim, hidden_dim)
        self.qpos_to_src = nn.Linear(state_dim, hidden_dim)
        self.extra_src_pos = nn.Embedding(2, hidden_dim)

        # ---- Decoder queries + action head ----
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    # -----------------------------------------------------------------------

    def _encode_style(
        self,
        qpos: torch.Tensor,     # (B, state_dim)
        actions: torch.Tensor,  # (B, k, action_dim)
        is_pad: torch.Tensor,   # (B, k) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = qpos.size(0)
        device = qpos.device

        cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)         # (B, 1, D)
        qpos_tok = self.style_qpos_proj(qpos).unsqueeze(1)                 # (B, 1, D)
        act_toks = self.style_action_proj(actions)                         # (B, k, D)
        seq = torch.cat([cls, qpos_tok, act_toks], dim=1)                  # (B, 2+k, D)
        seq = seq.permute(1, 0, 2).contiguous()                            # (2+k, B, D)

        pos = self.style_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)   # (2+k, B, D)

        always_valid = torch.zeros(B, 2, dtype=torch.bool, device=device)
        pad_mask = torch.cat([always_valid, is_pad], dim=1)                # (B, 2+k)

        out = self.style_encoder(seq, src_key_padding_mask=pad_mask, pos=pos)
        cls_out = out[0]                                                   # (B, D)
        mu, logvar = self.latent_proj(cls_out).chunk(2, dim=-1)
        return mu, logvar

    def _encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: (B, num_cam, 3, H, W)
        B, N = image.shape[:2]
        flat = image.flatten(0, 1)                                         # (B*N, 3, H, W)
        feat, pos = self.backbone(flat)
        # Single-encoder backbones return 4D (B*N, D, Hp, Wp); dual-encoder
        # backbones already return pre-flattened tokens (S, B*N, D).
        if feat.dim() == 4:
            D, Hp, Wp = feat.shape[1:]
            feat = feat.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
            pos = pos.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
            feat = feat.permute(2, 0, 1).contiguous()                      # (N*Hp*Wp, B, D)
            pos = pos.permute(2, 0, 1).contiguous()
        else:
            # feat: (S, B*N, D) — multi-camera reshape: each camera contributes
            # S tokens; concat along the token dimension.
            S, BN, D = feat.shape
            if N > 1:
                feat = feat.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
                pos = pos.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
        return feat, pos

    # -----------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,                       # (B, num_cam, 3, H, W)
        qpos: torch.Tensor,                        # (B, state_dim)
        actions: Optional[torch.Tensor] = None,    # (B, k, action_dim) train-only
        is_pad: Optional[torch.Tensor] = None,     # (B, k)             train-only
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = qpos.size(0)
        device = qpos.device

        if actions is not None:
            mu, logvar = self._encode_style(qpos, actions, is_pad)
            z = reparameterize(mu, logvar)
        else:
            mu = torch.zeros(B, self.latent_dim, device=device)
            logvar = torch.zeros(B, self.latent_dim, device=device)
            z = torch.zeros(B, self.latent_dim, device=device)

        img_feat, img_pos = self._encode_image(image)                      # (S_img, B, D)
        latent_tok = self.latent_to_src(z).unsqueeze(0)                    # (1, B, D)
        qpos_tok = self.qpos_to_src(qpos).unsqueeze(0)                     # (1, B, D)
        extra_pos = self.extra_src_pos.weight.unsqueeze(1).expand(-1, B, -1)  # (2, B, D)

        src = torch.cat([latent_tok, qpos_tok, img_feat], dim=0)           # (2+S_img, B, D)
        src_pos = torch.cat([extra_pos, img_pos], dim=0)

        hs = self.transformer(src, src_pos, self.query_embed.weight)       # (k, B, D)
        a_hat = self.action_head(hs.transpose(0, 1))                       # (B, k, action_dim)
        return a_hat, (mu, logvar)


def build_cvae(**kwargs) -> ACTCVAE:
    return ACTCVAE(**kwargs)
