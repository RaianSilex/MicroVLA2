"""Policy wrapper for MicroVLA training and rollout."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import vla_config as C
from data.vla_dataset import VocabBundle
from model.vla_cvae import VLACVAE


def _coerce_vocabs(vocabs) -> VocabBundle:
    if isinstance(vocabs, VocabBundle):
        return vocabs
    return VocabBundle(**vocabs)


def _lookup(vocab: Dict[str, int], value: str) -> int:
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))


class VLAPolicy(nn.Module):
    """Adds masked heterogeneous loss and raw-unit inference around VLACVAE."""

    def __init__(
        self,
        stats: dict,
        vocabs,
        kl_weight: float = C.KL_WEIGHT,
        action_space: str = C.DEFAULT_ACTION_SPACE,
        **vla_kwargs,
    ):
        super().__init__()
        self.vocabs = _coerce_vocabs(vocabs)
        self.kl_weight = float(kl_weight)
        # "delta": predictions are relative to the base state and added back at
        # inference so .inference() always returns ABSOLUTE targets (robot-native).
        # "absolute": predictions are absolute targets directly.
        self.action_space = str(action_space)

        vla_kwargs.setdefault("num_robot_ids", len(self.vocabs.robot_ids))
        vla_kwargs.setdefault("num_lab_ids", len(self.vocabs.lab_ids))
        vla_kwargs.setdefault("num_embodiment_ids", len(self.vocabs.embodiment_ids))
        vla_kwargs.setdefault("num_action_type_ids", len(self.vocabs.action_type_ids))
        vla_kwargs.setdefault("num_task_family_ids", len(self.vocabs.task_family_ids))
        self.model = VLACVAE(**vla_kwargs)

        self.register_buffer("qpos_mean_table", torch.zeros(len(self.vocabs.robot_ids), C.MAX_STATE_DIM))
        self.register_buffer("qpos_std_table", torch.ones(len(self.vocabs.robot_ids), C.MAX_STATE_DIM))
        self.register_buffer("action_mean_table", torch.zeros(len(self.vocabs.robot_ids), C.MAX_ACTION_DIM))
        self.register_buffer("action_std_table", torch.ones(len(self.vocabs.robot_ids), C.MAX_ACTION_DIM))
        for robot_name, rid in self.vocabs.robot_ids.items():
            if robot_name == C.UNKNOWN_TOKEN or robot_name not in stats["by_robot"]:
                continue
            robot_stats = stats["by_robot"][robot_name]
            self.qpos_mean_table[rid] = torch.from_numpy(robot_stats["qpos_mean"])
            self.qpos_std_table[rid] = torch.from_numpy(robot_stats["qpos_std"])
            self.action_mean_table[rid] = torch.from_numpy(robot_stats["action_mean"])
            self.action_std_table[rid] = torch.from_numpy(robot_stats["action_std"])

        self.register_buffer(
            "image_mean", torch.from_numpy(stats["image_mean"]).view(3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.from_numpy(stats["image_std"]).view(3, 1, 1)
        )

    def forward(
        self,
        image: torch.Tensor,
        qpos: torch.Tensor,
        instructions,
        robot_id: torch.Tensor,
        lab_id: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_type_id: torch.Tensor,
        task_family_id: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        img_primary_feat: Optional[torch.Tensor] = None,
        img_aux_feat: Optional[torch.Tensor] = None,
    ):
        if actions is not None:
            if action_mask is None or is_pad is None:
                raise ValueError("action_mask and is_pad are required for training")
            a_hat, (mu, logvar) = self.model(
                image,
                qpos,
                instructions,
                robot_id,
                lab_id,
                embodiment_id,
                action_type_id,
                task_family_id,
                state_mask=state_mask,
                action_mask=action_mask,
                actions=actions,
                is_pad=is_pad,
                img_primary_feat=img_primary_feat,
                img_aux_feat=img_aux_feat,
            )
            return self._compute_loss(a_hat, actions, is_pad, action_mask, mu, logvar)

        a_hat, _ = self.model(
            image,
            qpos,
            instructions,
            robot_id,
            lab_id,
            embodiment_id,
            action_type_id,
            task_family_id,
            state_mask=state_mask,
            action_mask=action_mask,
            img_primary_feat=img_primary_feat,
            img_aux_feat=img_aux_feat,
        )
        return a_hat

    def _compute_loss(
        self,
        a_hat: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        action_mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        l1_unreduced = F.l1_loss(a_hat, actions, reduction="none")
        valid = (~is_pad).unsqueeze(-1).float() * action_mask.unsqueeze(1).float()
        l1 = (l1_unreduced * valid).sum() / valid.sum().clamp_min(1.0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        total = l1 + self.kl_weight * kl
        return {"loss": total, "l1": l1.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def inference(
        self,
        image_np: np.ndarray,
        qpos_np: np.ndarray,
        instruction: str,
        robot_id: str = C.DEFAULT_ROBOT_ID,
        lab_id: str = C.DEFAULT_LAB_ID,
        embodiment: str = C.DEFAULT_EMBODIMENT,
        action_type: str = C.DEFAULT_ACTION_TYPE,
        task_family: str = C.DEFAULT_TASK_FAMILY,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> np.ndarray:
        self.eval()
        device = self.qpos_mean_table.device
        rid = _lookup(self.vocabs.robot_ids, robot_id)
        state_dim = int(state_dim if state_dim is not None else len(qpos_np))
        action_dim = int(action_dim if action_dim is not None else C.MAX_ACTION_DIM)

        img = self._preprocess_image(image_np).to(device).unsqueeze(0)
        qpos = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
        qpos[:state_dim] = np.asarray(qpos_np, dtype=np.float32).reshape(-1)[:state_dim]
        state_mask = np.zeros(C.MAX_STATE_DIM, dtype=bool)
        state_mask[:state_dim] = True
        action_mask = np.zeros(C.MAX_ACTION_DIM, dtype=bool)
        action_mask[:action_dim] = True

        qpos_raw_t = torch.from_numpy(qpos).to(device)            # absolute, padded
        qpos_t = ((qpos_raw_t - self.qpos_mean_table[rid]) / self.qpos_std_table[rid]).unsqueeze(0)
        state_mask_t = torch.from_numpy(state_mask).to(device).unsqueeze(0)
        action_mask_t = torch.from_numpy(action_mask).to(device).unsqueeze(0)

        robot_id_t = torch.tensor([rid], dtype=torch.long, device=device)
        lab_id_t = torch.tensor([_lookup(self.vocabs.lab_ids, lab_id)], dtype=torch.long, device=device)
        embodiment_id_t = torch.tensor(
            [_lookup(self.vocabs.embodiment_ids, embodiment)], dtype=torch.long, device=device
        )
        action_type_id_t = torch.tensor(
            [_lookup(self.vocabs.action_type_ids, action_type)], dtype=torch.long, device=device
        )
        task_family_id_t = torch.tensor(
            [_lookup(self.vocabs.task_family_ids, task_family)], dtype=torch.long, device=device
        )

        a_hat = self.forward(
            img,
            qpos_t,
            [instruction],
            robot_id_t,
            lab_id_t,
            embodiment_id_t,
            action_type_id_t,
            task_family_id_t,
            state_mask=state_mask_t,
            action_mask=action_mask_t,
        )
        a = a_hat[0] * self.action_std_table[rid] + self.action_mean_table[rid]
        if self.action_space == "delta":
            # delta[i] is relative to the current base state -> recover absolute.
            a[:, :action_dim] = a[:, :action_dim] + qpos_raw_t[:action_dim]
        return a[:, :action_dim].cpu().numpy().astype(np.float32)

    def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
        h, w = img_np.shape[:2]
        if (h, w) != (C.IMAGE_HEIGHT, C.IMAGE_WIDTH):
            pil = Image.fromarray(img_np).resize((C.IMAGE_WIDTH, C.IMAGE_HEIGHT), Image.BILINEAR)
            img_np = np.array(pil)
        x = torch.from_numpy(img_np).float() / 255.0
        x = x.permute(2, 0, 1).to(self.image_mean.device)
        x = (x - self.image_mean) / self.image_std
        return x.unsqueeze(0)


def build_vla_policy(
    stats: Optional[dict] = None,
    vocabs=None,
    stats_path: Path = C.VLA_STATS_PATH,
    kl_weight: float = C.KL_WEIGHT,
    action_space: str = C.DEFAULT_ACTION_SPACE,
    **vla_kwargs,
) -> VLAPolicy:
    if stats is None or vocabs is None:
        with open(stats_path, "rb") as f:
            payload = pickle.load(f)
        stats = payload["stats"] if stats is None else stats
        vocabs = payload["vocabs"] if vocabs is None else vocabs
    return VLAPolicy(
        stats=stats, vocabs=vocabs, kl_weight=kl_weight, action_space=action_space, **vla_kwargs
    )
