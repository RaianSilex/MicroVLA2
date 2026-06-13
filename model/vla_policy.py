"""Policy wrapper for MicroVLA training and rollout.

Adds, around ``VLACVAE``:
  * masked, **per-axis-weighted** L1 on the action chunk (near-constant axes are
    down-weighted automatically from data; see ``action_weight_table``),
  * a **contact-point** Gaussian negative-log-likelihood on the predicted goal,
  * the CVAE KL term,
  * optional resistance conditioning,
  * raw-unit ``inference()`` that returns ABSOLUTE targets regardless of the
    train-time action space.
"""

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
from data.vocab import VocabBundle
from model.vla_cvae import VLACVAE


def _coerce_vocabs(vocabs) -> VocabBundle:
    if isinstance(vocabs, VocabBundle):
        return vocabs
    return VocabBundle(**vocabs)


def _lookup(vocab: Dict[str, int], value: str) -> int:
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))


class VLAPolicy(nn.Module):
    def __init__(
        self,
        stats: dict,
        vocabs,
        kl_weight: float = C.KL_WEIGHT,
        goal_weight: float = C.GOAL_LOSS_WEIGHT,
        action_space: str = C.DEFAULT_ACTION_SPACE,
        use_resistance: Optional[bool] = None,
        **vla_kwargs,
    ):
        super().__init__()
        self.vocabs = _coerce_vocabs(vocabs)
        self.kl_weight = float(kl_weight)
        self.goal_weight = float(goal_weight)
        # "delta": predictions are relative to the base state and added back at
        # inference so .inference() always returns ABSOLUTE targets (robot-native).
        # "absolute": predictions are absolute targets directly.
        self.action_space = str(action_space)

        # Auto-detect resistance support from the stats unless explicitly set.
        if use_resistance is None:
            use_resistance = bool(stats.get("has_resistance", False))
        self.use_resistance = bool(use_resistance)

        n_robots = len(self.vocabs.robot_ids)
        vla_kwargs.setdefault("num_robot_ids", n_robots)
        vla_kwargs.setdefault("num_lab_ids", len(self.vocabs.lab_ids))
        vla_kwargs.setdefault("num_embodiment_ids", len(self.vocabs.embodiment_ids))
        vla_kwargs.setdefault("num_action_type_ids", len(self.vocabs.action_type_ids))
        vla_kwargs.setdefault("num_task_family_ids", len(self.vocabs.task_family_ids))
        self.model = VLACVAE(use_resistance=self.use_resistance, **vla_kwargs)

        # Per-robot normalization tables.
        self.register_buffer("qpos_mean_table", torch.zeros(n_robots, C.MAX_STATE_DIM))
        self.register_buffer("qpos_std_table", torch.ones(n_robots, C.MAX_STATE_DIM))
        self.register_buffer("action_mean_table", torch.zeros(n_robots, C.MAX_ACTION_DIM))
        self.register_buffer("action_std_table", torch.ones(n_robots, C.MAX_ACTION_DIM))
        # Per-axis loss weights (1.0 = neutral; near-0 for axes that don't move).
        self.register_buffer("action_weight_table", torch.ones(n_robots, C.MAX_ACTION_DIM))
        # Resistance normalization (scalar per robot).
        self.register_buffer("resistance_mean_table", torch.zeros(n_robots, 1))
        self.register_buffer("resistance_std_table", torch.ones(n_robots, 1))

        by_robot = stats.get("by_robot", {})
        for robot_name, rid in self.vocabs.robot_ids.items():
            if robot_name == C.UNKNOWN_TOKEN or robot_name not in by_robot:
                continue
            rs = by_robot[robot_name]
            self.qpos_mean_table[rid] = torch.from_numpy(rs["qpos_mean"])
            self.qpos_std_table[rid] = torch.from_numpy(rs["qpos_std"])
            self.action_mean_table[rid] = torch.from_numpy(rs["action_mean"])
            self.action_std_table[rid] = torch.from_numpy(rs["action_std"])
            if "action_weight" in rs:
                self.action_weight_table[rid] = torch.from_numpy(rs["action_weight"])
            if "resistance_mean" in rs:
                self.resistance_mean_table[rid, 0] = float(rs["resistance_mean"])
                self.resistance_std_table[rid, 0] = float(rs["resistance_std"])

        self.register_buffer("image_mean", torch.from_numpy(stats["image_mean"]).view(3, 1, 1))
        self.register_buffer("image_std", torch.from_numpy(stats["image_std"]).view(3, 1, 1))

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
        goal: Optional[torch.Tensor] = None,
        resistance: Optional[torch.Tensor] = None,
        img_primary_feat: Optional[torch.Tensor] = None,
        img_aux_feat: Optional[torch.Tensor] = None,
    ):
        a_hat, goal_params, (mu, logvar) = self.model(
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
            resistance=resistance,
        )
        if actions is not None:
            if action_mask is None or is_pad is None:
                raise ValueError("action_mask and is_pad are required for training")
            axis_w = self.action_weight_table[robot_id]            # (B, MAX_ACTION_DIM)
            return self._compute_loss(
                a_hat, actions, is_pad, action_mask, axis_w, goal, goal_params, mu, logvar
            )
        return a_hat

    def _compute_loss(
        self,
        a_hat: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        action_mask: torch.Tensor,
        axis_w: torch.Tensor,
        goal: Optional[torch.Tensor],
        goal_params,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        # Per-axis-weighted, masked L1 over the action chunk.
        l1_unreduced = F.l1_loss(a_hat, actions, reduction="none")     # (B, k, A)
        valid = (~is_pad).unsqueeze(-1).float() * action_mask.unsqueeze(1).float()
        weight = valid * axis_w.unsqueeze(1)                           # (B, k, A)
        l1 = (l1_unreduced * weight).sum() / weight.sum().clamp_min(1.0)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        out = {"l1": l1.detach(), "kl": kl.detach()}
        total = l1 + self.kl_weight * kl

        if goal_params is not None and goal is not None:
            goal_mu, goal_logvar = goal_params
            var = goal_logvar.exp()
            # Gaussian NLL (constant 0.5*log(2*pi) dropped); variance is learned.
            nll = 0.5 * ((goal - goal_mu).pow(2) / var + goal_logvar)  # (B, A)
            gmask = action_mask.float() * axis_w                        # (B, A)
            goal_loss = (nll * gmask).sum() / gmask.sum().clamp_min(1.0)
            total = total + self.goal_weight * goal_loss
            out["goal"] = goal_loss.detach()

        out["loss"] = total
        return out

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
        resistance: Optional[float] = None,
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

        def _id(vocab, value):
            return torch.tensor([_lookup(vocab, value)], dtype=torch.long, device=device)

        resistance_t = None
        if self.use_resistance:
            r = 0.0 if resistance is None else float(resistance)
            r = (r - float(self.resistance_mean_table[rid, 0])) / float(self.resistance_std_table[rid, 0])
            resistance_t = torch.tensor([[r]], dtype=torch.float32, device=device)

        a_hat = self.forward(
            img,
            qpos_t,
            [instruction],
            _id(self.vocabs.robot_ids, robot_id),
            _id(self.vocabs.lab_ids, lab_id),
            _id(self.vocabs.embodiment_ids, embodiment),
            _id(self.vocabs.action_type_ids, action_type),
            _id(self.vocabs.task_family_ids, task_family),
            state_mask=state_mask_t,
            action_mask=action_mask_t,
            resistance=resistance_t,
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
    goal_weight: float = C.GOAL_LOSS_WEIGHT,
    action_space: str = C.DEFAULT_ACTION_SPACE,
    use_resistance: Optional[bool] = None,
    **vla_kwargs,
) -> VLAPolicy:
    if stats is None or vocabs is None:
        with open(stats_path, "rb") as f:
            payload = pickle.load(f)
        stats = payload["stats"] if stats is None else stats
        vocabs = payload["vocabs"] if vocabs is None else vocabs
    return VLAPolicy(
        stats=stats,
        vocabs=vocabs,
        kl_weight=kl_weight,
        goal_weight=goal_weight,
        action_space=action_space,
        use_resistance=use_resistance,
        **vla_kwargs,
    )
