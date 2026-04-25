"""Small training helpers: seeding, optimizer builder, checkpoint IO, meters."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    policy: torch.nn.Module,
    lr: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """AdamW with two param groups: backbone (lower LR) vs. everything else."""
    backbone_params = list(policy.model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in policy.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in other_params    if p.requires_grad], "lr": lr},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def save_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"policy": policy.state_dict(), "epoch": epoch}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> int:
    """Loads weights (and optimizer state if provided). Returns the saved epoch."""
    ckpt = torch.load(path, map_location=map_location)
    policy.load_state_dict(ckpt["policy"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch") or 0


class AverageMeter:
    """Running mean for scalar metrics."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0


def format_meters(meters: dict) -> str:
    return "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
