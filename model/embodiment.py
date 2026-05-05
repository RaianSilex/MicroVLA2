"""Learned robot/lab/task/embodiment conditioning tokens for MicroVLA."""

from __future__ import annotations

import torch
import torch.nn as nn

from config import vla_config as C


class EmbodimentConditioner(nn.Module):
    """Turns categorical dataset metadata into source tokens.

    The output is sequence-first to match the DETR-style transformer stack:
    `(num_metadata_tokens, batch, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim: int = C.HIDDEN_DIM,
        num_robot_ids: int = C.NUM_ROBOT_IDS_FALLBACK,
        num_lab_ids: int = C.NUM_LAB_IDS_FALLBACK,
        num_embodiment_ids: int = C.NUM_EMBODIMENT_IDS_FALLBACK,
        num_action_type_ids: int = C.NUM_ACTION_TYPE_IDS_FALLBACK,
        num_task_family_ids: int = C.NUM_TASK_FAMILY_IDS_FALLBACK,
    ):
        super().__init__()
        self.robot_embed = nn.Embedding(num_robot_ids, hidden_dim)
        self.lab_embed = nn.Embedding(num_lab_ids, hidden_dim)
        self.embodiment_embed = nn.Embedding(num_embodiment_ids, hidden_dim)
        self.action_type_embed = nn.Embedding(num_action_type_ids, hidden_dim)
        self.task_family_embed = nn.Embedding(num_task_family_ids, hidden_dim)
        self.num_tokens = 5

    def forward(
        self,
        robot_id: torch.Tensor,
        lab_id: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_type_id: torch.Tensor,
        task_family_id: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.robot_embed(robot_id),
                self.lab_embed(lab_id),
                self.embodiment_embed(embodiment_id),
                self.action_type_embed(action_type_id),
                self.task_family_embed(task_family_id),
            ],
            dim=0,
        )
        return tokens
