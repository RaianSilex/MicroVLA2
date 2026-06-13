"""Categorical metadata vocabularies for MicroVLA.

A ``VocabBundle`` maps each dataset metadata field (robot / lab / embodiment /
action-type / task-family) to integer ids. It is saved inside every checkpoint so
inference and finetuning resolve the same ids the model was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VocabBundle:
    robot_ids: Dict[str, int]
    lab_ids: Dict[str, int]
    embodiment_ids: Dict[str, int]
    action_type_ids: Dict[str, int]
    task_family_ids: Dict[str, int]

    def as_dict(self) -> dict:
        return {
            "robot_ids": self.robot_ids,
            "lab_ids": self.lab_ids,
            "embodiment_ids": self.embodiment_ids,
            "action_type_ids": self.action_type_ids,
            "task_family_ids": self.task_family_ids,
        }
