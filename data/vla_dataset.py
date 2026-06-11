"""Shared MicroVLA data classes.

MicroVLA trains from **LeRobot** datasets — see ``data/lerobot_vla_dataset.py``
and ``dataset_vla/convert_microact_to_lerobot.py`` (v3) /
``convert_microact_to_lerobot_v21.py`` (v2.1). The legacy
``dataset_vla/episodes`` loader was removed; only the two metadata dataclasses
below remain, because they are shared by the model (``model/vla_policy.py``),
the LeRobot loader, and the finetune helpers (``model/finetune.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


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


@dataclass
class VLAEpisode:
    episode_dir: Path
    episode_id: str
    lab_id: str
    robot_id: str
    embodiment: str
    action_type: str
    task_family: str
    instruction: str
    camera_names: List[str]
    state_cols: List[str]
    action_cols: List[str]
    image_col: str
    timestep_col: str
    state_dim: int
    action_dim: int
    states: np.ndarray
    actions: np.ndarray
    image_paths: List[str]
    length: int
