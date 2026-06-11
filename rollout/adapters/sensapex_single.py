"""Single-Sensapex (uMp1, 4-DoF) adapter for MicroVLA rollout.

State / action vector (4-D):  [x1, y1, z1, d1]  (uMp1 only)
Mirrors SensapexDualAdapter but reads/commands only uMp1.
"""

from __future__ import annotations

import numpy as np

from config import vla_config as C
from rollout.main import clamp_action_4d, limit_step
from rollout.sensapex_env import SensapexEnv


class SensapexSingleAdapter:
    robot_id = "sensapex_single_ump4"   # MUST match the --robot-type used at dataset conversion
    lab_id = C.DEFAULT_LAB_ID
    embodiment = C.DEFAULT_EMBODIMENT
    action_type = C.DEFAULT_ACTION_TYPE
    task_family = C.DEFAULT_TASK_FAMILY
    state_dim = 4
    action_dim = 4

    def __init__(
        self,
        *,
        default_speed: int = 100,
        save_preview: bool = True,
        preview_path: str = "microact_vla_live.png",
        preview_every_n_frames: int = 5,
    ):
        self.env = SensapexEnv(
            n_ump=1,
            default_speed=default_speed,
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
        )

    def get_observation(self):
        return self.env.get_observation()

    def safe_command(self, state_4d: np.ndarray, action_4d: np.ndarray) -> np.ndarray:
        return limit_step(state_4d, clamp_action_4d(action_4d))

    def publish(self, command_4d: np.ndarray) -> None:
        self.env.step_absolute(command_4d)

    def hold_current(self) -> None:
        obs = self.get_observation()
        self.publish(obs.state.astype(np.float32).copy())

    def close(self) -> None:
        self.env.close()
