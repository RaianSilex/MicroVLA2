"""Dual-Sensapex adapter for MicroVLA rollout."""

from __future__ import annotations

import numpy as np

from config import vla_config as C
from rollout.main import clamp_action_8d, limit_step
from rollout.sensapex_env import SensapexEnv


class SensapexDualAdapter:
    robot_id = C.DEFAULT_ROBOT_ID
    lab_id = C.DEFAULT_LAB_ID
    embodiment = C.DEFAULT_EMBODIMENT
    action_type = C.DEFAULT_ACTION_TYPE
    task_family = C.DEFAULT_TASK_FAMILY
    state_dim = 8
    action_dim = 8

    def __init__(
        self,
        *,
        default_speed: int = 100,
        save_preview: bool = True,
        preview_path: str = "microact_vla_live.png",
        preview_every_n_frames: int = 5,
    ):
        self.env = SensapexEnv(
            default_speed=default_speed,
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
        )

    def get_observation(self):
        return self.env.get_observation()

    def safe_command(self, state_8d: np.ndarray, action_8d: np.ndarray) -> np.ndarray:
        action = clamp_action_8d(action_8d)
        return limit_step(state_8d, action)

    def publish(self, command_8d: np.ndarray) -> None:
        self.env.step_absolute(command_8d)

    def hold_current(self) -> None:
        obs = self.get_observation()
        self.publish(obs.state.astype(np.float32).copy())

    def close(self) -> None:
        self.env.close()
