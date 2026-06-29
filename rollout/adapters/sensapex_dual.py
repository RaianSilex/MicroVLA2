"""Sensapex adapter for MicroVLA rollout (1 or 2 manipulators).

Owns the robot-specific pieces the rig-agnostic policy must not know about:
observation acquisition, workspace clamping, per-tick step limits, and publishing
absolute targets to the ump_suite ROS topics.

The number of manipulators follows ``config.NUM_MANIPULATORS`` (override per run
with ``SensapexDualAdapter(num_manipulators=...)``):

    1 -> single pipette, state/action = [x, y, z, d]            (/ump only)
    2 -> dual pipette,   state/action = [x1,y1,z1,d1, x2,..,d2] (/ump + /ump2)

=== Safety limits ===
Units are centered Sensapex counts, matching /ump/live (and /ump2/live for dual).
EDIT the per-axis bounds for your workspace before commanding the motors.
"""

from __future__ import annotations

import numpy as np

from config import vla_config as C
from rollout.rollout import clamp

# Per-axis workspace bounds, ordered [x1, y1, z1, d1, x2, y2, z2, d2].
# Single-manipulator rollout uses only the first AXES_PER_MANIPULATOR rows.
_WORKSPACE_BOUNDS = (
    (17634, 18944),  # x1
    (17362, 18362),  # y1
    (14390, 14410),  # z1
    (15618, 15638),  # d1
    (10915, 12230),  # x2
    (10179, 11209),  # y2
    (18269, 18289),  # z2
    (12953, 12933),  # d2
)
# Per-axis max single-tick movement (so far-future targets ramp in safely).
_STEP_CAPS = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0)


def clamp_action(action: np.ndarray, n_dims: int) -> np.ndarray:
    """Clamp an absolute action to the per-axis safe box (first ``n_dims`` axes)."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)[:n_dims]
    out = np.empty(n_dims, dtype=np.float32)
    for i in range(n_dims):
        lo, hi = _WORKSPACE_BOUNDS[i]
        out[i] = clamp(a[i], lo, hi)
    return out


def limit_step(prev_state: np.ndarray, target_action: np.ndarray, n_dims: int) -> np.ndarray:
    """Cap each axis' per-tick movement so far targets ramp in safely."""
    prev = np.asarray(prev_state, dtype=np.float32).reshape(-1)[:n_dims]
    tgt = np.asarray(target_action, dtype=np.float32).reshape(-1)[:n_dims]
    out = np.empty(n_dims, dtype=np.float32)
    for i in range(n_dims):
        out[i] = prev[i] + clamp(tgt[i] - prev[i], -_STEP_CAPS[i], _STEP_CAPS[i])
    return out


class SensapexDualAdapter:
    robot_id = C.DEFAULT_ROBOT_ID
    lab_id = C.DEFAULT_LAB_ID
    embodiment = C.DEFAULT_EMBODIMENT
    action_type = C.DEFAULT_ACTION_TYPE
    task_family = C.DEFAULT_TASK_FAMILY

    def __init__(
        self,
        *,
        num_manipulators: int = C.NUM_MANIPULATORS,
        default_speed: int = 100,
        save_preview: bool = True,
        preview_path: str = "microvla_live.png",
        preview_every_n_frames: int = 5,
        resistance_topic: str | None = None,
        resistance_type: str = "float32",
    ):
        self.num_manipulators = int(num_manipulators)
        self.state_dim = self.num_manipulators * C.AXES_PER_MANIPULATOR
        self.action_dim = self.state_dim
        # Lazy import so the safety math above is importable without ROS (rclpy).
        from rollout.sensapex_env import SensapexEnv

        self.env = SensapexEnv(
            num_manipulators=self.num_manipulators,
            default_speed=default_speed,
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
            resistance_topic=resistance_topic,
            resistance_type=resistance_type,
        )

    def get_observation(self):
        return self.env.get_observation()

    def safe_command(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        clamped = clamp_action(action, self.action_dim)
        return limit_step(state, clamped, self.action_dim)

    def publish(self, command: np.ndarray) -> None:
        self.env.step_absolute(command)

    def hold_current(self) -> None:
        obs = self.get_observation()
        self.publish(obs.state.astype(np.float32).copy())

    def close(self) -> None:
        self.env.close()
