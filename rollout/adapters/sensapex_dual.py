"""Dual-Sensapex adapter for MicroVLA rollout.

Owns the robot-specific pieces the rig-agnostic policy must not know about:
observation acquisition, workspace clamping, per-tick step limits, and publishing
8-D absolute targets [x1, y1, z1, d1, x2, y2, z2, d2] to the ump_suite ROS topics.

=== Safety limits ===
Units are centered Sensapex counts, matching /ump/live and /ump2/live. EDIT
these for your workspace before commanding the motors.
"""

from __future__ import annotations

import numpy as np

from config import vla_config as C
from rollout.rollout import clamp
from rollout.sensapex_env import SensapexEnv


# Per-axis workspace bounds.
X1_MIN, X1_MAX = 17634, 18944
Y1_MIN, Y1_MAX = 17362, 18362
Z1_MIN, Z1_MAX = 14390, 14410
D1_MIN, D1_MAX = 15618, 15638

X2_MIN, X2_MAX = 10915, 12230
Y2_MIN, Y2_MAX = 10179, 11209
Z2_MIN, Z2_MAX = 18269, 18289
D2_MIN, D2_MAX = 12953, 12933

# Per-axis max single-tick movement (so far-future targets ramp in safely).
MAX_DX1 = MAX_DY1 = MAX_DZ1 = MAX_DD1 = 50.0
MAX_DX2 = MAX_DY2 = MAX_DZ2 = MAX_DD2 = 50.0


def clamp_action_8d(action_8d: np.ndarray) -> np.ndarray:
    """Clamp absolute action [x1,y1,z1,d1,x2,y2,z2,d2] to the safe box."""
    a = np.asarray(action_8d, dtype=np.float32).reshape(8,)
    return np.array(
        [
            clamp(a[0], X1_MIN, X1_MAX),
            clamp(a[1], Y1_MIN, Y1_MAX),
            clamp(a[2], Z1_MIN, Z1_MAX),
            clamp(a[3], D1_MIN, D1_MAX),
            clamp(a[4], X2_MIN, X2_MAX),
            clamp(a[5], Y2_MIN, Y2_MAX),
            clamp(a[6], Z2_MIN, Z2_MAX),
            clamp(a[7], D2_MIN, D2_MAX),
        ],
        dtype=np.float32,
    )


def limit_step(prev_state_8d: np.ndarray, target_action_8d: np.ndarray) -> np.ndarray:
    """Cap each axis' per-tick movement so far targets ramp in safely."""
    prev = np.asarray(prev_state_8d, dtype=np.float32).reshape(8,)
    tgt = np.asarray(target_action_8d, dtype=np.float32).reshape(8,)
    caps = (MAX_DX1, MAX_DY1, MAX_DZ1, MAX_DD1, MAX_DX2, MAX_DY2, MAX_DZ2, MAX_DD2)

    out = np.empty(8, dtype=np.float32)
    for i, cap in enumerate(caps):
        out[i] = prev[i] + clamp(tgt[i] - prev[i], -cap, cap)
    return out


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
        preview_path: str = "microvla_live.png",
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
