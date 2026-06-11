"""Closed-loop rollout for a local MicroACT policy checkpoint.

Pipeline per control tick:
    ROS obs -> ACTPolicy.inference(image_rgb, state_8d) -> action chunk
            -> clamp_action_8d -> limit_step -> optional EMA -> ROS targets

Run from the MicroACT repo root after sourcing ROS:
    python3 -m rollout.main --checkpoint checkpoints/policy_best.pt

The policy is local PyTorch, not an OpenPI websocket server. The state/action
vector is 8-D:
    [x1, y1, z1, d1, x2, y2, z2, d2]
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow `python rollout/main.py` as well as `python -m rollout.main`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config as C
from model.act_policy import build_policy
from utils import load_checkpoint

try:
    from .rollout import RolloutArgs, clamp, parse_args, start_estop_listener
except ImportError:  # pragma: no cover - direct script execution fallback
    from rollout.rollout import RolloutArgs, clamp, parse_args, start_estop_listener


# === Safety limits ===
# Units are centered Sensapex counts, matching /ump/live and /ump2/live.
# Edit these before running on a different workspace.

X1_MIN, X1_MAX = 17634, 18944
Y1_MIN, Y1_MAX = 17362, 18362
Z1_MIN, Z1_MAX = 14390, 14410
D1_MIN, D1_MAX = 15618, 15638

X2_MIN, X2_MAX = 10915, 12230
Y2_MIN, Y2_MAX = 10179, 11209
Z2_MIN, Z2_MAX = 18269, 18289
D2_MIN, D2_MAX = 12953, 12933

MAX_DX1 = 50.0
MAX_DY1 = 50.0
MAX_DZ1 = 50.0
MAX_DD1 = 50.0
MAX_DX2 = 50.0
MAX_DY2 = 50.0
MAX_DZ2 = 50.0
MAX_DD2 = 50.0


def _resolve_repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _stats_from_checkpoint(checkpoint: Path) -> dict:
    """Recover normalization stats from policy buffers if dataset_stats.pkl is absent."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["policy"]
    return {
        "qpos_mean": state["qpos_mean"].cpu().numpy(),
        "qpos_std": state["qpos_std"].cpu().numpy(),
        "action_mean": state["action_mean"].cpu().numpy(),
        "action_std": state["action_std"].cpu().numpy(),
        "image_mean": state["image_mean"].view(3).cpu().numpy(),
        "image_std": state["image_std"].view(3).cpu().numpy(),
    }


def _load_stats(stats_path: Path, checkpoint: Path) -> dict:
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            return pickle.load(f)
    print(f"[warn] stats file not found at {stats_path}; using checkpoint buffers")
    return _stats_from_checkpoint(checkpoint)


def load_microact_policy(args: RolloutArgs):
    checkpoint = _resolve_repo_path(args.checkpoint)
    stats_path = _resolve_repo_path(args.stats_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    stats = _load_stats(stats_path, checkpoint)
    policy = build_policy(
        stats=stats,
        pretrained_backbone=args.pretrained_backbone,
        backbone_name=args.backbone,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    epoch = load_checkpoint(checkpoint, policy, map_location=device)
    policy.eval()
    print(
        f"[microact] loaded {checkpoint} "
        f"(epoch={epoch}, backbone={args.backbone}, device={device})"
    )
    return policy


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


def clamp_action_4d(action_4d: np.ndarray) -> np.ndarray:
    """Clamp absolute uMp1 action [x1,y1,z1,d1] to the safe box."""
    a = np.asarray(action_4d, dtype=np.float32).reshape(4,)
    return np.array(
        [
            clamp(a[0], X1_MIN, X1_MAX),
            clamp(a[1], Y1_MIN, Y1_MAX),
            clamp(a[2], Z1_MIN, Z1_MAX),
            clamp(a[3], D1_MIN, D1_MAX),
        ],
        dtype=np.float32,
    )


def limit_step(prev_state: np.ndarray, target_action: np.ndarray) -> np.ndarray:
    """Cap each axis' per-tick movement. Works for 4-D (uMp1) or 8-D (dual)."""
    prev = np.asarray(prev_state, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_action, dtype=np.float32).reshape(-1)
    caps = (MAX_DX1, MAX_DY1, MAX_DZ1, MAX_DD1, MAX_DX2, MAX_DY2, MAX_DZ2, MAX_DD2)
    n = prev.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = prev[i] + clamp(tgt[i] - prev[i], -caps[i], caps[i])
    return out


def _fmt8(v: np.ndarray) -> str:
    return (
        f"[{v[0]:.0f},{v[1]:.0f},{v[2]:.0f},{v[3]:.0f}|"
        f"{v[4]:.0f},{v[5]:.0f},{v[6]:.0f},{v[7]:.0f}]"
    )


def _validate_action_chunk(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != C.ACTION_DIM:
        raise RuntimeError(
            f"Expected action chunk shape (T,{C.ACTION_DIM}), got {chunk.shape}"
        )
    return chunk


def _aggregate_temporal_action(chunk_history, t: int, k: float) -> np.ndarray:
    actions = []
    ages = []
    for start_t, chunk in chunk_history:
        age = t - start_t
        if 0 <= age < chunk.shape[0]:
            actions.append(chunk[age])
            ages.append(age)

    if not actions:
        raise RuntimeError("Temporal aggregation has no valid action for this tick")

    weights = np.exp(-float(k) * np.asarray(ages, dtype=np.float32))
    weights = weights / weights.sum()
    return (np.stack(actions, axis=0) * weights[:, None]).sum(axis=0).astype(np.float32)


def _get_env_cls():
    try:
        from .sensapex_env import SensapexEnv
    except ImportError:  # pragma: no cover - direct script execution fallback
        from rollout.sensapex_env import SensapexEnv
    return SensapexEnv


def main(args: RolloutArgs) -> None:
    if args.open_loop_horizon < 1:
        raise ValueError("--open-loop-horizon must be >= 1")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.temporal_agg_k < 0:
        raise ValueError("--temporal-agg-k must be >= 0")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1]")

    policy = load_microact_policy(args)
    SensapexEnv = _get_env_cls()
    env = SensapexEnv(
        save_preview=args.save_preview,
        preview_path=args.preview_path,
        preview_every_n_frames=args.preview_every_n_frames,
        default_speed=args.default_speed,
    )
    if args.save_preview:
        print(f"[sensapex] live preview will be saved to: {args.preview_path}")

    print("Running MicroACT rollout...")
    print("  - Press Ctrl+C to stop early")
    print("  - Type 'q' + Enter to E-STOP and hold current position")
    if args.temporal_agg:
        print("  - Temporal aggregation enabled: re-inferring every tick")
    else:
        print(f"  - Open-loop chunks: consuming {args.open_loop_horizon} actions per inference")
    if args.dry_run:
        print("  - DRY RUN: commands will be printed but not published")

    stop_flag = start_estop_listener()
    period = 1.0 / float(args.control_hz)
    actions_completed_in_chunk = 0
    max_actions_from_current_chunk = 0
    pred_action_chunk = None
    chunk_history = []
    ema_action = None

    try:
        for t in range(int(args.max_timesteps)):
            start_time = time.time()

            if stop_flag["stop"]:
                obs = env.get_observation()
                hold = obs.state.astype(np.float32).copy()
                print("[E-STOP] Holding current position and exiting.")
                if not args.dry_run:
                    env.step_absolute(hold)
                break

            obs = env.get_observation()
            img = obs.image_rgb
            state = obs.state.astype(np.float32)

            if args.temporal_agg:
                pred_action_chunk = _validate_action_chunk(policy.inference(img, state))
                chunk_history.append((t, pred_action_chunk))
                chunk_history = [
                    (start_t, chunk)
                    for start_t, chunk in chunk_history
                    if 0 <= t - start_t < chunk.shape[0]
                ]
                action = _aggregate_temporal_action(
                    chunk_history, t, args.temporal_agg_k
                )
            else:
                need_new_chunk = (
                    pred_action_chunk is None
                    or actions_completed_in_chunk >= max_actions_from_current_chunk
                )
                if need_new_chunk:
                    pred_action_chunk = _validate_action_chunk(policy.inference(img, state))
                    actions_completed_in_chunk = 0
                    max_actions_from_current_chunk = min(
                        int(args.open_loop_horizon), int(pred_action_chunk.shape[0])
                    )

                action = pred_action_chunk[actions_completed_in_chunk]
                actions_completed_in_chunk += 1

            action = clamp_action_8d(action)
            action = limit_step(state, action)

            if args.use_ema_smoothing:
                if ema_action is None:
                    ema_action = action.copy()
                else:
                    ema_action = args.ema_alpha * action + (1.0 - args.ema_alpha) * ema_action
                cmd = ema_action.astype(np.float32)
            else:
                cmd = action

            if not args.dry_run:
                env.step_absolute(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(f"[t={t:04d}] state={_fmt8(state)} cmd={_fmt8(cmd)}")

            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("Stopped early (Ctrl+C).")
    finally:
        env.close()


def main_entry() -> None:
    main(parse_args())


if __name__ == "__main__":
    main_entry()
