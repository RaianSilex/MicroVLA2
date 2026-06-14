"""Closed-loop rollout for a local MicroVLA checkpoint.

The VLA policy is robot-agnostic. Robot-specific state acquisition, safety, and
publishing are owned by adapters under rollout/adapters/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import vla_config as C
from model.vla_policy import build_vla_policy
from rollout.rollout import start_estop_listener


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a MicroVLA checkpoint through a robot adapter.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints_vla/vla_policy_best.pt"))
    p.add_argument("--adapter", choices=("sensapex_dual",), default="sensapex_dual")
    p.add_argument("--instruction", type=str, default="perform the cell manipulation task")
    p.add_argument("--backbone", type=str, default=None, help="Defaults to checkpoint config.")
    p.add_argument("--language-backend", choices=("hf", "simple"), default=None,
                   help="Defaults to checkpoint config.")
    p.add_argument("--text-model", type=str, default=None, help="Defaults to checkpoint config.")
    p.add_argument("--device", type=str, default=C.DEVICE)
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--max-timesteps", type=int, default=600)
    p.add_argument("--open-loop-horizon", type=int, default=C.OPEN_LOOP_HORIZON)
    p.add_argument("--control-hz", type=float, default=C.CONTROL_HZ)
    temporal_group = p.add_mutually_exclusive_group()
    temporal_group.add_argument("--temporal-agg", dest="temporal_agg", action="store_true")
    temporal_group.add_argument("--no-temporal-agg", dest="temporal_agg", action="store_false")
    p.set_defaults(temporal_agg=C.TEMPORAL_AGG)
    p.add_argument("--temporal-agg-k", type=float, default=C.TEMPORAL_AGG_K)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--default-speed", type=int, default=100)
    p.add_argument("--no-ema-smoothing", dest="use_ema_smoothing", action="store_false")
    p.add_argument("--ema-alpha", type=float, default=0.35)
    p.add_argument("--no-save-preview", dest="save_preview", action="store_false")
    p.add_argument("--preview-path", type=str, default="microvla_live.png")
    p.add_argument("--preview-every-n-frames", type=int, default=5)
    p.add_argument("--debug-every", type=int, default=10)
    p.add_argument("--lab-id", type=str, default=None)
    p.add_argument("--robot-id", type=str, default=None)
    p.add_argument("--embodiment", type=str, default=None)
    p.add_argument("--action-type", type=str, default=None)
    p.add_argument("--task-family", type=str, default=None)
    return p.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _get_adapter(args):
    if args.adapter == "sensapex_dual":
        from rollout.adapters.sensapex_dual import SensapexDualAdapter

        return SensapexDualAdapter(
            default_speed=args.default_speed,
            save_preview=args.save_preview,
            preview_path=args.preview_path,
            preview_every_n_frames=args.preview_every_n_frames,
        )
    raise ValueError(f"Unsupported adapter: {args.adapter}")


def _validate_action_chunk(chunk: np.ndarray, action_dim: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != action_dim:
        raise RuntimeError(f"Expected action chunk shape (T,{action_dim}), got {chunk.shape}")
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


def _fmt(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.0f}" for x in v.reshape(-1)) + "]"


def load_policy(args):
    checkpoint = _resolve_repo_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    backbone = args.backbone or ckpt_config.get("backbone", C.DEFAULT_BACKBONE)
    language_backend = args.language_backend or ckpt_config.get("language_backend", C.LANGUAGE_BACKEND)
    text_model = args.text_model or ckpt_config.get("text_model", C.DEFAULT_TEXT_MODEL)
    pretrained_backbone = args.pretrained_backbone or bool(
        ckpt_config.get("pretrained_backbone", False)
    )
    freeze_backbone = (
        not args.unfreeze_backbone
        if args.unfreeze_backbone
        else bool(ckpt_config.get("freeze_backbone", True))
    )
    action_space = ckpt_config.get("action_space", C.DEFAULT_ACTION_SPACE)
    policy = build_vla_policy(
        stats=ckpt["stats"],
        vocabs=ckpt["vocabs"],
        pretrained_backbone=pretrained_backbone,
        backbone_name=backbone,
        freeze_backbone=freeze_backbone,
        language_backend=language_backend,
        text_model_name=text_model,
        action_space=action_space,
        chunk_size=int(ckpt_config.get("chunk_size", C.CHUNK_SIZE)),
        goal_head=bool(ckpt_config.get("goal_head", C.GOAL_HEAD)),
        use_resistance=bool(ckpt_config.get("use_resistance", False)),
        cell_head=bool(ckpt_config.get("cell_head", False)),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(
        f"[microvla] loaded {checkpoint} "
        f"(epoch={ckpt.get('epoch')}, backbone={backbone}, language={language_backend}, "
        f"action_space={action_space}, device={device})"
    )
    return policy


def main() -> None:
    args = parse_args()
    if args.open_loop_horizon < 1:
        raise ValueError("--open-loop-horizon must be >= 1")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.temporal_agg_k < 0:
        raise ValueError("--temporal-agg-k must be >= 0")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in (0, 1]")

    policy = load_policy(args)
    adapter = _get_adapter(args)

    robot_id = args.robot_id or adapter.robot_id
    lab_id = args.lab_id or adapter.lab_id
    embodiment = args.embodiment or adapter.embodiment
    action_type = args.action_type or adapter.action_type
    task_family = args.task_family or adapter.task_family

    print("Running MicroVLA rollout...")
    print("  - Press Ctrl+C to stop early")
    print("  - Type 'q' + Enter to E-STOP and hold current position")
    print(f"  - instruction: {args.instruction!r}")
    if args.temporal_agg:
        print("  - Temporal aggregation enabled: re-inferring every tick")
    else:
        print(f"  - Open-loop chunks: consuming {args.open_loop_horizon} actions per inference")
    if args.dry_run:
        print("  - DRY RUN: commands will be printed but not published")

    stop_flag = start_estop_listener()
    period = 1.0 / float(args.control_hz)
    pred_action_chunk = None
    actions_completed_in_chunk = 0
    max_actions_from_current_chunk = 0
    chunk_history = []
    ema_action = None

    try:
        for t in range(int(args.max_timesteps)):
            start_time = time.time()

            if stop_flag["stop"]:
                print("[E-STOP] Holding current position and exiting.")
                if not args.dry_run:
                    adapter.hold_current()
                break

            obs = adapter.get_observation()
            img = obs.image_rgb
            state = obs.state.astype(np.float32)

            if args.temporal_agg:
                pred_action_chunk = _validate_action_chunk(
                    policy.inference(
                        img,
                        state,
                        args.instruction,
                        robot_id=robot_id,
                        lab_id=lab_id,
                        embodiment=embodiment,
                        action_type=action_type,
                        task_family=task_family,
                        state_dim=adapter.state_dim,
                        action_dim=adapter.action_dim,
                    ),
                    adapter.action_dim,
                )
                chunk_history.append((t, pred_action_chunk))
                chunk_history = [
                    (start_t, chunk)
                    for start_t, chunk in chunk_history
                    if 0 <= t - start_t < chunk.shape[0]
                ]
                action = _aggregate_temporal_action(chunk_history, t, args.temporal_agg_k)
            else:
                need_new_chunk = (
                    pred_action_chunk is None
                    or actions_completed_in_chunk >= max_actions_from_current_chunk
                )
                if need_new_chunk:
                    pred_action_chunk = _validate_action_chunk(
                        policy.inference(
                            img,
                            state,
                            args.instruction,
                            robot_id=robot_id,
                            lab_id=lab_id,
                            embodiment=embodiment,
                            action_type=action_type,
                            task_family=task_family,
                            state_dim=adapter.state_dim,
                            action_dim=adapter.action_dim,
                        ),
                        adapter.action_dim,
                    )
                    actions_completed_in_chunk = 0
                    max_actions_from_current_chunk = min(
                        int(args.open_loop_horizon), int(pred_action_chunk.shape[0])
                    )
                action = pred_action_chunk[actions_completed_in_chunk]
                actions_completed_in_chunk += 1

            safe = adapter.safe_command(state, action)
            if args.use_ema_smoothing:
                if ema_action is None:
                    ema_action = safe.copy()
                else:
                    ema_action = args.ema_alpha * safe + (1.0 - args.ema_alpha) * ema_action
                cmd = ema_action.astype(np.float32)
            else:
                cmd = safe

            if not args.dry_run:
                adapter.publish(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(f"[t={t:04d}] state={_fmt(state)} cmd={_fmt(cmd)}")

            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("Stopped early (Ctrl+C).")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
