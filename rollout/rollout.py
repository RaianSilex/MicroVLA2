"""Shared helpers for MicroACT robot rollouts.

This module intentionally has no OpenPI/websocket dependency. MicroACT runs a
local PyTorch policy checkpoint and consumes predicted action chunks directly.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import signal
import sys
import threading
from pathlib import Path

from config import config as C


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a scalar, accepting bounds in either order."""
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return lower if v < lower else (upper if v > upper else float(v))


def start_estop_listener() -> dict:
    """Watch stdin for `q` + Enter and flip a flag the rollout loop polls."""
    flag = {"stop": False}

    def _worker() -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                continue
            if line.strip().lower() == "q":
                flag["stop"] = True
                break

    threading.Thread(target=_worker, daemon=True).start()
    return flag


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until a critical section exits."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


@dataclasses.dataclass
class RolloutArgs:
    """CLI arguments for local MicroACT checkpoint rollout."""

    # Policy checkpoint.
    checkpoint: Path = Path("checkpoints/policy_best.pt")
    stats_path: Path = Path("checkpoints/dataset_stats.pkl")
    backbone: str = "resnet18"
    device: str = "cuda"
    pretrained_backbone: bool = False
    unfreeze_backbone: bool = False

    # Rollout loop.
    max_timesteps: int = 600
    open_loop_horizon: int = C.OPEN_LOOP_HORIZON
    control_hz: float = C.CONTROL_HZ
    temporal_agg: bool = C.TEMPORAL_AGG
    temporal_agg_k: float = C.TEMPORAL_AGG_K
    dry_run: bool = False

    # Robot params.
    default_speed: int = 100

    # Optional first-order smoothing on commanded actions.
    use_ema_smoothing: bool = True
    ema_alpha: float = 0.35

    # Live preview file, useful over SSH.
    save_preview: bool = True
    preview_path: str = "microact_live.png"
    preview_every_n_frames: int = 5

    # Print one debug line every N steps; 0 disables.
    debug_every: int = 10


def parse_args() -> RolloutArgs:
    p = argparse.ArgumentParser(description="Run a MicroACT checkpoint on the Sensapex rig.")
    p.add_argument("--checkpoint", type=Path, default=RolloutArgs.checkpoint)
    p.add_argument("--stats-path", type=Path, default=RolloutArgs.stats_path)
    p.add_argument("--backbone", type=str, default=RolloutArgs.backbone)
    p.add_argument("--device", type=str, default=RolloutArgs.device)
    p.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize ResNet with ImageNet weights before loading checkpoint.",
    )
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Build normally frozen DINOv2/Cellpose backbones as trainable.",
    )
    p.add_argument("--max-timesteps", type=int, default=RolloutArgs.max_timesteps)
    p.add_argument("--open-loop-horizon", type=int, default=RolloutArgs.open_loop_horizon)
    p.add_argument("--control-hz", type=float, default=RolloutArgs.control_hz)
    temporal_group = p.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--temporal-agg",
        dest="temporal_agg",
        action="store_true",
        help="Enable ACT-style temporal aggregation.",
    )
    temporal_group.add_argument(
        "--no-temporal-agg",
        dest="temporal_agg",
        action="store_false",
        help="Disable ACT-style temporal aggregation and use open-loop chunks instead.",
    )
    p.set_defaults(temporal_agg=RolloutArgs.temporal_agg)
    p.add_argument("--temporal-agg-k", type=float, default=RolloutArgs.temporal_agg_k)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--default-speed", type=int, default=RolloutArgs.default_speed)
    p.add_argument("--no-ema-smoothing", dest="use_ema_smoothing", action="store_false")
    p.add_argument("--ema-alpha", type=float, default=RolloutArgs.ema_alpha)
    p.add_argument("--no-save-preview", dest="save_preview", action="store_false")
    p.add_argument("--preview-path", type=str, default=RolloutArgs.preview_path)
    p.add_argument(
        "--preview-every-n-frames",
        type=int,
        default=RolloutArgs.preview_every_n_frames,
    )
    p.add_argument("--debug-every", type=int, default=RolloutArgs.debug_every)
    ns = p.parse_args()
    return RolloutArgs(**vars(ns))
