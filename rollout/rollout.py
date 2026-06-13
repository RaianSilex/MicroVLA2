"""Shared helpers for MicroVLA robot rollouts.

MicroVLA runs a local PyTorch policy checkpoint and consumes predicted action
chunks directly (no websocket / server). These helpers are robot-agnostic; the
robot-specific logic lives in ``rollout/adapters/``.
"""

from __future__ import annotations

import contextlib
import signal
import sys
import threading


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
