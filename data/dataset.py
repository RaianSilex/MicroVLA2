"""CSV + image dataset loader for MicroACT.

Each trial_N.csv under DATASET_ROOT/logs has one row per timestep with the
columns declared in config.config. Images live at
DATASET_ROOT/saved_frames/trial_N/frame_NNNNNN.png (path is also stored in
the `image_path` CSV column).

Each sample emitted here is:
    image  : (num_cameras, 3, H, W)  float32, ImageNet-normalized
    qpos   : (STATE_DIM,)            float32, dataset-normalized
    action : (CHUNK_SIZE, ACTION_DIM) float32, dataset-normalized, zero-padded
    is_pad : (CHUNK_SIZE,)           bool,  True where action was padded
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import config as C


class TrialData(NamedTuple):
    trial_id: int
    states: np.ndarray           # (T, STATE_DIM)
    actions: np.ndarray          # (T, ACTION_DIM)
    image_paths: List[str]       # length T; '' means no path recorded
    length: int


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_trials(logs_dir: Path = C.LOGS_DIR) -> List[Path]:
    files = sorted(
        logs_dir.glob("trial_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No trial_*.csv found under {logs_dir}")
    return files


def load_trial(csv_path: Path) -> TrialData:
    df = pd.read_csv(csv_path)
    trial_id = int(csv_path.stem.split("_")[-1])

    missing = [c for c in (*C.CSV_STATE_COLS, *C.CSV_ACTION_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    states = df[list(C.CSV_STATE_COLS)].to_numpy(dtype=np.float32)
    actions = df[list(C.CSV_ACTION_COLS)].to_numpy(dtype=np.float32)
    raw_paths = (
        df[C.CSV_IMAGE_COL].fillna("").astype(str).tolist()
        if C.CSV_IMAGE_COL in df.columns
        else [""] * len(df)
    )
    return TrialData(trial_id, states, actions, raw_paths, length=len(df))


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def _resolve_image_path(raw: str, trial_id: int, t: int) -> Optional[Path]:
    """Try a few conventions to find the frame file. Return None if unresolved."""
    # Conventional fallback: saved_frames/trial_N/frame_NNNNNN.png
    fallback = C.FRAMES_DIR / f"trial_{trial_id}" / f"frame_{t:06d}.png"

    raw = (raw or "").strip()
    if not raw:
        return fallback if fallback.exists() else None

    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (C.REPO_ROOT, C.DATASET_ROOT, C.FRAMES_DIR):
        q = base / p
        if q.exists():
            return q
    return fallback if fallback.exists() else None


def _load_image(path: Optional[Path], h: int, w: int) -> np.ndarray:
    if path is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def compute_norm_stats(trials: List[TrialData]) -> dict:
    all_states = np.concatenate([t.states for t in trials], axis=0)
    all_actions = np.concatenate([t.actions for t in trials], axis=0)

    # Clip std so constant dims don't divide by zero.
    state_std = np.clip(all_states.std(0), 1e-2, None)
    action_std = np.clip(all_actions.std(0), 1e-2, None)

    return {
        "qpos_mean":   all_states.mean(0).astype(np.float32),
        "qpos_std":    state_std.astype(np.float32),
        "action_mean": all_actions.mean(0).astype(np.float32),
        "action_std":  action_std.astype(np.float32),
        "image_mean":  np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "image_std":   np.array([0.229, 0.224, 0.225], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EpisodicDataset(Dataset):
    """One sample per (trial, start_t). Actions past the trial end are zero-padded."""

    def __init__(
        self,
        trials: List[TrialData],
        norm_stats: dict,
        chunk_size: int = C.CHUNK_SIZE,
        image_hw: tuple = (C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
    ):
        self.trials = trials
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.image_h, self.image_w = image_hw
        self.index = [
            (ti, t)
            for ti, tr in enumerate(trials)
            for t in range(tr.length)
        ]
        self._warned_missing_image = False

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        trial_idx, t = self.index[i]
        trial = self.trials[trial_idx]

        # ---- State at t ----
        qpos = trial.states[t]

        # ---- Future action chunk, zero-padded ----
        end = min(t + self.chunk_size, trial.length)
        avail = end - t
        action = np.zeros((self.chunk_size, C.ACTION_DIM), dtype=np.float32)
        action[:avail] = trial.actions[t:end]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[avail:] = True

        # ---- Image at t ----
        raw = trial.image_paths[t] if t < len(trial.image_paths) else ""
        path = _resolve_image_path(raw, trial.trial_id, t)
        if path is None and not self._warned_missing_image:
            warnings.warn(
                f"Trial {trial.trial_id}: image at t={t} unresolved "
                f"(csv='{raw}') — returning zeros. Further warnings suppressed.",
                stacklevel=2,
            )
            self._warned_missing_image = True
        img = _load_image(path, self.image_h, self.image_w)

        # ---- Normalize ----
        img = img.astype(np.float32) / 255.0
        img = (img - self.norm_stats["image_mean"]) / self.norm_stats["image_std"]
        img = np.transpose(img, (2, 0, 1))         # HWC -> CHW
        img = img[None]                            # (num_cam=1, C, H, W)

        qpos_n = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_n = (action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_n[is_pad] = 0.0                     # keep padded positions clean

        return {
            "image":  torch.from_numpy(img).float(),
            "qpos":   torch.from_numpy(qpos_n).float(),
            "action": torch.from_numpy(action_n).float(),
            "is_pad": torch.from_numpy(is_pad),
        }


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_dataset(
    logs_dir: Path = C.LOGS_DIR,
    stats_path: Path = C.STATS_PATH,
    recompute_stats: bool = False,
) -> EpisodicDataset:
    csv_paths = discover_trials(logs_dir)
    trials = [load_trial(p) for p in csv_paths]

    if stats_path.exists() and not recompute_stats:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = compute_norm_stats(trials)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

    return EpisodicDataset(trials, stats)
