"""Cellpose teacher: generate per-trial contact-point labels (Variant B).

This is the ONLY place Cellpose is used in the cell-aware path. It runs OFFLINE,
once, over the raw frames to produce a small ``cell_labels.csv``; the deployed
policy never runs Cellpose. The downstream pieces are:

    raw frames ──(this script: Cellpose-SAM segmentation)──► cell_labels.csv
              ──(convert_microact_to_lerobot.py --cell-labels)──► observation.goal_pixel
              ──(train_vla.py)──► cell-selection + image-space contact-point heads

For each trial it segments the **contact frame** (the last frame, where the tip
is on the target cell), takes the detected cell centroids, and picks the centroid
nearest the trial's labeled region center (from ``instruction_labels.csv``). That
gives the target cell's contact point in NORMALIZED pixels ``(u, v)`` in [0, 1].
The region label disambiguates WHICH cell; Cellpose refines it to a precise point.
If Cellpose finds no cells, the region center itself is used as a fallback.

Output ``cell_labels.csv`` columns: ``trial_id, goal_u, goal_v, region, n_cells``.

Usage:
    python dataset_vla/generate_cell_labels.py                      # all trials
    python dataset_vla/generate_cell_labels.py --limit-trials 3     # quick subset
    python dataset_vla/generate_cell_labels.py --frame last --gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import vla_config as C
from data.cell_grid import region_index_center
from dataset_vla.convert_microact_to_lerobot import (
    IMAGE_COL,
    REGIONS,
    _load_rgb_uint8,
    _resolve_image_path,
    _trial_idx,
    load_or_scaffold_labels,
    resolve_instruction,
)


# ---------------------------------------------------------------------------
# Pure helpers (no Cellpose) — unit-tested by the offline smoke test
# ---------------------------------------------------------------------------

def mask_centroids(masks: np.ndarray) -> List[Tuple[float, float]]:
    """Label image -> list of (u, v) cell centroids in normalized [0, 1] pixels.

    ``masks`` is the integer label map Cellpose returns (0 = background). ``u`` is
    horizontal (column / width), ``v`` is vertical (row / height).
    """
    masks = np.asarray(masks)
    h, w = masks.shape[:2]
    out: List[Tuple[float, float]] = []
    for cell_id in np.unique(masks):
        if int(cell_id) == 0:
            continue
        ys, xs = np.where(masks == cell_id)
        if xs.size == 0:
            continue
        out.append((float(xs.mean()) / max(w, 1), float(ys.mean()) / max(h, 1)))
    return out


def region_center_for(region: str) -> Tuple[float, float]:
    """Canonical region name -> its grid-cell center in normalized (u, v)."""
    try:
        idx = REGIONS.index(region)
    except ValueError:
        idx = REGIONS.index("center")
    return region_index_center(idx)


def pick_target_cell(
    centroids: Sequence[Tuple[float, float]],
    region_center: Tuple[float, float],
) -> Tuple[float, float]:
    """The contact point: the detected centroid nearest the labeled region center.

    Falls back to the region center itself when no cells were detected (so the
    label is always defined and at least region-accurate).
    """
    cu, cv = float(region_center[0]), float(region_center[1])
    if not centroids:
        return (cu, cv)
    best = min(centroids, key=lambda p: (p[0] - cu) ** 2 + (p[1] - cv) ** 2)
    return (float(best[0]), float(best[1]))


def _contact_frame_path(
    csv_path: Path, data_root: Path, frames_dir: Path, which: str
) -> Optional[Path]:
    """Resolve the frame to segment for a trial (default: the last/contact frame)."""
    trial_id = _trial_idx(csv_path)
    df = pd.read_csv(csv_path)
    if IMAGE_COL in df.columns:
        col = df[IMAGE_COL].astype(str)
        keep = col.notna() & col.str.strip().ne("")
        df = df[keep].reset_index(drop=True)
    if len(df) == 0:
        return None
    if which == "first":
        t = 0
    elif which == "middle":
        t = len(df) // 2
    else:
        t = len(df) - 1
    raw = str(df[IMAGE_COL].iloc[t]) if IMAGE_COL in df.columns else ""
    return _resolve_image_path(raw, data_root, frames_dir, trial_id, t)


# ---------------------------------------------------------------------------
# Cellpose runner (lazy import so the rest of the repo never needs cellpose)
# ---------------------------------------------------------------------------

def _build_cellpose(gpu: bool):
    try:
        from cellpose import models
    except ImportError as e:  # pragma: no cover - exercised only with cellpose absent
        raise SystemExit(
            "generate_cell_labels.py needs Cellpose >= 4.0 (Cellpose-SAM). Install with "
            "`python3 -m pip install 'cellpose>=4.0'`."
        ) from e
    # Cellpose 4 defaults CellposeModel to the cpsam (Cellpose-SAM) weights.
    return models.CellposeModel(gpu=gpu)


def _segment(model, img_rgb: np.ndarray, diameter: float,
             flow_threshold: float, cellprob_threshold: float) -> np.ndarray:
    out = model.eval(
        img_rgb,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = out[0]  # (masks, flows, styles[, diams]) across cellpose versions
    return np.asarray(masks)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    logs_dir = data_root / "logs" if (data_root / "logs").is_dir() else data_root
    frames_dir = data_root / "saved_frames"

    csv_files = sorted(logs_dir.glob("trial_*.csv"), key=_trial_idx)
    if not csv_files:
        raise FileNotFoundError(f"No trial_*.csv under {logs_dir}")
    if args.limit_trials and args.limit_trials > 0:
        csv_files = csv_files[: args.limit_trials]
    trial_ids = [_trial_idx(p) for p in csv_files]
    labels = load_or_scaffold_labels(args.labels.expanduser().resolve(), trial_ids)

    model = _build_cellpose(gpu=args.gpu)

    rows = []
    for csv_path in csv_files:
        trial_id = _trial_idx(csv_path)
        _, region = resolve_instruction(trial_id, labels)
        center = region_center_for(region)

        frame_path = _contact_frame_path(csv_path, data_root, frames_dir, args.frame)
        if frame_path is None:
            print(f"[cells] trial_{trial_id}: no resolvable frame; using region center.")
            u, v, n_cells = center[0], center[1], 0
        else:
            img = _load_rgb_uint8(frame_path)
            masks = _segment(model, img, args.diameter, args.flow_threshold,
                             args.cellprob_threshold)
            centroids = mask_centroids(masks)
            u, v = pick_target_cell(centroids, center)
            n_cells = len(centroids)
        rows.append({"trial_id": trial_id, "goal_u": round(u, 6), "goal_v": round(v, 6),
                     "region": region, "n_cells": n_cells})
        print(f"[OK] trial_{trial_id}: region={region} cells={n_cells} "
              f"contact=({u:.3f}, {v:.3f})")

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {len(rows)} contact-point label(s) -> {out_path}")
    print("Next: python dataset_vla/convert_microact_to_lerobot.py "
          f"--cell-labels {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cellpose teacher: per-trial contact-point labels.")
    p.add_argument("--data-root", type=Path, default=C.DATASET_ROOT)
    p.add_argument("--labels", type=Path, default=C.DATASET_ROOT / "instruction_labels.csv")
    p.add_argument("--out", type=Path, default=C.DATASET_ROOT / "cell_labels.csv")
    p.add_argument("--frame", choices=("last", "first", "middle"), default="last",
                   help="Which frame to segment per trial. 'last' is the contact frame.")
    p.add_argument("--diameter", type=float, default=C.CELLPOSE4_DIAMETER)
    p.add_argument("--flow-threshold", type=float, default=C.CELLPOSE4_FLOW_THRESHOLD)
    p.add_argument("--cellprob-threshold", type=float, default=C.CELLPOSE4_CELLPROB_THRESHOLD)
    p.add_argument("--gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="gpu", action="store_false")
    p.add_argument("--limit-trials", type=int, default=0, help="0 = all; >0 for a quick subset.")
    return p.parse_args()


if __name__ == "__main__":
    main()
