"""Convert classic MicroACT trials into a LeRobot dataset on the Hugging Face Hub.

This is the MicroVLA equivalent of the OpenPI ``convert_data_to_lerobot.py`` the
lab already uses: it produces a real LeRobot dataset (via the ``lerobot`` library)
so MicroVLA, SmolVLA, OpenPI / pi0, and the LeRobot tooling can all train from the
same HF dataset.

Differences from the OpenPI converter:

* **Standard LeRobot feature names** (``observation.images.cam_main`` /
  ``observation.state`` / ``action``) so the broader LeRobot ecosystem reads it
  out of the box. (OpenPI's data config just needs its repack keys pointed here.)
* **Varied, grounded instructions.** Each trial's ``task`` string is built from
  the target cell's frame position (``top_left`` ... ``center`` ... ``bottom_right``)
  read from an editable labels CSV, so the language channel carries real signal
  instead of one constant prompt.

Actions are stored **ABSOLUTE** (straight from the ``target_*`` columns), exactly
like the robot/ROS commands. The delta-vs-absolute choice is a *training-time*
transform (see ``data/lerobot_vla_dataset.py`` / ``train_vla.py --action-space``),
which keeps this dataset robot-native and reusable by any VLA.

Layout consumed:
    <data_root>/logs/trial_N.csv          (or <data_root>/trial_N.csv)
    <data_root>/saved_frames/trial_N/frame_NNNNNN.png

Output (one LeRobot repo under HF_LEROBOT_HOME, not pushed unless --push-to-hub):
    HF_LEROBOT_HOME/<repo_id>/

Usage:
    python dataset_vla/convert_microact_to_lerobot.py            # build locally
    python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3   # quick smoke
    python dataset_vla/convert_microact_to_lerobot.py --push-to-hub      # you run this
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import vla_config as C

STATE_COLS = list(C.CSV_STATE_COLS)
ACTION_COLS = list(C.CSV_ACTION_COLS)
IMAGE_COL = C.CSV_IMAGE_COL
RESISTANCE_COL = C.CSV_RESISTANCE_COL


# ---------------------------------------------------------------------------
# Instruction generation from a per-trial target-cell region label
# ---------------------------------------------------------------------------

# Canonical regions (a 3x3 frame grid). The labels CSV may use any of these or
# the many natural aliases below (e.g. "middle_left", "top_center", "lower-right").
REGIONS = (
    "top_left", "top", "top_right",
    "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
)

# Several phrasings per region so wording varies while the region word stays
# grounded. Picked deterministically by trial id (reproducible).
_REGION_PHRASES = {
    "top_left":     ["top-left", "upper-left"],
    "top":          ["top", "top-center", "upper"],
    "top_right":    ["top-right", "upper-right"],
    "left":         ["left", "middle-left", "center-left"],
    "center":       ["center", "middle"],
    "right":        ["right", "middle-right", "center-right"],
    "bottom_left":  ["bottom-left", "lower-left"],
    "bottom":       ["bottom", "bottom-center", "lower"],
    "bottom_right": ["bottom-right", "lower-right"],
}

# Forgiving aliases -> canonical region. Keys are normalized (lowercase, any
# spaces/hyphens collapsed to single underscores) before lookup.
_REGION_ALIASES = {
    # middle row
    "middle_left": "left", "mid_left": "left", "center_left": "left",
    "left_middle": "left", "left_center": "left", "left_side": "left",
    "middle_right": "right", "mid_right": "right", "center_right": "right",
    "right_middle": "right", "right_center": "right", "right_side": "right",
    "middle": "center", "middle_center": "center", "center_center": "center",
    "centre": "center", "mid": "center", "middle_middle": "center", "middle_centre": "center",
    # top row
    "top_center": "top", "top_middle": "top", "top_centre": "top",
    "upper": "top", "upper_center": "top", "center_top": "top", "top_center_": "top",
    "upper_left": "top_left", "left_top": "top_left", "top_left_corner": "top_left",
    "upper_right": "top_right", "right_top": "top_right", "top_right_corner": "top_right",
    # bottom row
    "bottom_center": "bottom", "bottom_middle": "bottom", "bottom_centre": "bottom",
    "lower": "bottom", "lower_center": "bottom", "center_bottom": "bottom",
    "lower_left": "bottom_left", "left_bottom": "bottom_left", "bottom_left_corner": "bottom_left",
    "lower_right": "bottom_right", "right_bottom": "bottom_right", "bottom_right_corner": "bottom_right",
}

# Every region spelling the labels CSV may use.
ACCEPTED_REGIONS = sorted(set(REGIONS) | set(_REGION_ALIASES))

# Instruction phrasing depends on how many manipulators the dataset uses, so a
# single-uMp dataset doesn't get "both manipulators" / "the two pipettes" prompts.
_TEMPLATES_SINGLE = [
    "move the manipulator toward the {r} cell",
    "guide the pipette to the cell in the {r}",
    "advance the needle to the {r} cell",
    "target the {r} cell with the manipulator",
    "bring the pipette to the {r} cell",
]
_TEMPLATES_DUAL = [
    "move both manipulators toward the {r} cell",
    "guide the pipettes to the cell in the {r}",
    "advance both needles to the {r} cell",
    "target the {r} cell with both manipulators",
    "bring the two pipettes to the {r} cell",
]


def _templates_for(n_manipulators: int) -> list:
    return _TEMPLATES_DUAL if int(n_manipulators) >= 2 else _TEMPLATES_SINGLE


def normalize_region(raw: str) -> tuple[str, bool]:
    """Map any accepted spelling to a canonical region.

    Returns (canonical_region, known). Unknown spellings fall back to 'center'
    with known=False so the caller can warn.
    """
    key = re.sub(r"[\s\-]+", "_", str(raw).strip().lower())
    key = re.sub(r"_+", "_", key).strip("_")
    if key in _REGION_PHRASES:
        return key, True
    if key in _REGION_ALIASES:
        return _REGION_ALIASES[key], True
    return "center", False


def instruction_for(trial_id: int, region: str, n_manipulators: int = C.NUM_MANIPULATORS) -> str:
    """Deterministic, grounded, lexically-varied instruction for a trial.

    Phrasing matches the manipulator count (singular for 1, "both"/"two" for 2).
    """
    canon, _ = normalize_region(region)
    phrases = _REGION_PHRASES[canon]
    phrase = phrases[trial_id % len(phrases)]
    templates = _templates_for(n_manipulators)
    template = templates[trial_id % len(templates)]
    return template.format(r=phrase)


# ---------------------------------------------------------------------------
# Labels CSV (trial_id, region, instruction)
# ---------------------------------------------------------------------------

def _scaffold_labels(path: Path, trial_ids: list[int]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"trial_id": trial_ids, "region": ["center"] * len(trial_ids), "instruction": [""] * len(trial_ids)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_or_scaffold_labels(path: Path, trial_ids: list[int]) -> dict[int, dict]:
    """Return {trial_id: {"region": str, "instruction": str}}.

    Creates an editable scaffold (region=center) if the file is missing so smoke
    tests run; warns loudly so the user knows to fill in real regions.
    """
    if not path.exists():
        _scaffold_labels(path, trial_ids)
        print(
            "\n" + "!" * 78 +
            f"\n[labels] No labels file found. Scaffolded {path}\n"
            "[labels] Every trial defaulted to region='center' (instructions WON'T vary).\n"
            "[labels] Edit the 'region' column (one of: "
            + ", ".join(REGIONS) + ")\n"
            "[labels] aliases also accepted, e.g. middle_left, middle_right, top_center,\n"
            "[labels]   bottom_center, lower-right, upper-left, centre ...\n"
            "[labels] or write a free-text 'instruction' to override, then re-run.\n"
            + "!" * 78 + "\n"
        )
    df = pd.read_csv(path)
    if "trial_id" not in df.columns:
        raise ValueError(f"{path} must have a 'trial_id' column")
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        tid = int(row["trial_id"])
        region = row.get("region", "center")
        region = "center" if pd.isna(region) else (str(region).strip().lower() or "center")
        instr = row.get("instruction", "")
        instr = "" if pd.isna(instr) else str(instr).strip()
        out[tid] = {"region": region, "instruction": instr}
    return out


def resolve_instruction(
    trial_id: int, labels: dict[int, dict], n_manipulators: int = C.NUM_MANIPULATORS
) -> tuple[str, str]:
    """(instruction, canonical_region) for a trial, honoring a free-text override."""
    entry = labels.get(trial_id, {"region": "center", "instruction": ""})
    raw_region = entry.get("region", "center") or "center"
    canon, known = normalize_region(raw_region)
    if not known and str(raw_region).strip():
        print(f"[labels] trial_{trial_id}: unknown region {raw_region!r}; using 'center'. "
              f"Accepted: {', '.join(REGIONS)} (+ aliases like middle_left, top_center).")
    override = entry.get("instruction", "")
    if override:
        return override, canon
    return instruction_for(trial_id, canon, n_manipulators), canon


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _trial_idx(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_letterbox(img: np.ndarray, out_h: int, out_w: int, pad: int = 0) -> np.ndarray:
    """Aspect-preserving resize + center pad to (out_h, out_w)."""
    im = Image.fromarray(img)
    in_w, in_h = im.size
    scale = min(out_w / in_w, out_h / in_h)
    new_w, new_h = int(round(in_w * scale)), int(round(in_h * scale))
    im_rs = im.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), (pad, pad, pad))
    canvas.paste(im_rs, ((out_w - new_w) // 2, (out_h - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _resize_exact(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


def _trial_resistance(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Per-row resistance for one trial as float32, or None if the column has no
    usable values. Missing cells become 0.0."""
    if RESISTANCE_COL not in df.columns:
        return None
    col = pd.to_numeric(df[RESISTANCE_COL], errors="coerce")
    if not np.isfinite(col.to_numpy(dtype=np.float64)).any():
        return None
    return col.fillna(0.0).to_numpy(dtype=np.float32)


def _dataset_has_resistance(csv_files: list[Path]) -> bool:
    """True if ANY trial carries real resistance values (so the whole dataset
    gets the optional observation.resistance feature)."""
    for p in csv_files:
        try:
            df = pd.read_csv(p, usecols=[RESISTANCE_COL])
        except (ValueError, KeyError):
            continue
        if _trial_resistance(df) is not None:
            return True
    return False


def _resolve_image_path(raw: str, data_root: Path, frames_dir: Path, trial_id: int, t: int) -> Optional[Path]:
    raw = (raw or "").strip()
    fallback = frames_dir / f"trial_{trial_id}" / f"frame_{t:06d}.png"
    if not raw:
        return fallback if fallback.exists() else None
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (data_root, REPO_ROOT, frames_dir):
        q = base / p
        if q.exists():
            return q
    return fallback if fallback.exists() else None


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Lazy import so the rest of the repo doesn't require lerobot.
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    # Which manipulators to convert (1 = single pipette xyzd; 2 = dual). The raw
    # CSV may contain both; we select only the first --manipulators of them.
    state_cols = list(C.state_cols_for(args.manipulators))
    action_cols = list(C.action_cols_for(args.manipulators))
    print(f"[manipulators] using {args.manipulators} -> state/action dims = {len(state_cols)}")

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

    out_root = (Path(args.root).expanduser().resolve() / args.repo_id) if args.root else (HF_LEROBOT_HOME / args.repo_id)
    if out_root.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to rebuild it.")

    # Optional: pipette resistance, included only if the raw logs carry real values.
    has_resistance = args.resistance and _dataset_has_resistance(csv_files)
    features = {
        C.LEROBOT_CAMERA_KEY: {"dtype": "image", "shape": (args.down_h, args.down_w, 3),
                               "names": ["height", "width", "channel"]},
        C.LEROBOT_STATE_KEY: {"dtype": "float32", "shape": (len(state_cols),), "names": ["state"]},
        C.LEROBOT_ACTION_KEY: {"dtype": "float32", "shape": (len(action_cols),), "names": ["action"]},
    }
    if has_resistance:
        features[C.LEROBOT_RESISTANCE_KEY] = {"dtype": "float32", "shape": (1,), "names": ["resistance"]}
        print(f"[resistance] found values -> adding feature {C.LEROBOT_RESISTANCE_KEY!r}")

    out_hw = (args.down_h, args.down_w)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=out_root,
        robot_type=args.robot_type,
        use_videos=False,  # store frames as PNG (no ffmpeg/av needed)
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    region_counts: dict[str, int] = {}
    total_rows = 0
    for csv_path in csv_files:
        trial_id = _trial_idx(csv_path)
        df = pd.read_csv(csv_path)
        missing = [c for c in (*state_cols, *action_cols) if c not in df.columns]
        if missing:
            raise KeyError(f"{csv_path.name} missing columns: {missing}")

        # Drop rows without an image reference.
        img_col = df[IMAGE_COL] if IMAGE_COL in df.columns else pd.Series([""] * len(df))
        keep = img_col.notna() & img_col.astype(str).str.strip().ne("")
        # If the column is absent entirely, keep all rows (fallback path resolves frames).
        if IMAGE_COL not in df.columns:
            keep = pd.Series([True] * len(df))
        df = df[keep].reset_index(drop=True)
        if len(df) == 0:
            print(f"[skip] trial_{trial_id}: no valid rows")
            continue

        states = df[state_cols].to_numpy(dtype=np.float32)
        actions = df[action_cols].to_numpy(dtype=np.float32)

        # Repair uninitialized commanded targets (target_* == 0 is a logging
        # sentinel): hold at the current state so the (absolute) action is a
        # zero-delta hold instead of a huge bogus jump that wrecks stats.
        if args.fix_uninitialized_targets:
            zero = actions == 0
            n_fixed = int(zero.any(axis=1).sum())
            if n_fixed:
                actions = np.where(zero, states, actions)
                print(f"[clean] trial_{trial_id}: held {n_fixed} uninitialized-target row(s)")

        instruction, region = resolve_instruction(trial_id, labels, args.manipulators)
        region_counts[region] = region_counts.get(region, 0) + 1

        resist = _trial_resistance(df) if has_resistance else None
        if has_resistance and resist is None:
            resist = np.zeros(len(df), dtype=np.float32)

        raw_paths = (
            df[IMAGE_COL].astype(str).tolist() if IMAGE_COL in df.columns else [""] * len(df)
        )
        wrote = 0
        for t in range(len(df)):
            src = _resolve_image_path(raw_paths[t], data_root, frames_dir, trial_id, t)
            if src is None:
                raise FileNotFoundError(
                    f"trial_{trial_id} t={t}: could not resolve image (csv={raw_paths[t]!r})"
                )
            img = _load_rgb_uint8(src)
            img = (_resize_letterbox(img, *out_hw) if args.keep_aspect else _resize_exact(img, *out_hw))
            frame = {
                C.LEROBOT_CAMERA_KEY: img,
                C.LEROBOT_STATE_KEY: states[t],
                C.LEROBOT_ACTION_KEY: actions[t],
                "task": instruction,
            }
            if has_resistance:
                frame[C.LEROBOT_RESISTANCE_KEY] = np.array([resist[t]], dtype=np.float32)
            ds.add_frame(frame)
            wrote += 1
        ds.save_episode()
        total_rows += wrote
        print(f"[OK] trial_{trial_id}: {wrote} frames | region={region} | task={instruction!r}")

    ds.finalize()

    print("\n=== summary ===")
    print(f"episodes={len(csv_files)}  frames={total_rows}")
    print(f"region_counts={region_counts}")
    print(f"robot_type={args.robot_type}  fps={args.fps}  image={args.down_h}x{args.down_w}")
    print(f"local dataset: {out_root}")
    if len(region_counts) <= 1:
        print("[warn] all trials share one region -> language won't vary. "
              "Edit the labels CSV and re-run for real language signal.")

    if args.push_to_hub:
        print(f"[push] pushing to hub: {args.repo_id} (private)")
        ds.push_to_hub(
            tags=["sensapex", "micromanipulation", "single_ump", "microvla"],
            private=True,
            push_videos=False,
            license="apache-2.0",
        )
        print(f"[push] done -> https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("[push] skipped (run with --push-to-hub, or use push_to_huggingface.py)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MicroACT trials to a LeRobot dataset.")
    p.add_argument("--data-root", type=Path, default=C.DATASET_ROOT)
    p.add_argument("--labels", type=Path, default=C.DATASET_ROOT / "instruction_labels.csv")
    p.add_argument("--repo-id", type=str, default=C.DEFAULT_DATASET_REPO_ID)
    p.add_argument("--root", type=Path, default=None,
                   help="Output base dir for the LeRobot repo. Default: HF_LEROBOT_HOME.")
    p.add_argument("--robot-type", type=str, default=C.DEFAULT_ROBOT_ID,
                   help="Stored as the dataset robot_type AND used as the robot id for "
                        "per-robot normalization. Must match the rollout adapter's robot_id.")
    p.add_argument("--manipulators", type=int, default=C.NUM_MANIPULATORS, choices=(1, 2),
                   help="How many manipulators to convert from the raw CSV. 1 = single "
                        "pipette (xyzd, 4-dim); 2 = dual (8-dim). Default from config "
                        f"(NUM_MANIPULATORS={C.NUM_MANIPULATORS}).")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--down-h", type=int, default=540)
    p.add_argument("--down-w", type=int, default=720)
    p.add_argument("--keep-aspect", action="store_true", default=True,
                   help="Letterbox (aspect-preserving + pad) to (down_h, down_w).")
    p.add_argument("--no-keep-aspect", dest="keep_aspect", action="store_false")
    p.add_argument("--fix-uninitialized-targets", action="store_true", default=True)
    p.add_argument("--no-fix-uninitialized-targets", dest="fix_uninitialized_targets", action="store_false")
    p.add_argument("--resistance", action="store_true", default=True,
                   help="Auto-include observation.resistance when the logs carry real values.")
    p.add_argument("--no-resistance", dest="resistance", action="store_false")
    p.add_argument("--limit-trials", type=int, default=0, help="0 = all; >0 for quick smoke tests.")
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--push-to-hub", action="store_true", default=False,
                   help="Push to the HF Hub (needs your HF login). Off by default.")
    return p.parse_args()


if __name__ == "__main__":
    main()
