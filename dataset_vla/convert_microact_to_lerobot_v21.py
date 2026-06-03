"""Convert MicroACT trials into a LeRobot **v2.1** dataset (for OpenPI / pi0).

Identical data + instruction logic to ``dataset_vla/convert_microact_to_lerobot.py``,
but writes the older **v2.1** on-disk LeRobot format that OpenPI's training pipeline
requires (OpenPI does not support v3.0). SmolVLA and MicroVLA can read either, so
this script is specifically for the OpenPI / pi0 / pi0.5 path. By default it targets
a NEW repo id (``<default>_v21``) so it never clobbers your v3.0 dataset.

----------------------------------------------------------------------------
IMPORTANT — the on-disk format version is decided by the *installed* ``lerobot``,
not by this script:

    * lerobot 0.3.x   -> writes v2.1   (what you want here)
    * lerobot >= 0.4  -> writes v3.0

So run this in a SEPARATE venv with an older lerobot:

    python3 -m venv .lerobot-v21-venv
    source .lerobot-v21-venv/bin/activate
    pip install "lerobot==0.3.3"
    python dataset_vla/convert_microact_to_lerobot_v21.py

The script reads back the format that ``LeRobotDataset.create`` actually wrote and
aborts early if it is not v2.x, so you cannot accidentally produce v3.0.
----------------------------------------------------------------------------

Usage:
    python dataset_vla/convert_microact_to_lerobot_v21.py             # build locally (v2.1)
    python dataset_vla/convert_microact_to_lerobot_v21.py --limit-trials 3   # quick smoke
    python dataset_vla/convert_microact_to_lerobot_v21.py --push-to-hub      # build + push
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import config as ACT
from config import vla_config as C

# Reuse the EXACT instruction + image logic from the v3.0 converter so the two
# datasets differ only in on-disk format, never in content. These helpers do not
# import lerobot, so importing them is safe inside the v2.1 venv.
from convert_microact_to_lerobot import (  # noqa: E402
    STATE_COLS,
    ACTION_COLS,
    IMAGE_COL,
    load_or_scaffold_labels,
    resolve_instruction,
    _trial_idx,
    _load_rgb_uint8,
    _resize_letterbox,
    _resize_exact,
    _resolve_image_path,
)


def _import_lerobot():
    """Import LeRobotDataset + HF_LEROBOT_HOME across the 0.3.x / 0.4+ module layouts."""
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    return LeRobotDataset, HF_LEROBOT_HOME


def _dataset_codebase_version(ds) -> str:
    """Best-effort read of the format version that LeRobotDataset.create just wrote."""
    for getter in (
        lambda: ds.meta.info["codebase_version"],
        lambda: ds.meta._version,
        lambda: ds._version,
    ):
        try:
            v = getter()
            if v:
                return str(v)
        except Exception:
            pass
    return ""


def _finalize(ds) -> None:
    """Finish writing the dataset.

    Older lerobot (0.1/0.2) finished with consolidate(); v3.0 (0.4.x) renamed it
    finalize(). lerobot 0.3.x (which writes the v2.1 format we want here) writes
    each episode incrementally in save_episode() and exposes NEITHER method, so in
    that case there is nothing left to do.
    """
    fn = getattr(ds, "consolidate", None) or getattr(ds, "finalize", None)
    if fn is not None:
        fn()


def main() -> None:
    args = parse_args()
    LeRobotDataset, HF_LEROBOT_HOME = _import_lerobot()

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

    out_root = (
        (Path(args.root).expanduser().resolve() / args.repo_id)
        if args.root
        else (HF_LEROBOT_HOME / args.repo_id)
    )
    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to rebuild it.")

    out_hw = (args.down_h, args.down_w)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=out_root,
        robot_type=args.robot_type,
        use_videos=False,  # store frames as PNG (no ffmpeg/av needed)
        features={
            C.LEROBOT_CAMERA_KEY: {"dtype": "image", "shape": (args.down_h, args.down_w, 3),
                                   "names": ["height", "width", "channel"]},
            C.LEROBOT_STATE_KEY: {"dtype": "float32", "shape": (len(STATE_COLS),), "names": ["state"]},
            C.LEROBOT_ACTION_KEY: {"dtype": "float32", "shape": (len(ACTION_COLS),), "names": ["action"]},
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    # Fail fast if the installed lerobot did not write v2.x (e.g. lerobot >= 0.4).
    version = _dataset_codebase_version(ds)
    if version and not version.lstrip("v").startswith("2"):
        shutil.rmtree(out_root, ignore_errors=True)
        raise SystemExit(
            f"[abort] active lerobot wrote dataset format {version!r}, not v2.x.\n"
            "        OpenPI needs v2.1. Build this in a separate venv with an older\n"
            '        lerobot:  pip install "lerobot==0.3.3"   then re-run.'
        )
    if not version:
        print("[warn] could not read codebase_version; verify meta/info.json says v2.1 after the build.")
    else:
        print(f"[ok] writing LeRobot dataset format {version}")

    region_counts: dict[str, int] = {}
    total_rows = 0
    for csv_path in csv_files:
        trial_id = _trial_idx(csv_path)
        df = pd.read_csv(csv_path)
        missing = [c for c in (*STATE_COLS, *ACTION_COLS) if c not in df.columns]
        if missing:
            raise KeyError(f"{csv_path.name} missing columns: {missing}")

        img_col = df[IMAGE_COL] if IMAGE_COL in df.columns else pd.Series([""] * len(df))
        keep = img_col.notna() & img_col.astype(str).str.strip().ne("")
        if IMAGE_COL not in df.columns:
            keep = pd.Series([True] * len(df))
        df = df[keep].reset_index(drop=True)
        if len(df) == 0:
            print(f"[skip] trial_{trial_id}: no valid rows")
            continue

        states = df[STATE_COLS].to_numpy(dtype=np.float32)
        actions = df[ACTION_COLS].to_numpy(dtype=np.float32)

        # Repair uninitialized commanded targets (target_* == 0 sentinel): hold at
        # the current state so the absolute action is a zero-delta hold.
        if args.fix_uninitialized_targets:
            zero = actions == 0
            n_fixed = int(zero.any(axis=1).sum())
            if n_fixed:
                actions = np.where(zero, states, actions)
                print(f"[clean] trial_{trial_id}: held {n_fixed} uninitialized-target row(s)")

        instruction, region = resolve_instruction(trial_id, labels)
        region_counts[region] = region_counts.get(region, 0) + 1

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
            }
            try:
                # lerobot 0.3.x (v2.1): task is a separate positional argument.
                ds.add_frame(frame, task=instruction)
            except TypeError:
                # lerobot 0.4.x (v3.0): task is a key inside the frame dict.
                ds.add_frame({**frame, "task": instruction})
            wrote += 1
        ds.save_episode()
        total_rows += wrote
        print(f"[OK] trial_{trial_id}: {wrote} frames | region={region} | task={instruction!r}")

    _finalize(ds)

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
        tags = ["sensapex", "micromanipulation", "dual_ump", "microvla", "lerobot-v2.1", "openpi"]
        try:
            ds.push_to_hub(tags=tags, private=True, license="apache-2.0", push_videos=False)
        except TypeError:
            # Older push_to_hub signatures don't accept push_videos.
            ds.push_to_hub(tags=tags, private=True, license="apache-2.0")
        print(f"[push] done -> https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("[push] skipped (run with --push-to-hub, or use push_to_huggingface.py)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MicroACT trials to a LeRobot v2.1 dataset (OpenPI-compatible)."
    )
    p.add_argument("--data-root", type=Path, default=ACT.DATASET_ROOT)
    p.add_argument("--labels", type=Path, default=ACT.DATASET_ROOT / "instruction_labels.csv")
    p.add_argument("--repo-id", type=str, default=f"{C.DEFAULT_DATASET_REPO_ID}_v21",
                   help="Destination repo id. Defaults to the v3.0 repo id + '_v21' so the "
                        "two datasets live in separate repos.")
    p.add_argument("--root", type=Path, default=None,
                   help="Output base dir for the LeRobot repo. Default: HF_LEROBOT_HOME.")
    p.add_argument("--robot-type", type=str, default=C.DEFAULT_ROBOT_ID,
                   help="Stored as dataset robot_type AND used as the per-robot normalization key.")
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--down-h", type=int, default=540)
    p.add_argument("--down-w", type=int, default=720)
    p.add_argument("--keep-aspect", action="store_true", default=True,
                   help="Letterbox (aspect-preserving + pad) to (down_h, down_w).")
    p.add_argument("--no-keep-aspect", dest="keep_aspect", action="store_false")
    p.add_argument("--fix-uninitialized-targets", action="store_true", default=True)
    p.add_argument("--no-fix-uninitialized-targets", dest="fix_uninitialized_targets", action="store_false")
    p.add_argument("--limit-trials", type=int, default=0, help="0 = all; >0 for quick smoke tests.")
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--push-to-hub", action="store_true", default=False,
                   help="Push to the HF Hub (needs `huggingface-cli login`). Off by default.")
    return p.parse_args()


if __name__ == "__main__":
    main()
