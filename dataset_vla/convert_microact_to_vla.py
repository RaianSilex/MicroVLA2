"""Convert classic MicroACT trials into MicroVLA episode folders.

The classic dataset layout is:

    dataset/logs/trial_N.csv
    dataset/saved_frames/trial_N/frame_000000.png

MicroVLA expects:

    dataset_vla/episodes/trial_N/
        metadata.json
        trajectory.csv
        frames/cam_main/frame_000000.png

By default this script creates relative symlinks for frames so converting a
large dataset does not duplicate image storage. Use --frame-mode copy if you
need a fully self-contained dataset_vla/ tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MICROACT_DATASET = REPO_ROOT / "dataset"
VLA_EPISODES_DIR = REPO_ROOT / "dataset_vla" / "episodes"

STATE_COLS = (
    "current_x", "current_y", "current_z", "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2",
)
ACTION_COLS = (
    "target_x", "target_y", "target_z", "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2",
)
IMAGE_COL = "image_path"
TIMESTEP_COL = "timestep"


def _trial_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _resolve_source_image(
    raw: str,
    *,
    microact_dataset: Path,
    trial_id: int,
    row_index: int,
) -> Optional[Path]:
    raw = (raw or "").strip()
    fallback = (
        microact_dataset
        / "saved_frames"
        / f"trial_{trial_id}"
        / f"frame_{row_index:06d}.png"
    )

    if not raw:
        return fallback if fallback.exists() else None

    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None

    for base in (REPO_ROOT, microact_dataset, microact_dataset / "saved_frames"):
        candidate = base / path
        if candidate.exists():
            return candidate

    return fallback if fallback.exists() else None


def _write_frame(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if mode == "none":
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "symlink":
        rel_src = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel_src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported frame mode: {mode}")


def _build_metadata(args: argparse.Namespace, episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "lab_id": args.lab_id,
        "robot_id": args.robot_id,
        "embodiment": args.embodiment,
        "action_type": args.action_type,
        "task_family": args.task_family,
        "instruction": args.instruction,
        "camera_names": ["cam_main"],
        "state_dim": len(STATE_COLS),
        "action_dim": len(ACTION_COLS),
        "state_cols": list(STATE_COLS),
        "action_cols": list(ACTION_COLS),
        "image_col": IMAGE_COL,
        "timestep_col": TIMESTEP_COL,
        "trajectory_file": "trajectory.csv",
    }


def _fix_or_filter_actions(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    zero_mask = df[list(ACTION_COLS)].eq(0)
    report = {
        "rows_with_any_zero_action": int(zero_mask.any(axis=1).sum()),
        "rows_with_all_zero_actions": int(zero_mask.all(axis=1).sum()),
        "replaced_zero_action_cells": 0,
        "dropped_rows": 0,
    }

    if args.replace_zero_targets_with_state:
        for state_col, action_col in zip(STATE_COLS, ACTION_COLS):
            mask = df[action_col].eq(0)
            report["replaced_zero_action_cells"] += int(mask.sum())
            df.loc[mask, action_col] = df.loc[mask, state_col]
        return df, report

    if args.drop_any_zero_target_row:
        keep = ~zero_mask.any(axis=1)
        report["dropped_rows"] = int((~keep).sum())
        return df.loc[keep].copy(), report

    if args.drop_all_zero_target_row:
        keep = ~zero_mask.all(axis=1)
        report["dropped_rows"] = int((~keep).sum())
        return df.loc[keep].copy(), report

    return df, report


def convert_trial(csv_path: Path, args: argparse.Namespace) -> dict:
    trial_id = _trial_num(csv_path)
    episode_id = f"trial_{trial_id}"
    episode_dir = args.output / episode_id

    if episode_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{episode_dir} already exists. Pass --overwrite to replace/update it."
        )

    df = pd.read_csv(csv_path)
    required = (*STATE_COLS, *ACTION_COLS)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {missing}")

    df, action_report = _fix_or_filter_actions(df, args)
    episode_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = episode_dir / "frames" / "cam_main"
    frame_dir.mkdir(parents=True, exist_ok=True)

    new_image_paths = []
    missing_images = 0
    frames_written = 0
    source_image_col = (
        df[IMAGE_COL].fillna("").astype(str).tolist()
        if IMAGE_COL in df.columns
        else [""] * len(df)
    )

    for row_index, raw in zip(df.index.tolist(), source_image_col):
        src = _resolve_source_image(
            raw,
            microact_dataset=args.microact_dataset,
            trial_id=trial_id,
            row_index=int(row_index),
        )
        dst_name = f"frame_{int(row_index):06d}.png"
        dst = frame_dir / dst_name
        new_image_paths.append(f"frames/cam_main/{dst_name}")

        if src is None:
            missing_images += 1
            continue
        before_exists = dst.exists() or dst.is_symlink()
        _write_frame(src, dst, args.frame_mode, args.overwrite)
        after_exists = dst.exists() or dst.is_symlink()
        if after_exists and (args.overwrite or not before_exists):
            frames_written += 1

    df = df.copy()
    df[IMAGE_COL] = new_image_paths
    df.to_csv(episode_dir / "trajectory.csv", index=False)

    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_build_metadata(args, episode_id), f, indent=2)
        f.write("\n")

    return {
        "episode_id": episode_id,
        "rows": int(len(df)),
        "missing_images": int(missing_images),
        "frames_written": int(frames_written),
        **action_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dataset/logs + dataset/saved_frames into MicroVLA episodes."
    )
    parser.add_argument("--microact-dataset", type=Path, default=MICROACT_DATASET)
    parser.add_argument("--output", type=Path, default=VLA_EPISODES_DIR)
    parser.add_argument(
        "--frame-mode",
        choices=("symlink", "hardlink", "copy", "none"),
        default="symlink",
        help="How to place frames in each VLA episode. Default avoids duplicating image data.",
    )
    parser.add_argument("--overwrite", action="store_true")

    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--replace-zero-targets-with-state",
        action="store_true",
        help="Replace any zero action component with the matching current_* state component.",
    )
    cleanup.add_argument(
        "--drop-any-zero-target-row",
        action="store_true",
        help="Drop rows where any target/action component is zero.",
    )
    cleanup.add_argument(
        "--drop-all-zero-target-row",
        action="store_true",
        help="Drop rows only when all target/action components are zero.",
    )

    parser.add_argument(
        "--instruction",
        default="move both manipulators toward the selected cell",
    )
    parser.add_argument("--lab-id", default="local_lab")
    parser.add_argument("--robot-id", default="sensapex_dual_ump4")
    parser.add_argument("--embodiment", default="dual_manipulator")
    parser.add_argument("--action-type", default="absolute_position")
    parser.add_argument("--task-family", default="cell_manipulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.microact_dataset = args.microact_dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    logs_dir = args.microact_dataset / "logs"
    csv_paths = sorted(logs_dir.glob("trial_*.csv"), key=_trial_num)
    if not csv_paths:
        raise FileNotFoundError(f"No trial_*.csv files found under {logs_dir}")

    summaries = [convert_trial(path, args) for path in csv_paths]
    rows = sum(item["rows"] for item in summaries)
    missing_images = sum(item["missing_images"] for item in summaries)
    zero_rows = sum(item["rows_with_any_zero_action"] for item in summaries)
    replaced = sum(item["replaced_zero_action_cells"] for item in summaries)
    dropped = sum(item["dropped_rows"] for item in summaries)
    frames_written = sum(item["frames_written"] for item in summaries)

    print(f"converted_episodes={len(summaries)}")
    print(f"rows_written={rows}")
    print(f"frames_written_or_present={frames_written}")
    print(f"missing_images={missing_images}")
    print(f"rows_with_any_zero_action_before_cleanup={zero_rows}")
    print(f"replaced_zero_action_cells={replaced}")
    print(f"dropped_rows={dropped}")
    if zero_rows and not (replaced or dropped):
        print(
            "[warn] zero target/action rows were preserved. Consider "
            "--replace-zero-targets-with-state or --drop-any-zero-target-row."
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
