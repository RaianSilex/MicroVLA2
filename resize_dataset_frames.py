"""Pre-resize MicroACT dataset frames and point CSVs at the resized copies.

The training dataloader resizes images on every epoch. This script does that
work once by creating:

    dataset/saved_frames_240x320/trial_N/frame_000000.png

and updating `dataset/logs/trial_N.csv` so `image_path` points to the resized
frame tree. Original CSVs are backed up before they are overwritten.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_LOGS_DIR = DEFAULT_DATASET_ROOT / "logs"
DEFAULT_FRAMES_DIR = DEFAULT_DATASET_ROOT / "saved_frames"
DEFAULT_OUT_FRAMES_DIR = DEFAULT_DATASET_ROOT / "saved_frames_240x320"
DEFAULT_BACKUP_LOGS_DIR = DEFAULT_DATASET_ROOT / "logs_before_resize_paths"


@dataclass(frozen=True)
class ResizeTask:
    src: Path
    dst: Path
    width: int
    height: int
    overwrite: bool


def _trial_num(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _resolve_image_path(
    raw: str,
    *,
    trial_id: int,
    timestep: int,
    dataset_root: Path,
    frames_dir: Path,
) -> Optional[Path]:
    raw = (raw or "").strip()
    fallback = frames_dir / f"trial_{trial_id}" / f"frame_{timestep:06d}.png"
    if not raw:
        return fallback if fallback.exists() else None

    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None

    for base in (REPO_ROOT, dataset_root, frames_dir):
        candidate = base / p
        if candidate.exists():
            return candidate

    return fallback if fallback.exists() else None


def _path_for_csv(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resize_one(task: ResizeTask) -> tuple[Path, bool, Optional[str]]:
    """Return (dst, wrote_file, error_message)."""
    if task.dst.exists() and not task.overwrite:
        return task.dst, False, None

    task.dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(task.src) as img:
            img = img.convert("RGB").resize(
                (task.width, task.height), Image.Resampling.BILINEAR
            )
            img.save(task.dst)
    except Exception as exc:  # pragma: no cover - best reported at runtime
        return task.dst, False, f"{task.src}: {exc}"
    return task.dst, True, None


def _source_csv_for(csv_path: Path, backup_logs_dir: Path) -> Path:
    backup = backup_logs_dir / csv_path.name
    return backup if backup.exists() else csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize dataset/saved_frames once and update dataset/logs image_path "
            "columns to point at the resized frame tree."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--out-frames-dir", type=Path, default=DEFAULT_OUT_FRAMES_DIR)
    parser.add_argument("--backup-logs-dir", type=Path, default=DEFAULT_BACKUP_LOGS_DIR)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--image-col", type=str, default="image_path")
    parser.add_argument("--timestep-col", type=str, default="timestep")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Regenerate resized images even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-backups",
        action="store_true",
        help="Replace CSV backups if they already exist.",
    )
    parser.add_argument(
        "--no-update-csv",
        action="store_true",
        help="Only write resized images; leave dataset/logs CSV files unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned work without writing images or CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    logs_dir = args.logs_dir.expanduser().resolve()
    frames_dir = args.frames_dir.expanduser().resolve()
    out_frames_dir = args.out_frames_dir.expanduser().resolve()
    backup_logs_dir = args.backup_logs_dir.expanduser().resolve()

    csv_paths = sorted(logs_dir.glob("trial_*.csv"), key=_trial_num)
    if not csv_paths:
        raise FileNotFoundError(f"No trial_*.csv files found under {logs_dir}")

    tasks: list[ResizeTask] = []
    csv_updates: list[tuple[Path, pd.DataFrame]] = []
    missing_images: list[str] = []

    for csv_path in csv_paths:
        source_csv = _source_csv_for(csv_path, backup_logs_dir)
        df = pd.read_csv(source_csv)
        trial_id = _trial_num(csv_path)

        image_paths: list[str] = []
        for row_index, row in df.iterrows():
            timestep = (
                int(row[args.timestep_col])
                if args.timestep_col in df.columns
                else int(row_index)
            )
            raw = str(row[args.image_col]) if args.image_col in df.columns else ""
            src = _resolve_image_path(
                raw,
                trial_id=trial_id,
                timestep=timestep,
                dataset_root=dataset_root,
                frames_dir=frames_dir,
            )
            dst = out_frames_dir / f"trial_{trial_id}" / f"frame_{timestep:06d}.png"
            image_paths.append(_path_for_csv(dst, dataset_root))

            if src is None:
                missing_images.append(f"{csv_path.name}: t={timestep} raw={raw!r}")
                continue
            tasks.append(
                ResizeTask(
                    src=src,
                    dst=dst,
                    width=args.width,
                    height=args.height,
                    overwrite=args.overwrite_images,
                )
            )

        if not args.no_update_csv:
            out_df = df.copy()
            out_df[args.image_col] = image_paths
            csv_updates.append((csv_path, out_df))

    print(f"trials={len(csv_paths)}")
    print(f"resize_tasks={len(tasks)}")
    print(f"missing_images={len(missing_images)}")
    print(f"out_frames_dir={out_frames_dir}")
    print(f"update_csv={not args.no_update_csv}")

    if missing_images:
        print("first_missing_images:")
        for item in missing_images[:20]:
            print(f"  {item}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")

    if args.dry_run:
        print("dry_run=True; no files written")
        return

    wrote = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_resize_one, task) for task in tasks]
        for future in as_completed(futures):
            _dst, did_write, error = future.result()
            wrote += int(did_write)
            if error is not None:
                errors.append(error)

    if errors:
        print("resize_errors:")
        for item in errors[:20]:
            print(f"  {item}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        raise SystemExit(f"Failed to resize {len(errors)} images")

    if csv_updates:
        backup_logs_dir.mkdir(parents=True, exist_ok=True)
        for csv_path, out_df in csv_updates:
            backup_path = backup_logs_dir / csv_path.name
            if args.overwrite_backups or not backup_path.exists():
                shutil.copy2(csv_path, backup_path)
            out_df.to_csv(csv_path, index=False)

    print(f"resized_images_written={wrote}")
    print(f"csvs_updated={len(csv_updates)}")
    if csv_updates:
        print(f"csv_backups={backup_logs_dir}")


if __name__ == "__main__":
    main()
