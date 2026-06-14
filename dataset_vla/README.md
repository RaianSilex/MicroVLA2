# MicroVLA dataset conversion

MicroVLA trains from a **LeRobot-format dataset** (SmolVLA / OpenPI / π0 style).
These scripts convert the raw micromanipulation logs into that format.

Raw layout consumed:

```
dataset/
├── logs/trial_N.csv
├── saved_frames/trial_N/frame_NNNNNN.png
└── instruction_labels.csv        # trial_id, region, [instruction]
```

## v3.0 (MicroVLA / SmolVLA) — the main path

```bash
# Build locally under HF_LEROBOT_HOME (default repo id RaianSilex/microvla_ump_dataset):
python dataset_vla/convert_microact_to_lerobot.py
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3   # quick subset
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub      # upload (needs HF login)
```

Produces standard LeRobot features so any LeRobot/SmolVLA tooling reads it:

- `observation.images.cam_main` — frame (letterboxed to 540×720 by default)
- `observation.state` — 8-D absolute Sensapex state
- `action` — 8-D **absolute** Sensapex target (robot-native)
- `task` — the per-trial instruction
- `observation.resistance` — **only if** the logs carry real `resistance_mohm`
  values (auto-detected; `--no-resistance` to skip)
- `observation.goal_pixel` — **only if** a `cell_labels.csv` exists (Variant B;
  auto-detected; `--no-cells` to skip). See *Cell labels* below.

### Grounded, varied instructions

The `task` string is built from the **target cell's region in the frame**, read
from `dataset/instruction_labels.csv`. The converter scaffolds that file (every
trial defaulting to `center`) on the first run and warns — **edit the `region`
column** per trial (or write a free-text `instruction` to override), then re-run.
Canonical regions: the 3×3 grid `top_left, top, top_right, left, center, right,
bottom_left, bottom, bottom_right`; many aliases normalize automatically
(`middle_left`→`left`, `top_center`→`top`, `lower-right`, `upper-left`, `centre`, …).
Wording varies deterministically per trial so language carries real signal.

### Action space: absolute on disk, delta at train time

Actions are stored **absolute** (matching the robot/ROS commands). Training
converts them to **deltas** relative to the current state
(`train_vla.py --action-space delta`, the default); `VLAPolicy.inference` converts
the predicted delta back to absolute, so the robot side never changes. Use
`--action-space absolute` to train on raw targets.

## Cell labels (Variant B — Cellpose as a teacher)

Optional. Generates a per-trial contact point so the policy can learn the
cell-aware selection + image-space contact-point heads. Cellpose runs **offline
only**; the trained policy never runs it.

```bash
# Needs `pip install 'cellpose>=4.0'`. Segments each trial's contact (last) frame,
# picks the detected cell nearest the labeled region -> dataset/cell_labels.csv.
python dataset_vla/generate_cell_labels.py            # --limit-trials 3 for a subset

# The converter then auto-adds observation.goal_pixel (disable with --no-cells):
python dataset_vla/convert_microact_to_lerobot.py
```

`cell_labels.csv` columns: `trial_id, goal_u, goal_v` (normalized pixels in
`[0, 1]`; `region`/`n_cells` are written for inspection). You can also hand-edit
or hand-author this file — the converter only needs `trial_id, goal_u, goal_v`.

## v2.1 (OpenPI / π0 / π0.5)

OpenPI pins an older `lerobot` and needs **v2.1**, not the v3.0 that lerobot ≥ 0.4
writes. The on-disk version is decided by the *installed* `lerobot`, so build the
v2.1 copy in a separate venv:

```bash
python3 -m venv .lerobot-v21-venv && source .lerobot-v21-venv/bin/activate
pip install "lerobot==0.3.3"
python dataset_vla/convert_microact_to_lerobot_v21.py     # -> <repo-id>_v21
```

Same content and region/instruction logic as v3.0; only the on-disk format
differs. The script aborts if the installed `lerobot` would write v3.0.
