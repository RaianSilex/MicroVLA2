# MicroVLA Dataset Layout

Place heterogeneous demonstrations under:

```text
dataset_vla/
└── episodes/
    └── <episode_id>/
        ├── metadata.json
        ├── trajectory.csv
        └── frames/
            └── cam_main/
                ├── frame_000000.png
                └── frame_000001.png
```

Minimal `metadata.json`:

```json
{
  "episode_id": "lab_a_trial_0001",
  "lab_id": "lab_a",
  "robot_id": "sensapex_dual_ump4",
  "embodiment": "dual_manipulator",
  "action_type": "absolute_position",
  "task_family": "cell_manipulation",
  "instruction": "move both manipulators toward the selected cell",
  "camera_names": ["cam_main"],
  "state_dim": 8,
  "action_dim": 8,
  "state_cols": [
    "current_x", "current_y", "current_z", "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2"
  ],
  "action_cols": [
    "target_x", "target_y", "target_z", "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2"
  ],
  "image_col": "image_path",
  "timestep_col": "timestep"
}
```

Single-manipulator episodes can use `state_dim: 4` and `action_dim: 4`.
The loader pads all state/action tensors to the shared VLA maximums and masks
invalid dimensions during loss computation.

## Convert an existing MicroACT dataset

If you already have the classic MicroACT layout:

```text
dataset/
├── logs/trial_N.csv
└── saved_frames/trial_N/frame_000000.png
```

convert it before running `train_vla.py`:

```bash
python3 dataset_vla/convert_microact_to_vla.py \
  --replace-zero-targets-with-state
```

The converter writes `dataset_vla/episodes/trial_N/metadata.json`,
`trajectory.csv`, and `frames/cam_main/`. By default frames are symlinked, not
copied, so the conversion does not duplicate the image dataset. Use
`--frame-mode copy` if you need a standalone `dataset_vla/` tree.

## LeRobot format on Hugging Face (recommended — SmolVLA / OpenPI / π0 style)

MicroVLA can also train from a **LeRobot dataset** that lives on the HF Hub, the
same convention SmolVLA, OpenPI and π0 use. This is the recommended path: the
dataset is robot-native and reusable by any VLA.

```bash
# Build the LeRobot dataset locally under HF_LEROBOT_HOME (no push):
python dataset_vla/convert_microact_to_lerobot.py

# Quick subset for a smoke test:
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3

# Push to your HF account (needs `huggingface-cli login` first):
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub
```

What it produces (standard LeRobot v2/v3 schema, so `lerobot`/smolvla tooling
reads it directly):

- `observation.images.cam_main` — frame (PNG, letterboxed to 540×720 by default)
- `observation.state` — 8-D absolute Sensapex state
- `action` — 8-D **absolute** Sensapex target (robot-native; see action space below)
- `task` — the per-trial instruction

### Varied, grounded instructions

The `task` string is built from the **target cell's position in the frame**, read
from an editable labels file `dataset/instruction_labels.csv`
(`trial_id,region,instruction`). The converter auto-scaffolds it (every trial
defaulted to `center`) on the first run and prints a warning — **edit the
`region` column** per trial, or write a free-text `instruction` to override, then
re-run. Canonical regions are the 3×3 grid (`top_left, top, top_right, left,
center, right, bottom_left, bottom, bottom_right`), and many natural aliases are
accepted and normalized (`middle_left`/`middle_right` → `left`/`right`,
`top_center` → `top`, `bottom_center` → `bottom`, `lower-right`, `upper-left`,
`centre`, …). Wording is varied deterministically per trial so the
language channel carries real signal (e.g. *"guide the pipettes to the cell in
the top-left"*) instead of one constant prompt. If every trial stays `center`,
language won't vary and the converter warns you.

### Action space: absolute on disk, delta at train time

The dataset stores **absolute** targets (matching the robot/ROS commands).
Training converts them to **deltas**
relative to the current state (`--action-space delta`, the default), which are
small and workspace-translation invariant. At rollout `VLAPolicy.inference`
converts the predicted delta **back to absolute**, so the ROS adapter and robot
side never change. Use `--action-space absolute` to train on raw targets instead.

### Train MicroVLA from the LeRobot dataset

```bash
# Offline smoke (no model downloads):
python train_vla.py --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --action-space delta --backbone resnet18 --language-backend simple --no-pretrained \
  --epochs 1 --batch-size 2 --num-workers 0

# Real run (frozen DistilBERT + DINOv2 + Cellpose4):
python train_vla.py --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --backbone dinov2_vits14+cellpose4 --language-backend hf
```

> Note: the dataset's `robot_type` (default `sensapex_dual_ump4`) is used as the
> per-robot normalization key and **must match the rollout adapter's `robot_id`**.
> OpenPI uses the same `observation.state` / `action` / `observation.images.cam_main`
> keys, but needs a **v2.1** copy of the dataset — see the next section.

### OpenPI / pi0 / pi0.5: build a v2.1 copy

OpenPI's training pipeline pins an older `lerobot` and **does not support the v3.0
dataset format** that `convert_microact_to_lerobot.py` produces. SmolVLA and
MicroVLA read v3.0 natively; OpenPI needs **v2.1**. Build a separate v2.1 copy with
`convert_microact_to_lerobot_v21.py` — same content and same instruction/region
logic, only the on-disk format differs (it defaults to a new repo id,
`<v3.0 id>_v21`, so it never clobbers the v3.0 dataset).

The on-disk version is decided by the *installed* `lerobot`, not the script:
`lerobot 0.3.x` writes v2.1; `lerobot >= 0.4` writes v3.0. So run it in a **separate
venv** (keep your main env on `lerobot>=0.4` for MicroVLA/SmolVLA):

```bash
python3 -m venv .lerobot-v21-venv
source .lerobot-v21-venv/bin/activate
pip install "lerobot==0.3.3"

# build locally -> RaianSilex/microvla_ump_dataset_v21:
python dataset_vla/convert_microact_to_lerobot_v21.py
# quick smoke (3 trials):
python dataset_vla/convert_microact_to_lerobot_v21.py --limit-trials 3
```

The script reads back the format `LeRobotDataset.create` actually wrote and aborts
if it is not v2.x, so you can't accidentally produce v3.0.

Push it (after `huggingface-cli login` with a write token):

```bash
# A) push during the build (private):
python dataset_vla/convert_microact_to_lerobot_v21.py --push-to-hub

# B) or build first, then push the local folder:
python push_to_huggingface.py \
  --local-dir ~/.cache/huggingface/lerobot/RaianSilex/microvla_ump_dataset_v21 \
  --repo-id RaianSilex/microvla_ump_dataset_v21 --repo-type dataset --private
```

Action representation is still **absolute on disk**. To match MicroVLA's delta in
OpenPI, enable its `DeltaActions` transform (per-dimension mask) so actions are
predicted relative to the current state and converted back to absolute at inference,
and compute OpenPI's norm stats *after* that transform.
