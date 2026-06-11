# MicroVLA Dataset (LeRobot format)

MicroVLA trains from a **LeRobot dataset** on the Hugging Face Hub — the same
convention SmolVLA, OpenPI and π0 use. Build one from your raw MicroACT data
(`dataset/` = `logs/trial_N.csv` + `saved_frames/trial_N/`) with the converters
in this folder. The dataset is robot-native and reusable by any VLA.

> The older `dataset_vla/episodes/` intermediate format (and its
> `convert_microact_to_vla.py` converter) was removed — MicroVLA reads LeRobot
> datasets directly now. For the full end-to-end upload recipe (v2.1 + v3, HF
> auth, parallel builds) see [../HUGGINGFACE_LEROBOT_UPLOAD.md](../HUGGINGFACE_LEROBOT_UPLOAD.md).

```bash
# Build the LeRobot dataset locally under HF_LEROBOT_HOME (no push):
python dataset_vla/convert_microact_to_lerobot.py

# Quick subset for a smoke test:
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3

# Push to your HF account (needs `huggingface-cli login` first):
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub
```

What it produces (standard LeRobot v3 schema, so `lerobot`/smolvla tooling
reads it directly):

- `observation.images.cam_main` — frame (PNG, letterboxed to 540×720 by default)
- `observation.state` — 8-D absolute Sensapex state (4-D for single-uMp datasets)
- `action` — 8-D **absolute** Sensapex target (robot-native; see action space below)
- `task` — the per-trial instruction

> Single-uMp (4-DoF) datasets: set `CSV_STATE_COLS`/`CSV_ACTION_COLS` in
> `config/config.py` to the uMp1 columns only, and convert with a new
> `--robot-type` (e.g. `sensapex_single_ump4`). The CSV can keep all 8 columns —
> the converter reads only the ones you list.

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
