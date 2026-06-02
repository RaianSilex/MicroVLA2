# MicroVLA End-to-End: Raw CSVs → LeRobot on Hugging Face → Train on MSI → Inference Locally

This is the complete walk-through for the LeRobot/Hugging-Face MicroVLA pipeline:

```
┌─ Your computer ──────────┐     ┌─ MSI (UMN HPC) ─────────┐     ┌─ Your computer ─────────┐
│ raw dataset/             │     │ pull dataset from HF    │     │ download checkpoint     │
│  logs/trial_N.csv        │     │ train_vla.py            │     │ rollout.vla_main        │
│  saved_frames/...        │ HF  │  --dataset-repo-id ...  │ scp │  --adapter sensapex_dual│
│ convert → LeRobot ──────────►  │  --backbone ...         │ ──► │  → ROS2 → Sensapex rig  │
│ push to Hugging Face     │     │ checkpoints_vla_*/      │     │                         │
└──────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
   Stage 1                           Stage 2                          Stage 3
```

Why this shape: the dataset lives on Hugging Face in **LeRobot format** (the same
convention SmolVLA / OpenPI / π0 use), so MicroVLA — or any other VLA — can train
from it. The dataset stores **absolute** Sensapex targets; MicroVLA trains on
**deltas** and converts them back to absolute at inference, so the robot side is
unchanged. **MicroACT is untouched** — this is the MicroVLA path only.

---

## 0. Conventions and prerequisites

Paths used below (adjust to your machines):

| Where | Path | Notes |
|---|---|---|
| Your computer, repo | `~/MicroVLA` | your local clone |
| MSI username | `chowd207` | |
| MSI project | `/projects/standard/suhasabk` | |
| MSI repo | `/projects/standard/suhasabk/shared/MicroVLA` | |
| MSI conda env | `/projects/standard/suhasabk/shared/conda_envs/microvla` | |
| HF dataset repo | `RaianSilex/microvla_ump_dataset` | private by default |

You need:

- A **Hugging Face account** + a **write token** (https://huggingface.co/settings/tokens).
- An **MSI account** with GPU access (see `MSI_MICROVLA_TRAINING_GUIDE.md`).
- Python 3.10 on your computer.

One key rule that prevents silent breakage: the dataset's `robot_type` (default
`sensapex_dual_ump4`) is used as the **per-robot normalization key** and **must
match the rollout adapter's `robot_id`**. The defaults already line up; only
change one if you change both.

---

## Stage 1 — Your computer: raw CSVs → LeRobot dataset on Hugging Face

### 1.1 Lay out the raw dataset

Put your trials in the classic MicroACT layout under `~/MicroVLA/dataset/`:

```text
dataset/
├── logs/
│   ├── trial_1.csv
│   ├── trial_2.csv
│   └── ...
└── saved_frames/
    ├── trial_1/frame_000000.png
    ├── trial_1/frame_000001.png
    └── ...
```

Each `trial_N.csv` must contain (other columns are ignored):

```
timestep,
current_x, current_y, current_z, current_d,    current_x2, current_y2, current_z2, current_d2,
target_x,  target_y,  target_z,  target_d,     target_x2,  target_y2,  target_z2,  target_d2,
image_path
```

`image_path` may be relative (e.g. `saved_frames/trial_1/frame_000000.png`) or
empty — the converter falls back to `saved_frames/trial_N/frame_NNNNNN.png`.

### 1.2 One-time local environment

```bash
cd ~/MicroVLA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # installs lerobot (+ huggingface_hub, datasets)
```

`lerobot` requires numpy 2.x; this repo is numpy-2 compatible. Conversion is
CPU-only (no GPU needed). `transformers`/`cellpose` are only needed later for
training/inference with the heavy backbones.

### 1.3 Label the target-cell regions (this is what makes language matter)

MicroVLA's instructions are grounded in **where the target cell sits in the
frame**, which is not in the CSVs — so you label it once. Run the converter once
to scaffold the labels file:

```bash
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 1
```

It creates `dataset/instruction_labels.csv` with every trial defaulted to
`center` and prints a warning. Open it and set the `region` per trial:

```csv
trial_id,region,instruction
1,top_left,
2,middle_right,
3,bottom,
4,center,
...
```

- **Canonical regions** (3×3 grid): `top_left, top, top_right, left, center,
  right, bottom_left, bottom, bottom_right`.
- **Aliases also accepted** and normalized: `middle_left`/`middle_right` →
  `left`/`right`, `top_center` → `top`, `bottom_center` → `bottom`,
  `lower-right`, `upper-left`, `centre`, …
- Leave `instruction` blank to auto-generate a grounded, lexically-varied prompt
  (e.g. *"guide the pipettes to the cell in the top-left"*), or type your own to
  override that trial.

> If every trial stays `center`, the language channel won't vary and the
> converter warns you — the model would then ignore language (it becomes a
> vision+state policy). Variation in `region` that correlates with the motion is
> what gives the "L" real signal.

### 1.4 Build the LeRobot dataset locally and sanity-check it

```bash
python dataset_vla/convert_microact_to_lerobot.py
```

This writes `~/.cache/huggingface/lerobot/RaianSilex/microvla_ump_dataset/`
(features `observation.images.cam_main` / `observation.state` / `action`, images
letterboxed to 540×720, `robot_type=sensapex_dual_ump4`, fps 3). Check the
summary line `region_counts=...` shows variety. Quick inspection:

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
ds = LeRobotDataset("RaianSilex/microvla_ump_dataset",
                    root=HF_LEROBOT_HOME / "RaianSilex/microvla_ump_dataset")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames)
print("tasks:\n", ds.meta.tasks)
PY
```

Useful flags: `--limit-trials N` (quick test), `--down-h/--down-w` (image size),
`--no-keep-aspect` (exact resize instead of letterbox), `--robot-type NAME`
(must match the rollout adapter), `--overwrite` (rebuild, default on).

### 1.5 Push to Hugging Face

Use the Hugging Face CLI's resumable large-folder uploader for the built dataset.
It is more reliable than a single `upload_folder` commit for multi-GB LeRobot
datasets, and if it crashes or your network drops, re-run the exact same command.

```bash
hf auth login            # paste your write token (once per machine)

hf upload-large-folder \
  RaianSilex/microvla_ump_dataset \
  ~/.cache/huggingface/lerobot/RaianSilex/microvla_ump_dataset \
  --repo-type dataset \
  --private \
  --num-workers 2
```

If the upload stalls or exits midway, retry:

```bash
hf upload-large-folder \
  RaianSilex/microvla_ump_dataset \
  ~/.cache/huggingface/lerobot/RaianSilex/microvla_ump_dataset \
  --repo-type dataset \
  --private \
  --num-workers 1
```

`--num-workers 1` is slower but gentler on unstable connections. The dataset is
pushed **private** when the repo is created. Verify at
`https://huggingface.co/datasets/RaianSilex/microvla_ump_dataset`.

Re-uploading to the same dataset repo overwrites files with the same path and
adds new files, but it does not delete old files that are absent locally. If the
local folder structure changed and you need a clean replacement, delete/recreate
the dataset repo on Hugging Face first or upload to a new repo id.

Avoid the `push_to_huggingface.py` helper for this large dataset
unless you specifically need it; it uses the normal `upload_folder` API, which is
less resilient for large resumable uploads.

---

## Stage 2 — MSI: train MicroVLA from the HF dataset

Full MSI environment setup (conda env, caches, GPU `srun`/Slurm partitions) is in
`MSI_MICROVLA_TRAINING_GUIDE.md`. The condensed LeRobot-specific steps:

### 2.1 Repo + env on MSI

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA
module purge && module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

# Caches in project storage (not home quota):
export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"
```

Make sure the repo on MSI is up to date (it must contain
`data/lerobot_vla_dataset.py` and `dataset_vla/convert_microact_to_lerobot.py`)
and that `lerobot` is installed in the env:

```bash
python -c "import lerobot, data.lerobot_vla_dataset; print('lerobot path OK')"
# If lerobot is missing:  python -m pip install "lerobot>=0.4"
```

You do **not** need to rsync the 26 GB raw dataset to MSI — training pulls the
(downsized) LeRobot dataset from HF.

### 2.2 Authenticate to Hugging Face on MSI (to pull the private dataset)

```bash
huggingface-cli login            # token cached under $HF_HOME, reused by Slurm jobs
# (or, non-interactively in a job:  export HF_TOKEN=hf_xxx)
```

### 2.3 Choose a training mode

| Mode | Flags | Needs |
|---|---|---|
| Smoke | `--backbone resnet18 --language-backend simple --no-pretrained` | nothing extra |
| Light VLA | `--backbone resnet18 --language-backend hf` | `transformers` |
| **Full VLA** | `--backbone dinov2_vits14+cellpose4 --language-backend hf` | `transformers`, `cellpose>=4.0` |
| Lighter cell-aware | `--backbone dinov2_vits14+cellpose --language-backend hf` | `transformers`, `cellpose` |

Action space (default `delta`, recommended): add `--action-space delta` or
`--action-space absolute`. It is saved in the checkpoint and used automatically
at rollout.

### 2.4 Slurm script (LeRobot `--dataset-repo-id`)

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
mkdir -p logs checkpoints_vla_lerobot

cat > train_vla_lerobot.sbatch <<'EOF'
#!/bin/bash -l
#SBATCH --job-name=microvla-lerobot
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100g
#SBATCH --time=24:00:00
#SBATCH --tmp=100g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu

set -euo pipefail
cd /projects/standard/suhasabk/shared/MicroVLA
module purge && module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"

echo "Node: $(hostname)"; nvidia-smi

REPO_ID="RaianSilex/microvla_ump_dataset"
CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_lerobot"
mkdir -p "$CKPT_DIR"

# Optional speedup: pre-stage the cached dataset to fast node-local storage and
# train from there with --dataset-root. (Requires it to be cached under $HF_HOME
# first, e.g. by one prior run, or remove this block to pull from HF directly.)
DATASET_ROOT_ARG=()
SRC="$HF_HOME/lerobot/$REPO_ID"
if [[ -d "$SRC" ]]; then
  JOB_DS="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}/lerobot/$REPO_ID"
  mkdir -p "$JOB_DS"
  rsync -a --delete "$SRC/" "$JOB_DS/"
  DATASET_ROOT_ARG=(--dataset-root "$JOB_DS")
  echo "Training from node-local copy: $JOB_DS"
fi

RESUME_ARG=()
[[ -f "$CKPT_DIR/vla_policy_last.pt" ]] && RESUME_ARG=(--resume "$CKPT_DIR/vla_policy_last.pt")

python train_vla.py \
  --dataset-repo-id "$REPO_ID" \
  "${DATASET_ROOT_ARG[@]}" \
  --action-space delta \
  --backbone dinov2_vits14+cellpose4 \
  --language-backend hf \
  --text-model distilbert-base-uncased \
  --epochs 2000 \
  --batch-size 2 \
  --num-workers 4 \
  --device cuda \
  --ckpt-dir "$CKPT_DIR" \
  --save-every 100 \
  "${RESUME_ARG[@]}"
EOF

sbatch train_vla_lerobot.sbatch
```

For a quick smoke first, swap the python args for
`--backbone resnet18 --language-backend simple --no-pretrained --epochs 1 --batch-size 8`.

> Offline alternative (no HF pull at train time): rsync your locally-built
> dataset folder to MSI and train with `--dataset-root <that folder>` and the
> same `--dataset-repo-id`. The repo id is still used as the per-robot stats key.

### 2.5 Monitor / resume

```bash
squeue -u chowd207
tail -f logs/microvla-lerobot-<JOBID>.out
```

Resubmitting the same script auto-resumes from `vla_policy_last.pt`. Resume
rebuilds the exact prior setup (backbone, action space, LoRA/freeze) from the
checkpoint's saved `config`. Don't mix backbones into one checkpoint dir.

### 2.6 Download the trained checkpoint back to your computer

```bash
mkdir -p ~/MicroVLA/checkpoints_msi_vla
rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_lerobot/ \
  ~/MicroVLA/checkpoints_msi_vla/
```

The VLA checkpoint is self-contained: it stores `stats`, `vocabs`, and `config`
(backbone, language backend, **action_space**, dataset repo id). You do **not**
need a separate stats file at inference.

---

## Stage 3 — Your computer: inference

### 3.0 Inference environment

Inference does **not** need `lerobot`. It needs:

```bash
# torch/torchvision + numpy + pillow (already in .venv), plus:
pip install transformers            # if the checkpoint used --language-backend hf
pip install "cellpose>=4.0"         # if the backbone includes cellpose / cellpose4
# rclpy comes from your ROS2 install (for the real rollout in 3.2)
```

The backbone, language backend and action space are read from the checkpoint, so
you usually don't pass them.

### 3.1 Quick offline sanity check (no robot)

Confirms the checkpoint loads and emits sane absolute targets:

```bash
cd ~/MicroVLA
python - <<'PY'
import numpy as np, torch
from model.vla_policy import build_vla_policy

ckpt = torch.load("checkpoints_msi_vla/vla_policy_best.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
policy = build_vla_policy(
    stats=ckpt["stats"], vocabs=ckpt["vocabs"],
    pretrained_backbone=False, backbone_name=cfg["backbone"],
    freeze_backbone=cfg.get("freeze_backbone", True),
    language_backend=cfg["language_backend"], text_model_name=cfg["text_model"],
    action_space=cfg["action_space"],
)
policy.load_state_dict(ckpt["policy"]); policy.eval()
print("backbone=%s language=%s action_space=%s" % (cfg["backbone"], cfg["language_backend"], cfg["action_space"]))

# EDIT this to a real current 8-D state [x1,y1,z1,d1, x2,y2,z2,d2]:
state = np.array([18587,18065,14322,15480, 11506,10861,18594,12945], np.float32)
img = (np.random.rand(540,720,3)*255).astype(np.uint8)   # or a real microscope frame (RGB)
chunk = policy.inference(img, state, "guide the pipettes to the cell in the top-left",
                         robot_id="sensapex_dual_ump4", state_dim=8, action_dim=8)
print("action chunk shape:", chunk.shape, "| finite:", bool(np.isfinite(chunk).all()))
print("first target:", chunk[0].astype(int), "  (delta mode -> near current state)")
PY
```

### 3.2 Real closed-loop rollout on the rig (ROS2)

> ⚠️ **Safety first.** Edit the per-axis workspace bounds and per-tick step caps
> at the top of `rollout/main.py` (`X1_MIN/X1_MAX … D2_MIN/D2_MAX`, `MAX_DX1 …`)
> for your real workspace **before** publishing commands. The shipped values are
> placeholders. The Sensapex dual adapter reuses these limits.

```bash
cd ~/MicroVLA
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash          # your ROS workspace
python3 -c "import rclpy, torch; print('rclpy + torch OK')"

# Dry run first (computes commands but does NOT publish):
python3 -m rollout.vla_main \
  --checkpoint checkpoints_msi_vla/vla_policy_best.pt \
  --adapter sensapex_dual \
  --instruction "guide the pipettes to the cell in the top-left" \
  --dry-run

# Real run (publishes /ump/target + /ump2/target):
python3 -m rollout.vla_main \
  --checkpoint checkpoints_msi_vla/vla_policy_best.pt \
  --adapter sensapex_dual \
  --instruction "guide the pipettes to the cell in the top-left"
```

During a rollout: `Ctrl+C` stops; `q` + Enter is the E-stop (holds current
position and exits). Useful flags: `--control-hz`, `--no-temporal-agg`,
`--open-loop-horizon`, `--ema-alpha`, `--debug-every`. The instruction should
match the style/region words you trained on.

ROS topics used (via `rollout/sensapex_env.py`): subscribes
`/camera/image/compressed`, `/ump/live`, `/ump2/live`; publishes `/ump/target`,
`/ump2/target` (`[x,y,z,d,speed]`).

### 3.3 What the action space means here

Whatever you trained with, **`rollout.vla_main` always sends absolute targets**:

- If trained `delta`: the policy predicts small deltas and adds the current
  measured state inside `.inference()` → absolute targets near the current pose.
- If trained `absolute`: the policy predicts absolute targets directly.

Either way the adapter clamps to the workspace, caps per-tick motion, optionally
EMA-smooths, and publishes absolute commands — your ROS package and the Sensapex
controllers are unchanged.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RepositoryNotFoundError` / 401 when pulling on MSI | Not logged in / private repo. `huggingface-cli login` (with `$HF_HOME` exported) or `export HF_TOKEN=hf_xxx`. |
| Instructions all identical (`region_counts` shows one region) | You didn't edit `dataset/instruction_labels.csv`. Set varied `region`s and re-run the converter (+ re-push). |
| Robot moves to a weird fixed pose / ignores state | `robot_id` mismatch: the checkpoint's `robot_type` ≠ what the adapter passes, so inference used the UNKNOWN (identity) stats row. Keep both `sensapex_dual_ump4`. |
| `ModuleNotFoundError: transformers` at train/inference | `pip install transformers` (only for `--language-backend hf`). |
| `cellpose4` import/weights fail | `pip install "cellpose>=4.0"`; CP-SAM weights download on first use (set `CELLPOSE_LOCAL_MODELS_PATH`). |
| OOM on `dinov2_vits14+cellpose4` | Lower `--batch-size` (2 → 1); or use `dinov2_vits14+cellpose` / `resnet18`. |
| `numpy` conflict after installing lerobot | Expected — lerobot needs numpy 2.x; this repo is compatible. |
| Robot jitter between chunks | Keep temporal aggregation on; lower `--temporal-agg-k` or `--ema-alpha`. |
| BGR vs RGB (robot lurches) | The adapter decodes to RGB already; if you feed frames elsewhere, convert BGR→RGB. |

---

## Command cheat-sheet

```bash
# 1. Your computer: build + push dataset
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 1   # scaffold labels
#   ... edit dataset/instruction_labels.csv ...
python dataset_vla/convert_microact_to_lerobot.py                    # build all
hf auth login
hf upload-large-folder RaianSilex/microvla_ump_dataset \
  ~/.cache/huggingface/lerobot/RaianSilex/microvla_ump_dataset \
  --repo-type dataset --private --num-workers 2

# 2. MSI: train (after env setup + huggingface-cli login)
sbatch train_vla_lerobot.sbatch
rsync -avhP chowd207@login.msi.umn.edu:.../checkpoints_vla_lerobot/ ~/MicroVLA/checkpoints_msi_vla/

# 3. Your computer: inference
python3 -m rollout.vla_main --checkpoint checkpoints_msi_vla/vla_policy_best.pt \
  --adapter sensapex_dual --instruction "guide the pipettes to the cell in the top-left" --dry-run
```
