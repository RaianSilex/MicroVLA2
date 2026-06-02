# MSI MicroVLA Training Guide

This guide is the end-to-end recipe for training MicroVLA on MSI from the home
laptop.

Current assumptions:

- Local laptop repo: `/home/raianlaptop/MicroVLA`
- MSI username: `chowd207`
- MSI project directory: `/projects/standard/suhasabk`
- MSI repo copy: `/projects/standard/suhasabk/shared/MicroVLA`
- Current source dataset: classic MicroACT trials under `dataset/`
- VLA dataset on MSI: generated under `dataset_vla/episodes/`

MSI references:

- Slurm partitions and GPU requests: https://msi.umn.edu/computing/slurm-scheduler/slurm-partitions
- Agate interactive GPU example: https://msi.umn.edu/about-msi-services/high-performance-computing/agate/agate-interactive-nodes
- Interactive `srun`: https://msi.umn.edu/about-msi-services/interactive-hpc/interactive-computing-srun
- Conda module: https://msi.umn.edu/software/msi-software/conda

## 0. What Is Different From MicroACT

MicroACT training reads this directly:

```text
dataset/logs/trial_N.csv
dataset/saved_frames/trial_N/frame_000000.png
```

MicroVLA training reads this:

```text
dataset_vla/episodes/trial_N/
  metadata.json
  trajectory.csv
  frames/cam_main/frame_000000.png
```

For the current dual-Sensapex dataset, generate VLA episodes from the existing
MicroACT dataset on MSI using:

```bash
python dataset_vla/convert_microact_to_vla.py --replace-zero-targets-with-state
```

The converter symlinks frames by default, so it does not duplicate the 26 GB
image dataset.

There are three useful MicroVLA training modes:

```text
Smoke test:
  backbone=resnet18
  language-backend=simple
  no external model downloads

Light VLA:
  backbone=resnet18
  language-backend=hf
  uses frozen DistilBERT language tokens

Full VLA:
  backbone=dinov2_vits14+cellpose4
  language-backend=hf
  uses DINOv2, Cellpose4 / CP-SAM, and DistilBERT
```

Start with the smoke test. Then use either the light VLA or full VLA Slurm
script depending on what you want to train.

## 1. Copy The Repo And Dataset To MSI

Run this from the laptop, not from MSI:

```bash
rsync -avhP --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*/__pycache__/' \
  --exclude 'onnx_exports/' \
  --exclude 'torchviz_exports/' \
  --exclude '_report_gen/' \
  --exclude 'checkpoints/*.pt' \
  --exclude 'checkpoints/*.pth' \
  --exclude 'checkpoints_resnet/' \
  --exclude 'checkpoints_vla*/' \
  --exclude 'logs/' \
  --exclude 'train_resnet*.sbatch' \
  --exclude 'train_vla*.sbatch' \
  --exclude 'dataset_vla/episodes/' \
  /home/raianlaptop/MicroVLA/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

Why `dataset_vla/episodes/` is excluded:

- For the current workflow, episodes should be regenerated on MSI.
- Regenerating on MSI creates symlinks that point correctly to the MSI copy of
  `dataset/saved_frames/`.

Verify the upload on MSI:

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA

find dataset/logs -maxdepth 1 -name 'trial_*.csv' | wc -l
find dataset/saved_frames -type f -name '*.png' | wc -l
du -sh dataset
```

Expected current values:

```text
66 trial CSV files
11355 PNG frames
about 26G dataset size
```

## 1A. Native VLA Episodes Instead Of Converted MicroACT Data

If you later receive true heterogeneous VLA episodes from another robot or lab,
put them on the laptop under:

```text
/home/raianlaptop/MicroVLA/dataset_vla/episodes/<episode_id>/
```

Then upload without excluding `dataset_vla/episodes/`:

```bash
rsync -avhP \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*/__pycache__/' \
  --exclude 'onnx_exports/' \
  --exclude 'torchviz_exports/' \
  --exclude '_report_gen/' \
  --exclude 'checkpoints/*.pt' \
  --exclude 'checkpoints/*.pth' \
  --exclude 'checkpoints_vla*/' \
  --exclude 'logs/' \
  --exclude 'train_vla*.sbatch' \
  /home/raianlaptop/MicroVLA/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

If you use native VLA episodes, skip the conversion section below.

## 2. Start An Interactive GPU Session For Setup

Run this after SSH, from an MSI login node.

For A100:

```bash
srun -N 1 --ntasks-per-node=16 --mem=60gb \
  --gres=gpu:a100:1 -t 8:00:00 -p msigpu \
  --tmp 100gb --pty bash -l
```

For L40S:

```bash
srun -N 1 --ntasks-per-node=16 --mem=60gb \
  --gres=gpu:l40s:1 -t 8:00:00 -p interactive-gpu \
  --tmp 100gb --pty bash -l
```

For H100:

```bash
srun -N 1 --ntasks-per-node=16 --mem=80gb \
  --gres=gpu:h100:1 -t 8:00:00 -p msigpu \
  --tmp 100gb --pty bash -l
```

If the command says `queued and waiting for resources`, wait. Slurm is looking
for a matching GPU.

After allocation:

```bash
hostname
nvidia-smi
```

## 3. Create Or Activate The MicroVLA Python Environment

Run this on the GPU node from the interactive session.

Use a project-directory conda environment so the large Python packages do not
fill the MSI home quota:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda

ENV_PREFIX=/projects/standard/suhasabk/shared/conda_envs/microvla
```

If the environment already exists:

```bash
source activate "$ENV_PREFIX"
```

If it does not exist:

```bash
mkdir -p /projects/standard/suhasabk/shared/conda_envs
conda create -p "$ENV_PREFIX" python=3.10 -y
source activate "$ENV_PREFIX"
```

Set model caches in project storage:

```bash
export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"
```

Install basic packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install "numpy>=1.24,<2" "pandas>=2.0" "pillow>=10.0" "transformers>=4.40"
```

If you will train the full `dinov2_vits14+cellpose4` model, also install
Cellpose:

```bash
python -m pip install "cellpose>=4.0"
```

Verify the environment:

```bash
which python

python - <<'PY'
import numpy, pandas, PIL, torch, transformers
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("PIL ok")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

If you installed Cellpose:

```bash
python - <<'PY'
import importlib.metadata
print("cellpose", importlib.metadata.version("cellpose"))
PY
```

## 4. Convert The Current MicroACT Dataset To VLA Episodes

Run this on MSI with the `microvla` environment activated. It can run on the GPU
node or on a login node after activating the same environment.

```bash
cd /projects/standard/suhasabk/shared/MicroVLA

python dataset_vla/convert_microact_to_vla.py \
  --replace-zero-targets-with-state
```

If you are rerunning the converter over existing episode directories, add
`--overwrite`:

```bash
python dataset_vla/convert_microact_to_vla.py \
  --overwrite \
  --replace-zero-targets-with-state
```

Expected output includes:

```text
converted_episodes=66
rows_written=11355
output=/projects/standard/suhasabk/shared/MicroVLA/dataset_vla/episodes
```

Verify:

```bash
find dataset_vla/episodes -mindepth 1 -maxdepth 1 -type d -name 'trial_*' | wc -l
ls dataset_vla/episodes/trial_1
head -n 20 dataset_vla/episodes/trial_1/metadata.json
```

The first command should currently print:

```text
66
```

## 5. Run A Fast MicroVLA Smoke Test

This confirms VLA data loading, metadata tokens, language path, ResNet backbone,
and checkpoint writing.

Run this on the interactive GPU node:

```bash
python train_vla.py \
  --episodes-dir dataset_vla/episodes \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --backbone resnet18 \
  --language-backend simple \
  --no-pretrained \
  --ckpt-dir /tmp/microvla_smoke_resnet
```

If this finishes with:

```text
done. best val loss: ...
```

the VLA pipeline is wired correctly.

Optional Hugging Face download check:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModel
name = "distilbert-base-uncased"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print("hf ok")
PY
```

Optional DINOv2 download check:

```bash
python - <<'PY'
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
print("dinov2 ok")
PY
```

Leave the interactive GPU session:

```bash
exit
```

You should return to an MSI login node prompt such as `ahl01`.

## 6. Train Light MicroVLA: ResNet18 + DistilBERT

This is the easiest real VLA training mode. It uses language conditioning with
frozen DistilBERT, but keeps the image backbone as ResNet18.

Run this on an MSI login node:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
mkdir -p logs checkpoints_vla_resnet
```

Create the Slurm script:

```bash
cat > train_vla_resnet.sbatch <<'EOF'
#!/bin/bash -l
#SBATCH --job-name=microvla-resnet
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=24:00:00
#SBATCH --tmp=100g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu

set -euo pipefail

cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

JOB_TMP="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}"
JOB_VLA_ROOT="$JOB_TMP/dataset_vla"
JOB_VLA_EPISODES="$JOB_VLA_ROOT/episodes"
mkdir -p "$JOB_VLA_EPISODES"

echo "Copying VLA episodes to node-local storage: $JOB_VLA_EPISODES"
rsync -aL --delete dataset_vla/episodes/ "$JOB_VLA_EPISODES/"
export MICROVLA_VLA_DATASET_ROOT="$JOB_VLA_ROOT"

CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_resnet"
mkdir -p "$CKPT_DIR"

RESUME_ARG=()
if [[ -f "$CKPT_DIR/vla_policy_last.pt" ]]; then
  RESUME_ARG=(--resume "$CKPT_DIR/vla_policy_last.pt")
  echo "Resuming from $CKPT_DIR/vla_policy_last.pt"
fi

python train_vla.py \
  --episodes-dir "$JOB_VLA_EPISODES" \
  --epochs 2000 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --backbone resnet18 \
  --language-backend hf \
  --text-model distilbert-base-uncased \
  --ckpt-dir "$CKPT_DIR" \
  --save-every 100 \
  "${RESUME_ARG[@]}"

echo "End: $(date)"
EOF
```

The dataset-copy block uses node-local temporary storage from:

```bash
#SBATCH --tmp=100g
```

It copies `dataset_vla/episodes/` into `$TMPDIR` and trains from:

```bash
--episodes-dir "$JOB_VLA_EPISODES"
```

The `-L` in `rsync -aL` matters for converted MicroACT episodes because their
frame files are symlinks by default. `-L` follows those symlinks and copies the
actual images into node-local storage, avoiding repeated image reads from MSI
project storage during training.

Submit:

```bash
sbatch train_vla_resnet.sbatch
```

Monitor:

```bash
squeue -u chowd207
tail -f logs/microvla-resnet-JOBID.out
```

Replace `JOBID` with the real number printed by `sbatch`.

## 7. Train Full MicroVLA: DINOv2 + Cellpose4 + DistilBERT

This matches the repo's default VLA backbone:

```text
dinov2_vits14+cellpose4
```

This is much heavier than ResNet. Start with batch size 2 on A100 40 GB. If it
is stable and GPU memory has room, try batch size 4 later.

Make sure Cellpose is installed first:

```bash
module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla
python -m pip install "cellpose>=4.0"
```

Create the Slurm script from an MSI login node:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
mkdir -p logs checkpoints_vla_cp4
```

```bash
cat > train_vla_cp4.sbatch <<'EOF'
#!/bin/bash -l
#SBATCH --job-name=microvla-cp4
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100g
#SBATCH --time=24:00:00
#SBATCH --tmp=150g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu

set -euo pipefail

cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

JOB_TMP="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}"
JOB_VLA_ROOT="$JOB_TMP/dataset_vla"
JOB_VLA_EPISODES="$JOB_VLA_ROOT/episodes"
mkdir -p "$JOB_VLA_EPISODES"

echo "Copying VLA episodes to node-local storage: $JOB_VLA_EPISODES"
rsync -aL --delete dataset_vla/episodes/ "$JOB_VLA_EPISODES/"
export MICROVLA_VLA_DATASET_ROOT="$JOB_VLA_ROOT"

CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_cp4"
mkdir -p "$CKPT_DIR"

RESUME_ARG=()
if [[ -f "$CKPT_DIR/vla_policy_last.pt" ]]; then
  RESUME_ARG=(--resume "$CKPT_DIR/vla_policy_last.pt")
  echo "Resuming from $CKPT_DIR/vla_policy_last.pt"
fi

python train_vla.py \
  --episodes-dir "$JOB_VLA_EPISODES" \
  --epochs 2000 \
  --batch-size 2 \
  --num-workers 4 \
  --device cuda \
  --backbone dinov2_vits14+cellpose4 \
  --language-backend hf \
  --text-model distilbert-base-uncased \
  --ckpt-dir "$CKPT_DIR" \
  --save-every 100 \
  "${RESUME_ARG[@]}"

echo "End: $(date)"
EOF
```

Submit:

```bash
sbatch train_vla_cp4.sbatch
```

Monitor:

```bash
squeue -u chowd207
tail -f logs/microvla-cp4-JOBID.out
```

If it runs out of memory, edit the script and reduce:

```bash
--batch-size 2
```

to:

```bash
--batch-size 1
```

## 7A. (Recommended) Train From A LeRobot Dataset On Hugging Face

Sections 4–7 read the local `dataset_vla/episodes/` tree. The **recommended**
path instead trains from a **LeRobot dataset on Hugging Face** (SmolVLA / OpenPI
/ π0 convention), so the dataset is robot-native and reusable by any VLA, and you
never rsync the 26 GB raw dataset to MSI.

The full laptop→MSI→local-inference walkthrough (building the dataset, the
cell-position instruction labels, and running inference) is in
**`MICROVLA_END_TO_END_GUIDE.md`**. The MSI-only steps:

### 7A.1 Make sure the dataset is on HF and the repo/env are ready

On your laptop you should have already run (see the end-to-end guide):

```bash
python dataset_vla/convert_microact_to_lerobot.py            # build locally
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub   # publish to HF
```

On MSI, confirm `lerobot` is installed and the repo is current:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
module purge && module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla
python -c "import lerobot, data.lerobot_vla_dataset; print('lerobot path OK')"
# if missing:  python -m pip install "lerobot>=0.4"
```

### 7A.2 Authenticate to Hugging Face on MSI (private dataset pull)

Export the caches first (so the token and downloads land in project storage),
then log in once on the login node — Slurm jobs reuse the cached token:

```bash
export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"
huggingface-cli login        # paste your HF token (or: export HF_TOKEN=hf_xxx in the job)
```

### 7A.3 Slurm script (`--dataset-repo-id`)

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

echo "Node: $(hostname)"; echo "Start: $(date)"; nvidia-smi

REPO_ID="RaianSilex/microvla_ump_dataset"
CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_lerobot"
mkdir -p "$CKPT_DIR"

# Optional speedup: if the dataset is already cached under $HF_HOME, stage it to
# fast node-local storage and train from there. Otherwise it pulls from HF.
DATASET_ROOT_ARG=()
SRC="$HF_HOME/lerobot/$REPO_ID"
if [[ -d "$SRC" ]]; then
  JOB_DS="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}/lerobot/$REPO_ID"
  mkdir -p "$JOB_DS"; rsync -a --delete "$SRC/" "$JOB_DS/"
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

echo "End: $(date)"
EOF

sbatch train_vla_lerobot.sbatch
```

Notes:

- **Smoke first:** swap the python args for
  `--backbone resnet18 --language-backend simple --no-pretrained --epochs 1 --batch-size 8`.
- **Action space:** `--action-space delta` (default, recommended) or `absolute`.
  It is saved in the checkpoint and applied automatically at rollout — the robot
  side always receives absolute targets.
- **Resume** reuses the same script (auto-detects `vla_policy_last.pt`) and
  rebuilds the exact prior backbone/action-space/LoRA from the checkpoint config.
  Use a fresh `--ckpt-dir` per backbone — don't mix backbones in one dir.
- **`robot_type` must match the rollout adapter's `robot_id`** (both
  `sensapex_dual_ump4`), or normalization falls back to the UNKNOWN row.
- **Offline alternative:** rsync your locally-built LeRobot dataset folder to MSI
  and pass `--dataset-root <that folder>` (still pass `--dataset-repo-id`, used as
  the per-robot stats key) — no HF pull at train time.

Download the result back to your laptop exactly like the other modes (Section 11),
using the `checkpoints_vla_lerobot/` folder. Then run inference locally per Stage 3
of `MICROVLA_END_TO_END_GUIDE.md`.

## 8. Switch GPU Type If A100 Waits Too Long

Do not run two jobs into the same checkpoint directory at once.

Cancel the pending job first:

```bash
scancel JOBID
squeue -u chowd207
```

For L40S, edit the Slurm header:

```bash
#SBATCH --job-name=microvla-l40s
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
```

For A40:

```bash
#SBATCH --job-name=microvla-a40
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:a40:1
```

For H100:

```bash
#SBATCH --job-name=microvla-h100
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120g
```

For this repo, request one GPU. `train_vla.py` is a single-GPU script.

## 8A. Full H100 Script For MicroVLA CP4

If you want to train full MicroVLA directly on H100, create a separate file:

```bash
nano train_vla_cp4_h100.sbatch
```

Paste:

```bash
#!/bin/bash -l
#SBATCH --job-name=microvla-h100
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --tmp=150g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu

set -euo pipefail

cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH"

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

JOB_TMP="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}"
JOB_VLA_ROOT="$JOB_TMP/dataset_vla"
JOB_VLA_EPISODES="$JOB_VLA_ROOT/episodes"
mkdir -p "$JOB_VLA_EPISODES"

echo "Copying VLA episodes to node-local storage: $JOB_VLA_EPISODES"
rsync -aL --delete dataset_vla/episodes/ "$JOB_VLA_EPISODES/"
export MICROVLA_VLA_DATASET_ROOT="$JOB_VLA_ROOT"

CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_cp4_h100"
mkdir -p "$CKPT_DIR"

RESUME_ARG=()
if [[ -f "$CKPT_DIR/vla_policy_last.pt" ]]; then
  RESUME_ARG=(--resume "$CKPT_DIR/vla_policy_last.pt")
  echo "Resuming from $CKPT_DIR/vla_policy_last.pt"
fi

python train_vla.py \
  --episodes-dir "$JOB_VLA_EPISODES" \
  --epochs 100 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --backbone dinov2_vits14+cellpose4 \
  --language-backend hf \
  --text-model distilbert-base-uncased \
  --ckpt-dir "$CKPT_DIR" \
  --save-every 25 \
  "${RESUME_ARG[@]}"

echo "End: $(date)"
```

Save nano:

```text
Ctrl+O
Enter
Ctrl+X
```

Submit:

```bash
sbatch train_vla_cp4_h100.sbatch
```

If batch size 4 runs out of memory, edit the file and change:

```bash
--batch-size 4
```

to:

```bash
--batch-size 2
```

## 9. Resume Training After Time Limit Or Interruption

The VLA scripts automatically resume from:

```text
checkpoints_vla_resnet/vla_policy_last.pt
checkpoints_vla_cp4/vla_policy_last.pt
```

If a job hits the 24-hour walltime, resubmit the same script:

```bash
sbatch train_vla_resnet.sbatch
```

or:

```bash
sbatch train_vla_cp4.sbatch
```

Do not resume a ResNet VLA checkpoint with the CP4 script, or a CP4 checkpoint
with the ResNet script. The backbone shapes are different.

## 10. Add More Trials Later And Rebuild VLA Episodes

After adding trials 67 through 200 on the laptop, upload the updated classic
dataset from the laptop:

```bash
rsync -avhP \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*/__pycache__/' \
  --exclude 'onnx_exports/' \
  --exclude 'torchviz_exports/' \
  --exclude '_report_gen/' \
  --exclude 'checkpoints/*.pt' \
  --exclude 'checkpoints/*.pth' \
  --exclude 'checkpoints_vla*/' \
  --exclude 'logs/' \
  --exclude 'train_vla*.sbatch' \
  --exclude 'dataset_vla/episodes/' \
  /home/raianlaptop/MicroVLA/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

Then regenerate VLA episodes on MSI:

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla

python dataset_vla/convert_microact_to_vla.py \
  --overwrite \
  --replace-zero-targets-with-state
```

Verify:

```bash
find dataset/logs -maxdepth 1 -name 'trial_*.csv' | wc -l
find dataset_vla/episodes -mindepth 1 -maxdepth 1 -type d -name 'trial_*' | wc -l
```

After trials 1 through 200 are uploaded and converted, both should print:

```text
200
```

Start a fresh checkpoint folder for the 200-trial dataset:

```text
checkpoints_vla_resnet_200
checkpoints_vla_cp4_200
```

Reason: VLA checkpoints include dataset stats and vocabs. A new dataset should
get a clean training run unless you are intentionally doing a finetune workflow.

## 11. Download Checkpoints Back To The Laptop

Run these from the laptop.

For ResNet VLA:

```bash
mkdir -p /home/raianlaptop/MicroVLA/checkpoints_msi_vla_resnet

rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_resnet/ \
  /home/raianlaptop/MicroVLA/checkpoints_msi_vla_resnet/
```

For full CP4 VLA:

```bash
mkdir -p /home/raianlaptop/MicroVLA/checkpoints_msi_vla_cp4

rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_cp4/ \
  /home/raianlaptop/MicroVLA/checkpoints_msi_vla_cp4/
```

For 200-trial runs, change the remote and local folder names:

```bash
mkdir -p /home/raianlaptop/MicroVLA/checkpoints_msi_vla_resnet_200

rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_vla_resnet_200/ \
  /home/raianlaptop/MicroVLA/checkpoints_msi_vla_resnet_200/
```

Important VLA files:

```text
vla_policy_best.pt
vla_policy_last.pt
vla_stats.pkl
```

## 12. Optional Rollout Command

After copying a VLA checkpoint back to the local machine, rollout on the current
dual-Sensapex rig uses the adapter-based entry point.

Example:

```bash
python3 -m rollout.vla_main \
  --checkpoint checkpoints_msi_vla_resnet/vla_policy_best.pt \
  --adapter sensapex_dual \
  --instruction "move both manipulators toward the selected cell" \
  --backbone resnet18 \
  --language-backend hf \
  --text-model distilbert-base-uncased
```

For a CP4 checkpoint:

```bash
python3 -m rollout.vla_main \
  --checkpoint checkpoints_msi_vla_cp4/vla_policy_best.pt \
  --adapter sensapex_dual \
  --instruction "move both manipulators toward the selected cell"
```

The CP4 command can usually omit `--backbone`, `--language-backend`, and
`--text-model` because `rollout.vla_main` reads them from the checkpoint config.

## 13. Common Problems

### `No VLA episodes directory found`

You did not convert the classic dataset yet, or `dataset_vla/episodes/` was not
uploaded.

For the current dataset, run:

```bash
python dataset_vla/convert_microact_to_vla.py \
  --replace-zero-targets-with-state
```

### `python: command not found`

Load conda and activate the environment:

```bash
module purge
module load conda
source activate /projects/standard/suhasabk/shared/conda_envs/microvla
which python
```

### `ModuleNotFoundError: transformers`

Install Transformers in the active environment:

```bash
python -m pip install "transformers>=4.40"
```

### `cellpose4` import or weight loading fails

For full CP4 VLA, install Cellpose:

```bash
python -m pip install "cellpose>=4.0"
```

If CP4 still blocks you, first train the ResNet VLA script. That verifies the
VLA dataset and language-conditioned policy without Cellpose.

### Hugging Face, DINOv2, Or Cellpose Downloads Fill Home

Make sure these are set in the shell or Slurm script:

```bash
export HF_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/huggingface
export TORCH_HOME=/projects/standard/suhasabk/shared/MicroVLA/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=/projects/standard/suhasabk/shared/MicroVLA/.cache/cellpose
```

### Job is `PD (Resources)`

The job is waiting for the requested GPU.

Options:

- Wait.
- Cancel and switch A100 to L40S or A40.
- Try H100 if your group has access.

### OOM During Full CP4 Training

Lower batch size:

```bash
--batch-size 1
```

### Do Not Mix Checkpoint Folders

These should be separate:

```text
checkpoints_vla_resnet/
checkpoints_vla_cp4/
checkpoints_vla_resnet_200/
checkpoints_vla_cp4_200/
```

Do not run two active jobs into the same checkpoint folder.
