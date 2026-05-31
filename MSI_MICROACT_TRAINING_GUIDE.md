# MSI MicroACT ResNet Training Guide

This guide is the end-to-end recipe for training the basic MicroACT ResNet18
policy on MSI from the home laptop.

Current assumptions:

- Local laptop repo: `/home/raianlaptop/MicroVLA`
- MSI username: `chowd207`
- MSI project directory: `/projects/standard/suhasabk`
- MSI repo copy: `/projects/standard/suhasabk/shared/MicroVLA`
- Training target: basic MicroACT, `resnet18`, not MicroVLA
- Checkpoints: `/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet`

MSI references:

- Slurm partitions and GPU requests: https://msi.umn.edu/computing/slurm-scheduler/slurm-partitions
- Agate interactive GPU example: https://msi.umn.edu/about-msi-services/high-performance-computing/agate/agate-interactive-nodes
- Interactive `srun`: https://msi.umn.edu/about-msi-services/interactive-hpc/interactive-computing-srun
- Conda module: https://msi.umn.edu/software/msi-software/conda

## 0. Know Where You Are

There are three places you will type commands.

Laptop terminal:

```bash
/home/raianlaptop/MicroVLA
```

MSI login node:

```text
chowd207@ahl01 [...]
chowd207@ahl04 [...]
```

MSI GPU compute node:

```text
chowd207@aga11 [...]
chowd207@agc10 [...]
chowd207@agd...
```

Rule of thumb:

- Commands using `/home/raianlaptop/...` run on the laptop.
- Commands using `/projects/standard/...` usually run on MSI.
- `srun` starts a GPU session from an MSI login node.
- Training jobs are submitted from an MSI login node with `sbatch`.

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
  --exclude 'logs/' \
  --exclude 'checkpoints_resnet/' \
  --exclude 'train_resnet*.sbatch' \
  /home/raianlaptop/MicroVLA/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

Why the excludes matter:

- `dataset/` is copied because training needs it.
- `.git/`, generated visualizations, and report-generation files are skipped.
- Remote training outputs are excluded so future `rsync --delete` runs do not
  delete logs, Slurm scripts, or checkpoints.

Verify the copy on MSI:

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA
find dataset/logs -maxdepth 1 -name 'trial_*.csv' | wc -l
find dataset/saved_frames -type f -name '*.png' | wc -l
du -sh dataset
```

Expected values for the current dataset:

```text
66 trial CSV files
11355 PNG frames
about 26G dataset size
```

## 1A. Add More Trials Later And Upload Only The New Data

If more demonstrations are collected later, first make the laptop copy of the
dataset complete.

For example, after adding 134 new trials, the laptop should contain:

```text
/home/raianlaptop/MicroVLA/dataset/logs/trial_1.csv
...
/home/raianlaptop/MicroVLA/dataset/logs/trial_200.csv

/home/raianlaptop/MicroVLA/dataset/saved_frames/trial_1/
...
/home/raianlaptop/MicroVLA/dataset/saved_frames/trial_200/
```

Then upload from the laptop:

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
  --exclude 'logs/' \
  --exclude 'checkpoints_resnet/' \
  --exclude 'checkpoints_resnet_200/' \
  --exclude 'train_resnet*.sbatch' \
  /home/raianlaptop/MicroVLA/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

This version does not use `--delete`, so it is gentle around remote-only files.
It will still transfer changed and new dataset files.

If the laptop dataset is the exact source of truth and MSI should match it
exactly, use `--delete`, but keep the excludes above so logs, Slurm scripts, and
checkpoints are not removed.

Verify on MSI:

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA

find dataset/logs -maxdepth 1 -name 'trial_*.csv' | wc -l
find dataset/saved_frames -mindepth 1 -maxdepth 1 -type d -name 'trial_*' | wc -l
```

After uploading trials 1 through 200, both commands should print:

```text
200
```

For a larger dataset, start a new checkpoint folder instead of resuming the old
66-trial run. In the Slurm script, change:

```bash
CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet"
```

to:

```bash
CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet_200"
```

Reason: `train.py` recomputes normalization stats for the active dataset at the
start of training, and old checkpoints contain old normalization buffers. A new
checkpoint folder keeps the 200-trial training run clean.

## 2. Start An Interactive GPU Session For Setup

Run this after SSH, from an MSI login node.

For A100:

```bash
srun -N 1 --ntasks-per-node=16 --mem=60gb \
  --gres=gpu:a100:1 -t 8:00:00 -p msigpu \
  --tmp 100gb --pty bash -l
```

For L40S, usually easier to get than A100:

```bash
srun -N 1 --ntasks-per-node=16 --mem=60gb \
  --gres=gpu:l40s:1 -t 8:00:00 -p interactive-gpu \
  --tmp 100gb --pty bash -l
```

For A40:

```bash
srun -N 1 --ntasks-per-node=16 --mem=60gb \
  --gres=gpu:a40:1 -t 8:00:00 -p interactive-gpu \
  --tmp 100gb --pty bash -l
```

For H100:

```bash
srun -N 1 --ntasks-per-node=16 --mem=80gb \
  --gres=gpu:h100:1 -t 8:00:00 -p msigpu \
  --tmp 100gb --pty bash -l
```

If the command says:

```text
queued and waiting for resources
```

that is normal. Slurm is waiting for a matching GPU.

After allocation, confirm the GPU:

```bash
hostname
nvidia-smi
```

## 3. Create Or Activate The Python Environment

Run this on the GPU node from the interactive session.

```bash
cd /projects/standard/suhasabk/shared/MicroVLA

module purge
module load conda
conda env list
```

If `microact` already exists:

```bash
source activate microact
```

If it does not exist:

```bash
conda create -n microact python=3.10 -y
source activate microact
```

Install packages once inside the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install "numpy>=1.24,<2" "pandas>=2.0" "pillow>=10.0"
```

Basic ResNet MicroACT does not need `cellpose` or `transformers`.

Verify:

```bash
which python

python - <<'PY'
import numpy, pandas, PIL, torch
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("PIL ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Good output should include:

```text
cuda True
gpu NVIDIA ...
```

## 4. Run A Smoke Test

Run this on the interactive GPU node.

Do not use `--batch-size 1` unless you really want a long test. One epoch means
one pass over the whole dataset. Batch size 32 or 64 is much faster.

```bash
python train.py \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --backbone resnet18 \
  --no-pretrained \
  --ckpt-dir /tmp/microact_smoke_resnet
```

If this finishes with:

```text
done. best val loss: ...
```

the data, code, environment, and GPU are wired correctly.

Leave the interactive GPU session:

```bash
exit
```

You should return to an MSI login node prompt such as `ahl01`.

## 5. Create The A100 Batch Training Script

Run this on an MSI login node:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
mkdir -p logs checkpoints_resnet
```

Create the Slurm script:

```bash
cat > train_resnet.sbatch <<'EOF'
#!/bin/bash -l
#SBATCH --job-name=microact-resnet
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
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
source activate microact

echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

JOB_TMP="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}"
JOB_DATASET="$JOB_TMP/dataset"
mkdir -p "$JOB_DATASET"

echo "Copying dataset to node-local storage: $JOB_DATASET"
rsync -a --delete dataset/logs "$JOB_DATASET/"
if grep -q "saved_frames_240x320" dataset/logs/trial_*.csv; then
  rsync -a --delete dataset/saved_frames_240x320 "$JOB_DATASET/"
else
  rsync -a --delete dataset/saved_frames "$JOB_DATASET/"
fi
export MICROVLA_DATASET_ROOT="$JOB_DATASET"

CKPT_DIR="/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet"
mkdir -p "$CKPT_DIR"

RESUME_ARG=()
if [[ -f "$CKPT_DIR/policy_last.pt" ]]; then
  RESUME_ARG=(--resume "$CKPT_DIR/policy_last.pt")
  echo "Resuming from $CKPT_DIR/policy_last.pt"
fi

python train.py \
  --epochs 2000 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --backbone resnet18 \
  --ckpt-dir "$CKPT_DIR" \
  --save-every 100 \
  "${RESUME_ARG[@]}"

echo "End: $(date)"
EOF
```

The dataset-copy block uses the Slurm node-local temporary disk requested by:

```bash
#SBATCH --tmp=100g
```

It copies `dataset/logs` and the active frame tree into `$TMPDIR`, then sets:

```bash
export MICROVLA_DATASET_ROOT="$JOB_DATASET"
```

`config/config.py` reads that environment variable before `train.py` builds the
dataset, so training loads frames from fast node-local storage instead of
project storage. If your CSVs point at `saved_frames_240x320`, the script copies
only that resized frame tree. Otherwise it copies the original `saved_frames`.

Submit:

```bash
sbatch train_resnet.sbatch
```

Example output:

```text
Submitted batch job 10059255
```

## 6. Monitor The Job

Check queue state:

```bash
squeue -u chowd207
```

Important states:

```text
PD = pending, waiting in queue
R  = running
```

If you see:

```text
PD ... (Resources)
```

that means the job is waiting for the requested GPU. Nothing is broken.

Once running, view logs using the real job id:

```bash
tail -f logs/microact-resnet-10059255.out
```

Check errors:

```bash
cat logs/microact-resnet-10059255.err
```

More job detail:

```bash
scontrol show job 10059255
```

Cancel if needed:

```bash
scancel 10059255
```

## 7. Switch GPU Type If A100 Waits Too Long

Do not run two training jobs into the same checkpoint directory at once.

If the A100 job is still pending and you want to switch, cancel it first:

```bash
scancel 10059255
squeue -u chowd207
```

Copy the script:

```bash
cp train_resnet.sbatch train_resnet_l40s.sbatch
nano train_resnet_l40s.sbatch
```

For L40S, change:

```bash
#SBATCH --job-name=microact-resnet
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
```

to:

```bash
#SBATCH --job-name=microact-l40s
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:l40s:1
```

Save nano:

```text
Ctrl+O
Enter
Ctrl+X
```

Submit:

```bash
sbatch train_resnet_l40s.sbatch
```

For A40 instead:

```bash
#SBATCH --job-name=microact-a40
#SBATCH -p interactive-gpu
#SBATCH --gres=gpu:a40:1
```

For H100:

```bash
#SBATCH --job-name=microact-h100
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80g
```

For this repository, request only one GPU. `train.py` is currently a single-GPU
script, so extra GPUs would sit unused unless the code is modified for
distributed training.

## 8. Resume Training After Time Limit Or Interruption

The batch script automatically resumes if this file exists:

```text
checkpoints_resnet/policy_last.pt
```

So if the job ends because of the 24-hour walltime limit, resubmit:

```bash
cd /projects/standard/suhasabk/shared/MicroVLA
sbatch train_resnet.sbatch
```

or, if you switched to L40S:

```bash
sbatch train_resnet_l40s.sbatch
```

The script will print:

```text
Resuming from .../checkpoints_resnet/policy_last.pt
```

## 9. Download Checkpoints Back To The Laptop

Run this from the laptop:

```bash
mkdir -p /home/raianlaptop/MicroVLA/checkpoints_msi_resnet

rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet/ \
  /home/raianlaptop/MicroVLA/checkpoints_msi_resnet/
```

Important files:

```text
policy_best.pt
policy_last.pt
dataset_stats.pkl
```

If the 200-trial run used `checkpoints_resnet_200`, download that folder
instead:

```bash
mkdir -p /home/raianlaptop/MicroVLA/checkpoints_msi_resnet_200

rsync -avhP \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/checkpoints_resnet_200/ \
  /home/raianlaptop/MicroVLA/checkpoints_msi_resnet_200/
```

Check locally:

```bash
ls -lh /home/raianlaptop/MicroVLA/checkpoints_msi_resnet
ls -lh /home/raianlaptop/MicroVLA/checkpoints_msi_resnet_200
```

Only the folder that you actually trained will exist.

## 10. Common Problems

### `python: command not found`

You forgot to load conda and activate the environment.

Run:

```bash
module purge
module load conda
source activate microact
which python
```

### Job is `PD (Resources)`

This means Slurm is waiting for the GPU you requested.

Options:

- Wait.
- Cancel and switch from A100 to L40S/A40.
- Try H100 if your group has access and the queue looks better.

### No log file exists

The job has probably not started yet. Logs are created after the job begins.

Use:

```bash
squeue -u chowd207
```

### `tail` says no such file

Use the slash:

```bash
tail -f logs/microact-resnet-JOBID.out
```

not:

```bash
tail -f logs.microact-resnet-JOBID.out
```

Replace `JOBID` with the real number from `sbatch`.

### Smoke test seems slow

One epoch means the full dataset. With batch size 1 that is over 10,000 gradient
steps. Use:

```bash
--batch-size 32
```

or:

```bash
--batch-size 64
```

### Do I need to reinstall numpy?

Only once per conda environment. If this works, you are done:

```bash
python - <<'PY'
import numpy, pandas, PIL, torch
print(numpy.__version__)
print(pandas.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### Do not run two jobs into the same checkpoint folder

This is unsafe:

```text
job A -> checkpoints_resnet/
job B -> checkpoints_resnet/
```

Only one active training job should write to `checkpoints_resnet/`.

If you intentionally want experiments in parallel, use separate directories:

```bash
checkpoints_resnet_a100
checkpoints_resnet_l40s
```
