# MicroACT Training on MSI — raw dataset → trained checkpoint

Minimal, copy-paste recipe for the **ACT policy** (`train.py`).

> **Important:** MicroACT does **not** use a HuggingFace dataset. `train.py` reads the **local raw
> `dataset/`** directly (`dataset/saved_frames/` + `dataset/instruction_labels.csv`) via
> `build_dataset`. There is **no** `--dataset-repo-id` and **no** feature cache here.
> (The HuggingFace flow is only for MicroVLA / OpenPI — see the other guides.)

> **Placeholders to change before pasting:**
> - `chowd207@login.msi.umn.edu` → your MSI login
> - `/projects/standard/suhasabk/shared` → your MSI project dir

---

## 0. The rules that bit us

| Rule | Why |
|---|---|
| **Never rsync `.venv/` or `.cache/`** | Slow, and `--delete` wipes MSI's model caches. |
| **Pre-download weights on the LOGIN node** | Compute nodes have **no internet**. |
| **Train with `"$PY"`, not bare `python`** | A silent `conda activate` failure → wrong interpreter → `ModuleNotFoundError: torch`. |
| **For ResNet backbones, train end-to-end (no freezing trick)** | ResNet18 is fine-tuned (frozen BN only); that's the intended ACT recipe. |

---

## 1. Sync code **and the raw dataset** to MSI

Code (only changed files; from repo root on your laptop):
```bash
cd ~/MicroVLA
rsync -avhPR ./train.py ./model/ ./data/dataset.py \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

Raw dataset — MicroACT trains from these, so they **must** be on MSI (the MicroVLA full-sync excludes them):
```bash
rsync -avhP --delete \
  ~/MicroVLA/dataset/saved_frames/ \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/dataset/saved_frames/
rsync -avhP ~/MicroVLA/dataset/instruction_labels.csv \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/dataset/
```

---

## 2. One-time on MSI: pre-download the backbone (LOGIN node)

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/standard/suhasabk/shared/conda_envs/microvla
export TORCH_HOME=$PWD/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=$PWD/.cache/cellpose

# Default ACT backbone is resnet18. Change the name if you use a different one (§5).
python -c "from model.backbone import build_backbone; build_backbone(backbone_name='resnet18', pretrained=True, freeze=True); print('backbone cached')"
```

---

## 3. The sbatch

Save as `train_act.sbatch` on MSI. **Lines to change are marked `# CHANGE`.**

```bash
#!/bin/bash -l
#SBATCH --job-name=microact-resnet         # CHANGE: unique per run
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --tmp=100g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu        # CHANGE

set -euo pipefail
ENV=/projects/standard/suhasabk/shared/conda_envs/microvla   # CHANGE if different
PY="$ENV/bin/python"
cd /projects/standard/suhasabk/shared/MicroVLA               # CHANGE

module purge; module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV" || true

export TORCH_HOME=$PWD/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=$PWD/.cache/cellpose
mkdir -p "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH" logs

echo "Node: $(hostname)  Start: $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi

CKPT_DIR="$PWD/checkpoints_act_resnet"      # CHANGE: unique per run
mkdir -p "$CKPT_DIR"
RESUME_ARG=()
[[ -f "$CKPT_DIR/policy_last.pt" ]] && RESUME_ARG=(--resume "$CKPT_DIR/policy_last.pt")

export PYTHONUNBUFFERED=1
"$PY" -u train.py \
  --backbone resnet18 \                     # CHANGE: see §5
  --epochs 2000 \
  --batch-size 32 --num-workers 8 --device cuda \
  --val-split 0.1 \
  --ckpt-dir "$CKPT_DIR" --save-every 100 \
  "${RESUME_ARG[@]}"
echo "End: $(date)"
```

> Confirm the checkpoint filename `train.py` writes and match it in `RESUME_ARG`:
> `grep -n "torch.save\|_last\|ckpt" train.py`.

Submit + watch:
```bash
sbatch train_act.sbatch
squeue --me
tail -f logs/microact-resnet-*.err
```

---

## 4. Change the dataset

Re-sync `dataset/saved_frames/` and `dataset/instruction_labels.csv` (§1). `train.py` rebuilds
normalization stats each run (`recompute_stats=True`), so no extra step. CSV columns:
`trial_id,region,instruction`.

---

## 5. Change the backbone

`--backbone` accepts `resnet18`, `resnet18+cellpose4`, `dinov2_vits14`, `dinov2_vits14+cellpose4`.
Pre-download it in §2 first.

- `resnet18` (default) is **fine-tuned** end-to-end (frozen BN only) — the standard, fast ACT setup.
- **`train.py` has no feature cache.** With a frozen ViT backbone (`dinov2_*`, `*+cellpose4`) the two
  ViTs run every step → **slow**. If you want a frozen-ViT backbone trained fast, use **MicroVLA**
  (`train_vla.py --cache-features`) instead — see [MSI_MICROVLA_TRAINING_GUIDE.md](MSI_MICROVLA_TRAINING_GUIDE.md).
- `--unfreeze-backbone` unfreezes the ViT encoders (does nothing for ResNet, already trainable).

---

## 6. Gotchas

| Symptom | Fix |
|---|---|
| download hangs / `hubconf.py` missing | pre-download on the **login node** (§2) |
| `ModuleNotFoundError: torch` | bare `python` instead of `"$PY"` |
| `.out` idle on `nvidia-smi` | progress is in **`.err`** |
| `FileNotFoundError` on a frame/CSV path | raw `dataset/` not synced to MSI (§1) |
| 2 runs clobbering checkpoints | give each its own `--ckpt-dir` + `--job-name` |
