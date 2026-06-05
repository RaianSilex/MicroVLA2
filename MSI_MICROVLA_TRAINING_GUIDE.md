# MicroVLA Training on MSI — HuggingFace dataset → trained checkpoint

Minimal, copy-paste recipe. Trains `train_vla.py` from a **LeRobot v3 (lerobot 0.4.x)** dataset
on HuggingFace. For uploading a dataset first, see [HUGGINGFACE_LEROBOT_UPLOAD.md](HUGGINGFACE_LEROBOT_UPLOAD.md).

> **Placeholders to change before pasting** (used throughout):
> - `chowd207@login.msi.umn.edu` → your MSI login
> - `/projects/standard/suhasabk/shared` → your MSI project dir
> - `RaianSilex/multibeads_165episodes` → your HuggingFace dataset repo id

---

## 0. The rules that bit us (read once)

| Rule | Why |
|---|---|
| **Never rsync `.venv/` or `.cache/` to MSI** | Huge + slow, and `--delete` will *wipe* MSI's model caches. Always exclude them. |
| **Pre-download model weights on the LOGIN node** | MSI **compute nodes have no internet** — DINOv2/ResNet/Cellpose downloads hang/fail there. |
| **Run training with `"$PY"`, not bare `python`** | If `conda activate` silently fails, bare `python` is the wrong interpreter → `ModuleNotFoundError: torch`. |
| **`--cache-features` only with a FULLY FROZEN backbone** | It caches frozen encoder features. With a trainable encoder (any `resnet18*`) it silently freezes it. See §5. |

---

## 1. Sync code to MSI

**Preferred — only the files you changed locally** (run on your laptop, from the repo root):

```bash
cd ~/MicroVLA
rsync -avhPR \
  ./train_vla.py ./model/backbone.py ./model/vla_cvae.py ./model/vla_policy.py \
  ./data/lerobot_vla_dataset.py ./data/feature_cache.py \
  chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```
`-R` keeps the `model/`, `data/` subpaths. Add/remove files as needed; verify with `rsync … -c` (checksum) — no files listed = identical.

**Full-tree sync** (only if you must) — note the mandatory excludes:
```bash
rsync -avhP --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '*venv*/' \
  --exclude '.cache/' --exclude '__pycache__/' --exclude '*/__pycache__/' \
  --exclude 'dataset/' --exclude 'dataset_vla/episodes/' --exclude 'saved_frames/' \
  --exclude 'checkpoints*/' --exclude 'logs/' --exclude '*.pt' --exclude '*.pth' \
  ~/MicroVLA/ chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/MicroVLA/
```

---

## 2. One-time on MSI: env + pre-download weights (LOGIN node)

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/MicroVLA
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/standard/suhasabk/shared/conda_envs/microvla

export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=$PWD/.cache/cellpose

# Pre-fetch the backbone weights for the backbone you will train (login node has internet).
# Change the backbone name to match your run (see §5):
python -c "from model.backbone import build_backbone; build_backbone(backbone_name='dinov2_vits14+cellpose4', pretrained=True, freeze=True); print('backbone cached')"
```
Wait for `backbone cached`. (DistilBERT downloads to `HF_HOME` the same way — it's fetched automatically once the dataset is set; if a run hangs on a `distilbert` download, run `python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"` here too.)

> If the env doesn't exist yet, create it once:
> ```bash
> conda create -p /projects/standard/suhasabk/shared/conda_envs/microvla python=3.10 -y
> conda activate /projects/standard/suhasabk/shared/conda_envs/microvla
> pip install -r requirements.txt    # or your pinned MicroVLA deps
> ```

---

## 3. The sbatch

Save as `train_vla_cp4_165.sbatch` on MSI. **Lines you change are marked `# CHANGE`.**

```bash
#!/bin/bash -l
#SBATCH --job-name=microvla-cp4-165        # CHANGE: unique per run
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --time=24:00:00
#SBATCH --tmp=256g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu       # CHANGE

set -euo pipefail
ENV=/projects/standard/suhasabk/shared/conda_envs/microvla   # CHANGE if different
PY="$ENV/bin/python"
cd /projects/standard/suhasabk/shared/MicroVLA               # CHANGE

module purge; module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV" || true                # $PY below is the real guarantee

export HF_HOME=$PWD/.cache/huggingface
export TORCH_HOME=$PWD/.cache/torch
export CELLPOSE_LOCAL_MODELS_PATH=$PWD/.cache/cellpose
mkdir -p "$HF_HOME" "$TORCH_HOME" "$CELLPOSE_LOCAL_MODELS_PATH" logs

echo "Node: $(hostname)  Start: $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi

REPO_ID="RaianSilex/multibeads_165episodes"                              # CHANGE: your HF dataset
CKPT_DIR="$PWD/checkpoints_vla_cp4_165"                                   # CHANGE: unique per run
mkdir -p "$CKPT_DIR"

# Stage the HF-cached dataset to fast node-local disk (speeds the one-time precompute).
DATASET_ROOT_ARG=()
SRC="$HF_HOME/lerobot/$REPO_ID"
if [[ -d "$SRC" ]]; then
  JOB_DS="${TMPDIR:-/tmp}/lerobot/$REPO_ID"; mkdir -p "$JOB_DS"
  rsync -a --delete "$SRC/" "$JOB_DS/"; DATASET_ROOT_ARG=(--dataset-root "$JOB_DS")
fi

RESUME_ARG=()
[[ -f "$CKPT_DIR/vla_policy_last.pt" ]] && RESUME_ARG=(--resume "$CKPT_DIR/vla_policy_last.pt")

export PYTHONUNBUFFERED=1
"$PY" -u train_vla.py \
  --dataset-repo-id "$REPO_ID" \
  --action-space delta \
  --backbone dinov2_vits14+cellpose4 \      # CHANGE: see §5
  --language-backend hf --text-model distilbert-base-uncased \
  --epochs 2000 \
  --batch-size 32 --num-workers 8 --device cuda \
  --cache-features \                         # REMOVE this line for any resnet18* backbone (§5)
  --feature-cache-dir "${TMPDIR:-/tmp}/feat_cache" \
  --precompute-batch 64 \
  --ckpt-dir "$CKPT_DIR" --save-every 100 \
  "${DATASET_ROOT_ARG[@]}" "${RESUME_ARG[@]}"
echo "End: $(date)"
```

Submit + watch (**progress is in `.err`**, not `.out`):
```bash
cd /projects/standard/suhasabk/shared/MicroVLA
sbatch train_vla_cp4_165.sbatch
squeue --me
tail -f logs/microvla-cp4-165-*.err     # also check .out for [feature-cache] + [epoch …]
```
Healthy run order: `torch … cuda True` → `[feature-cache] building … N/N` (one-time, ~10–40 min) → `[epoch 1/2000] train … | val …`. With the cache, epochs are **minutes**, not hours.

---

## 4. Change the **dataset**

One place: the `REPO_ID="…"` line in the sbatch. The dataset must be **LeRobot v3** (lerobot 0.4.x), single camera key `observation.images.cam_main`, state/action keys matching `config/vla_config.py`. First run downloads it into `HF_HOME`; **private datasets need a read token** → `huggingface-cli login` once on the login node (paste a READ token; never put a token in a script or doc).

---

## 5. Change the **backbone** (and the cache rule)

Set `--backbone` to one of the below, and **pre-download it in §2** first.

| `--backbone` | Encoders frozen? | `--cache-features`? |
|---|---|---|
| `dinov2_vits14+cellpose4` | both frozen | **Keep it** — valid, ~30–50× faster |
| `dinov2_vits14` | frozen | Keep it |
| `resnet18` | **trainable** (frozen BN only) | **Remove it** — caching would freeze ResNet at ImageNet init |
| `resnet18+cellpose4` | ResNet trainable, Cellpose frozen | **Remove it** — same reason |

- The cache caches **raw frozen-encoder features**; the trainable projection layers still train. It is only correct when *every* encoder is frozen. `train_vla.py` auto-disables the cache under `--unfreeze-backbone`, but **not** for `resnet18*` (ResNet ignores the freeze flag and is always fine-tuned) — so you must remove the flag yourself for those.
- Each run needs its **own `CKPT_DIR`** and job name, or two runs will resume each other's checkpoints.

---

## 6. Stop early / pick a checkpoint

- Checkpoints save every epoch: `vla_policy_last.pt` (latest) and `vla_policy_best.pt` (lowest val), plus `vla_policy_epoch<N>.pt` every `--save-every`.
- For inference use **`vla_policy_best.pt`**, not `last`.
- Resubmitting the **same** sbatch auto-resumes from `vla_policy_last.pt` (the `RESUME_ARG` block). To start fresh, `rm "$CKPT_DIR"/vla_policy_*.pt` first.
- The 24 h wall is a hard kill; it saves after every completed epoch, so you lose at most the in-progress epoch. The node-local feature cache (`$TMPDIR`) is rebuilt each job (~10–40 min); that's expected.

---

## 7. Gotchas

| Symptom | Fix |
|---|---|
| `FileNotFoundError … hub/…/hubconf.py` or a download hangs | weights not cached on a no-internet compute node → run §2 pre-download on the **login node** |
| `ModuleNotFoundError: No module named 'torch'` | bare `python` instead of `"$PY"`; the activate failed silently |
| `.out` stuck on `nvidia-smi`, GPU idle | that's the startup snapshot — real progress is in **`.err`** |
| rsync slow / deletes caches | you didn't exclude `.venv/` and `.cache/` — see §1 |
| Policy "ignores the image" / `kl ≈ 0` | posterior collapse; consider `--unfreeze-backbone` (drops the cache, slower) and shortcut-breaking data |
| 2 runs corrupting checkpoints | give each its own `--ckpt-dir` and `--job-name` |
