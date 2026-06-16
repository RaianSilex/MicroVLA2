# MicroVLA

A compact **vision-language-action** policy for micromanipulation under a
microscope (e.g. driving a Sensapex uMp4 pipette toward a target cell; one
manipulator by default, two with `NUM_MANIPULATORS=2`). It is an ACT-style
action-chunking CVAE (Zhao et al. 2023) conditioned on:

- one microscope frame (DINOv2 / Cellpose-SAM / ResNet18 backbones),
- a frozen-text instruction (DistilBERT by default),
- the current robot state + learned robot/lab/embodiment metadata tokens,

with three micromanipulation-specific additions:

- **Contact-point Gaussian head** — predicts *where the tip is heading* (the
  episode's final target) as a learned mean + per-dim variance, trained with a
  Gaussian NLL so the variance is a calibrated confidence, and used to
  goal-condition the trajectory.
- **Per-axis adaptive loss weighting** — axes that barely move (e.g. a fixed
  depth) are auto-down-weighted from data, so they stop diluting the loss; axes
  that *start* moving in a future dataset are picked up with no config change.
- **Optional resistance conditioning** — if the dataset carries per-frame pipette
  resistance, the policy uses it as an extra input (with modality dropout so the
  same checkpoint still runs when the sensor is absent).
- **Cell-aware contact heads (Variant B, this branch)** — Cellpose is used as an
  *offline teacher* (not a vision encoder): it labels each trial's contact point,
  and the policy learns two auxiliary heads — *which cell* (a grid-region
  selection) and *where on it* (an image-space contact-point Gaussian) — from the
  ordinary image features. Inference stays backbone-only; no Cellpose in the loop.
  See [Cell-aware heads](#cell-aware-heads-variant-b) below.

MicroVLA trains from a **LeRobot-format dataset** (the SmolVLA / OpenPI / π0
convention), so the dataset is robot-native and reusable by any VLA.

---

## Install

```bash
git clone <this-repo> MicroVLA && cd MicroVLA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU: install the matching CUDA torch wheel first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Backbone weights download on first use: DINOv2 via `torch.hub`, Cellpose-SAM
(`cpsam`) from Hugging Face, ResNet18 from torchvision.

---

## Pipeline: raw trials → inference

```
dataset/logs/trial_N.csv + saved_frames/   ─(1)─►  LeRobot dataset (HF)
                                            ─(2)─►  train_vla.py  ─►  checkpoint
                                            ─(3)─►  offline_replay (sanity)
                                            ─(4)─►  rollout.vla_main (robot)
```

### 0. Raw data layout

```
dataset/
├── logs/trial_N.csv                      # one row per timestep
├── saved_frames/trial_N/frame_NNNNNN.png
└── instruction_labels.csv                # trial_id, region, [instruction]
```

CSV columns: `current_{x,y,z,d}` (+ `current_{x2,y2,z2,d2}` for a 2nd
manipulator) for the state, matching `target_*` for the absolute action,
`image_path`, optional `resistance_mohm`. By default only the **first**
manipulator (`x,y,z,d`) is converted (`NUM_MANIPULATORS=1` /
`--manipulators 1`); the `*2` columns may be absent or simply ignored.
`instruction_labels.csv` maps each trial to the **target cell's region** in the
frame (`top_left … center … bottom_right`, many aliases accepted) so the language
channel carries real signal. The converter scaffolds it (all `center`) on first
run — edit the `region` column per trial, then re-convert.

### 1. Convert to a LeRobot dataset

```bash
# Build locally under HF_LEROBOT_HOME (no upload). Default repo id:
#   RaianSilex/microvla_ump_dataset
python dataset_vla/convert_microact_to_lerobot.py

# Quick subset / push to the Hub (needs `huggingface-cli login`):
python dataset_vla/convert_microact_to_lerobot.py --limit-trials 3
python dataset_vla/convert_microact_to_lerobot.py --push-to-hub
```

Stores `observation.images.cam_main`, `observation.state`, `action` (absolute),
`task` (instruction), and `observation.resistance` **only if** the raw logs carry
real resistance values (auto-detected). Actions are stored **absolute**;
delta-vs-absolute is a train-time choice (next step).

> **OpenPI / π0 needs LeRobot v2.1**, not the v3.0 that lerobot ≥ 0.4 writes. Build
> a separate v2.1 copy in a venv with `lerobot==0.3.3`:
> `python dataset_vla/convert_microact_to_lerobot_v21.py`. SmolVLA and MicroVLA
> read v3.0 directly.

### 2. Train

```bash
# Recommended for a single rig: unfrozen ResNet18 (adapts to your scope) +
# frozen DistilBERT language. Delta actions by default.
python train_vla.py \
  --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --backbone resnet18 --unfreeze-backbone --language-backend hf

# Cell-aware frozen generalist (better cross-lab transfer, heavier):
python train_vla.py \
  --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --backbone dinov2_vits14+cellpose4 --language-backend hf

# Offline smoke test (no downloads):
python train_vla.py --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --backbone resnet18 --language-backend simple --no-pretrained \
  --epochs 1 --batch-size 2 --num-workers 0
```

Checkpoints (self-contained: weights + stats + vocabs + config) land in
`checkpoints_vla/vla_policy_{best,last}.pt`. Speed-ups: `--cache-features`
(precompute frozen-encoder features once; frozen backbones only), `--amp` (bf16).

### 3. Sanity-check the checkpoint (no hardware)

```bash
python -m rollout.offline_replay \
  --checkpoint checkpoints_vla/vla_policy_best.pt \
  --dataset-repo-id RaianSilex/microvla_ump_dataset
```

Reports per-axis error vs a "predict-the-mean" baseline (`ratio << 1` = the policy
is conditioning on its inputs; `~1` = mean-collapse) and the contact-point goal
correlation per axis.

### 4. Inference on the robot (ROS2)

The checkpoint is self-contained, so the same command runs any backbone — only
`--checkpoint` changes. Always `--dry-run` first (infers + writes a preview, no
motion).

```bash
# Terminal 1: bring up camera + manipulators (ump_suite).
# Terminal 2 (ROS sourced, repo .venv):
python -m rollout.vla_main \
  --checkpoint checkpoints_vla/vla_policy_best.pt \
  --instruction "move both manipulators toward the top-left cell" \
  --dry-run
```

Keep the instruction **in-distribution** (same templates/regions as training).
Drop `--dry-run` to command the motors. Workspace bounds and per-tick step caps
live in [rollout/adapters/sensapex_dual.py](rollout/adapters/sensapex_dual.py) —
**edit them for your rig before live motion.** `Ctrl+C` stops; `q`+Enter
E-stops (hold position).

---

## Finetune a pretrained checkpoint

```bash
python train_vla.py \
  --dataset-repo-id <new/dataset> \
  --finetune checkpoints_vla/vla_policy_best.pt \
  --freeze-mode trunk --lora-r 8 --lora-alpha 16 \
  --epochs 200 --lr 5e-5
```

Vocabs and per-robot stats extend to cover the new data (old ids preserved);
weights partial-load (grown embeddings are corner-copied); the pretrained chunk
size / action space / goal-head / resistance settings are read from the
checkpoint. `--freeze-mode {none,trunk,head_only}`, `--lora-r` (FFN linears),
`--resume <ckpt>` to continue a run.

---

## Cell-aware heads (Variant B)

This branch (`variant-b-cell-teacher`) adds optional cell grounding using
**Cellpose as a training-time teacher**, not as a vision encoder. The deployed
policy stays whatever backbone you trained (e.g. unfrozen ResNet18) and runs with
**no Cellpose in the loop** — Cellpose only produces labels offline.

What it adds, both auxiliary (they shape the image features; they do **not** feed
the action head, so a checkpoint runs unchanged where no cell labels exist):

- **cell-selection head** — *which* `CELL_GRID×CELL_GRID` frame region holds the
  target cell (cross-entropy);
- **image-space contact-point Gaussian** — *where on it*, a diagonal Gaussian over
  the target cell's `(u, v)` in `[0, 1]` (Gaussian NLL).

The whole feature is **auto-gated**: it activates only when the dataset carries an
`observation.goal_pixel` feature, and is silently off otherwise.

```
raw frames ─(A)─► cell_labels.csv ─(B)─► observation.goal_pixel ─(C)─► cell heads
```

```bash
# (A) Cellpose teacher: segment the contact frame of each trial and pick the
#     detected cell nearest the labeled region -> dataset/cell_labels.csv.
#     Needs `pip install 'cellpose>=4.0'`; this is the ONLY Cellpose use.
python dataset_vla/generate_cell_labels.py            # --limit-trials 3 for a subset

# (B) Re-convert: the converter auto-adds observation.goal_pixel when
#     dataset/cell_labels.csv exists (--no-cells to skip).
python dataset_vla/convert_microact_to_lerobot.py

# (C) Train exactly as usual — the cell heads turn on automatically:
python train_vla.py \
  --dataset-repo-id RaianSilex/microvla_ump_dataset \
  --backbone resnet18 --unfreeze-backbone --language-backend hf
#   --no-cell-head to disable even when labels are present;
#   --cell-goal-weight / --cell-select-weight to retune the aux losses.
```

`offline_replay` then also reports the cell-selection accuracy and the normalized
pixel error of the predicted contact point. The contact-point *label* comes from
the demonstrated trajectory's region; Cellpose only refines it to a precise cell
centroid, so segmentation quality is worth eyeballing before relying on it.

---

## Repository layout

```
config/vla_config.py          all shapes, hyperparameters, feature switches
data/
  lerobot_vla_dataset.py       LeRobot loader (+ goal label, per-axis weights, resistance, goal_pixel)
  feature_cache.py             memmap cache of frozen-encoder features
  vocab.py                     metadata vocabularies
  cell_grid.py                 pixel<->grid-region math for the cell-aware heads
model/
  backbone.py                  ResNet18 / DINOv2 / Cellpose(-SAM) + dual fusion
  transformer.py               DETR-style encoder/decoder
  language_encoder.py          frozen HF text encoder (+ offline 'simple' fallback)
  embodiment.py                robot/lab/embodiment/action/task tokens
  vla_cvae.py                  the model: CVAE + contact-point goal head
  vla_policy.py                loss (weighted L1 + goal NLL + KL) + inference
  finetune.py                  vocab/stats extension, partial load, LoRA, freezing
rollout/
  vla_main.py                  closed-loop rollout entry point
  offline_replay.py            mean-collapse / goal diagnostic
  adapters/sensapex_dual.py    dual-Sensapex adapter + safety limits
  sensapex_env.py              ROS2 camera + 2 manipulators bridge
dataset_vla/
  generate_cell_labels.py              Cellpose teacher -> cell_labels.csv (Variant B)
  convert_microact_to_lerobot.py       raw trials -> LeRobot v3.0
  convert_microact_to_lerobot_v21.py   raw trials -> LeRobot v2.1 (OpenPI)
train_vla.py                   training entry point
push_to_huggingface.py         upload a checkpoint or dataset folder to the Hub
```

---

## Key config knobs (`config/vla_config.py`)

| Constant | Default | Notes |
|---|---|---|
| `NUM_MANIPULATORS` | `1` | Pipettes used (1 = `xyzd`, 2 = dual). Re-convert + re-train to change. |
| `CHUNK_SIZE` | `30` | Actions predicted per inference (~10 s at 3 Hz). |
| `DEFAULT_ACTION_SPACE` | `"delta"` | Train on deltas; inference returns absolute. |
| `GOAL_HEAD` / `GOAL_LOSS_WEIGHT` | `True` / `1.0` | Contact-point Gaussian head + NLL weight. |
| `AXIS_WEIGHTING` | `True` | Data-driven per-axis loss weights. |
| `RESISTANCE_DROPOUT` | `0.3` | Modality dropout when resistance is present. |
| `CELL_HEAD` / `CELL_GRID` | `True` / `3` | Cell-aware heads (Variant B), auto-gated on `goal_pixel`; `3×3` selection grid. |
| `CELL_GOAL_WEIGHT` / `CELL_SELECT_WEIGHT` | `1.0` / `0.5` | Image-space contact Gaussian NLL + selection CE weights. |
| `DEFAULT_BACKBONE` | `dinov2_vits14+cellpose4` | Override with `--backbone`. |
| `KL_WEIGHT` | `10.0` | CVAE β. |
| `MAX_STATE_DIM` / `MAX_ACTION_DIM` | `16` | Heterogeneous-robot padding maxima. |
