# MicroVLA

A from-scratch implementation of **ACT** (Action Chunking with Transformers,
Zhao et al. 2023) for a dual-Sensapex uMp4 micromanipulator rig with one
microscope camera. Trains a visuomotor policy from teleoperated demonstrations.
This is combined with a vision-encoder backbone created by DINOv2 + Cellpose
(Cellpose 3 truncated U-Net or Cellpose 4 / Cellpose-SAM) along with a frozen
language encoder to create MicroVLA.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [What the Model Does](#2-what-the-model-does)
3. [Repository Layout](#3-repository-layout)
4. [The Model](#4-the-model)
5. [Dataset Format](#5-dataset-format)
6. [Training](#6-training)
7. [`train.py` CLI Reference](#7-trainpy-cli-reference)
8. [Inference (ROS2 Integration)](#8-inference-ros2-integration)
9. [Visualizing the Architecture](#9-visualizing-the-architecture)
10. [Tuning Notes & Gotchas](#10-tuning-notes--gotchas)
11. [Glossary](#11-glossary)
12. [MicroVLA Pipeline](#12-microvla-pipeline)

---

## 1. Quick Start

```bash
# Clone and install
git clone <repo-url> MicroACT
cd MicroACT
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For GPU machines, use the CUDA torch wheel index:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Drop your trials into dataset/
#   dataset/logs/trial_N.csv
#   dataset/saved_frames/trial_N/frame_NNNNNN.png

# Train
python train.py
```

That's it for the default config. The first run computes dataset normalization
statistics, downloads ImageNet ResNet18 weights, and starts training. Checkpoints
are written to `checkpoints/` automatically.

To use a different backbone, pass `--backbone`:

```bash
python train.py --backbone dinov2_vits14+cellpose4  # strongest cell-aware option
```

See [§4.2](#42-backbone-variants) for the full list.

---

## 2. What the Model Does

**Inputs:** one microscope frame (240×320 RGB) + the current 8-dim Sensapex state
`[x1, y1, z1, d1, x2, y2, z2, d2]`.

**Output:** the next **100** absolute target positions for both stages
(an "action chunk").

**Why a chunk?** Predicting many actions at once and executing them open-loop for
several ticks before re-inferring is the central trick of ACT. It dramatically
reduces compounding error and produces temporally coherent behavior — both
critical when the actuator takes hundreds of ms per move.

**Why a CVAE?** Human teleop is **multimodal** — for the same starting state, an
operator may legitimately approach from the left or the right, fast or slow.
A deterministic regressor averages these into a meaningless middle path.
The CVAE adds a small "style code" `z` that, during training, captures *which*
demo-style is being imitated. At inference `z = 0` selects a canonical style.

The full loss is:

```
loss = masked_L1(predicted_chunk, true_chunk)  +  β · KL( N(μ, σ²) || N(0, I) )
```

with `β = 10` (paper default).

---

## 3. Repository Layout

```
MicroACT/
├── config/config.py              all hyperparameters and shapes
├── data/dataset.py               CSV+image dataset, normalization, padding
├── model/
│   ├── backbone.py               image encoders + dispatcher
│   ├── transformer.py            DETR-style encoder/decoder primitives
│   ├── cvae.py                   ACTCVAE: backbone + style encoder + main transformer
│   └── act_policy.py             ACTPolicy: loss + numpy inference helper
├── dataset/                      data lives here (logs/ + saved_frames/)
├── checkpoints/                  written automatically (gitignored)
├── train.py                      CLI training entry point
├── utils.py                      seeding, optimizer, checkpoint IO, meters
├── evaluate.py                   intentionally empty (offline sanity script TODO)
├── export_onnx.py                exports model to ONNX (for Netron viewing)
├── requirements.txt
└── README.md                     this file
```

### File-by-file purpose

| File | Purpose |
|---|---|
| `config/config.py` | Single source of truth for shapes, hyperparameters, paths. Change `BACKBONE`, `STATE_DIM`, `IMAGE_HEIGHT`, etc. here — every other file imports from this. |
| `data/dataset.py` | Loads `trial_N.csv`, resolves image paths (with zero-image fallback for missing frames), builds a flat `(trial, timestep)` index, computes normalization stats, and emits `(image, qpos, action_chunk, is_pad)` tuples. |
| `model/backbone.py` | Image encoders. Dispatches between `resnet18`, `dinov2_*`, `cellpose`, `cellpose4`, and dual modes (`<primary>+cellpose[4]`). All produce a unified token sequence for the transformer. |
| `model/transformer.py` | DETR-style transformer primitives — encoder/decoder layers, stacks, factory. Sequence-first conventions. |
| `model/cvae.py` | The actual ACT model. Combines backbone + CVAE style encoder + main encoder-decoder + action head. |
| `model/act_policy.py` | Thin wrapper around `ACTCVAE` adding (a) training loss with masked L1 + KL, (b) numpy-in/numpy-out `.inference()` for rollout, (c) dataset stats stored as buffers so checkpoints are self-contained. |
| `train.py` | CLI entry point. Builds dataset, splits train/val, builds policy + AdamW, runs the train/val/checkpoint loop. |
| `utils.py` | `set_seed`, `build_optimizer` (two-group AdamW), `save_checkpoint`/`load_checkpoint`, `AverageMeter`. |
| `export_onnx.py` | Exports inference + training graphs to ONNX so you can drag them into [Netron](https://netron.app). |

---

## 4. The Model

### 4.1 Architecture overview

```
   image (B,1,3,H,W) ───► backbone ────► image tokens
                                              │
   qpos  (B, 8)        ───► linear ─────► qpos token
                                              │
                                              ▼
                                  ┌─────► main encoder (4 layers)
                                  │             │
                                  │             ▼
                  query embeddings ►  main decoder (7 layers)
                  (100 learned)                 │
                                                ▼
                                          action head
                                          (Linear → 8)
                                                │
                                                ▼
                                  predicted action chunk (B, 100, 8)

   During training, an extra branch:
   actions + qpos ─► style encoder ─► (μ, σ²) ─► z ─► extra source token
```

The default architecture follows the ACT paper:

| Component | Value |
|---|---|
| Hidden dim | 512 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Attention heads | 8 |
| FFN dim | 3200 |
| Dropout | 0.1 |
| Latent dim (CVAE) | 32 |
| Chunk size | 100 |
| KL weight (β) | 10 |

### 4.2 Backbone variants

Selected via `--backbone`:

| Name | Architecture | Total params | Trainable | Best for |
|---|---|---|---|---|
| `resnet18` *(default)* | ResNet18 + FrozenBN | 11.2 M | yes | Fast iteration, small datasets, ACT paper baseline |
| `dinov2_vits14` | DINOv2 ViT-S/14 | 22.1 M | no (frozen) | Better generalization across lighting / focus / morphology |
| `dinov2_vitb14` | DINOv2 ViT-B/14 | 86.6 M | no (frozen) | Slightly stronger; usually overkill |
| `dinov2_vitl14` | DINOv2 ViT-L/14 | 304 M | no (frozen) | Benchmarking only |
| `cellpose` | Cellpose 3 cyto3 U-Net encoder | 6.6 M | no (frozen) | Specialist; **only useful as the aux stream** |
| `cellpose4` | Cellpose 4 / Cellpose-SAM CP-SAM neck + readout | ~304 M | no (frozen) | Strong cell-aware single stream; heavy |
| `dinov2_vits14+cellpose` | DINOv2 + Cellpose 3 dual | 28.7 M (frozen) | no | Lighter cell-aware dual stream |
| **`dinov2_vits14+cellpose4`** | DINOv2 + Cellpose 4 dual | ~326 M (frozen) | no | **Recommended when CP4 works best on your raw frames** |
| `resnet18+cellpose` | ResNet18 + Cellpose dual | 17.8 M | partial | ResNet trains, Cellpose frozen |
| `resnet18+cellpose4` | ResNet18 + Cellpose 4 dual | ~315 M | partial | Trainable ResNet + CP-SAM specialist |

DINOv2 weights download from `torch.hub` on first use. Cellpose 3 cyto3 weights
download from `cellpose.org` (~25 MB); Cellpose 4 `cpsam` weights download from
Hugging Face and are much larger.

### 4.3 Token counts and step time

| Backbone | Image tokens (240×320) | Approx. train step time vs ResNet18 |
|---|---|---|
| `resnet18` | 80 | 1.0× (baseline) |
| `dinov2_vits14` | 374 | ~2× |
| `dinov2_vits14+cellpose` | 674 | ~2.5–3× |
| `cellpose4` | ~35 with `CELLPOSE4_DIAMETER=180` | heavy; benchmark locally |
| `dinov2_vits14+cellpose4` | ~409 with `CELLPOSE4_DIAMETER=180` | heavy; benchmark locally |

DINOv2 produces ~5× more spatial tokens than ResNet18, and the dual encoder
adds another stream from Cellpose. Cellpose 4 is compute-heavy because CP-SAM is
a SAM-style transformer, even though the default diameter scaling keeps its
token count small. Attention compute scales with sequence length.

### 4.4 Dual-encoder design (`<primary>+cellpose[4]`)

When you use `dinov2_vits14+cellpose4`, `dinov2_vits14+cellpose`,
`resnet18+cellpose4`, or `resnet18+cellpose`, the design is:

| Decision | Choice |
|---|---|
| Fusion strategy | Late concatenation along token sequence dimension |
| Position embeddings | DETR-style 2D sine, normalized to `[0, 2π]` per feature map → grids align approximately at corresponding image positions |
| Token-type identity | Learned `nn.Embedding(2, hidden_dim)`: index 0 = primary tokens, index 1 = Cellpose tokens |
| Cellpose token budget | 2×2 avg-pool before flatten for large aux grids; skipped for already-small Cellpose 4 diameter-scaled grids |
| Default freeze policy | Both encoders frozen; only the two 1×1 projections + type embed train |

The trainable surface added by the second encoder is ~330 K params — negligible.

Cellpose 4 defaults in `config/config.py` mirror the raw-image settings that
worked best in manual tests:

```python
CELLPOSE4_DIAMETER = 180.0
CELLPOSE4_CELLPROB_THRESHOLD = -2.0
CELLPOSE4_FLOW_THRESHOLD = 1.5
```

The backbone uses `CELLPOSE4_DIAMETER` to match Cellpose's canonical scale
before extracting CP-SAM features. The thresholds are recorded beside the
backbone because they matter for full mask post-processing, but the training
path uses CP-SAM feature/readout tensors directly instead of generating masks
for every batch.

---

## 5. Dataset Format

```
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

### CSV columns (per row = one timestep)

| Column | Meaning |
|---|---|
| `timestep` | Integer frame index |
| `current_x, current_y, current_z, current_d` | Stage 1 state in centered Sensapex counts |
| `current_x2, current_y2, current_z2, current_d2` | Stage 2 state |
| `target_x, target_y, target_z, target_d` | Stage 1 commanded target |
| `target_x2, target_y2, target_z2, target_d2` | Stage 2 commanded target |
| `image_path` | Relative or absolute path to the frame PNG |

Other columns (motor encoder, pressure, etc.) are ignored unless added to
`CSV_STATE_COLS` / `CSV_ACTION_COLS` in `config/config.py`.

**Image-path resolution.** The dataset tolerates several conventions:

1. Absolute path → used as-is.
2. Relative path → tried against repo root, dataset root, then `saved_frames/`.
3. Empty cell → falls back to `dataset/saved_frames/trial_N/frame_NNNNNN.png`.
4. Unresolvable → returns a zero image and warns once per trial.

---

## 6. Training

### 6.1 First-time setup

```bash
git clone <repo-url> MicroACT && cd MicroACT
python3 -m venv .venv && source .venv/bin/activate

# CPU/Mac:
pip install -r requirements.txt

# Linux GPU (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

**Optional (GPU only): install xFormers to silence DINOv2 warnings and get faster attention.**
DINOv2 prints `UserWarning: xFormers is not available` for SwiGLU / Attention / Block when
its preferred optimized kernels are missing. The model still runs correctly with the
PyTorch fallback, but on GPU you can pick up a 5–15% speedup (and reduced VRAM on
ViT-B/L) by installing xFormers. Skip this on CPU/Mac — there's no benefit.

```bash
# Match the CUDA version to your torch wheel:
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

### 6.2 Smoke test (CPU, one batch, no weight downloads)

```bash
python train.py \
  --epochs 1 --batch-size 1 \
  --num-workers 0 --device cpu \
  --no-pretrained \
  --ckpt-dir /tmp/microact_test
```

If this completes without errors, your data pipeline + model are wired up correctly.

### 6.3 Real training

```bash
python train.py
# Equivalent to:
# --epochs 2000 --batch-size 8 --lr 1e-5 --lr-backbone 1e-5 \
# --weight-decay 1e-4 --val-split 0.1 --save-every 100 --backbone resnet18
```

Expect ~10 GB VRAM at batch 8 with the default config. Drop to `--batch-size 4`
if you OOM.

### 6.4 What gets saved

Each `policy_*.pt` is **~960 MB** (84 M params × 4 bytes × ~3 for params + AdamW
moments).

| File | Written when |
|---|---|
| `checkpoints/policy_last.pt` | Every epoch |
| `checkpoints/policy_best.pt` | Whenever val loss improves |
| `checkpoints/policy_epochN.pt` | Every `--save-every` epochs |
| `checkpoints/dataset_stats.pkl` | Once at the start of each run, under the active `--ckpt-dir` |

**Plan disk accordingly** — over 2000 epochs with `--save-every 100`, expect
~22 GB total. Use `--ckpt-dir /mnt/data/microact_ckpts` to point checkpoints
and `dataset_stats.pkl` at a roomier filesystem.

If `train.py` exits with status 1 and no log line, the most likely cause is
disk-out-of-space during a checkpoint write.

### 6.5 Resume an interrupted run

```bash
python train.py --resume checkpoints/policy_last.pt
```

Restores model weights, optimizer state, epoch counter, and best validation
loss tracking.

### 6.6 Switching backbones

```bash
python train.py --backbone dinov2_vits14
python train.py --backbone dinov2_vits14+cellpose4
python train.py --backbone dinov2_vits14+cellpose
python train.py --backbone resnet18 --no-pretrained         # offline / smoke-test
```

Backbones produce checkpoints with different shapes as you can't load a
ResNet18 checkpoint into a DINOv2 model. Switching = retrain from scratch.

### 6.7 Validation split

Default is **timestep-level random split** (90/10 by default). Two timesteps
from the same trial may land on opposite sides of the split, which makes val
loss optimistic but keeps it useful as a "is training progressing?" sanity
metric. With ≥10 trials, switch to:

```bash
python train.py --val-by-trial --val-split 0.2   # holds out whole trials
```

---

## 7. `train.py` CLI Reference

| Flag | Default | Effect |
|---|---|---|
| `--epochs` | `2000` | Number of training epochs. |
| `--batch-size` | `8` | Samples per gradient step. |
| `--lr` | `1e-5` | AdamW LR for everything except the backbone. |
| `--lr-backbone` | `1e-5` | AdamW LR for backbone parameters (no effect on frozen DINOv2/Cellpose). |
| `--weight-decay` | `1e-4` | AdamW weight decay. |
| `--seed` | `0` | Random seed for python/numpy/torch + dataset split. |
| `--device` | `"cuda"` | Falls back to CPU with a warning if CUDA unavailable. Use `cuda:1` to pick a specific GPU. |
| `--val-split` | `0.1` | Validation fraction. Of *samples* in default mode, of *trials* under `--val-by-trial`. |
| `--val-by-trial` | off | Split val by whole trials instead of timesteps. Use only with ≥10 trials. |
| `--num-workers` | `4` | DataLoader subprocesses. Set to `0` for debugging. |
| `--save-every` | `100` | Numbered checkpoint frequency. |
| `--ckpt-dir` | `checkpoints/` | Where to write checkpoint files and `dataset_stats.pkl`. |
| `--resume` | none | Path to a checkpoint to resume from. |
| `--no-pretrained` | off | Skip ImageNet ResNet18 weights and Cellpose 4 `cpsam` weight loading. DINOv2 still downloads through `torch.hub`; legacy `cellpose` still loads cyto3. |
| `--backbone` | `resnet18` | See §[4.2](#42-backbone-variants). Single: `resnet18`, `dinov2_vits14/b14/l14`, `cellpose`, `cellpose4`. Dual: `dinov2_vits14+cellpose4`, `dinov2_vits14+cellpose`, `resnet18+cellpose4`, `resnet18+cellpose`. |
| `--unfreeze-backbone` | off | Fine-tune DINOv2/Cellpose. Rarely useful, much slower. No effect on `resnet18`. |

---

## 8. Inference (ROS2 Integration)

MicroACT now includes local robot-side rollout scripts under `rollout/`.

The rollout state/action vector is **8-D**:

```
[x1, y1, z1, d1, x2, y2, z2, d2]
```

There is no ODrive / focus-motor ninth dimension in this MicroACT policy.

### 8.1 Files

| File | Purpose |
|---|---|
| `rollout/main.py` | Main closed-loop rollout. Loads `ACTPolicy`, calls `policy.inference(image_rgb, state_8d)`, consumes action chunks, clamps/step-limits commands, and publishes targets. |
| `rollout/sensapex_env.py` | ROS2 bridge for camera + two Sensapex stages. Subscribes to live state/image topics and publishes absolute Sensapex targets. |
| `rollout/rollout.py` | Shared CLI args, scalar clamp helper, Ctrl+C handling helper, and `q` + Enter E-stop listener. |

### 8.2 Environment setup

Run from the MicroACT repo root in a Python environment that can import both
ROS2 `rclpy` and the MicroACT dependencies (`torch`, `torchvision`, `numpy`,
`PIL`, etc.).

```bash
cd /home/raianlaptop/MicroACT

# Source ROS / your workspace first.
source /opt/ros/humble/setup.bash
source /home/raianlaptop/ros2_ws/install/setup.bash

# If you use a venv, activate one that still has access to rclpy.
# Then verify imports:
python3 -c "import rclpy, torch; print('rclpy + torch OK')"
```

Before running on hardware, open `rollout/main.py` and verify the workspace
bounds near the top of the file:

```python
X1_MIN, X1_MAX = 4600, 5700
Y1_MIN, Y1_MAX = 4900, 5500
Z1_MIN, Z1_MAX = 8750, 8250
D1_MIN, D1_MAX = 5900, 6100

X2_MIN, X2_MAX = 4600, 5700
Y2_MIN, Y2_MAX = 4900, 5500
Z2_MIN, Z2_MAX = 8750, 8250
D2_MIN, D2_MAX = 5900, 6100
```

Those limits are safety-critical placeholders copied from the previous rollout
shape. Edit them for the actual workspace before publishing commands.

### 8.3 Run a rollout

The usual command is:

```bash
python3 -m rollout.main \
  --checkpoint checkpoints/policy_best.pt \
  --stats-path checkpoints/dataset_stats.pkl \
  --backbone resnet18 \
  --device cuda
```

For a DINOv2 + Cellpose 4 checkpoint:

```bash
python3 -m rollout.main \
  --checkpoint checkpoints/policy_best.pt \
  --stats-path checkpoints/dataset_stats.pkl \
  --backbone dinov2_vits14+cellpose4 \
  --device cuda
```

If CUDA is requested but unavailable, the script prints a warning and falls
back to CPU. If `dataset_stats.pkl` is missing, the script attempts to recover
normalization stats from the checkpoint buffers.

### 8.4 Dry run and help

Show all rollout flags:

```bash
python3 -m rollout.main --help
```

Run the policy and ROS subscribers without publishing target commands:

```bash
python3 -m rollout.main \
  --checkpoint checkpoints/policy_best.pt \
  --backbone resnet18 \
  --dry-run
```

During a real rollout:

| Input | Effect |
|---|---|
| `Ctrl+C` | Stop the rollout loop early and shut down the ROS node. |
| `q` + Enter | E-stop path: send one hold-current-position command, then exit. |

### 8.5 Important flags

| Flag | Default | Effect |
|---|---|---|
| `--checkpoint` | `checkpoints/policy_best.pt` | Policy checkpoint to load. |
| `--stats-path` | `checkpoints/dataset_stats.pkl` | Dataset normalization stats. Falls back to stats stored in checkpoint if absent. |
| `--backbone` | `resnet18` | Must match the backbone used for the checkpoint. |
| `--device` | `cuda` | Torch device. Falls back to CPU if CUDA is unavailable. |
| `--open-loop-horizon` | `8` | Number of actions consumed from each predicted chunk before re-inferring when temporal aggregation is disabled. |
| `--control-hz` | `5.0` | Rollout control frequency. Match the data collection/training rate when possible. |
| `--temporal-agg` | follows `TEMPORAL_AGG` | Enable ACT-style temporal aggregation. |
| `--no-temporal-agg` | off | Disable ACT-style temporal aggregation and use open-loop chunk execution. |
| `--temporal-agg-k` | `0.01` | Exponential age penalty for temporal aggregation. Larger values favor newer chunks more strongly. |
| `--default-speed` | `100` | Speed appended to each `/ump/target` and `/ump2/target` message. |
| `--no-ema-smoothing` | off | Disable first-order smoothing on commanded targets. |
| `--ema-alpha` | `0.35` | EMA coefficient. `1.0` means no smoothing; smaller values smooth more. |
| `--dry-run` | off | Compute commands but do not publish them. |
| `--debug-every` | `10` | Print state/command every N ticks. Use `0` to disable. |

### 8.6 ROS topics used

`rollout/sensapex_env.py` uses these topics:

| Direction | Topic | Message | Meaning |
|---|---|---|---|
| Subscribe | `/camera/image/compressed` | `sensor_msgs/CompressedImage` | Microscope RGB image source. Decoded with PIL, so rollout receives RGB. |
| Subscribe | `/ump/live` | `std_msgs/Int32MultiArray` | Stage 1 live `[x, y, z, d]`. |
| Subscribe | `/ump2/live` | `std_msgs/Int32MultiArray` | Stage 2 live `[x, y, z, d]`. |
| Publish | `/ump/target` | `std_msgs/Int32MultiArray` | Stage 1 absolute target `[x, y, z, d, speed]`. |
| Publish | `/ump2/target` | `std_msgs/Int32MultiArray` | Stage 2 absolute target `[x, y, z, d, speed]`. |

### 8.7 Rollout logic

Each control tick does:

1. Read the latest camera frame and 8-D Sensapex state from ROS.
2. With temporal aggregation enabled, call
   `ACTPolicy.inference(image_rgb, state_8d)` every tick, keep recent chunks,
   and exponentially average all predictions that target the current tick.
3. With `--no-temporal-agg`, call inference only when the current open-loop
   chunk is exhausted and take the next action from that chunk.
4. Clamp it to the configured per-axis workspace bounds.
5. Limit each axis' single-tick delta relative to the current measured state.
6. Optionally apply EMA smoothing.
7. Publish absolute targets to `/ump/target` and `/ump2/target`.

The policy expects RGB images and raw centered Sensapex counts. Do not add the
OpenPI ODrive ninth dimension, and do not pre-shift coordinates before calling
MicroACT.

---

## 9. Visualizing the Architecture

`export_onnx.py` produces a Netron-loadable `.onnx` for answering "how do the
modules connect?" — a clickable architecture diagram.

It accepts the same `--backbone` flag as training and writes to a per-backbone
subdirectory (`onnx_exports/<backbone>/`).

```bash
python export_onnx.py --backbone dinov2_vits14+cellpose4
```

For Netron: drag `onnx_exports/<backbone>/act_inference.onnx` into
[netron.app](https://netron.app), or `pip install netron && netron <path>` for
a local server with full weight loading.

---

## 10. Tuning Notes & Gotchas

### 10.1 Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `train.py` exits with status 1, no log | Disk full during checkpoint write | Free space, use `--ckpt-dir` on a roomier disk |
| KL → 0 within a few epochs | Style encoder being ignored | Lower `KL_WEIGHT` to 1.0 or 0.1 in `config.py` |
| KL > 50 | Latent too expressive | Lower `LATENT_DIM` or raise `KL_WEIGHT` |
| Val loss "looks great" but robot fails | Random val split leaks across trials | Use `--val-by-trial` (needs ≥10 trials) |
| Inference jittery between chunks | Open-loop execution or smoothing too weak | Keep temporal aggregation enabled, lower `--temporal-agg-k`, lower `--ema-alpha`, or raise `--open-loop-horizon` when using `--no-temporal-agg` |
| Robot moves wildly during inference | BGR vs RGB mismatch | Convert your OpenCV-decoded frames to RGB |
| OOM at default batch size | VRAM tight | `--batch-size 4`, or use `resnet18` instead of DINOv2 |
| `UserWarning: xFormers is not available (SwiGLU/Attention/Block)` | DINOv2 wants optimized kernels that aren't installed | Harmless — model runs correctly. On GPU, see §[6.1](#61-first-time-setup) for the optional xFormers install. |

### 10.2 Key hyperparameters

Edit in `config/config.py`:

| Constant | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 100 | At 5 Hz this is 20 s of future actions. **Probably too long for short micromanipulation moves** — try 30-50. |
| `HIDDEN_DIM` | 512 | Reduce to 256 with small datasets to fight overfitting. |
| `ENC_LAYERS` / `DEC_LAYERS` | 4 / 7 | Reduce to 3 / 4 for faster training on small datasets. |
| `KL_WEIGHT` | 10 | Beta on KL term. Tune if KL collapses or explodes. |
| `OPEN_LOOP_HORIZON` | 8 | How many actions to execute per inference at deployment. |
| `TEMPORAL_AGG` / `TEMPORAL_AGG_K` | `True` / `0.01` | Defaults for ACT-style temporal aggregation in `rollout/main.py`. |
| `BACKBONE` | `"resnet18"` | Defaults; override per-run with `--backbone`. |
| `CELLPOSE4_DIAMETER` | `180.0` | CP-SAM scale setting used by the `cellpose4` backbone. |

### 10.3 Adding new control modalities

To add a focus motor / pressure solenoids later:

1. Append the new column names to `CSV_STATE_COLS` / `CSV_ACTION_COLS` in `config/config.py`.
2. Bump `STATE_DIM` and/or `ACTION_DIM` to match.
3. Re-collect or re-export trials with those columns populated.
4. Retrain. Nothing in the model needs to change — every shape flows from these constants.

For categorical action modalities (e.g. 3-state pressure: compressed / vacuum / atm), the
action head + loss will need a small refactor — separate output heads with mixed losses.
Out of scope for the current release.

### 10.4 Rollout: temporal aggregation

The ACT paper averages overlapping chunk predictions with an exponential weight to
smooth jitter. `rollout/main.py` implements this by re-inferring every control
tick, storing recent action chunks, and averaging the predictions that point at
the current tick with weights:

```
weight(age) = exp(-TEMPORAL_AGG_K * age)
```

Temporal aggregation is on by default. Disable it with:

```bash
python3 -m rollout.main --no-temporal-agg
```

Use `--temporal-agg-k` to tune how quickly older chunks lose influence.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **ACT** | Action Chunking with Transformers (Zhao et al. 2023). |
| **Action chunk** | The next `CHUNK_SIZE` actions predicted from one observation. |
| **CVAE** | Conditional Variational Autoencoder. Style encoder over `[CLS, qpos, actions]`; main transformer decodes conditioned on `z`. |
| **DETR** | Detection Transformer (Carion et al. 2020). Source of the transformer architecture and frozen-BN convention. |
| **`is_pad`** | Boolean mask over the action chunk: `True` where the trial ended before `CHUNK_SIZE` real actions were available. |
| **Open-loop horizon** | Number of actions executed from each predicted chunk before re-inferring. |
| **qpos** | "Joint position." Used loosely here for the 8-dim Sensapex state vector. |
| **Sensapex centered counts** | Symmetric-around-zero integer position units (0 = middle of travel). |
| **Style encoder** | The CVAE encoder that produces `(μ, log σ²)` from a demo's `(qpos, actions)`. Used during training only. |
| **Temporal aggregation** | Averaging predictions from overlapping chunks at deployment to reduce jitter. |

---

## 12. MicroVLA Pipeline

MicroVLA is a parallel vision-language-action stack built on top of the same
ACT action-chunking idea. The original MicroACT files remain usable for the
fixed dual-Sensapex 8-D policy. The VLA path adds:

| File | Purpose |
|---|---|
| `config/vla_config.py` | Shared VLA dimensions, default DINOv2+Cellpose 4 backbone, text model, paths, and rollout defaults. |
| `data/vla_dataset.py` | Metadata-driven heterogeneous episode loader with padded state/action tensors and masks. |
| `model/language_encoder.py` | Frozen Hugging Face text encoder by default, plus a simple offline smoke-test encoder. |
| `model/embodiment.py` | Learned robot/lab/embodiment/action/task metadata tokens. |
| `model/vla_cvae.py` | ACT-style CVAE conditioned on image, language, state, and embodiment tokens. |
| `model/vla_policy.py` | Masked heterogeneous loss and raw-unit VLA inference helper. |
| `model/finetune.py` | Helpers for adapting a pretrained checkpoint to new data: vocab extension, per-robot stats merging, partial state-dict loading, selective freezing, and LoRA on transformer FFN linears. |
| `train_vla.py` | VLA training entry point with episode-level validation, optional lab/robot holdouts, and `--finetune` mode. |
| `rollout/vla_main.py` | Adapter-based VLA rollout entry point. |
| `rollout/adapters/sensapex_dual.py` | First robot adapter: current dual-Sensapex ROS2 setup. |

### 12.1 Dataset layout

Put VLA episodes under:

```text
dataset_vla/episodes/<episode_id>/
├── metadata.json
├── trajectory.csv
└── frames/cam_main/frame_000000.png
```

Each `metadata.json` declares the robot/lab/task metadata, language instruction,
state/action dimensions, and the CSV column names for that episode. Single-arm
robots can use `state_dim=4` / `action_dim=4`; the loader pads to
`MAX_STATE_DIM=16` and `MAX_ACTION_DIM=16` and masks invalid dimensions.

See `dataset_vla/README.md` for a minimal metadata example.

**Recommended: LeRobot dataset on Hugging Face.** MicroVLA can instead train from
a LeRobot-format dataset on the HF Hub — the same convention SmolVLA / OpenPI / π0
use, so the dataset is robot-native and reusable by any VLA. Convert with
`dataset_vla/convert_microact_to_lerobot.py` (which also injects varied,
cell-position-grounded instructions), then train with `--dataset-repo-id`. The
dataset stores **absolute** actions; training uses **delta** actions by default
(`--action-space delta`) and inference converts them back to absolute, so the
rollout/robot side is unchanged. Full details in `dataset_vla/README.md`.

### 12.2 Train VLA

```bash
python train_vla.py \
  --backbone dinov2_vits14+cellpose4 \
  --language-backend hf \
  --text-model distilbert-base-uncased
```

For offline smoke tests without downloading a text model:

```bash
python train_vla.py --language-backend simple --backbone resnet18 --no-pretrained
```

Use whole-lab or whole-robot validation holdouts when you have enough data:

```bash
python train_vla.py --holdout-lab lab_b
python train_vla.py --holdout-robot sensapex_single_ump4
```

### 12.3 Rollout VLA on the current rig

```bash
python3 -m rollout.vla_main \
  --checkpoint checkpoints_vla/vla_policy_best.pt \
  --adapter sensapex_dual \
  --instruction "move both manipulators toward the selected cell"
```

The VLA rollout uses the checkpoint's saved backbone/language settings by
default. The Sensapex adapter still applies workspace clamping, per-tick step
limits, optional EMA smoothing, and publishes only the 8-D dual-Sensapex target
shape expected by the current ROS2 rig.

### 12.4 Pretrain → finetune workflow

A pretrained MicroVLA checkpoint can be handed to a downstream user who then
adapts it to their own rig, instructions, or task variants on a much smaller
dataset. The flow is built into `train_vla.py` via `--finetune <ckpt>`.

**What pretraining produces.** Standard `python train_vla.py …` with no
finetune flags writes a vanilla checkpoint — no LoRA wrappers, no extra
freezing — at `checkpoints_vla/vla_policy_best.pt`. This is exactly what
inference and downstream finetuning expect.

**Finetune on new data.** The downstream user runs:

```bash
python train_vla.py \
  --episodes-dir /path/to/their/episodes \
  --finetune checkpoints_vla/your_pretrained.pt \
  --freeze-mode trunk \
  --lora-r 8 --lora-alpha 16 \
  --epochs 200 --batch-size 8 --lr 5e-5
```

Under the hood:

1. **Vocab extension** — `extend_vocabs` appends any new robot / lab /
   embodiment / action-type / task-family names to the pretrained vocabs.
   Old IDs are preserved verbatim so trained embedding rows still mean
   what they meant during pretraining.
2. **Stats merging** — `merge_stats` keeps the pretrained per-robot
   normalization for robots only seen during pretraining and uses the new
   dataset's stats for shared robots.
3. **Partial state-dict load** — `load_finetune_state_dict` copies
   exactly-matching tensors directly and corner-copies grown tensors
   (e.g. an embedding table that gained rows). The per-robot stat tables
   are skipped during load so the merged stats stay in place.
4. **Optional freezing** — `--freeze-mode trunk` freezes the main
   transformer + style encoder; `head_only` additionally freezes the
   backbones, language encoder, projections, and most embeddings,
   leaving only metadata embeddings + action head trainable.
5. **Optional LoRA** — `--lora-r 8` wraps the FFN `linear1` / `linear2`
   modules in transformer + style encoder with low-rank adapters. The
   base weights stay frozen; only the rank-r `A` and `B` matrices train.

**Resume a finetune run.** `--resume <ckpt>` rebuilds the prior
architecture (including freeze mode + LoRA settings) from the checkpoint's
saved config before loading weights. `--resume` wins if both `--resume`
and `--finetune` are passed.

#### Recommended finetune recipes

| Dataset size | Recommended flags |
|---|---|
| 50–200 demos, new rig in same robot family | `--freeze-mode trunk --lora-r 8 --lora-alpha 16 --lr 5e-5` |
| 500–2000 demos, varied conditions | `--freeze-mode none --lr 1e-5` |
| New robot DOF count | Write a new `rollout/adapters/<name>.py` (~50 lines) with `state_dim` / `action_dim` and rig-specific safety bounds; pretrained model adapts via vocab extension. |

#### Caveats

- **LoRA covers FFN linears only.** `nn.MultiheadAttention` uses a fused
  QKV weight rather than `nn.Linear`, so QKV LoRA would need a separate
  replacement module. Not implemented.
- **Rollout adapters are robot-specific.** The default
  `SensapexDualAdapter` is hardcoded for the 8-DOF dual-Sensapex rig
  with workspace bounds in `rollout/main.py`. Different robots need
  their own adapter — the policy itself is rig-agnostic.
- **The corner-copy loader handles "grew", not "shrank".** Reducing
  `chunk_size` or `MAX_ACTION_DIM` between pretrain and finetune skips
  the affected weights instead of truncating them.

#### `train_vla.py` finetune flags

| Flag | Default | Effect |
|---|---|---|
| `--finetune` | none | Pretrained checkpoint to start from. |
| `--freeze-mode` | `none` | `none` / `trunk` / `head_only`. |
| `--lora-r` | `0` | LoRA rank. `0` disables LoRA. |
| `--lora-alpha` | `16.0` | LoRA scaling: effective update is multiplied by `alpha / r`. |
| `--lora-targets` | `transformer,style_encoder` | Comma-separated submodules under `policy.model` to wrap. |
| `--lora-dropout` | `0.0` | Dropout applied to the LoRA input. |
