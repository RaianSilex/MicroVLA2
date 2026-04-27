# MicroACT

A from-scratch implementation of **ACT** (Action Chunking with Transformers,
Zhao et al. 2023) for a dual-Sensapex uMp4 micromanipulator rig with one
microscope camera. Trains a visuomotor policy from teleoperated demonstrations.

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
python train.py --backbone dinov2_vits14+cellpose   # recommended for cell targeting
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
├── viz_torchviz.py               renders autograd-graph SVGs
├── viz_summary.py                Keras-style layer summary table
├── requirements.txt
└── README.md                     this file
```

### File-by-file purpose

| File | Purpose |
|---|---|
| `config/config.py` | Single source of truth for shapes, hyperparameters, paths. Change `BACKBONE`, `STATE_DIM`, `IMAGE_HEIGHT`, etc. here — every other file imports from this. |
| `data/dataset.py` | Loads `trial_N.csv`, resolves image paths (with zero-image fallback for missing frames), builds a flat `(trial, timestep)` index, computes normalization stats, and emits `(image, qpos, action_chunk, is_pad)` tuples. |
| `model/backbone.py` | Image encoders. Dispatches between `resnet18`, `dinov2_*`, `cellpose`, and dual modes (`<primary>+cellpose`). All produce a unified token sequence for the transformer. |
| `model/transformer.py` | DETR-style transformer primitives — encoder/decoder layers, stacks, factory. Sequence-first conventions. |
| `model/cvae.py` | The actual ACT model. Combines backbone + CVAE style encoder + main encoder-decoder + action head. |
| `model/act_policy.py` | Thin wrapper around `ACTCVAE` adding (a) training loss with masked L1 + KL, (b) numpy-in/numpy-out `.inference()` for rollout, (c) dataset stats stored as buffers so checkpoints are self-contained. |
| `train.py` | CLI entry point. Builds dataset, splits train/val, builds policy + AdamW, runs the train/val/checkpoint loop. |
| `utils.py` | `set_seed`, `build_optimizer` (two-group AdamW), `save_checkpoint`/`load_checkpoint`, `AverageMeter`. |
| `export_onnx.py` | Exports inference + training graphs to ONNX so you can drag them into [Netron](https://netron.app). |
| `viz_torchviz.py` | Renders autograd-graph SVGs at four scopes (backbone, style encoder, full inference, full training). |
| `viz_summary.py` | Prints a Keras-style layer table — every named layer, output shape, param count. Most scannable. |

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
| **`dinov2_vits14+cellpose`** | DINOv2 + Cellpose dual | 28.7 M (frozen) | no | **Recommended for cell-targeting micromanipulation** |
| `resnet18+cellpose` | ResNet18 + Cellpose dual | 17.8 M | partial | ResNet trains, Cellpose frozen |

DINOv2 weights download from `torch.hub` on first use. Cellpose weights download
from `cellpose.org` (~25 MB) on first use.

### 4.3 Token counts and step time

| Backbone | Image tokens (240×320) | Approx. train step time vs ResNet18 |
|---|---|---|
| `resnet18` | 80 | 1.0× (baseline) |
| `dinov2_vits14` | 374 | ~2× |
| `dinov2_vits14+cellpose` | 674 | ~2.5–3× |

DINOv2 produces ~5× more spatial tokens than ResNet18, and the dual encoder
adds another ~300 from Cellpose. Attention compute scales with sequence length.

### 4.4 Dual-encoder design (`<primary>+cellpose`)

When you use `dinov2_vits14+cellpose` (or `resnet18+cellpose`), the design is:

| Decision | Choice |
|---|---|
| Fusion strategy | Late concatenation along token sequence dimension |
| Position embeddings | DETR-style 2D sine, normalized to `[0, 2π]` per feature map → grids align approximately at corresponding image positions |
| Token-type identity | Learned `nn.Embedding(2, hidden_dim)`: index 0 = primary tokens, index 1 = Cellpose tokens |
| Cellpose token budget | 2×2 avg-pool before flatten (1200 → ~300 tokens) |
| Default freeze policy | Both encoders frozen; only the two 1×1 projections + type embed train |

The trainable surface added by the second encoder is ~330 K params — negligible.

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
| `checkpoints/dataset_stats.pkl` | Once at the start of each run |

**Plan disk accordingly** — over 2000 epochs with `--save-every 100`, expect
~22 GB total. Use `--ckpt-dir /mnt/data/microact_ckpts` to point at a roomier
filesystem.

If `train.py` exits with status 1 and no log line, the most likely cause is
disk-out-of-space during a checkpoint write.

### 6.5 Resume an interrupted run

```bash
python train.py --resume checkpoints/policy_last.pt
```

Restores model weights, optimizer state, and the epoch counter exactly.

### 6.6 Switching backbones

```bash
python train.py --backbone dinov2_vits14
python train.py --backbone dinov2_vits14+cellpose
python train.py --backbone resnet18 --no-pretrained         # offline / smoke-test
```

Backbones produce checkpoints with different shapes — you can't load a
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
| `--ckpt-dir` | `checkpoints/` | Where to write all checkpoint files. |
| `--resume` | none | Path to a checkpoint to resume from. |
| `--no-pretrained` | off | Skip ImageNet ResNet18 download (no effect on DINOv2/Cellpose, which always pretrained). |
| `--backbone` | `resnet18` | See §[4.2](#42-backbone-variants). Single: `resnet18`, `dinov2_vits14/b14/l14`, `cellpose`. Dual: `dinov2_vits14+cellpose`, `resnet18+cellpose`. |
| `--unfreeze-backbone` | off | Fine-tune DINOv2/Cellpose. Rarely useful, much slower. No effect on `resnet18`. |

---

## 8. Inference (ROS2 Integration)

This codebase ends at `ACTPolicy.inference(image_np, qpos_np)`. The closed-loop
rollout (camera/state subscribers, target publishers, E-stop, safety clamping)
should live in your existing `ump_suite` ROS2 package.

### 8.1 The inference call

```python
from model.act_policy import build_policy
from utils import load_checkpoint
import torch

# Build a policy with stats from the checkpoint dir, load weights.
policy = build_policy(stats_path="checkpoints/dataset_stats.pkl",
                      backbone_name="dinov2_vits14+cellpose").eval()
load_checkpoint("checkpoints/policy_best.pt", policy, map_location="cuda")
policy = policy.to("cuda")

# At each rollout tick:
image_np = ...                          # (H, W, 3) uint8 RGB — ensure RGB, not BGR
qpos_np = ...                           # (8,) float, raw centered Sensapex counts
action_chunk = policy.inference(image_np, qpos_np)   # (100, 8) float32 absolute targets
```

### 8.2 Open-loop rollout pattern

```python
chunk = None
chunk_idx = 0
period = 1.0 / 5.0          # match training control rate

for t in range(max_timesteps):
    if estop.is_set():
        env.hold(); break

    obs = env.get_observation()
    if chunk is None or chunk_idx >= OPEN_LOOP_HORIZON:
        chunk = policy.inference(obs.image_rgb, obs.state[:8])
        chunk_idx = 0

    target = clamp_action_8d(chunk[chunk_idx])     # your safety clamp
    env.step_absolute(target)
    chunk_idx += 1
    time.sleep(period)
```

### 8.3 Things you must add on the ROS2 side

| Concern | Fix |
|---|---|
| **BGR vs RGB** | If your camera node decodes JPEGs with OpenCV, convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling `inference`. Training images go through PIL (RGB). |
| **Safety clamping** | Per-axis position bounds and max step deltas. Your existing rollout loop has these. |
| **State slicing** | Drop any non-Sensapex state dims (e.g. focus motor encoder) if they're not part of `STATE_DIM=8`. |
| **Sensapex centered counts** | The policy is trained on the same centered-counts space the rig uses (0 = mid-travel). Don't pre-shift before calling `inference`. |

---

## 9. Visualizing the Architecture

Three tools, picked by purpose:

| Tool | Output | Best for |
|---|---|---|
| `viz_summary.py` | ~300 lines text | "What layers exist? What sizes? How many params?" |
| `export_onnx.py` | Netron-loadable `.onnx` | "How do the modules connect?" — clickable architecture diagram |
| `viz_torchviz.py` | 4× SVG (~2.5 MB total) | "Why is this gradient zero?" — debugging gradient flow |

All three accept the same `--backbone` flag and write to per-backbone subdirectories
(`onnx_exports/<backbone>/`, `torchviz_exports/<backbone>/`).

```bash
pip install torchinfo torchviz       # graphviz system pkg also required for torchviz

python viz_summary.py --backbone dinov2_vits14+cellpose > arch_dual.txt
python export_onnx.py --backbone dinov2_vits14+cellpose
python viz_torchviz.py --backbone dinov2_vits14+cellpose
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
| Inference jittery between chunks | `OPEN_LOOP_HORIZON` too short | Raise it in `config.py`, or implement temporal aggregation in your ROS2 loop |
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
| `BACKBONE` | `"resnet18"` | Defaults; override per-run with `--backbone`. |

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
smooth jitter. If you want this on, set `TEMPORAL_AGG = True` and
`TEMPORAL_AGG_K = 0.01` in `config.py` and implement the averaging in your rollout
loop. Not implemented in MicroACT itself — config flags are placeholders.

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
