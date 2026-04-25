# MicroACT — Codebase Report

A from-scratch implementation of **ACT** (Action Chunking with Transformers,
Zhao et al. 2023) targeted at a dual–Sensapex uMp4 micromanipulator rig.
This document explains what every file does, how they connect, the theory
behind the model, how to train it, and how to run it on the robot.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Background](#2-theoretical-background)
3. [Repository Layout](#3-repository-layout)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
   - [`config/config.py`](#51-configconfigpy)
   - [`data/dataset.py`](#52-datadatasetpy)
   - [`model/backbone.py`](#53-modelbackbonepy)
   - [`model/transformer.py`](#54-modeltransformerpy)
   - [`model/cvae.py`](#55-modelcvaepy)
   - [`model/act_policy.py`](#56-modelact_policypy)
   - [`utils.py`](#57-utilspy)
   - [`train.py`](#58-trainpy)
   - [`evaluate.py`](#59-evaluatepy)
   - [`export_onnx.py`](#510-export_onnxpy)
   - [`viz_torchviz.py`](#511-viz_torchvizpy)
   - [`viz_summary.py`](#512-viz_summarypy)
6. [How to Train](#6-how-to-train)
7. [How to Run Inference (ROS2 Integration)](#7-how-to-run-inference-ros2-integration)
8. [`train.py` CLI Reference](#8-trainpy-cli-reference)
9. [Tuning Guide & Common Gotchas](#9-tuning-guide--common-gotchas)
10. [Visualizing the Model Architecture](#10-visualizing-the-model-architecture)
11. [Glossary](#11-glossary)

---

## 1. Overview

**Goal.** Train a single neural-network policy that watches a microscope
camera and drives two Sensapex uMp4 stages (4-DoF each: X, Y, Z, D) toward
absolute target positions, by imitating teleoperated demonstrations.

**Inputs the policy sees.**
- A single RGB image (microscope frame), resized to 240 × 320.
- The current 8-dim state vector `[x1, y1, z1, d1, x2, y2, z2, d2]` in
  centered Sensapex counts.

**Outputs the policy produces.**
- A *chunk* of `CHUNK_SIZE` future actions (default 100), each an absolute
  8-dim target vector in the same space as the state.

**Why action chunking?**
Predicting many actions at once and executing them open-loop for several
ticks before re-inferring is the central trick of ACT. It dramatically
reduces compounding error and makes behaviour temporally coherent — both
critical when the underlying actuator (Sensapex) takes hundreds of ms to
finish a move.

**What this codebase is *not*.**
- Not a ROS2 package. It is a pure-PyTorch training repo. The closed-loop
  rollout (subscribing to camera/state topics, publishing target topics,
  E-stop, safety clamping) belongs in your existing `ump_suite` ROS2
  package, which imports `model.act_policy.ACTPolicy` and calls
  `policy.inference(...)` each tick.

---

## 2. Theoretical Background

### 2.1 Behavior Cloning Baseline

Behavior cloning trains a policy `π_θ(a | o)` to imitate
expert action–observation pairs. With a single observation `o_t` predicting
a single action `a_t`, **compounding error** is the central failure mode:
small prediction errors take the system off the demonstrated manifold,
where the policy was never trained.

### 2.2 Action Chunking

ACT replaces `a_t = π(o_t)` with `a_{t:t+k} = π(o_t)`. The policy predicts a
**chunk** of `k` future actions from one observation. At deployment, the
robot executes some prefix of that chunk open-loop before the policy is
queried again. The effective control horizon per inference is `k` (≈ 100 in
ACT), so:

- One observation grounds many actions ⇒ more temporal coherence.
- Temporal aggregation (averaging predictions from overlapping chunks)
  smooths jitter and recovers from stale predictions.

### 2.3 Conditional VAE (CVAE)

Human teleop is **multimodal** — there are several valid action chunks for
the same observation. A deterministic regressor smears across these modes
and predicts an unrealistic average. ACT solves this with a CVAE:

- A **style encoder** observes both the demonstration action chunk and the
  current state, and emits a Gaussian latent `z ~ N(μ, σ²)` capturing the
  particular *style* of that demonstration.
- The **decoder** is conditioned on `z` (in addition to image + state) and
  reconstructs the action chunk.
- A **KL term** pulls `q(z | a, s)` toward `N(0, I)`, the prior used at
  inference.

At inference the style encoder is dropped and `z = 0` (the prior mean)
selects a "canonical" mode.

The training loss is:

```
L = L1( a_hat , a_true )      # masked over real (non-padded) timesteps
  + β · KL( N(μ, σ²) || N(0, I) )
```

With `β = KL_WEIGHT = 10` (paper default).

### 2.4 Transformer Architecture (DETR Lineage)

ACT's encoder-decoder is a DETR-style transformer:

- **Encoder** consumes a token sequence: `[ z_token, qpos_token,
  img_token_1, ..., img_token_M ]` where image tokens are the
  flattened spatial cells of a ResNet18 feature map.
- **Decoder** has `CHUNK_SIZE` learned **query embeddings** that
  cross-attend to the encoder memory; each decoder output is linearly
  projected to an 8-dim action.
- **Position embeddings are added at every layer**, not just the input —
  a stability detail inherited from DETR.
- **Frozen BatchNorm** in the ResNet backbone (DETR convention) is more
  stable than tracking BN at small batch sizes.

### 2.5 Why a CVAE-shaped Style Encoder?

The style encoder is itself a transformer encoder over
`[CLS, qpos_token, action_1, ..., action_k]`. The CLS position is read out
and linearly projected to `(μ, logσ²)`. Padded action positions (when a
demonstration ends before the chunk) are masked out via
`src_key_padding_mask`. This is why the dataset emits an `is_pad` boolean
vector alongside each action chunk.

---

## 3. Repository Layout

```
MicroACT/
├── config/
│   ├── __init__.py
│   └── config.py              # all hyperparameters and shapes
├── data/
│   ├── __init__.py
│   └── dataset.py             # CSV+image dataset, normalization, padding
├── model/
│   ├── __init__.py
│   ├── backbone.py            # ResNet18 + DINOv2 (selectable) + 2D sine pos embed
│   ├── transformer.py         # DETR-style encoder + decoder blocks
│   ├── cvae.py                # ACTCVAE: style encoder + main encoder-decoder
│   └── act_policy.py          # ACTPolicy: loss + numpy inference
├── dataset/                   # data lives here
│   ├── logs/trial_N.csv
│   ├── saved_frames/trial_N/frame_NNNNNN.png
│   └── saved_videos/trial_N.mp4
├── checkpoints/               # created at first save
│   ├── dataset_stats.pkl
│   ├── policy_last.pt
│   └── policy_best.pt
├── train.py                   # CLI training entry point
├── evaluate.py                # (intentionally empty; offline sanity script TODO)
├── export_onnx.py             # exports model to ONNX for Netron visualization
├── viz_torchviz.py            # renders autograd-graph SVGs with torchviz
├── viz_summary.py             # prints layer-level Keras-style summary table
├── utils.py                   # seeding, optimizer, AverageMeter, ckpt IO
├── requirements.txt
└── README.md                  # this file
```

### 3.1 Module Dependency Graph

```
                        +--------------------+
                        |   config/config.py | ← single source of truth
                        +---------+----------+
                                  |
        +------------------+------+------+------------------+
        |                  |             |                  |
+-------v-------+  +-------v-------+ +---v----------+  +----v-------+
| data/dataset  |  | model/backbone | | model/       |  |  utils.py  |
|     .py       |  |      .py       | |  transformer |  |            |
+-------+-------+  +-------+--------+ +---+----------+  +----+-------+
        |                  |              |                  |
        |                  +------+-------+                  |
        |                         |                          |
        |                  +------v-------+                  |
        |                  | model/cvae   |                  |
        |                  |    .py       |                  |
        |                  +------+-------+                  |
        |                         |                          |
        |                  +------v---------+                |
        +------------------> model/         <----------------+
                           | act_policy.py  |
                           +------+---------+
                                  |
                          +-------v--------+
                          |    train.py    |
                          +----------------+
```

---

## 4. End-to-End Pipeline

### 4.1 Data Collection (outside this repo)

Performed by the `ump_suite` ROS2 package. Each trial produces:

- `dataset/logs/trial_N.csv` — one row per ~200 ms tick. Columns:
  `timestep`, `current_x..d`, `current_motor`, `target_x..d`,
  `target_motor`, `current_x2..d2`, `target_x2..d2`, `image_path`.
- `dataset/saved_frames/trial_N/frame_NNNNNN.png` — one frame per row.
- `dataset/saved_videos/trial_N.mp4` — visual reference, not used in training.

The motor and pressure columns exist in the CSV but are **not used**;
MicroACT consumes only the 8 Sensapex columns.

### 4.2 Training Pipeline

```
   trial_N.csv + frame_NNNNNN.png
              │
              ▼
   data.dataset.build_dataset()
   ─────────────────────────────
   • discover trial_*.csv
   • per-row: load 8-dim state, 8-dim action, image path
   • build (trial_idx, t) sample index
   • compute (and cache) qpos/action mean+std
   • emit one sample per __getitem__:
        image  (1, 3, 240, 320)
        qpos   (8,)
        action (100, 8)   ← future chunk, zero-padded
        is_pad (100,)     ← bool
              │
              ▼
   torch.utils.data.DataLoader (batched, shuffled)
              │
              ▼
   model.act_policy.ACTPolicy(image, qpos, action, is_pad)
   ───────────────────────────────────────────────────────
   • run ACTCVAE
       - style encoder → (μ, logσ²) → z
       - backbone(image) → spatial tokens
       - main transformer → action chunk a_hat
   • compute masked L1 + β·KL → loss dict
              │
              ▼
   optimizer.step() (AdamW, two param groups)
              │
              ▼
   utils.save_checkpoint() → policy_last.pt / policy_best.pt
```

### 4.3 Inference Pipeline (ROS2 side)

```
   /camera/image/compressed  ──┐
                               │
   /ump/live + /ump2/live    ──┼──→  SensapexEnv.get_observation()
                               │       (your existing helper in ump_suite)
                               │
                       (image_np, qpos_np)
                               │
                               ▼
              policy.inference(image_np, qpos_np)
              ──────────────────────────────────
              • resize + ImageNet-normalize image
              • normalize qpos with cached stats
              • forward ACTCVAE with z = 0
              • de-normalize predictions
                       │
                       ▼
              chunk: (100, 8) absolute targets
                       │
                       ▼
              for k in OPEN_LOOP_HORIZON:
                  clamp + step-limit
                  publish on /ump/target + /ump2/target
                  sleep 1/CONTROL_HZ
              re-infer
```

---

## 5. File-by-File Walkthrough

### 5.1 `config/config.py`

**Role:** the single source of truth for shapes, hyperparameters, and paths.
Every other file imports from here so nothing is hardcoded twice.

**Key sections:**

| Block | Purpose |
|---|---|
| Paths (`REPO_ROOT`, `DATASET_ROOT`, `LOGS_DIR`, `FRAMES_DIR`, `CKPT_DIR`, `STATS_PATH`) | All file system locations. |
| Robot shapes (`STATE_DIM=8`, `ACTION_DIM=8`, `NUM_CAMERAS=1`, `IMAGE_HEIGHT=240`, `IMAGE_WIDTH=320`) | The "data contract" — change these and the model auto-resizes. |
| CSV column tuples (`CSV_STATE_COLS`, `CSV_ACTION_COLS`, `CSV_IMAGE_COL`) | Exactly the columns that `data/dataset.py` reads. Adding the focus motor or solenoid logging later is a one-line change here. |
| ACT hyperparameters (`CHUNK_SIZE=100`, `HIDDEN_DIM=512`, `DIM_FEEDFORWARD=3200`, `ENC_LAYERS=4`, `DEC_LAYERS=7`, `NHEAD=8`, `LATENT_DIM=32`, `KL_WEIGHT=10`) | Match the ACT paper; safe defaults. |
| Backbone (`BACKBONE="resnet18"`, `BACKBONE_PRETRAINED=True`) | Default backbone. Override via `--backbone resnet18 \| dinov2_vits14 \| dinov2_vitb14 \| dinov2_vitl14` on any entry-point script — see [§5.3](#53-modelbackbonepy). |
| Training (`BATCH_SIZE=8`, `NUM_EPOCHS=2000`, `LR=1e-5`, `LR_BACKBONE=1e-5`, `WEIGHT_DECAY=1e-4`, `SEED=0`, `DEVICE="cuda"`) | Paper defaults. |
| Rollout (`OPEN_LOOP_HORIZON=8`, `CONTROL_HZ=5.0`, `TEMPORAL_AGG=True`, `TEMPORAL_AGG_K=0.01`) | Used by the ROS2 rollout script you'll write. |

**Why constants instead of a dataclass?** This matches the ACT reference
style and lets every consumer write `from config import config as C; ...
C.STATE_DIM` with zero ceremony. Everything is overridable from the command
line via `train.py` flags.

---

### 5.2 `data/dataset.py`

**Role:** turn the CSV + image folder into batched PyTorch tensors,
including normalization and action padding.

**Key components:**

#### `TrialData` (NamedTuple)
A single trial held in memory: `trial_id`, `states (T, 8)`,
`actions (T, 8)`, `image_paths (list of T strings)`, `length (T)`.

#### `discover_trials(logs_dir)`
Globs `trial_*.csv` and sorts them by trial number. Raises if the dataset
folder is empty.

#### `load_trial(csv_path)`
Reads one CSV with `pandas`. Validates that the configured state and action
columns are present. Returns a `TrialData`.

#### `_resolve_image_path(raw, trial_id, t)`
Tolerates several conventions for the `image_path` column:
1. Empty cell → fall back to
   `dataset/saved_frames/trial_N/frame_NNNNNN.png`.
2. Absolute path → used directly.
3. Relative path → tried against `REPO_ROOT`, `DATASET_ROOT`, `FRAMES_DIR`.
4. None of the above resolve → returns `None`, dataset returns a zero
   image and warns once.

This is what made the empty-image-path sample CSVs you provided work
without any special-case code.

#### `compute_norm_stats(trials)`
Concatenates all states and actions across trials and returns
`(mean, std)` for each. The std is **clipped to `1e-2`** to prevent
division by zero on dimensions that never moved (which is exactly what
happened on your all-zeros sample data). ImageNet stats are baked in for
the image channels.

The first time `build_dataset()` runs (or `recompute_stats=True`), the
stats are pickled to `checkpoints/dataset_stats.pkl` so subsequent runs
load instantly.

#### `EpisodicDataset` (`torch.utils.data.Dataset`)
- `__init__` builds a flat index `[(trial_idx, t) for t in range(T)]`
  across all trials.
- `__getitem__(i)` returns the four tensors:
  - `image (1, 3, 240, 320)` — RGB float32, ImageNet-normalized.
  - `qpos (8,)` — current state, dataset-normalized.
  - `action (100, 8)` — future chunk, zero-padded if the trial ends.
  - `is_pad (100,)` — bool mask over padded positions.

**Key invariant:** padded action positions are zeroed *after*
normalization, so they don't pollute the loss when masked.

#### `build_dataset(logs_dir, stats_path, recompute_stats)`
Convenience factory. Returns a ready-to-use `EpisodicDataset`. Set
`recompute_stats=True` whenever the underlying trials change (which
`train.py` does by default).

---

### 5.3 `model/backbone.py`

**Role:** turn an image into a sequence of tokens for the transformer.
Two backbone families are supported; selection is by string name and
the rest of the model sees an identical `(feat, pos)` interface
regardless of which backbone is active.

**Supported backbones**

| Name | Architecture | Params | Trainable by default | Notes |
|---|---|---|---|---|
| `resnet18` | torchvision ResNet18 + FrozenBN | 11.2 M | yes | ImageNet-pretrained, fine-tuned end-to-end. ACT paper default. |
| `dinov2_vits14` | DINOv2 ViT-S/14 | 21.6 M | no (frozen) | Self-supervised foundation features. ~21 M frozen + tiny trainable head. |
| `dinov2_vitb14` | DINOv2 ViT-B/14 | 86.2 M | no (frozen) | Bigger DINOv2 — slower, marginally better features. |
| `dinov2_vitl14` | DINOv2 ViT-L/14 | 304 M | no (frozen) | Largest stable DINOv2; rarely worth the cost for behavior cloning. |

**Components:**

#### `FrozenBatchNorm2d`
A `BatchNorm2d` whose running statistics and affine parameters are
frozen as buffers. This is the DETR convention for low-batch fine-tuning;
otherwise the BN running statistics would lurch around at batch size 8.
Only used by `ResNet18Backbone`.

#### `ResNet18Backbone`
- Builds a torchvision ResNet18 with `norm_layer=FrozenBatchNorm2d`.
- Optionally loads ImageNet weights (`pretrained=True`).
- Uses `IntermediateLayerGetter` to extract `layer4`'s feature map
  (1/32 resolution, 512 channels) and discard everything past it
  (avgpool + fc are unused).

For 240×320 input → output is `(B, 512, 8, 10)` = **80 spatial tokens**
per image after flattening.

#### `DinoV2Backbone`
- Loads a DINOv2 ViT from `torch.hub` (`facebookresearch/dinov2`). First
  call downloads ~85 MB (ViT-S) → ~1 GB (ViT-L) into
  `~/.cache/torch/hub/`.
- Frozen by default (`freeze=True`). All DINOv2 parameters get
  `requires_grad=False` and the submodule is forced into `eval()` mode
  even when the parent module is in train mode — this disables any
  training-only ops (e.g. drop path) that DINOv2 has internally.
- Bilinearly resizes input to the nearest multiple of patch size (14)
  on the fly. For 240×320 → 238×322 → **17×23 = 391 spatial tokens**.
- Uses `forward_features(x)["x_norm_patchtokens"]` to get patch tokens,
  drops the `[CLS]` token, and reshapes back to `(B, D, H', W')` so the
  rest of the pipeline doesn't have to know which backbone produced
  the features.

Embedding dim per variant: ViT-S=384, ViT-B=768, ViT-L=1024. The 1×1
projection in `Backbone` rescales whichever down to `HIDDEN_DIM=512`.

#### `PositionEmbeddingSine2D`
A 2D sinusoidal position encoding. Each `(row, col)` position gets a
fixed embedding with sinusoids at multiple frequencies — half the
channels encode row, the other half encode column. Identical to the
DETR formulation, allowing the transformer to recover spatial structure
that the flatten step destroyed.

> **Note:** DINOv2 already adds *its own* learned positional encoding
> internally. We add a sine pos on top regardless because the main
> ACT transformer expects one. Empirically harmless — the redundant
> info is just additional positional signal.

#### `Backbone` (combined wrapper)
- Dispatches on `backbone_name` to instantiate either
  `ResNet18Backbone` or `DinoV2Backbone`,
- Stores the right submodule under `self.resnet` or `self.dinov2`
  (preserves existing checkpoint compatibility for ResNet runs),
- Projects the backbone's channel count down to `HIDDEN_DIM` with a
  1×1 conv,
- Generates a matching `PositionEmbeddingSine2D`.

Returns `(feat, pos)` both of shape `(B, hidden_dim, H', W')`.

#### `build_backbone(backbone_name=None, freeze=True, ...)`
Factory. `backbone_name=None` falls back to `config.BACKBONE`
(default `"resnet18"`). `freeze` only matters for DINOv2.

#### Memory + token-count tradeoff

| Backbone | Spatial tokens (240×320) | Main encoder seq length | Trainable params (with rest of ACT) |
|---|---|---|---|
| `resnet18` | 80 | 82 (80 img + z + qpos) | ~84 M |
| `dinov2_vits14` (frozen) | 391 | 393 | ~73 M |
| `dinov2_vitb14` (frozen) | 391 | 393 | ~73 M |

DINOv2 paths produce **~5× more image tokens** than ResNet18, which
means the main encoder/decoder do more work per sample (longer
attention), even though the *backbone* itself is frozen. Plan for
~1.5–2× the train step time on DINOv2 vs ResNet18.

---

### 5.4 `model/transformer.py`

**Role:** DETR-style transformer primitives — encoder layer, decoder layer,
their stacks, and a convenience `Transformer` that wires them up.

**Conventions:**
- All tensors are **sequence-first** `(L, B, D)` to match
  `nn.MultiheadAttention`.
- **Position embeddings are added at every layer** (in both
  self-attention queries/keys *and* cross-attention queries/keys),
  not once at the input. This was a stability finding in DETR.

**Components:**

#### `_with_pos(x, pos)`
Adds the position embedding (or returns `x` if `pos is None`).

#### `TransformerEncoderLayer`
Standard pre-attention LayerNorm-after-residual structure:

```
q = k = src + pos
src ← LN( src + dropout( SelfAttn(q, k, src, mask) ) )
src ← LN( src + dropout( FFN(src) ) )
```

#### `TransformerEncoder`
Stack of N encoder layers, optional final LayerNorm.

#### `TransformerDecoderLayer`
Three sub-blocks:
1. **Self-attention** over decoder queries (positionally encoded by
   `query_pos`).
2. **Cross-attention** from queries to encoder memory (memory tokens
   carry `pos`, queries carry `query_pos`).
3. **Feed-forward**.

#### `TransformerDecoder`
Stack of M decoder layers + final LayerNorm.

#### `Transformer`
Convenience class that:
- Builds `ENC_LAYERS=4` encoder layers and `DEC_LAYERS=7` decoder layers.
- `forward(src, pos_embed, query_embed, src_key_padding_mask)`:
  1. Encode the source.
  2. Broadcast `query_embed` over batch if needed; initialize decoder
     input `tgt = zeros_like(query_embed)`.
  3. Decode and return shape `(Q, B, D)` where `Q = CHUNK_SIZE`.
- Calls `xavier_uniform_` on every multi-dim parameter (DETR init).

#### `build_transformer()` and `build_encoder()`
Factories — the second returns an encoder-only stack used by the CVAE
style encoder.

---

### 5.5 `model/cvae.py`

**Role:** the actual ACT model. Combines backbone + style encoder + main
encoder-decoder + action head.

**Components:**

#### `reparameterize(mu, logvar)`
Standard VAE trick: `z = μ + σ · ε`, where `ε ~ N(0, I)` and
`σ = exp(0.5 · logσ²)`.

#### `ACTCVAE.__init__`
Builds:
- `self.backbone = build_backbone()` — image → tokens.
- `self.transformer = build_transformer()` — main encoder-decoder.
- `self.style_encoder = build_encoder()` — encoder-only for CVAE.
- IO projections:
  - `cls_embed` — learnable `[CLS]` token for the style encoder.
  - `style_qpos_proj`, `style_action_proj` — project state and actions
    into hidden dim for the style encoder.
  - `style_pos_embed` — learnable 1D position embedding over
    `[CLS, qpos, a_1..a_k]`.
  - `latent_proj` — projects the CLS output to `(μ, logσ²)` of dim
    `2 * LATENT_DIM`.
  - `latent_to_src`, `qpos_to_src` — project `z` and `qpos` into hidden
    dim for the *main* encoder.
  - `extra_src_pos` — learnable 2-token position embedding for the
    `[latent, qpos]` prefix in the main encoder.
  - `query_embed` — `CHUNK_SIZE` learnable decoder queries.
  - `action_head` — final linear projecting decoder outputs to 8-dim.

#### `_encode_style(qpos, actions, is_pad)`
1. Build the sequence `[CLS_B, qpos_B, action_B(k tokens)]` after
   per-token projection.
2. Permute to sequence-first `(2 + k, B, D)`.
3. Build a `pad_mask` that always lets CLS + qpos through and uses
   `is_pad` for action tokens.
4. Run the style encoder with the position embedding.
5. Read the CLS output (position 0), project to `(μ, logσ²)`.

#### `_encode_image(image)`
1. Flatten the cameras dim into the batch dim → `(B*N, 3, H, W)`.
2. Run the backbone.
3. Reshape back into `(B, N, D, H', W')`, then flatten cameras and
   spatial dims into a token sequence `(N·H'·W', B, D)`.

This is what makes adding more cameras a one-line config change — the
extra cameras simply lengthen the token sequence.

#### `forward(image, qpos, actions=None, is_pad=None)`
- **Training path** (actions provided):
  1. `μ, logσ² = _encode_style(qpos, actions, is_pad)`
  2. `z = reparameterize(μ, logσ²)`
- **Inference path** (no actions):
  1. `μ = logσ² = z = 0` (the prior mean).
- Build the encoder source: `[ z_token, qpos_token, image_tokens... ]`.
- Decode with `query_embed`.
- Project to actions → `a_hat (B, k, 8)`.
- Return `(a_hat, (μ, logσ²))`.

#### `build_cvae()`
Factory.

---

### 5.6 `model/act_policy.py`

**Role:** the thin layer that the rest of the world talks to. Wraps
`ACTCVAE` with loss computation (training) and a numpy-in/numpy-out
inference helper (rollout). Carries the dataset normalization stats as
**buffers** so a saved checkpoint is self-contained.

**Components:**

#### `ACTPolicy.__init__(stats, kl_weight, **cvae_kwargs)`
- Builds an `ACTCVAE` from `cvae_kwargs`.
- Registers `qpos_mean`, `qpos_std`, `action_mean`, `action_std`,
  `image_mean`, `image_std` as **buffers**.

  Buffers (vs parameters) are:
  - Saved in `state_dict()` ⇒ loaded on `load_state_dict()`.
  - Moved with `policy.to(device)`.
  - Not optimized.

  This means a saved checkpoint contains everything needed to run
  inference — no separate stats file required.

#### `forward(image, qpos, actions=None, is_pad=None)`
- **Training** (actions given): returns a dict
  `{loss, l1, kl}` where `l1` and `kl` are detached for logging.
- **Eval** (no actions): returns the predicted action chunk (still in
  normalized space). Used for unit tests; production inference uses
  `.inference()`.

#### `_compute_loss(a_hat, actions, is_pad, μ, logσ²)`
- L1 element-wise, then masked: padded positions multiply by 0 before
  the mean. This matches the ACT reference and is more stable than a
  proper masked-mean.
- KL closed-form for diagonal Gaussian to standard normal:
  `KL = -0.5 · Σ (1 + logσ² - μ² - exp(logσ²))`, then averaged across
  batch.
- `total = l1 + KL_WEIGHT * kl`.

#### `inference(image_np, qpos_np)` (no_grad)
The rollout entry point. Takes:
- `image_np` shape `(H, W, 3)` uint8 RGB (any size; resized internally).
- `qpos_np` shape `(8,)` float in raw centered Sensapex counts.

Does:
1. `_preprocess_image` — resize to `IMAGE_HEIGHT × IMAGE_WIDTH`,
   ImageNet normalize, HWC → CHW, prepend cam dim.
2. Normalize qpos with cached `qpos_mean`/`qpos_std`.
3. Forward through the CVAE (with `z = 0`).
4. **Denormalize** predicted actions back to absolute Sensapex counts
   using `action_mean`/`action_std`.
5. Return `(CHUNK_SIZE, 8)` float32 numpy array.

Important: **expects RGB input.** If your ROS2 rollout decodes JPEGs
with OpenCV (which gives BGR), convert with
`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` first. Training images go through
PIL which is RGB.

#### `build_policy(stats, stats_path, kl_weight, **cvae_kwargs)`
Factory. If `stats` is None, loads from `stats_path` (the pickle written
by `data/dataset.py`).

---

### 5.7 `utils.py`

**Role:** small training helpers. Kept tiny so `train.py` stays readable.

**Components:**

#### `set_seed(seed)`
Seeds `random`, `numpy`, `torch`, and (if available) all CUDA RNGs.

#### `build_optimizer(policy, lr, lr_backbone, weight_decay)`
Builds an `AdamW` with **two parameter groups**:

1. ResNet backbone parameters → `lr_backbone`.
2. Everything else → `lr`.

This is the ACT paper convention: the pretrained backbone usually
benefits from a lower learning rate than the from-scratch transformer.
Defaults give both `1e-5`, but the CLI lets you set them independently.

#### `save_checkpoint(path, policy, optimizer, epoch)`
Saves a flat dict `{policy: state_dict, optimizer: state_dict, epoch: int}`.
Creates parent directories as needed.

#### `load_checkpoint(path, policy, optimizer=None, map_location=None)`
Reverse of save. Returns the saved epoch (0 if missing). Optimizer state
is only restored if you pass one in — this means you can load a
checkpoint for *inference* without needing an optimizer.

#### `AverageMeter`
Tiny running-mean class. `update(val, n=1)`, `.avg` property, `.reset()`.

#### `format_meters(meters: dict) → str`
Formats `{name: AverageMeter}` as `"loss=0.83  l1=0.12  kl=8.0"` for
one-line epoch logs.

---

### 5.8 `train.py`

**Role:** training entry point. Composes everything else.

**Flow:**

1. **`parse_args()`** — argparse with sane defaults from `config.config`.
2. **`set_seed(args.seed)`**.
3. **Device selection.** If you asked for `cuda` but it's unavailable,
   warns and falls back to `cpu`.
4. **Build dataset** with `recompute_stats=True` so stats reflect the
   current data (the cached pickle is rewritten).
5. **Train/val split** — `random_split` with a seeded generator gives a
   deterministic 90/10 split. Adjustable via `--val-split`.
6. **DataLoaders** — shuffle for train, deterministic for val. Pinned
   memory and persistent workers when CUDA is available.
7. **Build policy** with the just-computed stats. Move to device.
8. **Build optimizer** (two-group AdamW).
9. **Optional resume** from a `--resume` checkpoint.
10. **Main loop:** for each epoch
    - `run_epoch(train=True)` — forward, backward, step, update meters.
    - `run_epoch(train=False)` — forward only under `no_grad`, update
      meters.
    - Print one summary line.
    - Save `policy_last.pt` every epoch.
    - Save `policy_best.pt` if val loss improved.
    - Save numbered checkpoint every `--save-every` epochs.

**`run_epoch` detail.** For each batch it moves all four tensors to the
device, then either trains (with `optimizer.zero_grad → loss.backward
→ optimizer.step`) or evaluates (under `torch.no_grad`). Loss components
returned by `policy(...)` are accumulated into `AverageMeter`s keyed by
their dict names.

---

### 5.9 `evaluate.py`

**Role:** intentionally empty. Two reasons:

1. The **closed-loop** rollout (subscribe to `/camera/image/compressed`,
   publish `/ump/target` + `/ump2/target`, E-stop, safety clamping)
   belongs in your ROS2 package, not here. That code already exists in
   `ump_suite/main.py`; the MicroACT version of it would just import
   `model.act_policy.ACTPolicy` and replace the OpenPI call with
   `policy.inference(...)`.

2. An **offline** sanity script — load a held-out trial CSV, run
   `policy(image, qpos)` at every timestep, compare predicted vs
   ground-truth actions per axis — would be useful but is not yet
   written. Plug to add when you want one.

---

### 5.10 `export_onnx.py`

**Role:** developer tool. Exports the model to ONNX so you can drag it
into [Netron](https://netron.app) and view the architecture as a
clickable node graph. Not part of the training or inference pipeline.

**Components:**

- `_InferenceWrapper(cvae)` — thin `nn.Module` whose `forward(image, qpos)`
  calls `cvae(image, qpos, actions=None, is_pad=None)` and returns only
  `a_hat`. Captures the deployment path.
- `_TrainingWrapper(cvae)` — thin `nn.Module` whose
  `forward(image, qpos, actions, is_pad)` returns `(a_hat, mu, logvar)`.
  Captures the training path including the CVAE style encoder.
- `main()` builds a fresh randomly-initialized CVAE, exports both
  wrappers to `onnx_exports/<backbone>/act_inference.onnx` and
  `act_training.onnx` with opset 18, and prints the next steps.

**Why two wrappers?** ONNX export traces a single execution path. The
real `ACTCVAE.forward` branches on whether `actions` is provided, so
one trace can't capture both cases. The two wrappers force the export
down each path explicitly.

**`--backbone` flag.** Pass `--backbone resnet18` (default) or
`--backbone dinov2_vits14` etc. to export either architecture. Outputs
go into a per-backbone subdirectory so the two never collide.

**Output layout** (`onnx_exports/<backbone>/`):

| File | Size (resnet18 / dinov2_vits14) | Purpose |
|---|---|---|
| `act_inference.onnx` | ~1.5 / ~1.9 MB | Graph topology — drag this into Netron |
| `act_inference.onnx.data` | ~254 / ~295 MB | Weight blobs (sidecar) |
| `act_training.onnx` | ~1.8 / ~2.2 MB | Graph topology with style encoder branch |
| `act_training.onnx.data` | ~321 / ~362 MB | Weight blobs (sidecar) |

The `.onnx` and `.onnx.data` pair go together. Both are gitignored.

See [§10](#10-visualizing-the-model-architecture) for how to view them.

---

### 5.11 `viz_torchviz.py`

**Role:** developer tool. Renders the model's **autograd graph** as SVG
files via `torchviz`. Complements `export_onnx.py` — Netron shows the
*architecture* (clean module-level boxes), torchviz shows the *operations*
(every add, matmul, layernorm, softmax, transpose).

**Components:**

- `render(out, output_tensor, params, label)` — thin wrapper around
  `torchviz.make_dot(...)` that writes a single SVG with `cleanup=True`
  (no leftover `.dot` files).
- `main()` builds a fresh CVAE and renders four SVGs at increasing
  scope into `torchviz_exports/<backbone>/`:

| File | Scope | Size (resnet18 / dinov2_vits14 frozen) |
|---|---|---|
| `01_backbone.svg` | Backbone + 1×1 projection only | ~90 KB / ~5 KB |
| `02_style_encoder.svg` | CVAE style encoder branch only | ~230 KB |
| `03_inference_full.svg` | Full inference forward pass | ~950 / ~860 KB |
| `04_training_full.svg` | Full training forward pass | ~1.2 / ~1.1 MB |

The DINOv2 `01_backbone.svg` is much smaller because the backbone is
frozen — the autograd graph stops at its output, so the SVG only shows
the trainable `input_proj` + sine pos. Pass `--unfreeze-backbone` if
you want to see DINOv2's internal ops too (much larger graph).

**Why four files?** The full graphs have thousands of nodes — readable
when zoomed but overwhelming as an entry point. The focused sub-graphs
let you understand one piece at a time.

**`--backbone` flag.** Same as `export_onnx.py` — pass `--backbone
resnet18 | dinov2_vits14 | ...`. Outputs go to per-backbone
subdirectories so they don't overwrite each other.

**Requirements:** `pip install torchviz` and a working `graphviz` system
package (`apt install graphviz` provides the `dot` binary). Both are
already installed in this environment.

See [§10.5](#105-when-to-use-which-tool) for when to use this vs Netron.

---

### 5.12 `viz_summary.py`

**Role:** developer tool. Prints a Keras-style **layer summary** to
stdout — every named layer, its input/output shape, and parameter
count. No graph, no math ops, no SVG. The most scannable of the three
visualization tools.

**Components:**

- `main(depth=4)` builds a fresh CVAE and prints two things:
  1. `print(cvae)` — the raw `nn.Module` hierarchy (names only, no
     shapes).
  2. `torchinfo.summary(cvae, ...)` for both the inference and training
     forward paths, with `col_names=("input_size", "output_size",
     "num_params")`.

**Requirements:** `pip install torchinfo`. No system packages.

**Output:** ~300 lines of formatted text to stdout. To save:

```bash
python viz_summary.py > architecture.txt
```

**Example excerpt** (top of the inference table):

```
ACTCVAE                                    [1,1,3,240,320]  [1,100,8]      17,480,768
├─Backbone: 1-1                            [1,3,240,320]    [1,512,8,10]   --
│    └─ResNet18Backbone: 2-1               [1,3,240,320]    [1,512,8,10]   --
│    └─Conv2d: 2-2                         [1,512,8,10]     [1,512,8,10]   262,656
├─Transformer: 1-4                         [82,1,512]       [100,1,512]    --
│    └─TransformerEncoder: 2-4 (×4)        [82,1,512]       [82,1,512]     17,332,736
│    └─TransformerDecoder: 2-5 (×7)        [100,1,512]      [100,1,512]    37,694,848
├─Linear: 1-5  (action head)               [1,100,512]      [1,100,8]      4,104
=========================================================================================
Total params: 83,963,528
Estimated Total Size (MB): 264.29
```

See [§10.5](#105-when-to-use-which-tool) for when to use this vs the
other two tools.

---

## 6. How to Train

### 6.1 First-time setup

```bash
# Inside the repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your trials under `dataset/`:

```
dataset/
├── logs/trial_1.csv
├── logs/trial_2.csv
├── ...
├── saved_frames/trial_1/frame_000000.png
├── saved_frames/trial_1/frame_000001.png
└── ...
```

### 6.2 Sanity check on CPU

Verify the pipeline runs end-to-end (won't converge, just checks plumbing):

```bash
python train.py \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --no-pretrained \
  --ckpt-dir /tmp/microact_test
```

Expect ~1 minute per batch on CPU; one epoch on the 55-sample dataset
will take a few minutes.

### 6.3 Real training on GPU

```bash
python train.py
# equivalent to:
# --epochs 2000 --batch-size 8 --lr 1e-5 --lr-backbone 1e-5 \
# --weight-decay 1e-4 --val-split 0.1 --save-every 100
```

Expect ~10 GB VRAM at batch size 8 with 240×320 images. Drop to
`--batch-size 4` if you run out.

### 6.4 What gets saved

After every epoch:
- `checkpoints/policy_last.pt` — most recent weights + optimizer state.
- `checkpoints/policy_best.pt` — best val-loss weights (for inference).
- `checkpoints/policy_epochN.pt` — every `--save-every` epochs.
- `checkpoints/dataset_stats.pkl` — written once at the start.

**Checkpoint footprint.** Each checkpoint is **~960 MB** (measured: 84 M
params × 4 bytes × ~3 for params + AdamW first/second moments). Plan disk
accordingly: ~2 GB minimum for `last + best`, ~22 GB if you keep a
numbered checkpoint every 100 epochs over a 2000-epoch run. Use
`--ckpt-dir` to point at a roomier filesystem if needed.

### 6.5 Resuming an interrupted run

```bash
python train.py --resume checkpoints/policy_last.pt
```

The optimizer state and the epoch counter are restored. Training picks
up exactly where it left off.

---

## 7. How to Run Inference (ROS2 Integration)

This codebase ends at `ACTPolicy.inference()`. The closed-loop rollout
should be a new file in your `ump_suite` ROS2 package, mirroring
`ump_suite/main.py` but with the OpenPI websocket call replaced by a
direct `policy.inference(...)` call.

### 7.1 Sketch of the rollout script

```python
# ump_suite/microact_rollout.py (in your ROS2 package)

import sys, time
import torch
sys.path.insert(0, "/path/to/MicroACT")  # or pip install -e .

from model.act_policy import build_policy
from utils import load_checkpoint
from ump_suite.sensapex_env import SensapexEnv
from ump_suite._rollout_common import (
    parse_args, start_estop_listener, prevent_keyboard_interrupt,
)

CKPT  = "/path/to/MicroACT/checkpoints/policy_best.pt"
STATS = "/path/to/MicroACT/checkpoints/dataset_stats.pkl"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build policy from stats, then overwrite weights from checkpoint.
    policy = build_policy(stats_path=STATS).to(device).eval()
    load_checkpoint(CKPT, policy, map_location=device)

    env = SensapexEnv(default_speed=args.default_speed)
    estop = start_estop_listener()

    chunk = None
    chunk_idx = 0
    period = 1.0 / 5.0   # match dataset CONTROL_HZ

    try:
        for t in range(args.max_timesteps):
            if estop.is_set():
                env.hold(); break

            obs = env.get_observation()                     # SensapexObs
            image_np = obs.image_rgb                         # ensure RGB
            qpos_np = obs.state[:8]                          # 8-dim Sensapex slice

            need_new_chunk = (
                chunk is None or chunk_idx >= args.open_loop_horizon
            )
            if need_new_chunk:
                with prevent_keyboard_interrupt():
                    chunk = policy.inference(image_np, qpos_np)
                chunk_idx = 0

            action = chunk[chunk_idx]                        # (8,)
            action = clamp_action_8d(action)                 # your safety box
            action = limit_step(action, prev=qpos_np)        # per-axis Δ cap

            env.step_absolute(action)                        # publishes targets
            chunk_idx += 1
            time.sleep(period)
    finally:
        env.close()
```

### 7.2 What you must add on the ROS2 side

1. **Safety clamping** (`clamp_action_8d`, `limit_step`) — exact values
   depend on your stage. Borrow the structure from `ump_suite/main.py`.
2. **BGR → RGB conversion** if your camera node decodes JPEGs with
   OpenCV.
3. **State slicing** — drop the focus motor encoder count if your
   `SensapexObs.state` includes it (training only saw 8 dims).
4. **`SensapexObs` adapter** — `get_observation()` already exists in
   `ump_suite/sensapex_env.py`; you just need to feed `image_rgb` and
   `state[:8]` into `policy.inference`.

### 7.3 Optional: temporal aggregation

The ACT paper averages overlapping chunk predictions with an exponential
weight, smoothing jitter further. If you want it on:

```
TEMPORAL_AGG = True   # in config.config
TEMPORAL_AGG_K = 0.01
```

Then in the rollout, instead of popping from a single chunk, store the
last `OPEN_LOOP_HORIZON` chunks and average with weights `exp(-k * Δt)`.
Not implemented in this repo's example; add to your ROS2 script if
needed.

---

## 8. `train.py` CLI Reference

| Flag | Default | Effect |
|---|---|---|
| `--epochs` | `2000` | Number of training epochs. One epoch = one full pass over the train split. |
| `--batch-size` | `8` | Samples per gradient step. Memory grows roughly linearly. |
| `--lr` | `1e-5` | AdamW learning rate for everything *except* the backbone. |
| `--lr-backbone` | `1e-5` | AdamW learning rate for backbone parameters. Lower than `--lr` is sometimes useful when fine-tuning ImageNet weights. Has no effect on frozen DINOv2 (no gradients to optimize). |
| `--weight-decay` | `1e-4` | AdamW weight decay (applied to all groups). |
| `--seed` | `0` | Random seed for python/numpy/torch + dataset split. |
| `--device` | `"cuda"` | `"cuda"`, `"cuda:1"`, `"cpu"`, etc. Falls back to CPU with a warning if CUDA isn't available. |
| `--val-split` | `0.1` | Fraction held out for validation. With timestep-level split (default), this is the fraction of *samples*; with `--val-by-trial`, it's the fraction of *trials*. |
| `--val-by-trial` | off | Split the val set by whole trials instead of by individual timesteps. Use only when you have ≥10 trials. See [§9.9](#99-validation-split-trial-vs-timestep). |
| `--num-workers` | `4` | DataLoader subprocess count. Set to `0` if multiprocessing causes issues (e.g. inside a debugger). |
| `--save-every` | `100` | Write a numbered `policy_epochN.pt` checkpoint every N epochs (in addition to `policy_last.pt` and `policy_best.pt`). |
| `--ckpt-dir` | `checkpoints/` | Where to write all checkpoint files and `dataset_stats.pkl`. Point to an external disk if `/` is tight. |
| `--resume` | none | Path to a checkpoint to resume from. Restores model weights, optimizer state, and the epoch counter. |
| `--no-pretrained` | off | Skip downloading the ImageNet ResNet18 weights. Has no effect on DINOv2 (always loads pretrained from torch.hub). Useful for offline boxes or smoke tests; you almost never want this for real training. |
| `--backbone` | `resnet18` | Image backbone selection: `resnet18`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`. DINOv2 variants download from torch.hub on first use. See [§5.3](#53-modelbackbonepy) and [§9.10](#910-backbone-choice). |
| `--unfreeze-backbone` | off | DINOv2 backbones are frozen by default. Pass this to fine-tune them — rarely needed and significantly increases VRAM/compute. No effect on `resnet18` (already trainable). |

### Examples

**Basic GPU training:**
```bash
python train.py
```

**Smaller batch (lower VRAM):**
```bash
python train.py --batch-size 4
```

**Two-GPU box, use the second one:**
```bash
python train.py --device cuda:1
```

**Resume a crashed run:**
```bash
python train.py --resume checkpoints/policy_last.pt
```

**Smoke test on CPU with a tiny config:**
```bash
python train.py \
  --epochs 1 --batch-size 1 \
  --num-workers 0 --device cpu \
  --no-pretrained \
  --ckpt-dir /tmp/microact_test
```

**External checkpoint dir (low root-disk):**
```bash
python train.py --ckpt-dir /mnt/data/microact_ckpts
```

**Train with a DINOv2 backbone (frozen):**
```bash
python train.py --backbone dinov2_vits14
# or:
python train.py --backbone dinov2_vitb14
```

**Fine-tune the DINOv2 backbone (rarely needed):**
```bash
python train.py --backbone dinov2_vits14 --unfreeze-backbone --lr-backbone 1e-6
```

---

## 9. Tuning Guide & Common Gotchas

### 9.1 KL collapse / KL explosion

- **KL → 0** within the first few epochs: the style encoder is being
  ignored. Often fine for toy tasks; if you actually need the latent to
  capture style, lower `KL_WEIGHT` to 1.0 or 0.1.
- **KL → very large** (>50): too much capacity in the latent. Lower
  `LATENT_DIM` (e.g. 16) or raise `KL_WEIGHT`.

### 9.2 L1 loss won't go below ~normalized noise floor

Normalization stats are computed per dimension and clipped at
`std ≥ 1e-2`. If a particular axis (say `d2`) barely moves in your
demos, its normalized loss will look bad even when predictions are
fine in absolute units. Inspect raw-units MSE (in the offline eval
script that goes in `evaluate.py`) before concluding the model is
broken.

### 9.3 Inference jitter

Open-loop horizon `OPEN_LOOP_HORIZON = 8` means a fresh chunk every
~1.6 s at 5 Hz. If the robot looks "shaky" between chunks:

- Increase `OPEN_LOOP_HORIZON` (run more of each chunk).
- Or implement temporal aggregation in the rollout (see §7.3).
- Or lower `CONTROL_HZ` (longer dwell on each command).

### 9.4 BGR vs RGB

Training images go through PIL → RGB. `policy.inference` expects RGB.
If your camera node decodes JPEGs with OpenCV, you'll get BGR — convert
before calling `policy.inference`.

### 9.5 Sensapex centered counts

The CSV records *centered* counts (`CENTER_OFFSET = 10000` in
`ump_suite/ump_driver_node.py`), so 0 means middle-of-travel. The
policy is trained on these centered values; do not pre-shift before
calling `policy.inference`. Just pass the same `[x1, y1, z1, d1, x2,
y2, z2, d2]` you'd publish on `/ump/target` and `/ump2/target`.

### 9.6 Camera resolution

Native camera resolution doesn't matter for training — `dataset.py`
resizes to `IMAGE_HEIGHT × IMAGE_WIDTH`, and `policy.inference` does
the same on the rollout side. But: if your demos were collected at
2× the resolution your policy will see, fine details may be lost.
If you ever need higher resolution, bump `IMAGE_HEIGHT` and
`IMAGE_WIDTH` in `config.config` and retrain — every other shape
flows from there.

### 9.7 Checkpoint disk usage

One ACT checkpoint with optimizer state is **~960 MB** (measured). With
`policy_last.pt` + `policy_best.pt` + numbered checkpoints every 100
epochs over 2000 epochs, that's ~22 GB. Use `--ckpt-dir` to point at
a roomier filesystem, or raise `--save-every` to drop fewer numbered
copies. Note: an early-stage symptom of running out of disk during
training is `train.py` exiting with status 1 and no log line — the
checkpoint write fails before any epoch summary flushes to stdout.

### 9.8 Adding the focus motor or pressure later

When you're ready to control the focusing motor and/or the solenoids:

1. In `config/config.py`, append the new column names to
   `CSV_STATE_COLS` and `CSV_ACTION_COLS`, and bump `STATE_DIM` and
   `ACTION_DIM` to match.
2. Re-collect (or re-export) trials with those columns populated.
3. Retrain. Nothing in the model needs to change — every shape is
   driven by `STATE_DIM` and `ACTION_DIM`.

For solenoids specifically, you may want to switch from L1 loss to
BCE on those dims (since they are binary), but L1 on `{0, 1}` still
works as a starting point.

### 9.9 Validation split: trial vs timestep

By default, `train.py` shuffles all `(trial, timestep)` samples and
random-splits them. This is **timestep-level** splitting and it has a
known caveat for behavior cloning: timestep `t` from trial 2 might
land in train while `t+1` from trial 2 lands in val. Adjacent
timesteps share ~99% of the next-100-action chunk and a near-identical
image, so the val example is essentially a one-step-shifted train
example. Val loss measures **interpolation**, not generalization to a
new trial.

**Why we keep this as the default:**

- ACT's reference code uses random split too. Authors validated on the
  real robot, not on val loss.
- With **<10 trials**, the alternative (trial-level split) gives val
  sets so small they're statistical noise — worse than the leakage.
- For early-development debugging, val loss as a "is training
  progressing?" signal is still useful even when leaky.

**When to switch.** Once you have **≥10 trials of realistic length**,
the leakage starts mattering and trial-level split becomes both viable
and informative. Pass:

```bash
python train.py --val-by-trial --val-split 0.2
```

This holds out **whole trials** (20% of them, rounded). The script
prints which trial IDs went to val so you can correlate with what was
in those trials. Same `--val-split` knob, but it now means "fraction of
trials" instead of "fraction of timesteps."

**Hierarchy of validation signals (most → least trustworthy):**

1. Closed-loop performance on the robot — the only metric that ultimately matters.
2. Trial-level held-out val loss (`--val-by-trial`) — predicts generalization.
3. Timestep-level random val loss (default) — sanity check that training is progressing; **not** a generalization metric.

### 9.10 Backbone choice

Pick `--backbone` based on dataset size, training budget, and how much
visual variation your task involves.

| Backbone | When to use | Tradeoffs |
|---|---|---|
| `resnet18` (default) | Small dataset (<100 trials), tight compute, you want the **ACT paper baseline** | Trainable end-to-end; smallest token budget (80 tokens) → fastest training. Pretrained on ImageNet which doesn't match microscope imagery. |
| `dinov2_vits14` | You suspect ImageNet features aren't general enough for your microscope view (focus drift, lighting changes, novel cell morphology) | ~21 M frozen params. Self-supervised features generalize better across domains. Produces ~5× more spatial tokens → ~2× train step time. |
| `dinov2_vitb14` | You have GPU headroom and want a stronger feature extractor | ~86 M frozen params. Marginal gain vs ViT-S unless you have a lot of visual variation. |
| `dinov2_vitl14` | You're benchmarking / writing an ablation table | ~300 M frozen. Significant memory; usually not worth the cost for behavior cloning. |

**Practical heuristic.** Start with `resnet18` for quick iteration. Once
you have a working pipeline with ≥50 trials, run `dinov2_vits14` as a
side-by-side experiment. If DINOv2 gives a meaningful val-loss drop or
better generalization to *new* trials (use `--val-by-trial`), switch.
If it doesn't, ResNet18's faster training wins.

**On freezing.** DINOv2 is designed as a frozen feature extractor —
the entire point is that the SSL pretraining produces features general
enough that downstream tasks don't need to fine-tune. Empirically:

- **Frozen (default)**: faster, lower VRAM, often equivalent quality.
- **Unfrozen** (`--unfreeze-backbone --lr-backbone 1e-6`): rarely
  helpful, much slower, easy to wreck pretrained features with too
  high a LR.

Stick with frozen unless you have a clear reason and a very small
backbone learning rate.

**Switching mid-project.** Checkpoints are not interchangeable across
backbones (different shapes for `input_proj` and the parameter set).
If you switch backbones, you start training from scratch (the rest of
the network is randomly initialized regardless of backbone choice).

---

## 10. Visualizing the Model Architecture

For a clickable, zoomable view of the entire network — every layer, every
shape, every connection — export the model to ONNX and open it in
[Netron](https://netron.app).

### 10.1 One-time export

```bash
python export_onnx.py                          # default: resnet18
python export_onnx.py --backbone dinov2_vits14 # DINOv2 variant
```

This writes four files into `onnx_exports/<backbone>/`:

```
onnx_exports/
├── resnet18/
│   ├── act_inference.onnx       1.5 MB  ← graph topology (drop in Netron)
│   ├── act_inference.onnx.data  254 MB  ← weight blobs (sidecar)
│   ├── act_training.onnx        1.8 MB  ← + style encoder branch
│   └── act_training.onnx.data   321 MB
└── dinov2_vits14/
    ├── act_inference.onnx       1.9 MB
    ├── act_inference.onnx.data  295 MB
    ├── act_training.onnx        2.2 MB
    └── act_training.onnx.data   362 MB
```

The script builds a **fresh, randomly-initialized** model — weight
values are meaningless, but the graph structure is exactly what
`train.py` would optimize. Re-run any time you change the architecture.

### 10.2 Viewing — two options

**Option A: Netron web app (zero install).** Open
[netron.app](https://netron.app) and drag in `act_inference.onnx`. The
browser sandbox can't auto-load the `.data` sidecar, so weight tensors
display as `<unloaded>`. That's fine for understanding the architecture
— layer types, shapes, and connections all render correctly.

**Option B: Netron desktop / pip (full weight loading).**

```bash
pip install netron
netron onnx_exports/resnet18/act_inference.onnx        # localhost:8080
# or for the DINOv2 variant:
netron onnx_exports/dinov2_vits14/act_inference.onnx
```

The local web server can read the `.data` sidecar, so you also see
weight values inside each node.

### 10.3 What you'll see

| File | Architecture content |
|---|---|
| `resnet18/act_inference.onnx` | ResNet18 backbone → 1×1 channel projection → main encoder (4 layers, 8 heads) → main decoder (7 layers) with 100 query tokens cross-attending to memory → action head. **No style encoder.** |
| `resnet18/act_training.onnx` | Same as above, **plus** the CVAE style encoder reading `(qpos, actions, is_pad)` and producing `(mu, logvar)`. |
| `dinov2_vits14/act_inference.onnx` | Same overall structure but with a DINOv2 ViT-S encoder block in place of ResNet18. The image token sequence is ~5× longer (391 vs 80 tokens). |
| `dinov2_vits14/act_training.onnx` | DINOv2 inference + style encoder branch. |

Compare `resnet18/act_inference.onnx` vs `dinov2_vits14/act_inference.onnx`
side by side to see exactly what swapping backbones changes (and what
stays the same — the entire transformer + style encoder is identical).

Comparing the two side-by-side is the cleanest visual demonstration of
what the CVAE adds: a parallel encoder branch whose only purpose is to
emit a small "style code" that conditions the rest of the network.

### 10.4 Notes & gotchas

- **`nn.MultiheadAttention` exports as one fused node.** Netron will show
  attention as a single block with input/output shapes rather than
  exposing the internal Q/K/V projections. This is the right level of
  abstraction for architecture viewing — drill into the source if you
  need the inner ops.
- **Opset 18.** Required by torch 2.10 for the internal `aten_split`
  decomposition. Older opsets fail with a domain-version mismatch error.
- **Disk usage.** ~575 MB total per export. `onnx_exports/` and
  `*.onnx`/`*.onnx.data` are in `.gitignore` so you won't commit them
  by accident.
- **Want a fast text-only summary instead?** Use `viz_summary.py`
  ([§5.12](#512-viz_summarypy)) — no graph, just a layer-by-layer table.

### 10.5 When to use which tool

Three visualization tools ship in this repo. They answer different
questions and produce wildly different output sizes — pick the one that
matches what you're actually trying to understand.

| Tool | Script | Output | Granularity | Use when... |
|---|---|---|---|---|
| **torchinfo** | `viz_summary.py` | ~300 lines text | One row per `nn.Module` (no math ops) | You want a scannable list: "what layers exist? what's their size? how many params?" |
| **Netron** | `export_onnx.py` | Interactive web view | One box per ONNX op (~50 boxes module-level) | You want a clickable architecture diagram with shape arrows |
| **torchviz** | `viz_torchviz.py` | 4 × SVG (90 KB – 1.2 MB) | One box per autograd op (~5,000 ops total) | You're debugging gradients: "is this branch traced?", "where does the loss flow?" |

#### Recommended order for a newcomer

1. **Start with `viz_summary.py`** to learn the model in 5 minutes. Pure
   text, no install beyond `pip install torchinfo`. Tells you *what
   pieces exist*.
2. **Then `export_onnx.py` + Netron** to see *how the pieces connect*.
   Module-level visual flow.
3. **Reach for `viz_torchviz.py` only when debugging.** Seeing every
   autograd op is overkill for understanding architecture — but
   essential when a gradient is zero and you need to know why.

#### Running each

All three scripts accept the same `--backbone` flag — pass `resnet18`
(default) or `dinov2_vits14` / `dinov2_vitb14` / `dinov2_vitl14`.

```bash
# Layer summary (fastest, no install beyond torchinfo)
pip install torchinfo
python viz_summary.py                           # defaults to config.BACKBONE
python viz_summary.py --backbone dinov2_vits14
python viz_summary.py --backbone dinov2_vits14 > architecture_dinov2.txt

# ONNX for Netron (drag .onnx into https://netron.app)
python export_onnx.py                           # → onnx_exports/resnet18/
python export_onnx.py --backbone dinov2_vits14  # → onnx_exports/dinov2_vits14/

# Autograd graph (graphviz system pkg + torchviz pip pkg)
pip install torchviz
python viz_torchviz.py                          # → torchviz_exports/resnet18/
python viz_torchviz.py --backbone dinov2_vits14 # → torchviz_exports/dinov2_vits14/
```

Outputs from `viz_torchviz.py` (`torchviz_exports/<backbone>/`):

```
01_backbone.svg          backbone + 1x1 projection only      (~90 KB / ~5 KB DINOv2 frozen)
02_style_encoder.svg     CVAE branch: (qpos, actions) -> mu/logvar  (~230 KB)
03_inference_full.svg    Full deployment forward pass        (~950 / ~860 KB)
04_training_full.svg     Full training forward pass          (~1.2 / ~1.1 MB)
```

The DINOv2 `01_backbone.svg` is tiny because the backbone is frozen —
the autograd graph stops at its output. Pass `--unfreeze-backbone` to
include DINOv2's internal ops (much larger graph).

SVG is vector — open in any browser and zoom freely. Start with
`01_backbone.svg` and `02_style_encoder.svg` to learn the conventions
(blue ovals = parameters, grey rectangles = autograd ops, orange =
saved tensors).

`onnx_exports/` and `torchviz_exports/` are gitignored.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **ACT** | Action Chunking with Transformers (Zhao et al. 2023). |
| **Action chunk** | The next `CHUNK_SIZE` actions predicted from one observation. |
| **Chunk size (k)** | Number of future actions the policy predicts per inference. Default 100. |
| **CVAE** | Conditional Variational Autoencoder. Here: encoder over `[CLS, qpos, actions]`, decoder is the main ACT transformer conditioned on `z`. |
| **DETR** | Detection Transformer (Carion et al. 2020). The transformer architecture and frozen-BN convention come from here. |
| **FrozenBatchNorm2d** | BatchNorm whose running stats and affine params are frozen. Stable at small batch sizes. |
| **`is_pad`** | Boolean mask over the action chunk: `True` where the trial ended before `CHUNK_SIZE` real future actions were available. |
| **Open-loop horizon** | Number of actions executed from each predicted chunk before re-inferring. |
| **qpos** | "Joint position." Used loosely here for the 8-dim Sensapex state vector. |
| **Sensapex centered counts** | Symmetric-around-zero integer position units used by the rig (0 = middle of travel). |
| **Style encoder** | The CVAE encoder that produces `(μ, logσ²)` from a demo's `(qpos, actions)`. |
| **Temporal aggregation** | Averaging predictions from overlapping chunks at deployment to reduce jitter. |

---

*End of report.*
