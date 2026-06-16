"""Configuration for MicroVLA.

Single source of truth for shapes, model hyperparameters, dataset/feature keys,
and the optional learnable-feature switches (contact-point goal head, per-axis
action weighting, optional resistance conditioning).

MicroVLA trains from a LeRobot-format dataset (SmolVLA / OpenPI / pi0 style).
The raw-CSV constants near the top are only used by the dataset converters under
``dataset_vla/`` to read the original micromanipulation logs.
"""

import os
from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Raw micromanipulation logs (only read by the dataset_vla converters).
_DATASET_ROOT_ENV = os.environ.get("MICROVLA_DATASET_ROOT")
DATASET_ROOT = (
    Path(_DATASET_ROOT_ENV).expanduser() if _DATASET_ROOT_ENV else REPO_ROOT / "dataset"
)

VLA_CKPT_DIR = REPO_ROOT / "checkpoints_vla"
VLA_STATS_PATH = VLA_CKPT_DIR / "vla_stats.pkl"

# ---------- Raw CSV layout (converters only) ----------
# Each manipulator contributes AXES_PER_MANIPULATOR axes (x, y, z, d). The raw
# logs may carry a second manipulator (x2, y2, z2, d2); NUM_MANIPULATORS chooses
# how many THIS dataset/model uses:
#   1 -> single pipette: state/action = [x, y, z, d]           (4-dim)
#   2 -> dual pipette:   state/action = [x1..d1, x2..d2]       (8-dim)
# The model is dimension-agnostic (padded to MAX_*_DIM with masks), so switching
# is just this one number + a re-convert + a re-train. Nothing is hard-coded to 8.
AXES_PER_MANIPULATOR = 4
NUM_MANIPULATORS = 1

_STATE_COLS_BY_MANIPULATOR = (
    ("current_x",  "current_y",  "current_z",  "current_d"),
    ("current_x2", "current_y2", "current_z2", "current_d2"),
)
_ACTION_COLS_BY_MANIPULATOR = (
    ("target_x",  "target_y",  "target_z",  "target_d"),
    ("target_x2", "target_y2", "target_z2", "target_d2"),
)


def state_cols_for(n_manipulators: int) -> tuple:
    """Flat raw state-column names for the first ``n_manipulators`` manipulators."""
    return tuple(c for g in _STATE_COLS_BY_MANIPULATOR[:int(n_manipulators)] for c in g)


def action_cols_for(n_manipulators: int) -> tuple:
    """Flat raw action-column names for the first ``n_manipulators`` manipulators."""
    return tuple(c for g in _ACTION_COLS_BY_MANIPULATOR[:int(n_manipulators)] for c in g)


CSV_STATE_COLS = state_cols_for(NUM_MANIPULATORS)
CSV_ACTION_COLS = action_cols_for(NUM_MANIPULATORS)
CSV_IMAGE_COL = "image_path"
CSV_TIMESTEP_COL = "timestep"
# Optional per-timestep pipette resistance (megaohms). Used automatically if the
# column exists and has real values; ignored otherwise. See "Resistance" below.
CSV_RESISTANCE_COL = "resistance_mohm"

# ---------- Vision ----------
NUM_CAMERAS = 1
CAMERA_NAMES = ("cam_main",)
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# ---------- Heterogeneous robot contract ----------
# Each episode can expose fewer dimensions. Samples are padded to these maxima
# and accompanied by state/action masks.
MAX_STATE_DIM = 16
MAX_ACTION_DIM = 16

STATE_MASK_KEY = "state_mask"
ACTION_MASK_KEY = "action_mask"

# ---------- Action chunking ----------
# At ~3 Hz, 30 steps is ~10 s of future actions — well matched to the short
# micromanipulation moves in this data. (The old ACT default of 100 covered up to
# half a trial, which forced the model to regress noisy far-future targets.)
CHUNK_SIZE = 30

# ---------- Model hyperparameters ----------
HIDDEN_DIM = 512
DIM_FEEDFORWARD = 3200
ENC_LAYERS = 4
DEC_LAYERS = 7
NHEAD = 8
DROPOUT = 0.1
LATENT_DIM = 32           # CVAE style-latent dimension
KL_WEIGHT = 10.0          # beta on the KL term (ACT paper default)

# ---------- Backbone ----------
DEFAULT_BACKBONE = "dinov2_vits14+cellpose4"
# Shared backbone module reads BACKBONE / BACKBONE_PRETRAINED as fallbacks.
BACKBONE = DEFAULT_BACKBONE
BACKBONE_PRETRAINED = True

# Cellpose 4 / Cellpose-SAM defaults (used by the `cellpose4` backbone).
CELLPOSE4_DIAMETER = 180.0
CELLPOSE4_CELLPROB_THRESHOLD = -2.0
CELLPOSE4_FLOW_THRESHOLD = 1.5
CELLPOSE4_INCLUDE_READOUT = True

# ---------- Language ----------
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
LANGUAGE_BACKEND = "hf"          # "hf" for frozen Transformers, "simple" for offline smoke tests
SIMPLE_TEXT_VOCAB_SIZE = 8192
MAX_LANGUAGE_TOKENS = 32

# ---------- LeRobot dataset (HF) ----------
DEFAULT_DATASET_REPO_ID = "RaianSilex/microvla_ump_dataset"
DEFAULT_ACTION_SPACE = "delta"   # "delta" (relative to base state) or "absolute"
# Standard LeRobot feature keys (so smolvla/lerobot tooling can read the dataset).
LEROBOT_CAMERA_KEY = "observation.images.cam_main"
LEROBOT_STATE_KEY = "observation.state"
LEROBOT_ACTION_KEY = "action"
# Optional: per-frame pipette resistance, written by the converter only when the
# raw logs contain real resistance values. The loader auto-detects it.
LEROBOT_RESISTANCE_KEY = "observation.resistance"
# Optional (Variant B): per-episode contact point in NORMALIZED image pixels
# (u, v) in [0, 1], generated offline by Cellpose (the "teacher") and written by
# the converter only when a cell-labels file is present. The loader auto-detects
# it. This grounds the cell-aware selection + image-space contact-point heads.
LEROBOT_GOAL_PIXEL_KEY = "observation.goal_pixel"

# ---------- Metadata vocab fallbacks ----------
# Real ids are built from dataset metadata and saved into VLA checkpoints.
UNKNOWN_TOKEN = "<unk>"
DEFAULT_ROBOT_ID = "sensapex_dual_ump4"
DEFAULT_LAB_ID = "local_lab"
DEFAULT_EMBODIMENT = "dual_manipulator"
DEFAULT_ACTION_TYPE = "absolute_position"
DEFAULT_TASK_FAMILY = "cell_manipulation"

NUM_ROBOT_IDS_FALLBACK = 64
NUM_LAB_IDS_FALLBACK = 64
NUM_EMBODIMENT_IDS_FALLBACK = 32
NUM_ACTION_TYPE_IDS_FALLBACK = 32
NUM_TASK_FAMILY_IDS_FALLBACK = 64

# ---------- Contact-point (goal) head ----------
# An auxiliary head predicts a Gaussian over the episode's final reached target
# (the "contact point") in the action representation: mean + per-dim log-variance.
# It is trained with a Gaussian negative-log-likelihood (so the variance is a
# learned, calibrated uncertainty) and the prediction conditions the trajectory
# decoder. See model/vla_cvae.py and model/vla_policy.py.
GOAL_HEAD = True
GOAL_LOSS_WEIGHT = 1.0
GOAL_LOGVAR_MIN = -6.0     # clamp predicted log-variance for numerical stability
GOAL_LOGVAR_MAX = 4.0

# ---------- Per-axis adaptive action weighting ----------
# The masked L1 over the action chunk is weighted per dimension by how much that
# axis actually moves in the dataset, so near-constant axes (e.g. a fixed depth)
# stop diluting the loss while axes that DO move (or start moving in a future
# dataset) are learned. Computed from data; nothing is hard-coded per rig.
AXIS_WEIGHTING = True
AXIS_WEIGHT_MIN = 0.05     # floor so a dim is never fully ignored
AXIS_WEIGHT_MAX = 3.0      # cap so no single dim dominates

# ---------- Cell-aware contact head (Variant B: Cellpose as a teacher) ----------
# Cellpose is run OFFLINE (training-time only) to produce a per-episode contact
# point in image pixels (see dataset_vla/generate_cell_labels.py). When that
# label is present the policy learns two AUXILIARY heads from the (ResNet/DINOv2)
# image features — so inference stays backbone-only, with no Cellpose in the loop:
#   * a cell-SELECTION head: which CELL_GRID x CELL_GRID frame region holds the
#     target cell (cross-entropy) — the "which cell" of technique 2;
#   * an image-space contact-point GAUSSIAN head: a diagonal Gaussian over the
#     target cell's (u, v) in [0, 1] (Gaussian NLL) — the "where on it".
# These shape the image representation to be cell-aware; they do not change the
# action head, so the same checkpoint runs unchanged when no cell labels exist.
CELL_HEAD = True               # auto-gated: only active when the data carries goal_pixel
CELL_GRID = 3                  # frame split into CELL_GRID x CELL_GRID selection regions
CELL_GOAL_WEIGHT = 1.0         # weight on the image-space contact-point Gaussian NLL
CELL_SELECT_WEIGHT = 0.5       # weight on the cell-selection cross-entropy
CELL_LOGVAR_MIN = -8.0         # clamp pixel-Gaussian log-variance (coords are in [0, 1])
CELL_LOGVAR_MAX = 2.0

# ---------- Resistance conditioning (optional) ----------
# If the dataset carries a per-frame resistance signal, the policy conditions on
# it via an extra source token. Training randomly zeroes it (modality dropout) so
# the same checkpoint still runs when the sensor is absent at rollout.
RESISTANCE_DROPOUT = 0.3

# ---------- Training ----------
BATCH_SIZE = 8
NUM_EPOCHS = 2000
LR = 1e-5
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 0
DEVICE = "cuda"
VAL_SPLIT = 0.1            # episode-level validation fraction

# ---------- Rollout ----------
OPEN_LOOP_HORIZON = 8       # actions consumed per inference during rollout
CONTROL_HZ = 5.0
TEMPORAL_AGG = True         # ACT-style exponential ensembling in rollout/vla_main.py
TEMPORAL_AGG_K = 0.01
