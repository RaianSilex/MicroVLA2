"""Configuration for the heterogeneous MicroVLA pipeline.

The original MicroACT path stays fixed to the current dual-Sensapex rig. This
module defines the wider contract used when demonstrations come from multiple
robots, labs, action conventions, and language-labeled tasks.
"""

import os
from pathlib import Path

from config import config as ACT

# ---------- Paths ----------
REPO_ROOT = ACT.REPO_ROOT
_VLA_DATASET_ROOT_ENV = os.environ.get("MICROVLA_VLA_DATASET_ROOT")
VLA_DATASET_ROOT = (
    Path(_VLA_DATASET_ROOT_ENV).expanduser()
    if _VLA_DATASET_ROOT_ENV
    else REPO_ROOT / "dataset_vla"
)
VLA_EPISODES_DIR = VLA_DATASET_ROOT / "episodes"
VLA_CKPT_DIR = REPO_ROOT / "checkpoints_vla"
VLA_STATS_PATH = VLA_CKPT_DIR / "vla_stats.pkl"

# ---------- Heterogeneous robot contract ----------
# Each episode can expose fewer dimensions. Samples are padded to these maxima
# and accompanied by state/action masks.
MAX_STATE_DIM = 16
MAX_ACTION_DIM = 16

STATE_MASK_KEY = "state_mask"
ACTION_MASK_KEY = "action_mask"

# ---------- Vision / language / action chunking ----------
NUM_CAMERAS = ACT.NUM_CAMERAS
IMAGE_HEIGHT = ACT.IMAGE_HEIGHT
IMAGE_WIDTH = ACT.IMAGE_WIDTH
CHUNK_SIZE = ACT.CHUNK_SIZE

DEFAULT_BACKBONE = "dinov2_vits14+cellpose4"
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
LANGUAGE_BACKEND = "hf"          # "hf" for frozen Transformers, "simple" for offline smoke tests
SIMPLE_TEXT_VOCAB_SIZE = 8192
MAX_LANGUAGE_TOKENS = 32

# ---------- LeRobot dataset (HF) ----------
# MicroVLA can train from a LeRobot-format dataset (like SmolVLA / OpenPI / pi0)
# instead of the local dataset_vla/episodes/ tree. The dataset is robot-native:
# it stores ABSOLUTE Sensapex targets; the action space (delta vs absolute) is a
# train-time transform chosen here / on the CLI. See data/lerobot_vla_dataset.py
# and dataset_vla/convert_microact_to_lerobot.py.
DEFAULT_DATASET_REPO_ID = "RaianSilex/microvla_ump_dataset"
DEFAULT_ACTION_SPACE = "delta"   # "delta" (relative to base state) or "absolute"
# Standard LeRobot feature keys (so smolvla/lerobot tooling can read the dataset).
LEROBOT_CAMERA_KEY = "observation.images.cam_main"
LEROBOT_STATE_KEY = "observation.state"
LEROBOT_ACTION_KEY = "action"

# Vocabulary fallback ids. Real ids are built from dataset metadata in
# data/vla_dataset.py and saved into VLA checkpoints.
UNKNOWN_TOKEN = "<unk>"
DEFAULT_ROBOT_ID = "sensapex_dual_ump4"
DEFAULT_LAB_ID = "local_lab"
DEFAULT_EMBODIMENT = "dual_manipulator"
DEFAULT_ACTION_TYPE = "absolute_position"
DEFAULT_TASK_FAMILY = "cell_manipulation"

# ---------- Model hyperparameters ----------
HIDDEN_DIM = ACT.HIDDEN_DIM
DIM_FEEDFORWARD = ACT.DIM_FEEDFORWARD
ENC_LAYERS = ACT.ENC_LAYERS
DEC_LAYERS = ACT.DEC_LAYERS
NHEAD = ACT.NHEAD
DROPOUT = ACT.DROPOUT
LATENT_DIM = ACT.LATENT_DIM
KL_WEIGHT = ACT.KL_WEIGHT

NUM_ROBOT_IDS_FALLBACK = 64
NUM_LAB_IDS_FALLBACK = 64
NUM_EMBODIMENT_IDS_FALLBACK = 32
NUM_ACTION_TYPE_IDS_FALLBACK = 32
NUM_TASK_FAMILY_IDS_FALLBACK = 64

# ---------- Training ----------
BATCH_SIZE = ACT.BATCH_SIZE
NUM_EPOCHS = ACT.NUM_EPOCHS
LR = ACT.LR
LR_BACKBONE = ACT.LR_BACKBONE
WEIGHT_DECAY = ACT.WEIGHT_DECAY
SEED = ACT.SEED
DEVICE = ACT.DEVICE

# Heterogeneous data should be split by whole episodes by default. Lab/robot
# holdouts can be added on top from the metadata without changing the model.
VAL_SPLIT = 0.1

# ---------- Rollout ----------
OPEN_LOOP_HORIZON = ACT.OPEN_LOOP_HORIZON
CONTROL_HZ = ACT.CONTROL_HZ
TEMPORAL_AGG = ACT.TEMPORAL_AGG
TEMPORAL_AGG_K = ACT.TEMPORAL_AGG_K
