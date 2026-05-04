"""Global configuration for MicroACT.

Shapes here are tied to the current rig: 2 Sensapex uMp4 stages, 1 camera,
no focusing motor and no pressure solenoids. To add those later, bump
STATE_DIM / ACTION_DIM and extend CSV_STATE_COLS / CSV_ACTION_COLS.
"""

from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
LOGS_DIR = DATASET_ROOT / "logs"                 # trial_N.csv
FRAMES_DIR = DATASET_ROOT / "saved_frames"       # trial_N/frame_NNNNNN.png
CKPT_DIR = REPO_ROOT / "checkpoints"
STATS_PATH = CKPT_DIR / "dataset_stats.pkl"

# ---------- Robot / data shapes ----------
# State  = [x1, y1, z1, d1, x2, y2, z2, d2]   (centered Sensapex counts)
# Action = absolute target in the same 8-dim space.
STATE_DIM = 8
ACTION_DIM = 8

NUM_CAMERAS = 1
CAMERA_NAMES = ("cam_main",)
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

# CSV columns consumed from each trial_N.csv.
# Anything else in the file (motor, image_path handled separately) is ignored.
CSV_STATE_COLS = (
    "current_x",  "current_y",  "current_z",  "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2",
)
CSV_ACTION_COLS = (
    "target_x",  "target_y",  "target_z",  "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2",
)
CSV_IMAGE_COL = "image_path"
CSV_TIMESTEP_COL = "timestep"

# ---------- ACT model hyperparameters ----------
CHUNK_SIZE = 100          # k: number of future actions predicted per inference
HIDDEN_DIM = 512
DIM_FEEDFORWARD = 3200
ENC_LAYERS = 4
DEC_LAYERS = 7
NHEAD = 8
DROPOUT = 0.1
LATENT_DIM = 32           # CVAE style-latent dimension
KL_WEIGHT = 10.0          # beta on the KL term (ACT paper default)

BACKBONE = "resnet18"
BACKBONE_PRETRAINED = True

# ---------- Training ----------
BATCH_SIZE = 8
NUM_EPOCHS = 2000
LR = 1e-5
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 0
DEVICE = "cuda"

# ---------- Rollout ----------
OPEN_LOOP_HORIZON = 8       # actions consumed per inference during rollout
CONTROL_HZ = 5.0            # matches logger_node default log_interval_ms = 200
TEMPORAL_AGG = True         # ACT-style exponential ensembling in rollout/main.py
TEMPORAL_AGG_K = 0.01
