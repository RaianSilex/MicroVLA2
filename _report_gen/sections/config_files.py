from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # ----- config/config.py -----
    h1(story, "config/config.py")
    h2(story, "Purpose")
    body(story, "Single source of truth for paths, robot/data shapes, ACT model "
                "hyperparameters, training defaults, and rollout defaults. The rest of "
                "the codebase imports it as <code>from config import config as C</code>.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<b>STATE_DIM = ACTION_DIM = 8</b>; image is <b>240 &times; 320</b> RGB; "
        "<b>CHUNK_SIZE = 100</b>; <b>HIDDEN_DIM = 512</b>; <b>LATENT_DIM = 32</b>.",
        "<b>NUM_CAMERAS = 1</b>, <b>BATCH_SIZE = 8</b>, default <b>BACKBONE = \"resnet18\"</b>.",
        "Rollout defaults: <b>OPEN_LOOP_HORIZON = 8</b>, <b>CONTROL_HZ = 5.0</b>, "
        "<b>TEMPORAL_AGG = True</b>, <b>TEMPORAL_AGG_K = 0.01</b>.",
    ])
    code_block(story, "config/config.py:8-16 - paths", """\
from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
LOGS_DIR = DATASET_ROOT / "logs"                 # trial_N.csv
FRAMES_DIR = DATASET_ROOT / "saved_frames"       # trial_N/frame_NNNNNN.png
CKPT_DIR = REPO_ROOT / "checkpoints"
STATS_PATH = CKPT_DIR / "dataset_stats.pkl\"""")
    bullets(story, [
        "<code>__file__.resolve().parent.parent</code> walks two levels up from "
        "<i>config/config.py</i> to reach the repo root, so all subsequent paths are "
        "absolute regardless of how the script is launched.",
        "<b>STATS_PATH</b> is the default location for the pickled normalization stats. "
        "<code>train.py</code> overrides it under <code>--ckpt-dir</code>.",
    ])
    code_block(story, "config/config.py:18-40 - data shapes and CSV schema", """\
# State  = [x1, y1, z1, d1, x2, y2, z2, d2]   (centered Sensapex counts)
# Action = absolute target in the same 8-dim space.
STATE_DIM = 8
ACTION_DIM = 8

NUM_CAMERAS = 1
CAMERA_NAMES = ("cam_main",)
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

CSV_STATE_COLS = (
    "current_x",  "current_y",  "current_z",  "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2",
)
CSV_ACTION_COLS = (
    "target_x",  "target_y",  "target_z",  "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2",
)
CSV_IMAGE_COL = "image_path"
CSV_TIMESTEP_COL = "timestep\"""")
    bullets(story, [
        "The 8-D state vector is the concatenation of two Sensapex uMp4 stages, each "
        "with x/y/z/d (depth) axes. <code>d</code> is the focus/depth motor specific to "
        "the rig, not gripper width.",
        "The CSV column tuples define the <i>order</i> in which dataframes are sliced "
        "into the qpos and action arrays — change the tuple to change the column order.",
    ])
    code_block(story, "config/config.py:42-69 - hyperparameters and rollout", """\
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
OPEN_LOOP_HORIZON = 8
CONTROL_HZ = 5.0
TEMPORAL_AGG = True
TEMPORAL_AGG_K = 0.01""")
    bullets(story, [
        "<b>CHUNK_SIZE = 100</b> is the action-chunking horizon <i>k</i>: the decoder "
        "always emits 100 future actions, even when only the first few will be executed.",
        "<b>DIM_FEEDFORWARD = 3200</b> is the inner MLP width inside transformer blocks; "
        "<b>HIDDEN_DIM = 512</b> is the token width <i>D</i>. The transformer is heavy on "
        "FFN and light on attention.",
        "<b>LATENT_DIM = 32</b> means the style encoder outputs <code>mu (B,32)</code> "
        "and <code>logvar (B,32)</code>.",
        "<b>KL_WEIGHT = 10.0</b> is the &beta; multiplier on the KL term in "
        "ACTPolicy._compute_loss.",
        "<b>OPEN_LOOP_HORIZON = 8</b> is how many of the 100 predicted actions to actually "
        "execute before re-running inference (when temporal aggregation is OFF).",
        "<b>TEMPORAL_AGG_K = 0.01</b> is the decay rate in <code>exp(-k * age)</code> "
        "weighting; smaller k means older predictions are still trusted.",
    ])

    # ----- config/vla_config.py -----
    h1(story, "config/vla_config.py (NEW)")
    h2(story, "Purpose")
    body(story, "Configuration for the heterogeneous MicroVLA pipeline. The original "
                "MicroACT path stays fixed to the dual-Sensapex rig; this file declares "
                "the wider contract used when demonstrations come from multiple robots, "
                "labs, action conventions, and language-labeled tasks.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<b>MAX_STATE_DIM = MAX_ACTION_DIM = 16</b>. Each episode declares its own "
        "<i>state_dim</i> &le; 16 and is right-padded with zeros to the maximum.",
        "<b>state_mask</b> and <b>action_mask</b> (boolean, length 16) mark which "
        "indices are real for each episode.",
        "<b>NUM_CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH, CHUNK_SIZE</b> are re-exported from "
        "<code>config.config</code>, so the VLA model expects identical 240&times;320 inputs and a "
        "100-step action horizon.",
        "<b>HIDDEN_DIM, LATENT_DIM, ENC_LAYERS, DEC_LAYERS, NHEAD, DIM_FEEDFORWARD</b> "
        "are also re-exported — the transformer geometry is identical to ACT's.",
    ])
    code_block(story, "config/vla_config.py:1-17 - module identity and paths", """\
from pathlib import Path

from config import config as ACT

# ---------- Paths ----------
REPO_ROOT = ACT.REPO_ROOT
VLA_DATASET_ROOT = REPO_ROOT / "dataset_vla"
VLA_EPISODES_DIR = VLA_DATASET_ROOT / "episodes"
VLA_CKPT_DIR = REPO_ROOT / "checkpoints_vla"
VLA_STATS_PATH = VLA_CKPT_DIR / "vla_stats.pkl\"""")
    bullets(story, [
        "VLA imports the ACT config as <code>ACT</code> and reuses its paths/shape "
        "constants. Anything that <i>differs</i> from ACT (paths, max dims, language "
        "hyperparameters) is redefined locally.",
        "Checkpoints live at <b>checkpoints_vla/</b> so ACT and VLA training never "
        "overwrite each other's files.",
    ])
    code_block(story, "config/vla_config.py:19-47 - heterogeneous robot contract", """\
MAX_STATE_DIM = 16
MAX_ACTION_DIM = 16

STATE_MASK_KEY = "state_mask"
ACTION_MASK_KEY = "action_mask"

NUM_CAMERAS = ACT.NUM_CAMERAS
IMAGE_HEIGHT = ACT.IMAGE_HEIGHT
IMAGE_WIDTH = ACT.IMAGE_WIDTH
CHUNK_SIZE = ACT.CHUNK_SIZE

DEFAULT_BACKBONE = "dinov2_vits14+cellpose"
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"
LANGUAGE_BACKEND = "hf"
SIMPLE_TEXT_VOCAB_SIZE = 8192
MAX_LANGUAGE_TOKENS = 32

UNKNOWN_TOKEN = "<unk>"
DEFAULT_ROBOT_ID = "sensapex_dual_ump4"
DEFAULT_LAB_ID = "local_lab"
DEFAULT_EMBODIMENT = "dual_manipulator"
DEFAULT_ACTION_TYPE = "absolute_position"
DEFAULT_TASK_FAMILY = "cell_manipulation\"""")
    bullets(story, [
        "<b>MAX_STATE_DIM/MAX_ACTION_DIM = 16</b> sets the upper bound on per-robot "
        "DOF. A single-arm 4-DOF demonstration packs into <code>qpos[:4]</code> with "
        "<code>state_mask[:4] = True</code>; everything past 4 is zero and masked.",
        "<b>DEFAULT_BACKBONE = \"dinov2_vits14+cellpose\"</b> uses the dual encoder "
        "(general scene features + cell-aware features) by default for VLA.",
        "<b>DEFAULT_TEXT_MODEL = \"distilbert-base-uncased\"</b> is a 66M-parameter "
        "frozen text encoder. Hidden size 768, output projected to D=512.",
        "<b>MAX_LANGUAGE_TOKENS = 32</b> caps tokenized instructions; longer ones get "
        "truncated. This determines the language-token sequence length added to the "
        "transformer source.",
        "The five DEFAULT_* string constants are fallback IDs used when an episode's "
        "metadata is missing a field — they ensure the vocab lookup always finds a "
        "sane value during inference.",
    ])
    code_block(story, "config/vla_config.py:59-82 - vocab fallbacks, training, rollout", """\
NUM_ROBOT_IDS_FALLBACK = 64
NUM_LAB_IDS_FALLBACK = 64
NUM_EMBODIMENT_IDS_FALLBACK = 32
NUM_ACTION_TYPE_IDS_FALLBACK = 32
NUM_TASK_FAMILY_IDS_FALLBACK = 64

BATCH_SIZE = ACT.BATCH_SIZE
NUM_EPOCHS = ACT.NUM_EPOCHS
LR = ACT.LR
LR_BACKBONE = ACT.LR_BACKBONE
WEIGHT_DECAY = ACT.WEIGHT_DECAY
SEED = ACT.SEED
DEVICE = ACT.DEVICE

VAL_SPLIT = 0.1

OPEN_LOOP_HORIZON = ACT.OPEN_LOOP_HORIZON
CONTROL_HZ = ACT.CONTROL_HZ
TEMPORAL_AGG = ACT.TEMPORAL_AGG
TEMPORAL_AGG_K = ACT.TEMPORAL_AGG_K""")
    bullets(story, [
        "<b>NUM_*_FALLBACK</b> values are used by <code>EmbodimentConditioner</code> "
        "when constructing its <code>nn.Embedding</code> tables before the dataset is "
        "scanned. <code>VLAPolicy</code> overrides them with the actual vocab size at "
        "construction time, so the fallbacks really only matter for unit tests / "
        "torchinfo dumps.",
        "<b>VAL_SPLIT = 0.1</b> drives episode-level (not timestep-level) splits — VLA "
        "training is split by whole episodes by default to avoid leaking adjacent "
        "frames between train and val.",
    ])
