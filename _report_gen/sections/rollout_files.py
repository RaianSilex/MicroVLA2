from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # =====================================================================
    # rollout/rollout.py
    # =====================================================================
    h1(story, "rollout/rollout.py")
    h2(story, "Purpose")
    body(story, "Pure-Python helpers shared by the MicroACT and MicroVLA rollouts: "
                "scalar clamp, an E-STOP keyboard listener, a Ctrl+C-deferring "
                "context manager, and the <code>RolloutArgs</code> dataclass + CLI "
                "parser. No PyTorch, no ROS — easy to import in any context.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>clamp(v, lo, hi)</code>: scalar in, scalar out; tolerates inverted "
        "bounds (lo &gt; hi).",
        "<code>start_estop_listener()</code>: returns a single-item dict "
        "<code>{\"stop\": bool}</code> the rollout polls each tick.",
        "<code>RolloutArgs</code>: dataclass with ~20 fields covering checkpoint, "
        "loop, robot, smoothing, preview and debug knobs.",
    ])

    code_block(story, "rollout/rollout.py:20-24 - clamp", """\
def clamp(v: float, lo: float, hi: float) -> float:
    \"\"\"Clamp a scalar, accepting bounds in either order.\"\"\"
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return lower if v < lower else (upper if v > upper else float(v))""")
    bullets(story, [
        "Tolerating inverted bounds is deliberate: the Z-axis safety limits in "
        "<code>main.py</code> are written <code>Z1_MIN, Z1_MAX = 8750, 8250</code> "
        "(decreasing) because the Sensapex Z axis runs in negative-down direction "
        "in raw counts. Without this, the user would have to remember the convention "
        "at every call site.",
    ])

    code_block(story, "rollout/rollout.py:27-41 - start_estop_listener", """\
def start_estop_listener() -> dict:
    flag = {"stop": False}

    def _worker() -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                continue
            if line.strip().lower() == "q":
                flag["stop"] = True
                break

    threading.Thread(target=_worker, daemon=True).start()
    return flag""")
    bullets(story, [
        "<b>Daemon thread on stdin</b>: blocks on <code>readline()</code> until the "
        "user types something. <code>q + Enter</code> flips the shared flag, the "
        "worker exits, and the rollout's main loop sees the change on its next tick.",
        "<b>Daemon=True</b> ensures the thread dies when the main process exits — "
        "no need to explicitly join it on Ctrl+C.",
        "Returning the dict (not a bool) lets the rollout call "
        "<code>stop_flag[\"stop\"]</code> as a live reference rather than a snapshot.",
    ])

    code_block(story, "rollout/rollout.py:44-60 - prevent_keyboard_interrupt", """\
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt""")
    bullets(story, [
        "Defers Ctrl+C until <code>finally:</code>. Useful for atomic sections like "
        "publishing a final E-STOP command — Python won't raise mid-publish.",
        "Currently unused in the active rollout loops but kept for future use; cheap "
        "and safe to import.",
    ])

    code_block(story, "rollout/rollout.py:63-97 - RolloutArgs", """\
@dataclasses.dataclass
class RolloutArgs:
    # Policy checkpoint.
    checkpoint: Path = Path("checkpoints/policy_best.pt")
    stats_path: Path = Path("checkpoints/dataset_stats.pkl")
    backbone: str = "resnet18"
    device: str = "cuda"
    pretrained_backbone: bool = False
    unfreeze_backbone: bool = False

    # Rollout loop.
    max_timesteps: int = 600
    open_loop_horizon: int = C.OPEN_LOOP_HORIZON      # 8
    control_hz: float = C.CONTROL_HZ                  # 5.0
    temporal_agg: bool = C.TEMPORAL_AGG               # True
    temporal_agg_k: float = C.TEMPORAL_AGG_K          # 0.01
    dry_run: bool = False

    # Robot params.
    default_speed: int = 100

    # Optional first-order smoothing on commanded actions.
    use_ema_smoothing: bool = True
    ema_alpha: float = 0.35

    # Live preview file, useful over SSH.
    save_preview: bool = True
    preview_path: str = "microact_live.png"
    preview_every_n_frames: int = 5

    # Print one debug line every N steps; 0 disables.
    debug_every: int = 10""")
    bullets(story, [
        "<b>checkpoint default</b> is the best-on-val checkpoint produced by "
        "<code>train.py</code>; <b>stats_path default</b> is the per-axis "
        "<code>qpos/action_mean/std</code> pickle.",
        "<b>open_loop_horizon = 8</b>: when temporal aggregation is OFF, the rollout "
        "executes 8 of the 100 predicted actions before re-running inference.",
        "<b>control_hz = 5.0</b>: 5 Hz rollout = 200 ms per tick. The DataLoader "
        "rate during training is much higher; this number governs the closed-loop "
        "rate at deploy.",
        "<b>temporal_agg = True</b> with <code>temporal_agg_k = 0.01</code>: tiny "
        "decay means even old chunk predictions still contribute to the smoothed "
        "current action.",
        "<b>EMA smoothing</b> on top of temporal agg: "
        "<code>ema = alpha * new + (1 - alpha) * ema</code> with <code>alpha=0.35</code> "
        "gives a soft floor on per-tick jerk.",
    ])

    code_block(story, "rollout/rollout.py:99-146 - parse_args (CLI surface)", """\
def parse_args() -> RolloutArgs:
    p = argparse.ArgumentParser(description="Run a MicroACT checkpoint on the Sensapex rig.")
    p.add_argument("--checkpoint", type=Path, default=RolloutArgs.checkpoint)
    p.add_argument("--stats-path", type=Path, default=RolloutArgs.stats_path)
    p.add_argument("--backbone", type=str, default=RolloutArgs.backbone)
    p.add_argument("--device", type=str, default=RolloutArgs.device)
    p.add_argument("--pretrained-backbone", action="store_true", ...)
    p.add_argument("--unfreeze-backbone", action="store_true", ...)
    p.add_argument("--max-timesteps", type=int, default=RolloutArgs.max_timesteps)
    p.add_argument("--open-loop-horizon", type=int, default=RolloutArgs.open_loop_horizon)
    p.add_argument("--control-hz", type=float, default=RolloutArgs.control_hz)
    temporal_group = p.add_mutually_exclusive_group()
    temporal_group.add_argument("--temporal-agg", dest="temporal_agg", action="store_true")
    temporal_group.add_argument("--no-temporal-agg", dest="temporal_agg", action="store_false")
    p.set_defaults(temporal_agg=RolloutArgs.temporal_agg)
    p.add_argument("--temporal-agg-k", type=float, default=RolloutArgs.temporal_agg_k)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--default-speed", type=int, default=RolloutArgs.default_speed)
    p.add_argument("--no-ema-smoothing", dest="use_ema_smoothing", action="store_false")
    p.add_argument("--ema-alpha", type=float, default=RolloutArgs.ema_alpha)
    p.add_argument("--no-save-preview", dest="save_preview", action="store_false")
    p.add_argument("--preview-path", type=str, default=RolloutArgs.preview_path)
    p.add_argument("--preview-every-n-frames", type=int,
                   default=RolloutArgs.preview_every_n_frames)
    p.add_argument("--debug-every", type=int, default=RolloutArgs.debug_every)
    ns = p.parse_args()
    return RolloutArgs(**vars(ns))""")
    bullets(story, [
        "<b>Mutually-exclusive group</b> <code>--temporal-agg / --no-temporal-agg</code>: "
        "either turn it on or off, not both. <code>set_defaults</code> sets the default "
        "to whatever <code>RolloutArgs.temporal_agg</code> says (True).",
        "<code>RolloutArgs(**vars(ns))</code>: argparse Namespace &rarr; dict &rarr; "
        "dataclass kwargs. Field names line up exactly with the dest names.",
    ])

    # =====================================================================
    # rollout/sensapex_env.py
    # =====================================================================
    h1(story, "rollout/sensapex_env.py")
    h2(story, "Purpose")
    body(story, "Thin ROS2 client owned by the rollout. Subscribes to one camera "
                "topic and two Sensapex stage live topics, publishes two Sensapex "
                "absolute target topics. Hides ROS plumbing behind a synchronous "
                "<code>get_observation()</code> + <code>step_absolute(action_8d)</code> "
                "interface.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>SensapexObs</code>: <code>image_rgb (H, W, 3) uint8</code>, "
        "<code>state (8,) float32</code> in centered Sensapex counts.",
        "Subscribed topics: <code>/camera/image/compressed</code> (CompressedImage), "
        "<code>/ump/live</code> and <code>/ump2/live</code> (Int32MultiArray, length &gt;=4).",
        "Published topics: <code>/ump/target</code> and <code>/ump2/target</code> "
        "(Int32MultiArray, length 5: [x, y, z, d, speed]).",
    ])

    code_block(story, "rollout/sensapex_env.py:27-35 - SensapexObs + JPEG decode", """\
@dataclass
class SensapexObs:
    image_rgb: np.ndarray
    state: np.ndarray


def _decode_compressed_jpeg_to_rgb(msg: CompressedImage) -> np.ndarray:
    pil = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    return np.array(pil, dtype=np.uint8)""")
    bullets(story, [
        "<b>SensapexObs</b> is a frozen-shape dataclass — exactly the two arrays the "
        "policy needs. No timestamp, no header, nothing extra to drop accidentally.",
        "<b>JPEG via PIL</b> avoids a hard OpenCV dependency. "
        "<code>BytesIO(bytes(msg.data))</code> wraps the ROS message bytes so PIL "
        "can read them as a file-like object.",
    ])

    code_block(story, "rollout/sensapex_env.py:38-71 - _SensapexROSNode.__init__", """\
class _SensapexROSNode(Node):
    def __init__(self, *, save_preview=True, preview_path="microact_live.png",
                 preview_every_n_frames=5):
        super().__init__("microact_sensapex_bridge")

        self.sub_img = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self._on_img, 10)
        self.sub_ump1_live = self.create_subscription(
            Int32MultiArray, "/ump/live", self._on_ump1_live, 10)
        self.sub_ump2_live = self.create_subscription(
            Int32MultiArray, "/ump2/live", self._on_ump2_live, 10)

        self.pub_ump1_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.pub_ump2_target = self.create_publisher(Int32MultiArray, "/ump2/target", 10)

        self._lock = threading.Lock()
        self._latest_image_rgb = None
        self._latest_ump1 = None
        self._latest_ump2 = None

        self._save_preview = bool(save_preview)
        self._preview_path = str(preview_path)
        self._preview_every_n_frames = max(1, int(preview_every_n_frames))
        self._frame_counter = 0""")
    bullets(story, [
        "<b>Queue size 10</b>: ROS QoS depth. With 5 Hz control and 5 Hz publish, "
        "this is two seconds of buffer — plenty for transient hiccups.",
        "<b>Three latest-message slots</b> protected by one lock. The ROS executor "
        "thread writes; <code>get_latest()</code> reads atomic copies.",
    ])

    code_block(story, "rollout/sensapex_env.py:73-108 - subscriber callbacks", """\
def _on_img(self, msg: CompressedImage) -> None:
    try:
        rgb = _decode_compressed_jpeg_to_rgb(msg)
    except Exception as e:
        self.get_logger().warn(f"Image decode failed: {e}")
        return

    with self._lock:
        self._latest_image_rgb = rgb

    if self._save_preview:
        self._frame_counter += 1
        if self._frame_counter % self._preview_every_n_frames == 0:
            try:
                PILImage.fromarray(rgb).save(self._preview_path)
            except Exception as e:
                self.get_logger().warn(f"Preview save failed: {e}")


def _on_ump1_live(self, msg: Int32MultiArray) -> None:
    if len(msg.data) < 4:
        return
    with self._lock:
        self._latest_ump1 = [int(v) for v in msg.data[:4]]


def _on_ump2_live(self, msg: Int32MultiArray) -> None:
    if len(msg.data) < 4:
        return
    with self._lock:
        self._latest_ump2 = [int(v) for v in msg.data[:4]]


def get_latest(self):
    with self._lock:
        img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
        ump1 = None if self._latest_ump1 is None else list(self._latest_ump1)
        ump2 = None if self._latest_ump2 is None else list(self._latest_ump2)
    return img, ump1, ump2""")
    bullets(story, [
        "<b>Lock scope is tiny</b>: only the assignment / read is locked, not the "
        "JPEG decode or the disk save — those happen outside the critical section.",
        "<b>get_latest() returns copies</b>: <code>image.copy()</code> and "
        "<code>list(ump1)</code> ensure the caller cannot mutate the live state.",
        "<b>Preview save</b> happens every Nth frame (default 5). At 30 Hz camera "
        "input that is one snapshot every ~167 ms, useful for monitoring an SSH'd "
        "rollout without streaming video.",
    ])

    code_block(story, "rollout/sensapex_env.py:110-128 - send_action_absolute", """\
def send_action_absolute(self, x1, y1, z1, d1, x2, y2, z2, d2, speed=100) -> None:
    ump1_msg = Int32MultiArray()
    ump1_msg.data = [int(x1), int(y1), int(z1), int(d1), int(speed)]
    self.pub_ump1_target.publish(ump1_msg)

    ump2_msg = Int32MultiArray()
    ump2_msg.data = [int(x2), int(y2), int(z2), int(d2), int(speed)]
    self.pub_ump2_target.publish(ump2_msg)""")
    bullets(story, [
        "<b>Two messages per tick</b>, one per stage. Each is "
        "<code>Int32MultiArray.data = [x, y, z, d, speed]</code> — exactly 5 ints.",
        "<b>speed is the 5th element</b>, not a separate field. The Sensapex driver "
        "interprets the trailing element as the move speed in raw units.",
    ])

    code_block(story, "rollout/sensapex_env.py:131-198 - SensapexEnv (synchronous wrapper)", """\
class SensapexEnv:
    def __init__(self, *, save_preview=True, preview_path="microact_live.png",
                 preview_every_n_frames=5, default_speed=100, wait_timeout_s=10.0):
        self.default_speed = int(default_speed)
        rclpy.init(args=None)
        self.node = _SensapexROSNode(...)
        self._executor_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self._executor_thread.start()
        self._wait_for_first_messages(timeout_s=wait_timeout_s)

    def _wait_for_first_messages(self, timeout_s=10.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            img, ump1, ump2 = self.node.get_latest()
            if img is not None and ump1 is not None and ump2 is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for /camera/image/compressed, /ump/live, /ump2/live")

    def get_observation(self) -> SensapexObs:
        img, ump1, ump2 = self.node.get_latest()
        if img is None or ump1 is None or ump2 is None:
            raise RuntimeError("Missing observation components (image/ump1/ump2).")
        x1, y1, z1, d1 = ump1
        x2, y2, z2, d2 = ump2
        state = np.array([x1, y1, z1, d1, x2, y2, z2, d2], dtype=np.float32)  # (8,)
        return SensapexObs(image_rgb=img, state=state)

    def step_absolute(self, action_8d: np.ndarray) -> None:
        action_8d = np.asarray(action_8d).reshape(-1)
        if action_8d.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action_8d.shape}")
        x1, y1, z1, d1, x2, y2, z2, d2 = action_8d
        self.node.send_action_absolute(x1, y1, z1, d1, x2, y2, z2, d2,
                                       speed=self.default_speed)

    def close(self) -> None:
        try: self.node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass""")
    bullets(story, [
        "<b>rclpy.spin in a thread</b>: the ROS executor needs its own thread so "
        "subscriber callbacks fire continuously while the main thread runs the "
        "policy. <code>daemon=True</code> means the thread exits when main does.",
        "<b>_wait_for_first_messages</b>: blocks construction until at least one "
        "image and both stage live messages have arrived. Without it, "
        "<code>get_observation()</code> would race the first messages.",
        "<b>state vector layout</b>: "
        "<code>[x1, y1, z1, d1, x2, y2, z2, d2]</code>. Stage 1 first, stage 2 next. "
        "Matches <code>config.CSV_STATE_COLS</code>.",
        "<b>step_absolute hard-checks shape (8,)</b>: catches accidental "
        "<code>(1, 8)</code> or <code>(8, 1)</code> from a missing squeeze.",
    ])

    # =====================================================================
    # rollout/main.py
    # =====================================================================
    h1(story, "rollout/main.py")
    h2(story, "Purpose")
    body(story, "Closed-loop rollout for the homogeneous MicroACT policy. Loads the "
                "checkpoint, opens a SensapexEnv, and runs the per-tick "
                "<code>obs &rarr; policy &rarr; clamp &rarr; step-limit &rarr; EMA "
                "&rarr; publish</code> chain at <code>control_hz</code> until "
                "<code>max_timesteps</code> or E-STOP.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Per tick: <code>state (8,)</code> + <code>image (H,W,3)</code> &rarr; "
        "<code>policy.inference</code> &rarr; <code>chunk (100, 8)</code> &rarr; "
        "single 8-D action via temporal agg or open-loop pick.",
        "Safety: <code>clamp_action_8d (8,) &rarr; (8,)</code>; "
        "<code>limit_step(prev (8,), tgt (8,)) &rarr; (8,)</code>.",
        "Output: <code>SensapexEnv.step_absolute(cmd_8d)</code> publishes two "
        "Int32MultiArrays of length 5 (4 axes + speed) per tick.",
    ])

    code_block(story, "rollout/main.py:25-37 - module bootstrap", """\
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config as C
from model.act_policy import build_policy
from utils import load_checkpoint

try:
    from .rollout import RolloutArgs, clamp, parse_args, start_estop_listener
except ImportError:  # pragma: no cover - direct script execution fallback
    from rollout.rollout import RolloutArgs, clamp, parse_args, start_estop_listener""")
    bullets(story, [
        "<b>Path bootstrap</b> lets <code>python rollout/main.py</code> work the "
        "same as <code>python -m rollout.main</code>. The first line resolves the "
        "repo root regardless of cwd.",
        "<b>Try/except import</b> handles both relative (<code>-m</code>) and "
        "script (<code>python rollout/main.py</code>) execution modes.",
    ])

    code_block(story, "rollout/main.py:40-61 - safety limits", """\
# === Safety limits ===
# Units are centered Sensapex counts, matching /ump/live and /ump2/live.

X1_MIN, X1_MAX = 4600, 5700
Y1_MIN, Y1_MAX = 4900, 5500
Z1_MIN, Z1_MAX = 8750, 8250    # inverted: clamp() handles either order
D1_MIN, D1_MAX = 5900, 6100
X2_MIN, X2_MAX = 4600, 5700
Y2_MIN, Y2_MAX = 4900, 5500
Z2_MIN, Z2_MAX = 8750, 8250
D2_MIN, D2_MAX = 5900, 6100

MAX_DX1 = 250.0
MAX_DY1 = 250.0
MAX_DZ1 = 250.0
MAX_DD1 = 250.0
MAX_DX2 = 250.0
MAX_DY2 = 250.0
MAX_DZ2 = 250.0
MAX_DD2 = 250.0""")
    bullets(story, [
        "<b>Workspace box per axis</b>, in raw centered Sensapex counts. Defines a "
        "hard cage the rollout cannot drive outside. Edit before running on a "
        "different rig.",
        "<b>Z bounds are intentionally inverted</b> (8750 &gt; 8250) — Sensapex Z "
        "decreases as the probe goes down. <code>clamp()</code> handles either "
        "order so the constants stay readable.",
        "<b>MAX_DX1 = ... = 250.0</b>: per-tick step caps. At control_hz=5 this is "
        "1250 counts/sec per axis — slow enough that a misbehaving policy can be "
        "interrupted before it crashes the probe.",
    ])

    code_block(story, "rollout/main.py:64-88 - stats recovery from checkpoint", """\
def _stats_from_checkpoint(checkpoint: Path) -> dict:
    \"\"\"Recover normalization stats from policy buffers if dataset_stats.pkl is absent.\"\"\"
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["policy"]
    return {
        "qpos_mean": state["qpos_mean"].cpu().numpy(),
        "qpos_std": state["qpos_std"].cpu().numpy(),
        "action_mean": state["action_mean"].cpu().numpy(),
        "action_std": state["action_std"].cpu().numpy(),
        "image_mean": state["image_mean"].view(3).cpu().numpy(),
        "image_std": state["image_std"].view(3).cpu().numpy(),
    }


def _load_stats(stats_path: Path, checkpoint: Path) -> dict:
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            return pickle.load(f)
    print(f"[warn] stats file not found at {stats_path}; using checkpoint buffers")
    return _stats_from_checkpoint(checkpoint)""")
    bullets(story, [
        "<b>Self-contained checkpoints</b>: ACTPolicy registers the stats as "
        "buffers, so they live in <code>state_dict()</code> as "
        "<code>qpos_mean (8,), qpos_std (8,), action_mean (8,), action_std (8,), "
        "image_mean (3,1,1), image_std (3,1,1)</code>.",
        "<b>image_mean.view(3)</b>: the buffer was stored as <code>(3, 1, 1)</code> "
        "for broadcast convenience; the stats dict expects 1-D <code>(3,)</code>.",
        "<b>Fallback warning</b>: only fires if the standalone "
        "<code>dataset_stats.pkl</code> is missing. The recovered stats are "
        "byte-equivalent to the original.",
    ])

    code_block(story, "rollout/main.py:91-115 - load_microact_policy", """\
def load_microact_policy(args: RolloutArgs):
    checkpoint = _resolve_repo_path(args.checkpoint)
    stats_path = _resolve_repo_path(args.stats_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    stats = _load_stats(stats_path, checkpoint)
    policy = build_policy(
        stats=stats,
        pretrained_backbone=args.pretrained_backbone,
        backbone_name=args.backbone,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    epoch = load_checkpoint(checkpoint, policy, map_location=device)
    policy.eval()
    return policy""")
    bullets(story, [
        "<b>Build then load</b>: <code>build_policy</code> creates the architecture "
        "with the right backbone, then <code>load_checkpoint</code> overlays the "
        "trained weights. Without the build, the architecture would default to "
        "ResNet18 and refuse to load DINOv2 keys.",
        "<b>policy.eval()</b> freezes BatchNorm / Dropout. The frozen DINOv2 "
        "/ Cellpose backbones would already be eval-only because of their custom "
        "<code>train()</code> overrides, but <code>policy.eval()</code> ensures the "
        "transformer's own dropout layers are off too.",
    ])

    code_block(story, "rollout/main.py:118-145 - clamp_action_8d + limit_step", """\
def clamp_action_8d(action_8d: np.ndarray) -> np.ndarray:
    \"\"\"Clamp absolute action [x1,y1,z1,d1,x2,y2,z2,d2] to the safe box.\"\"\"
    a = np.asarray(action_8d, dtype=np.float32).reshape(8,)
    return np.array(
        [
            clamp(a[0], X1_MIN, X1_MAX),
            clamp(a[1], Y1_MIN, Y1_MAX),
            clamp(a[2], Z1_MIN, Z1_MAX),
            clamp(a[3], D1_MIN, D1_MAX),
            clamp(a[4], X2_MIN, X2_MAX),
            clamp(a[5], Y2_MIN, Y2_MAX),
            clamp(a[6], Z2_MIN, Z2_MAX),
            clamp(a[7], D2_MIN, D2_MAX),
        ],
        dtype=np.float32,
    )


def limit_step(prev_state_8d: np.ndarray, target_action_8d: np.ndarray) -> np.ndarray:
    \"\"\"Cap each axis' per-tick movement so far targets ramp in safely.\"\"\"
    prev = np.asarray(prev_state_8d, dtype=np.float32).reshape(8,)
    tgt = np.asarray(target_action_8d, dtype=np.float32).reshape(8,)
    caps = (MAX_DX1, MAX_DY1, MAX_DZ1, MAX_DD1, MAX_DX2, MAX_DY2, MAX_DZ2, MAX_DD2)

    out = np.empty(8, dtype=np.float32)
    for i, cap in enumerate(caps):
        out[i] = prev[i] + clamp(tgt[i] - prev[i], -cap, cap)
    return out""")
    bullets(story, [
        "<b>clamp_action_8d</b>: per-axis absolute clamp to the workspace box. "
        "Shape stays <code>(8,)</code>.",
        "<b>limit_step</b>: per-axis delta clamp. <code>tgt[i] - prev[i]</code> is "
        "the requested movement; clamping it to <code>[-cap, +cap]</code> caps the "
        "per-tick speed. Adding <code>prev[i]</code> back yields the new absolute "
        "target.",
        "<b>Order matters</b>: <code>clamp_action_8d</code> first to keep targets "
        "inside the box, then <code>limit_step</code> to keep approach speed safe. "
        "Reversing the order would let a far-away clamped target still trigger a "
        "huge first step.",
    ])

    code_block(story, "rollout/main.py:155-178 - chunk validation + temporal aggregation", """\
def _validate_action_chunk(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != C.ACTION_DIM:
        raise RuntimeError(f"Expected action chunk shape (T,{C.ACTION_DIM}), got {chunk.shape}")
    return chunk


def _aggregate_temporal_action(chunk_history, t: int, k: float) -> np.ndarray:
    actions = []
    ages = []
    for start_t, chunk in chunk_history:
        age = t - start_t
        if 0 <= age < chunk.shape[0]:
            actions.append(chunk[age])     # (8,)
            ages.append(age)

    if not actions:
        raise RuntimeError("Temporal aggregation has no valid action for this tick")

    weights = np.exp(-float(k) * np.asarray(ages, dtype=np.float32))   # (n,)
    weights = weights / weights.sum()
    return (np.stack(actions, axis=0) * weights[:, None]).sum(axis=0).astype(np.float32)
    # stack: (n, 8); weights[:, None]: (n, 1); broadcast then sum axis=0 -> (8,)""")
    bullets(story, [
        "<b>chunk_history</b> is a list of <code>(start_t, chunk_(100,8))</code> "
        "tuples accumulated over the rollout. Each entry's chunk[age] is the action "
        "that chunk predicted for the current tick.",
        "<b>weights = exp(-k * ages)</b> with default k=0.01: an age=100 prediction "
        "gets weight <code>exp(-1.0) &asymp; 0.37</code>, age=0 gets 1.0. Tiny k "
        "means even old predictions still vote.",
        "<b>weighted average</b>: <code>stack (n, 8) * weights[:, None] (n, 1) = "
        "(n, 8) via broadcast; .sum(axis=0)</code> &rarr; <code>(8,)</code> — the "
        "smoothed action for this tick.",
    ])

    code_block(story, "rollout/main.py:189-296 - main loop", """\
def main(args: RolloutArgs) -> None:
    if args.open_loop_horizon < 1: raise ValueError("--open-loop-horizon must be >= 1")
    if args.control_hz <= 0: raise ValueError("--control-hz must be > 0")
    if args.temporal_agg_k < 0: raise ValueError("--temporal-agg-k must be >= 0")
    if not (0.0 < args.ema_alpha <= 1.0): raise ValueError("--ema-alpha must be in (0, 1]")

    policy = load_microact_policy(args)
    SensapexEnv = _get_env_cls()
    env = SensapexEnv(save_preview=args.save_preview, preview_path=args.preview_path,
                      preview_every_n_frames=args.preview_every_n_frames,
                      default_speed=args.default_speed)

    stop_flag = start_estop_listener()
    period = 1.0 / float(args.control_hz)         # seconds per tick
    actions_completed_in_chunk = 0
    max_actions_from_current_chunk = 0
    pred_action_chunk = None
    chunk_history = []
    ema_action = None

    try:
        for t in range(int(args.max_timesteps)):
            start_time = time.time()

            if stop_flag["stop"]:
                obs = env.get_observation()
                hold = obs.state.astype(np.float32).copy()
                if not args.dry_run:
                    env.step_absolute(hold)        # publish current pose to lock
                break

            obs = env.get_observation()             # SensapexObs
            img = obs.image_rgb                     # (H, W, 3) uint8
            state = obs.state.astype(np.float32)    # (8,)

            if args.temporal_agg:
                pred_action_chunk = _validate_action_chunk(policy.inference(img, state))
                # pred_action_chunk: (100, 8)
                chunk_history.append((t, pred_action_chunk))
                chunk_history = [(start_t, chunk) for start_t, chunk in chunk_history
                                 if 0 <= t - start_t < chunk.shape[0]]
                action = _aggregate_temporal_action(chunk_history, t, args.temporal_agg_k)
                # action: (8,)
            else:
                need_new_chunk = (
                    pred_action_chunk is None
                    or actions_completed_in_chunk >= max_actions_from_current_chunk
                )
                if need_new_chunk:
                    pred_action_chunk = _validate_action_chunk(policy.inference(img, state))
                    actions_completed_in_chunk = 0
                    max_actions_from_current_chunk = min(
                        int(args.open_loop_horizon), int(pred_action_chunk.shape[0]))
                action = pred_action_chunk[actions_completed_in_chunk]
                actions_completed_in_chunk += 1

            action = clamp_action_8d(action)         # (8,)
            action = limit_step(state, action)       # (8,)

            if args.use_ema_smoothing:
                if ema_action is None:
                    ema_action = action.copy()
                else:
                    ema_action = args.ema_alpha * action + (1.0 - args.ema_alpha) * ema_action
                cmd = ema_action.astype(np.float32)
            else:
                cmd = action

            if not args.dry_run:
                env.step_absolute(cmd)

            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("Stopped early (Ctrl+C).")
    finally:
        env.close()""")
    bullets(story, [
        "<b>E-STOP path</b>: when <code>q + Enter</code> is typed, the next tick "
        "reads current state and publishes it as the absolute target. The motor "
        "stops where it is, the loop exits, env closes.",
        "<b>Temporal-agg path</b>: re-runs inference every tick (slow but smooth). "
        "<code>chunk_history</code> is filtered down to only entries with a valid "
        "age &lt; 100; the aggregated action is the exp-weighted mean.",
        "<b>Open-loop path</b>: re-runs inference only when the previous "
        "chunk's first <code>open_loop_horizon=8</code> actions are exhausted. "
        "Faster but jerkier on long horizons.",
        "<b>EMA smoothing</b> is the final step: "
        "<code>ema = alpha * new + (1 - alpha) * ema</code>. With alpha=0.35 the "
        "command is 35% new + 65% old — visible damping without lag.",
        "<b>Tick budget</b>: <code>period = 1/control_hz = 0.2 s</code>. The loop "
        "sleeps the remainder so each tick lasts the full period regardless of how "
        "long inference took.",
    ])

    # =====================================================================
    # rollout/vla_main.py
    # =====================================================================
    h1(story, "rollout/vla_main.py (NEW)")
    h2(story, "Purpose")
    body(story, "Closed-loop rollout for the VLA policy. Same loop structure as "
                "<code>rollout/main.py</code>, but the policy is robot-agnostic: a "
                "robot-specific <b>adapter</b> (under <code>rollout/adapters/</code>) "
                "owns observation, safety, and publishing.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>policy.inference(image, state, instruction, robot_id, lab_id, "
        "embodiment, action_type, task_family, state_dim, action_dim)</code> &rarr; "
        "<code>(100, action_dim)</code> in raw absolute units.",
        "Adapter contract: <code>get_observation()</code>, "
        "<code>safe_command(state, action)</code>, <code>publish(cmd)</code>, "
        "<code>hold_current()</code>, <code>close()</code>; plus class attributes "
        "<code>robot_id, lab_id, embodiment, action_type, task_family, state_dim, "
        "action_dim</code>.",
    ])

    code_block(story, "rollout/vla_main.py:26-59 - parse_args (VLA-specific)", """\
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(...)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints_vla/vla_policy_best.pt"))
    p.add_argument("--adapter", choices=("sensapex_dual",), default="sensapex_dual")
    p.add_argument("--instruction", type=str,
                   default="perform the cell manipulation task")
    p.add_argument("--backbone", type=str, default=None,
                   help="Defaults to checkpoint config.")
    p.add_argument("--language-backend", choices=("hf", "simple"), default=None, ...)
    p.add_argument("--text-model", type=str, default=None, ...)
    p.add_argument("--device", type=str, default=C.DEVICE)
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--max-timesteps", type=int, default=600)
    p.add_argument("--open-loop-horizon", type=int, default=C.OPEN_LOOP_HORIZON)
    p.add_argument("--control-hz", type=float, default=C.CONTROL_HZ)
    temporal_group = p.add_mutually_exclusive_group()
    temporal_group.add_argument("--temporal-agg", dest="temporal_agg", action="store_true")
    temporal_group.add_argument("--no-temporal-agg", dest="temporal_agg", action="store_false")
    p.set_defaults(temporal_agg=C.TEMPORAL_AGG)
    p.add_argument("--temporal-agg-k", type=float, default=C.TEMPORAL_AGG_K)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--default-speed", type=int, default=100)
    p.add_argument("--no-ema-smoothing", dest="use_ema_smoothing", action="store_false")
    p.add_argument("--ema-alpha", type=float, default=0.35)
    p.add_argument("--no-save-preview", dest="save_preview", action="store_false")
    p.add_argument("--preview-path", type=str, default="microact_vla_live.png")
    p.add_argument("--preview-every-n-frames", type=int, default=5)
    p.add_argument("--debug-every", type=int, default=10)
    p.add_argument("--lab-id", type=str, default=None)        # override metadata
    p.add_argument("--robot-id", type=str, default=None)
    p.add_argument("--embodiment", type=str, default=None)
    p.add_argument("--action-type", type=str, default=None)
    p.add_argument("--task-family", type=str, default=None)
    return p.parse_args()""")
    bullets(story, [
        "<b>--adapter</b> is currently a single-choice flag (sensapex_dual). The "
        "design intentionally hardcodes the registration so adding a new robot "
        "means adding both the adapter file and a new choice here.",
        "<b>--backbone / --language-backend / --text-model</b> default to None, "
        "meaning \"use what the checkpoint says\". Pass an explicit value only when "
        "you intentionally want to override.",
        "<b>--lab-id / --robot-id / etc.</b> let the user override the adapter's "
        "default metadata at the CLI — useful when running a new lab's rig with the "
        "same physical robot but different lab_id.",
    ])

    code_block(story, "rollout/vla_main.py:67-77 - _get_adapter (registry)", """\
def _get_adapter(args):
    if args.adapter == "sensapex_dual":
        from rollout.adapters.sensapex_dual import SensapexDualAdapter
        return SensapexDualAdapter(
            default_speed=args.default_speed,
            save_preview=args.save_preview,
            preview_path=args.preview_path,
            preview_every_n_frames=args.preview_every_n_frames,
        )
    raise ValueError(f"Unsupported adapter: {args.adapter}")""")
    bullets(story, [
        "<b>Lazy import inside the branch</b>: importing "
        "<code>SensapexDualAdapter</code> pulls in <code>rclpy</code> via "
        "<code>SensapexEnv</code>. Keeping the import inside the if-branch lets the "
        "module be imported in environments without ROS for unit tests / docs.",
    ])

    code_block(story, "rollout/vla_main.py:106-144 - load_policy", """\
def load_policy(args):
    checkpoint = _resolve_repo_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(...)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    backbone = args.backbone or ckpt_config.get("backbone", C.DEFAULT_BACKBONE)
    language_backend = args.language_backend or ckpt_config.get("language_backend",
                                                                  C.LANGUAGE_BACKEND)
    text_model = args.text_model or ckpt_config.get("text_model", C.DEFAULT_TEXT_MODEL)
    pretrained_backbone = args.pretrained_backbone or bool(
        ckpt_config.get("pretrained_backbone", False))
    freeze_backbone = (
        not args.unfreeze_backbone if args.unfreeze_backbone
        else bool(ckpt_config.get("freeze_backbone", True))
    )
    policy = build_vla_policy(
        stats=ckpt["stats"], vocabs=ckpt["vocabs"],
        pretrained_backbone=pretrained_backbone, backbone_name=backbone,
        freeze_backbone=freeze_backbone, language_backend=language_backend,
        text_model_name=text_model,
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy""")
    bullets(story, [
        "<b>weights_only=False</b>: required because the checkpoint contains "
        "Python dicts (stats, vocabs, config) — not just tensors. Safe here because "
        "the file came from <code>save_vla_checkpoint</code>.",
        "<b>Override priority</b>: CLI flag &gt; checkpoint config &gt; module "
        "default. The <code>or</code> chain implements this naturally because all "
        "three layers return falsy (None / missing) when not set.",
        "<b>Build then load_state_dict</b>: same pattern as ACT — construct the "
        "exact architecture first (so embedding tables and projection sizes match) "
        "then overlay the trained weights.",
    ])

    code_block(story, "rollout/vla_main.py:147-215 - main loop (temporal-agg branch)", """\
def main() -> None:
    args = parse_args()
    # Validation of --open-loop-horizon, --control-hz, --temporal-agg-k, --ema-alpha ...

    policy = load_policy(args)
    adapter = _get_adapter(args)

    robot_id = args.robot_id or adapter.robot_id
    lab_id = args.lab_id or adapter.lab_id
    embodiment = args.embodiment or adapter.embodiment
    action_type = args.action_type or adapter.action_type
    task_family = args.task_family or adapter.task_family

    stop_flag = start_estop_listener()
    period = 1.0 / float(args.control_hz)
    pred_action_chunk = None
    actions_completed_in_chunk = 0
    max_actions_from_current_chunk = 0
    chunk_history = []
    ema_action = None

    try:
        for t in range(int(args.max_timesteps)):
            start_time = time.time()

            if stop_flag["stop"]:
                if not args.dry_run:
                    adapter.hold_current()
                break

            obs = adapter.get_observation()
            img = obs.image_rgb                          # (H, W, 3) uint8
            state = obs.state.astype(np.float32)          # (state_dim,)

            if args.temporal_agg:
                pred_action_chunk = _validate_action_chunk(
                    policy.inference(
                        img, state, args.instruction,
                        robot_id=robot_id, lab_id=lab_id, embodiment=embodiment,
                        action_type=action_type, task_family=task_family,
                        state_dim=adapter.state_dim, action_dim=adapter.action_dim,
                    ),
                    adapter.action_dim,
                )
                # pred_action_chunk: (100, action_dim)
                chunk_history.append((t, pred_action_chunk))
                chunk_history = [(start_t, chunk) for start_t, chunk in chunk_history
                                 if 0 <= t - start_t < chunk.shape[0]]
                action = _aggregate_temporal_action(chunk_history, t, args.temporal_agg_k)
            else:
                need_new_chunk = (...)
                if need_new_chunk:
                    pred_action_chunk = _validate_action_chunk(policy.inference(...),
                                                               adapter.action_dim)
                    actions_completed_in_chunk = 0
                    max_actions_from_current_chunk = min(int(args.open_loop_horizon),
                                                          int(pred_action_chunk.shape[0]))
                action = pred_action_chunk[actions_completed_in_chunk]
                actions_completed_in_chunk += 1

            safe = adapter.safe_command(state, action)   # (action_dim,) clamp+limit_step
            if args.use_ema_smoothing:
                if ema_action is None: ema_action = safe.copy()
                else: ema_action = args.ema_alpha * safe + (1.0 - args.ema_alpha) * ema_action
                cmd = ema_action.astype(np.float32)
            else:
                cmd = safe

            if not args.dry_run:
                adapter.publish(cmd)
            ...
    finally:
        adapter.close()""")
    bullets(story, [
        "<b>Adapter owns the safety / publish chain</b>. The VLA loop calls "
        "<code>adapter.safe_command(state, action)</code> instead of the standalone "
        "<code>clamp_action_8d / limit_step</code> functions — the adapter is "
        "free to use any robot-specific logic.",
        "<b>Adapter publishes plain numpy arrays</b> of shape "
        "<code>(adapter.action_dim,)</code>. The adapter knows how to translate "
        "those into ROS messages for its specific robot.",
        "<b>chunk shape is (100, action_dim)</b> not (100, 16) — the policy already "
        "sliced down inside <code>VLAPolicy.inference</code>, so the rollout loop "
        "never sees padded dims.",
    ])

    # =====================================================================
    # rollout/adapters/sensapex_dual.py
    # =====================================================================
    h1(story, "rollout/adapters/sensapex_dual.py (NEW)")
    h2(story, "Purpose")
    body(story, "Adapter that lets the VLA rollout drive the dual-Sensapex rig. "
                "Owns a <code>SensapexEnv</code>, exposes the robot's metadata IDs "
                "as class attributes, and reuses ACT's <code>clamp_action_8d</code> + "
                "<code>limit_step</code> for safety.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Class attributes (read by <code>vla_main.main</code>): "
        "<code>robot_id = \"sensapex_dual_ump4\", lab_id = \"local_lab\", "
        "embodiment = \"dual_manipulator\", action_type = \"absolute_position\", "
        "task_family = \"cell_manipulation\", state_dim = 8, action_dim = 8</code>.",
        "<code>get_observation()</code> &rarr; <code>SensapexObs(image_rgb (H,W,3), "
        "state (8,))</code>.",
        "<code>safe_command(state_8d (8,), action_8d (8,))</code> &rarr; <code>(8,)</code>.",
    ])

    code_block(story, "rollout/adapters/sensapex_dual.py:1-52 - SensapexDualAdapter", """\
from config import vla_config as C
from rollout.main import clamp_action_8d, limit_step
from rollout.sensapex_env import SensapexEnv


class SensapexDualAdapter:
    robot_id = C.DEFAULT_ROBOT_ID         # "sensapex_dual_ump4"
    lab_id = C.DEFAULT_LAB_ID             # "local_lab"
    embodiment = C.DEFAULT_EMBODIMENT     # "dual_manipulator"
    action_type = C.DEFAULT_ACTION_TYPE   # "absolute_position"
    task_family = C.DEFAULT_TASK_FAMILY   # "cell_manipulation"
    state_dim = 8
    action_dim = 8

    def __init__(self, *, default_speed=100, save_preview=True,
                 preview_path="microact_vla_live.png", preview_every_n_frames=5):
        self.env = SensapexEnv(
            default_speed=default_speed, save_preview=save_preview,
            preview_path=preview_path, preview_every_n_frames=preview_every_n_frames,
        )

    def get_observation(self):
        return self.env.get_observation()

    def safe_command(self, state_8d: np.ndarray, action_8d: np.ndarray) -> np.ndarray:
        action = clamp_action_8d(action_8d)         # workspace box clamp
        return limit_step(state_8d, action)         # per-tick step cap

    def publish(self, command_8d: np.ndarray) -> None:
        self.env.step_absolute(command_8d)

    def hold_current(self) -> None:
        obs = self.get_observation()
        self.publish(obs.state.astype(np.float32).copy())

    def close(self) -> None:
        self.env.close()""")
    bullets(story, [
        "<b>Class attributes (no <code>self</code>)</b>: <code>vla_main.main</code> "
        "reads them via <code>adapter.robot_id</code> etc. before the policy is even "
        "called. Defining them on the class (not in <code>__init__</code>) keeps "
        "construction lighter and lets the CLI <code>--robot-id</code> override "
        "fall through cleanly.",
        "<b>Reuses MicroACT's safety chain</b>: <code>clamp_action_8d</code> and "
        "<code>limit_step</code> live in <code>rollout/main.py</code>. The same "
        "workspace box and per-tick caps apply whether ACT or VLA is driving.",
        "<b>hold_current</b> is the E-STOP behavior — read current pose, publish it "
        "as the target, motor stops where it is.",
    ])
