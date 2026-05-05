from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # ----- data/dataset.py -----
    h1(story, "data/dataset.py")
    h2(story, "Purpose")
    body(story, "Loads the per-trial CSV and frame files for the original "
                "(homogeneous) MicroACT setup, computes normalization statistics, and "
                "emits per-timestep training samples with action-chunking and image "
                "preprocessing.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<b>TrialData</b>: states <b>(T, 8)</b>, actions <b>(T, 8)</b>, "
        "image_paths length T, length T.",
        "Per-sample emit: image <b>(1, 3, 240, 320)</b>, qpos <b>(8,)</b>, "
        "action <b>(100, 8)</b>, is_pad <b>(100,)</b>. After DataLoader collation: "
        "image <b>(B, 1, 3, 240, 320)</b>, qpos <b>(B, 8)</b>, "
        "action <b>(B, 100, 8)</b>, is_pad <b>(B, 100)</b>.",
    ])
    code_block(story, "data/dataset.py:31-36 - TrialData", """\
class TrialData(NamedTuple):
    trial_id: int
    states: np.ndarray           # (T, STATE_DIM)
    actions: np.ndarray          # (T, ACTION_DIM)
    image_paths: List[str]       # length T; '' means no path recorded
    length: int""")
    bullets(story, [
        "<b>NamedTuple</b> means TrialData is immutable — you cannot accidentally mutate "
        "<code>tr.states</code> after construction. Cleaner than a plain dict for a "
        "fixed schema.",
    ])
    code_block(story, "data/dataset.py:43-68 - discover_trials and load_trial", """\
def discover_trials(logs_dir: Path = C.LOGS_DIR) -> List[Path]:
    files = sorted(
        logs_dir.glob("trial_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No trial_*.csv found under {logs_dir}")
    return files


def load_trial(csv_path: Path) -> TrialData:
    df = pd.read_csv(csv_path)
    trial_id = int(csv_path.stem.split("_")[-1])

    missing = [c for c in (*C.CSV_STATE_COLS, *C.CSV_ACTION_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    states = df[list(C.CSV_STATE_COLS)].to_numpy(dtype=np.float32)
    actions = df[list(C.CSV_ACTION_COLS)].to_numpy(dtype=np.float32)
    raw_paths = (
        df[C.CSV_IMAGE_COL].fillna("").astype(str).tolist()
        if C.CSV_IMAGE_COL in df.columns
        else [""] * len(df)
    )
    return TrialData(trial_id, states, actions, raw_paths, length=len(df))""")
    bullets(story, [
        "Sorting by the integer trail number (parsed via <code>p.stem.split(\"_\")[-1]</code>) "
        "guarantees <code>trial_2.csv</code> precedes <code>trial_10.csv</code>, unlike "
        "lexicographic sort.",
        "Missing CSV columns are a <i>hard</i> error — the loader will not silently "
        "produce shape-mismatched arrays.",
        "<code>df[list(C.CSV_STATE_COLS)].to_numpy(dtype=np.float32)</code> selects "
        "the 8 columns in the order declared in config and converts to a "
        "<code>(T, 8)</code> float32 array directly.",
    ])
    code_block(story, "data/dataset.py:75-98 - image path resolution and loading", """\
def _resolve_image_path(raw: str, trial_id: int, t: int) -> Optional[Path]:
    fallback = C.FRAMES_DIR / f"trial_{trial_id}" / f"frame_{t:06d}.png"

    raw = (raw or "").strip()
    if not raw:
        return fallback if fallback.exists() else None

    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (C.REPO_ROOT, C.DATASET_ROOT, C.FRAMES_DIR):
        q = base / p
        if q.exists():
            return q
    return fallback if fallback.exists() else None


def _load_image(path: Optional[Path], h: int, w: int) -> np.ndarray:
    if path is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)""")
    bullets(story, [
        "Three search strategies before falling back to the conventional "
        "<code>saved_frames/trial_N/frame_NNNNNN.png</code> layout: absolute path, "
        "relative-to-repo-root, relative-to-dataset, relative-to-frames.",
        "<code>Image.open(...).convert(\"RGB\").resize((w, h), BILINEAR)</code> always "
        "produces a uint8 RGB array of shape <b>(h, w, 3)</b>. Note PIL's <code>.resize</code> "
        "takes <code>(width, height)</code> while NumPy uses <code>(height, width)</code>.",
        "Missing image becomes <b>zeros (H, W, 3)</b> so the dataloader never crashes; "
        "a single warning is emitted and suppressed thereafter.",
    ])
    code_block(story, "data/dataset.py:105-120 - compute_norm_stats", """\
def compute_norm_stats(trials: List[TrialData]) -> dict:
    all_states = np.concatenate([t.states for t in trials], axis=0)
    all_actions = np.concatenate([t.actions for t in trials], axis=0)

    state_std = np.clip(all_states.std(0), 1e-2, None)
    action_std = np.clip(all_actions.std(0), 1e-2, None)

    return {
        "qpos_mean":   all_states.mean(0).astype(np.float32),
        "qpos_std":    state_std.astype(np.float32),
        "action_mean": all_actions.mean(0).astype(np.float32),
        "action_std":  action_std.astype(np.float32),
        "image_mean":  np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "image_std":   np.array([0.229, 0.224, 0.225], dtype=np.float32),
    }""")
    bullets(story, [
        "<code>np.concatenate([t.states for t in trials], axis=0)</code> stacks all "
        "per-trial <code>(T_i, 8)</code> arrays into one big <code>(sum_T, 8)</code> "
        "matrix; <code>.mean(0)</code> and <code>.std(0)</code> reduce across timesteps "
        "and produce per-dimension stats of shape <b>(8,)</b>.",
        "<code>np.clip(std, 1e-2, None)</code> protects against constant axes "
        "producing division-by-zero during normalization.",
        "<b>image_mean / image_std</b> are the canonical ImageNet statistics, broadcast "
        "across H, W later in <code>__getitem__</code>.",
    ])
    code_block(story, "data/dataset.py:127-146 - EpisodicDataset.__init__", """\
class EpisodicDataset(Dataset):
    def __init__(
        self,
        trials: List[TrialData],
        norm_stats: dict,
        chunk_size: int = C.CHUNK_SIZE,
        image_hw: tuple = (C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
    ):
        self.trials = trials
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.image_h, self.image_w = image_hw
        self.index = [
            (ti, t)
            for ti, tr in enumerate(trials)
            for t in range(tr.length)
        ]
        self._warned_missing_image = False""")
    bullets(story, [
        "<code>self.index</code> is a flat list of <code>(trial_index, timestep)</code> "
        "pairs. Total length equals <code>sum(t.length for t in trials)</code>; that "
        "becomes <code>len(dataset)</code>.",
        "Storing the index this way means every timestep is a valid sample: there is no "
        "stride or window — the chunk is built on the fly and zero-padded past the end.",
    ])
    code_block(story, "data/dataset.py:151-193 - __getitem__: chunking and normalization", """\
def __getitem__(self, i: int) -> dict:
    trial_idx, t = self.index[i]
    trial = self.trials[trial_idx]

    # ---- State at t ----
    qpos = trial.states[t]                                  # (8,)

    # ---- Future action chunk, zero-padded ----
    end = min(t + self.chunk_size, trial.length)
    avail = end - t
    action = np.zeros((self.chunk_size, C.ACTION_DIM), dtype=np.float32)  # (100, 8)
    action[:avail] = trial.actions[t:end]                    # real rows
    is_pad = np.zeros(self.chunk_size, dtype=bool)
    is_pad[avail:] = True                                    # padded rows

    # ---- Image at t ----
    raw = trial.image_paths[t] if t < len(trial.image_paths) else ""
    path = _resolve_image_path(raw, trial.trial_id, t)
    img = _load_image(path, self.image_h, self.image_w)      # (240, 320, 3) uint8

    # ---- Normalize ----
    img = img.astype(np.float32) / 255.0
    img = (img - self.norm_stats["image_mean"]) / self.norm_stats["image_std"]
    img = np.transpose(img, (2, 0, 1))                       # (3, 240, 320) - HWC -> CHW
    img = img[None]                                          # (1, 3, 240, 320) - add cam axis

    qpos_n = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
    action_n = (action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
    action_n[is_pad] = 0.0                                   # keep padded positions clean

    return {
        "image":  torch.from_numpy(img).float(),             # (1, 3, 240, 320)
        "qpos":   torch.from_numpy(qpos_n).float(),          # (8,)
        "action": torch.from_numpy(action_n).float(),        # (100, 8)
        "is_pad": torch.from_numpy(is_pad),                  # (100,)
    }""")
    bullets(story, [
        "<b>end = min(t+100, trial.length)</b> caps the chunk at the trial's tail. "
        "<b>avail = end - t</b> is the number of real rows available; "
        "<code>action[:avail] = trial.actions[t:end]</code> copies them and the rest "
        "stays zero.",
        "<b>is_pad[avail:] = True</b> flags the padded slots so the loss can ignore them.",
        "Image normalization order: cast to float, divide by 255 to get [0,1], subtract "
        "ImageNet mean, divide by ImageNet std, transpose HWC&rarr;CHW, add the camera "
        "axis with <code>[None]</code>. The final NumPy shape is <b>(1, 3, 240, 320)</b> "
        "before <code>torch.from_numpy</code>.",
        "<b>action_n[is_pad] = 0.0</b> overwrites the padded rows after normalization. "
        "Zero is the normalized mean (since action_n is in zero-mean units), so padded "
        "rows look like \"the average action\" but they will be masked out of the loss.",
    ])
    code_block(story, "data/dataset.py:200-217 - build_dataset", """\
def build_dataset(
    logs_dir: Path = C.LOGS_DIR,
    stats_path: Path = C.STATS_PATH,
    recompute_stats: bool = False,
) -> EpisodicDataset:
    csv_paths = discover_trials(logs_dir)
    trials = [load_trial(p) for p in csv_paths]

    if stats_path.exists() and not recompute_stats:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = compute_norm_stats(trials)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

    return EpisodicDataset(trials, stats)""")
    bullets(story, [
        "<code>recompute_stats=True</code> forces a fresh stat computation and rewrites "
        "the pickle. <code>train.py</code> always passes True so adding new trials does "
        "not silently keep stale normalization stats.",
        "<code>EpisodicDataset(trials, stats)</code> is what <code>train.py</code> "
        "wraps in a <code>DataLoader</code>.",
    ])

    # ----- data/vla_dataset.py -----
    h1(story, "data/vla_dataset.py (NEW)")
    h2(story, "Purpose")
    body(story, "Metadata-driven dataset for heterogeneous MicroVLA training. Each "
                "episode lives in its own directory with a <i>metadata.json</i>, a "
                "<i>trajectory.csv</i> and a <i>frames/&lt;cam&gt;/</i> subfolder. "
                "Different episodes can come from different robots, declare different "
                "<i>state_dim</i>/<i>action_dim</i>, and ship language instructions. "
                "Samples are padded to <i>MAX_STATE_DIM</i> / <i>MAX_ACTION_DIM</i> and "
                "shipped with masks, so single- and dual-arm episodes mix in the same "
                "batch.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<b>VocabBundle</b>: five Dict[str, int] (robot/lab/embodiment/action_type/"
        "task_family). Always contains <code>UNKNOWN_TOKEN -&gt; 0</code>.",
        "<b>VLAEpisode</b>: includes states <b>(T, ep.state_dim)</b>, "
        "actions <b>(T, ep.action_dim)</b>, plus all metadata fields.",
        "Per-sample emit: image <b>(1,3,240,320)</b>, qpos <b>(16,)</b>, "
        "state_mask <b>(16,)</b>, action <b>(100,16)</b>, action_mask <b>(16,)</b>, "
        "is_pad <b>(100,)</b>, instruction <i>str</i>, plus 5 long-tensor scalars for "
        "the metadata IDs and 2 for episode_index/timestep.",
    ])
    code_block(story, "data/vla_dataset.py:34-72 - VocabBundle and VLAEpisode", """\
@dataclass(frozen=True)
class VocabBundle:
    robot_ids: Dict[str, int]
    lab_ids: Dict[str, int]
    embodiment_ids: Dict[str, int]
    action_type_ids: Dict[str, int]
    task_family_ids: Dict[str, int]

    def as_dict(self) -> dict:
        return {
            "robot_ids": self.robot_ids,
            "lab_ids": self.lab_ids,
            "embodiment_ids": self.embodiment_ids,
            "action_type_ids": self.action_type_ids,
            "task_family_ids": self.task_family_ids,
        }


@dataclass
class VLAEpisode:
    episode_dir: Path
    episode_id: str
    lab_id: str
    robot_id: str
    embodiment: str
    action_type: str
    task_family: str
    instruction: str
    camera_names: List[str]
    state_cols: List[str]
    action_cols: List[str]
    image_col: str
    timestep_col: str
    state_dim: int
    action_dim: int
    states: np.ndarray
    actions: np.ndarray
    image_paths: List[str]
    length: int""")
    bullets(story, [
        "<b>VocabBundle</b> is frozen so it is hashable and can be safely shared between "
        "the dataset, the policy, and the checkpoint. <code>as_dict()</code> serializes "
        "the five mappings in a form pickle can write.",
        "<b>VLAEpisode</b> stores everything an episode needs to be sampled from: "
        "metadata, the per-timestep state/action arrays sized to the episode's own "
        "<i>state_dim</i>/<i>action_dim</i>, and the raw image paths.",
    ])
    code_block(story, "data/vla_dataset.py:108-165 - load_episode", """\
def load_episode(episode_dir: Path) -> VLAEpisode:
    meta = _read_metadata(episode_dir)
    episode_id = str(meta.get("episode_id", episode_dir.name))
    trajectory_name = str(meta.get("trajectory_file", "trajectory.csv"))
    csv_path = episode_dir / trajectory_name

    state_dim = int(meta.get("state_dim", len(meta.get("state_cols", [])) or 0))
    action_dim = int(meta.get("action_dim", len(meta.get("action_cols", [])) or 0))
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError(...)
    if state_dim > C.MAX_STATE_DIM:
        raise ValueError(...)

    state_cols = list(meta.get("state_cols") or _default_state_cols(state_dim))
    action_cols = list(meta.get("action_cols") or _default_action_cols(action_dim))

    df = pd.read_csv(csv_path)
    missing = [c for c in (*state_cols, *action_cols) if c not in df.columns]
    if missing:
        raise ValueError(...)

    return VLAEpisode(
        ...,
        state_dim=state_dim,
        action_dim=action_dim,
        states=df[state_cols].to_numpy(dtype=np.float32),    # (T, ep.state_dim)
        actions=df[action_cols].to_numpy(dtype=np.float32),  # (T, ep.action_dim)
        image_paths=raw_paths,
        length=len(df),
    )""")
    bullets(story, [
        "<code>state_dim</code> is taken from metadata if present, else inferred from "
        "<code>len(state_cols)</code>. The check <code>state_dim &gt; MAX_STATE_DIM</code> "
        "is a hard error so a 32-DOF dataset cannot silently truncate.",
        "If metadata omits <code>state_cols</code>, <code>_default_state_cols(dim)</code> "
        "fabricates names matching the original Sensapex schema for the first 8 dims, "
        "then <code>state_8 ... state_15</code> beyond that.",
        "<b>states</b> shape is the per-episode <code>(T, state_dim)</code> &mdash; not "
        "yet padded to 16. Padding happens in <code>VLADataset.__getitem__</code>.",
    ])
    code_block(story, "data/vla_dataset.py:168-187 - vocabulary builders", """\
def _make_vocab(values: Iterable[str]) -> Dict[str, int]:
    vocab = {C.UNKNOWN_TOKEN: 0}
    for value in sorted({str(v) for v in values}):
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def build_vocabs(episodes: List[VLAEpisode]) -> VocabBundle:
    return VocabBundle(
        robot_ids=_make_vocab(e.robot_id for e in episodes),
        lab_ids=_make_vocab(e.lab_id for e in episodes),
        embodiment_ids=_make_vocab(e.embodiment for e in episodes),
        action_type_ids=_make_vocab(e.action_type for e in episodes),
        task_family_ids=_make_vocab(e.task_family for e in episodes),
    )


def _lookup(vocab: Dict[str, int], value: str) -> int:
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))""")
    bullets(story, [
        "Every vocab starts with <b>{UNKNOWN_TOKEN: 0}</b>. Real values are inserted in "
        "sorted order so the same dataset always yields the same IDs across runs (which "
        "matters because checkpoint embedding rows are indexed by these IDs).",
        "<code>_lookup</code> falls back to ID 0 (the unknown slot) if a name is "
        "missing — used at inference when the rollout adapter declares a robot the "
        "model has never seen.",
    ])
    code_block(story, "data/vla_dataset.py:190-246 - compute_vla_norm_stats", """\
def compute_vla_norm_stats(episodes: List[VLAEpisode]) -> dict:
    by_robot = {}
    for robot_id in sorted({e.robot_id for e in episodes}):
        robot_eps = [e for e in episodes if e.robot_id == robot_id]
        state_rows = []
        action_rows = []
        state_masks = []
        action_masks = []
        for e in robot_eps:
            sp = np.zeros((e.length, C.MAX_STATE_DIM), dtype=np.float32)   # (T, 16)
            ap = np.zeros((e.length, C.MAX_ACTION_DIM), dtype=np.float32)  # (T, 16)
            sm = np.zeros((e.length, C.MAX_STATE_DIM), dtype=bool)
            am = np.zeros((e.length, C.MAX_ACTION_DIM), dtype=bool)
            sp[:, :e.state_dim] = e.states                                  # left-aligned pack
            ap[:, :e.action_dim] = e.actions
            sm[:, :e.state_dim] = True
            am[:, :e.action_dim] = True
            state_rows.append(sp); action_rows.append(ap)
            state_masks.append(sm); action_masks.append(am)

        states = np.concatenate(state_rows, axis=0)            # (sum_T, 16)
        actions = np.concatenate(action_rows, axis=0)          # (sum_T, 16)
        state_masks_arr = np.concatenate(state_masks, axis=0)  # (sum_T, 16)
        action_masks_arr = np.concatenate(action_masks, axis=0)
        state_mean = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
        state_std = np.ones(C.MAX_STATE_DIM, dtype=np.float32)
        action_mean = np.zeros(C.MAX_ACTION_DIM, dtype=np.float32)
        action_std = np.ones(C.MAX_ACTION_DIM, dtype=np.float32)

        for dim in range(C.MAX_STATE_DIM):
            valid = state_masks_arr[:, dim]
            if valid.any():
                vals = states[valid, dim]
                state_mean[dim] = vals.mean()
                state_std[dim] = np.clip(vals.std(), 1e-2, None)
        # ... same loop for action_mean/action_std ...

        by_robot[robot_id] = {
            "qpos_mean": state_mean,    # (16,)
            "qpos_std": state_std,      # (16,)
            "action_mean": action_mean, # (16,)
            "action_std": action_std,   # (16,)
        }

    return {
        "by_robot": by_robot,
        "image_mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "image_std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
    }""")
    bullets(story, [
        "Stats are computed <i>per robot</i>, not globally. Two different robots can "
        "share the same dim 0 in padded space but live in entirely different unit "
        "systems; mixing their stats would produce nonsense.",
        "Per dimension, the loop only averages over <i>valid</i> rows (where the mask is "
        "True). For dimensions a particular robot never uses, the mean stays 0 and std "
        "stays 1 (the no-op normalization). Together with the action_mask later, this "
        "guarantees the model never sees nonsense values from padded slots.",
        "Final shapes per robot: <b>qpos_mean/qpos_std (16,)</b>, "
        "<b>action_mean/action_std (16,)</b>. The top-level dict also stores the same "
        "ImageNet image stats as ACT.",
    ])
    code_block(story, "data/vla_dataset.py:272-365 - VLADataset and __getitem__", """\
class VLADataset(Dataset):
    def __init__(self, episodes, stats, vocabs, chunk_size=..., image_hw=(240,320)):
        self.episodes = episodes
        self.stats = stats
        self.vocabs = vocabs
        self.chunk_size = int(chunk_size)
        self.image_h, self.image_w = image_hw
        self.index = [
            (ei, t)
            for ei, ep in enumerate(episodes)
            for t in range(ep.length)
        ]

    def __getitem__(self, i):
        episode_idx, t = self.index[i]
        ep = self.episodes[episode_idx]
        robot_stats = self._stats_for(ep.robot_id)            # dict with (16,) arrays

        qpos = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)    # (16,)
        qpos[:ep.state_dim] = ep.states[t]                    # left-pack real values
        state_mask = np.zeros(C.MAX_STATE_DIM, dtype=bool)
        state_mask[:ep.state_dim] = True

        end = min(t + self.chunk_size, ep.length)
        avail = end - t
        action = np.zeros((self.chunk_size, C.MAX_ACTION_DIM), dtype=np.float32)  # (100,16)
        action[:avail, :ep.action_dim] = ep.actions[t:end]    # 2D pad: rows + cols
        action_mask = np.zeros(C.MAX_ACTION_DIM, dtype=bool)
        action_mask[:ep.action_dim] = True
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[avail:] = True

        camera_name = ep.camera_names[0] if ep.camera_names else "cam_main"
        path = _resolve_image_path(ep, ep.image_paths[t], t, camera_name)
        img = _load_image(path, self.image_h, self.image_w)   # (240,320,3) uint8
        img = img.astype(np.float32) / 255.0
        img = (img - self.stats["image_mean"]) / self.stats["image_std"]
        img = np.transpose(img, (2, 0, 1))[None]              # (1,3,240,320)

        qpos_n = (qpos - robot_stats["qpos_mean"]) / robot_stats["qpos_std"]
        action_n = (action - robot_stats["action_mean"]) / robot_stats["action_std"]
        action_n[is_pad] = 0.0                                 # zero padded chunk rows
        action_n[:, ~action_mask] = 0.0                        # zero padded action dims
        qpos_n[~state_mask] = 0.0                              # zero padded state dims

        return {
            "image": torch.from_numpy(img).float(),                       # (1,3,240,320)
            "qpos": torch.from_numpy(qpos_n).float(),                     # (16,)
            "state_mask": torch.from_numpy(state_mask),                   # (16,) bool
            "action": torch.from_numpy(action_n).float(),                 # (100,16)
            "action_mask": torch.from_numpy(action_mask),                 # (16,) bool
            "is_pad": torch.from_numpy(is_pad),                           # (100,) bool
            "instruction": ep.instruction,                                # Python str
            "robot_id": torch.tensor(_lookup(self.vocabs.robot_ids, ep.robot_id), ...),
            ... lab_id, embodiment_id, action_type_id, task_family_id ...
            "episode_index": torch.tensor(episode_idx, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }""")
    bullets(story, [
        "<b>2-axis padding</b>: actions are padded along both axis 0 (timesteps past "
        "the trial end) and axis 1 (action dims past <code>ep.action_dim</code>). "
        "<code>is_pad</code> covers axis-0 padding; <code>action_mask</code> covers "
        "axis-1 padding. The loss multiplies them to mask both at once.",
        "<b>Per-robot normalization</b>: <code>self._stats_for(ep.robot_id)</code> looks "
        "up that robot's mean/std arrays. After normalization the padded action dims "
        "(where mask is False) and padded chunk rows are explicitly zeroed.",
        "Note <code>action_n[:, ~action_mask] = 0.0</code> uses NumPy advanced indexing: "
        "boolean index inverts the mask and zeroes the (chunk_size, k) slab where k is "
        "the count of False entries.",
        "<b>Instruction</b> is returned as a plain Python string. <code>DataLoader</code>'s "
        "default collate_fn turns a list of strings into a Python list, not a tensor; "
        "the language encoder accepts that list directly.",
    ])
    code_block(story, "data/vla_dataset.py:368-388 - build_vla_dataset", """\
def build_vla_dataset(
    episodes_dir=C.VLA_EPISODES_DIR,
    stats_path=C.VLA_STATS_PATH,
    recompute_stats=False,
) -> VLADataset:
    episodes = [load_episode(p) for p in discover_episode_dirs(episodes_dir)]
    vocabs = build_vocabs(episodes)

    if stats_path.exists() and not recompute_stats:
        with open(stats_path, "rb") as f:
            payload = pickle.load(f)
        stats = payload["stats"] if "stats" in payload else payload
        if "vocabs" in payload:
            vocabs = VocabBundle(**payload["vocabs"])
    else:
        stats = compute_vla_norm_stats(episodes)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump({"stats": stats, "vocabs": vocabs.as_dict()}, f)

    return VLADataset(episodes, stats, vocabs)""")
    bullets(story, [
        "Stats <i>and</i> vocabs are persisted together because the embedding rows in "
        "the saved policy are indexed by these vocab IDs — losing the mapping would "
        "make a checkpoint useless.",
        "When loading, the function tolerates an older payload that lacked the "
        "<code>vocabs</code> key (the legacy format was just the stats dict).",
    ])
