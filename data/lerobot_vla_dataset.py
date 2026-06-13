"""LeRobot-backed dataset for MicroVLA (SmolVLA / OpenPI-style data path).

Reads a LeRobot-format dataset (local under HF_LEROBOT_HOME, or pulled from the
HF Hub) and yields a per-sample dict consumed by ``VLAPolicy`` / ``train_vla.py``.

The dataset is robot-native: it stores **ABSOLUTE** Sensapex targets. The action
space used for training is chosen here:

* ``"delta"``  (default): ``action[i] = abs_target[t+i] - state_t``  — small,
  zero-centered, workspace-translation invariant. Inverted at rollout
  (``VLAPolicy.inference``) back to absolute, so the robot side never changes.
* ``"absolute"``: actions are the stored absolute targets.

Each sample also carries:
* ``goal``        — the episode's final reached target in the same action
  representation (the contact point the tip heads toward), normalized; supervises
  the policy's contact-point Gaussian head.
* ``resistance``  — per-frame pipette resistance, ONLY when the dataset provides
  ``observation.resistance``. Auto-detected; absent datasets train without it.

Per-axis action weights (down-weighting near-constant axes) are computed from the
realized action representation and stored in the per-robot stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import vla_config as C
from data.vocab import VocabBundle

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class LeRobotEpisodeMeta:
    episode_id: str
    episode_index: int
    robot_id: str
    lab_id: str
    embodiment: str
    action_type: str
    task_family: str
    instruction: str
    length: int
    from_index: int
    to_index: int


def _make_single_vocab(value: str) -> Dict[str, int]:
    return {C.UNKNOWN_TOKEN: 0, str(value): 1}


def _lookup(vocab: Dict[str, int], value: str) -> int:
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))


def _axis_weights(raw_std: np.ndarray, action_dim: int) -> np.ndarray:
    """Per-axis loss weights from each dim's movement std.

    Active axes land near 1.0; near-constant axes are floored to AXIS_WEIGHT_MIN so
    they stop diluting the loss without being fully ignored. Padded dims get 0.
    """
    weight = np.zeros(C.MAX_ACTION_DIM, dtype=np.float32)
    if not C.AXIS_WEIGHTING:
        weight[:action_dim] = 1.0
        return weight
    rs = raw_std[:action_dim].astype(np.float64)
    active = rs > 1e-6 * (rs.max() + 1e-9)
    ref = np.median(rs[active]) if active.any() else 1.0
    ref = ref if ref > 1e-9 else 1.0
    w = np.clip(rs / ref, C.AXIS_WEIGHT_MIN, C.AXIS_WEIGHT_MAX)
    weight[:action_dim] = w.astype(np.float32)
    return weight


def compute_lerobot_norm_stats(
    states_all: np.ndarray,        # (N, state_dim) absolute
    actions_all: np.ndarray,       # (N, action_dim) absolute targets
    episodes: List[LeRobotEpisodeMeta],
    action_space: str,
    chunk_size: int,
    robot_id: str,
    resistance_all: Optional[np.ndarray] = None,   # (N,) or None
) -> dict:
    """Per-robot stats padded to MAX_*_DIM, in the chosen action space."""
    state_dim = states_all.shape[1]
    action_dim = actions_all.shape[1]

    qpos_mean = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
    qpos_std = np.ones(C.MAX_STATE_DIM, dtype=np.float32)
    qpos_mean[:state_dim] = states_all.mean(0)
    qpos_std[:state_dim] = np.clip(states_all.std(0), 1e-2, None)

    action_mean = np.zeros(C.MAX_ACTION_DIM, dtype=np.float32)
    action_std = np.ones(C.MAX_ACTION_DIM, dtype=np.float32)
    raw_std = np.zeros(C.MAX_ACTION_DIM, dtype=np.float32)   # unclipped, for weights

    if action_space == "absolute":
        mean = actions_all.mean(0)
        std = actions_all.std(0)
        action_mean[:action_dim] = mean
        action_std[:action_dim] = np.clip(std, 1e-2, None)
        raw_std[:action_dim] = std
    elif action_space == "delta":
        # Accumulate over every realized chunk delta d[i] = action[t+i] - state[t].
        s = np.zeros(action_dim, dtype=np.float64)
        sq = np.zeros(action_dim, dtype=np.float64)
        count = 0
        for ep in episodes:
            A = actions_all[ep.from_index:ep.to_index]
            S = states_all[ep.from_index:ep.to_index]
            L = ep.length
            for t in range(L):
                e = min(t + chunk_size, L)
                d = A[t:e] - S[t]                      # (e-t, action_dim)
                s += d.sum(0)
                sq += (d * d).sum(0)
                count += (e - t)
        count = max(count, 1)
        mean = s / count
        var = np.clip(sq / count - mean * mean, 0.0, None)
        std = np.sqrt(var)
        action_mean[:action_dim] = mean.astype(np.float32)
        action_std[:action_dim] = np.clip(std, 1e-2, None).astype(np.float32)
        raw_std[:action_dim] = std.astype(np.float32)
    else:
        raise ValueError(f"action_space must be 'delta' or 'absolute', got {action_space!r}")

    robot_stats = {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "action_weight": _axis_weights(raw_std, action_dim),
    }

    has_resistance = resistance_all is not None
    if has_resistance:
        r = np.asarray(resistance_all, dtype=np.float32).reshape(-1)
        robot_stats["resistance_mean"] = float(r.mean())
        robot_stats["resistance_std"] = float(np.clip(r.std(), 1e-6, None))

    return {
        "by_robot": {robot_id: robot_stats},
        "image_mean": _IMAGENET_MEAN.copy(),
        "image_std": _IMAGENET_STD.copy(),
        "action_space": action_space,
        "has_resistance": has_resistance,
    }


class LeRobotVLADataset(Dataset):
    """One sample per (episode, timestep) from a LeRobot dataset."""

    def __init__(
        self,
        lerobot_ds,
        states_all: np.ndarray,
        actions_all: np.ndarray,
        episodes: List[LeRobotEpisodeMeta],
        stats: dict,
        vocabs: VocabBundle,
        action_space: str,
        chunk_size: int = C.CHUNK_SIZE,
        image_hw: tuple = (C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
        camera_key: str = C.LEROBOT_CAMERA_KEY,
        resistance_all: Optional[np.ndarray] = None,
    ):
        self.ds = lerobot_ds
        self.states_all = states_all
        self.actions_all = actions_all
        self.resistance_all = resistance_all
        self.has_resistance = resistance_all is not None
        self.episodes = episodes
        self.stats = stats
        self.vocabs = vocabs
        self.action_space = action_space
        self.chunk_size = int(chunk_size)
        self.image_h, self.image_w = image_hw
        self.camera_key = camera_key
        self.state_dim = states_all.shape[1]
        self.action_dim = actions_all.shape[1]

        # Optional FeatureCache (set externally by the trainer). When present,
        # __getitem__ returns precomputed raw encoder features instead of
        # decoding + resizing the frame, removing the per-step video decode.
        self.feature_cache = None

        self.index = [(ei, t) for ei, ep in enumerate(episodes) for t in range(ep.length)]

        self._image_mean = torch.from_numpy(stats["image_mean"]).view(3, 1, 1)
        self._image_std = torch.from_numpy(stats["image_std"]).view(3, 1, 1)

        # Precompute per-episode metadata token ids (constant per episode).
        self._ids = [
            {
                "robot_id": _lookup(vocabs.robot_ids, ep.robot_id),
                "lab_id": _lookup(vocabs.lab_ids, ep.lab_id),
                "embodiment_id": _lookup(vocabs.embodiment_ids, ep.embodiment),
                "action_type_id": _lookup(vocabs.action_type_ids, ep.action_type),
                "task_family_id": _lookup(vocabs.task_family_ids, ep.task_family),
            }
            for ep in episodes
        ]

    def __len__(self) -> int:
        return len(self.index)

    def _stats_for(self, robot_id: str) -> dict:
        return self.stats["by_robot"][robot_id]

    def _load_image(self, global_index: int) -> torch.Tensor:
        frame = self.ds[global_index]
        img = frame[self.camera_key]                       # (3, H, W) float in [0, 1]
        if not torch.is_tensor(img):
            img = torch.as_tensor(np.asarray(img))
        img = img.float()
        if img.max() > 1.5:                                # guard if returned as 0-255
            img = img / 255.0
        if img.shape[-2:] != (self.image_h, self.image_w):
            img = F.interpolate(
                img.unsqueeze(0), size=(self.image_h, self.image_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        img = (img - self._image_mean) / self._image_std
        return img.unsqueeze(0)                            # (num_cam=1, 3, H, W)

    def __getitem__(self, i: int) -> dict:
        ei, t = self.index[i]
        ep = self.episodes[ei]
        robot_stats = self._stats_for(ep.robot_id)
        g0 = ep.from_index
        g = g0 + t

        state_raw = self.states_all[g]                     # (state_dim,) absolute
        base = state_raw[: self.action_dim][None, :]       # (1, action_dim)

        end = min(t + self.chunk_size, ep.length)
        avail = end - t
        abs_targets = self.actions_all[g0 + t: g0 + end]   # (avail, action_dim) absolute
        abs_final = self.actions_all[g0 + ep.length - 1]   # (action_dim,) contact point
        if self.action_space == "delta":
            chunk = abs_targets - base
            goal_raw = abs_final - base[0]
        else:
            chunk = abs_targets
            goal_raw = abs_final

        qpos = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
        qpos[: self.state_dim] = state_raw
        state_mask = np.zeros(C.MAX_STATE_DIM, dtype=bool)
        state_mask[: self.state_dim] = True

        action = np.zeros((self.chunk_size, C.MAX_ACTION_DIM), dtype=np.float32)
        action[:avail, : self.action_dim] = chunk
        action_mask = np.zeros(C.MAX_ACTION_DIM, dtype=bool)
        action_mask[: self.action_dim] = True
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[avail:] = True

        goal = np.zeros(C.MAX_ACTION_DIM, dtype=np.float32)
        goal[: self.action_dim] = goal_raw

        qpos_n = (qpos - robot_stats["qpos_mean"]) / robot_stats["qpos_std"]
        action_n = (action - robot_stats["action_mean"]) / robot_stats["action_std"]
        goal_n = (goal - robot_stats["action_mean"]) / robot_stats["action_std"]
        action_n[is_pad] = 0.0
        action_n[:, ~action_mask] = 0.0
        goal_n[~action_mask] = 0.0
        qpos_n[~state_mask] = 0.0

        ids = self._ids[ei]
        sample = {
            "qpos": torch.from_numpy(qpos_n).float(),
            "state_mask": torch.from_numpy(state_mask),
            "action": torch.from_numpy(action_n).float(),
            "action_mask": torch.from_numpy(action_mask),
            "is_pad": torch.from_numpy(is_pad),
            "goal": torch.from_numpy(goal_n).float(),
            "instruction": ep.instruction,
            "robot_id": torch.tensor(ids["robot_id"], dtype=torch.long),
            "lab_id": torch.tensor(ids["lab_id"], dtype=torch.long),
            "embodiment_id": torch.tensor(ids["embodiment_id"], dtype=torch.long),
            "action_type_id": torch.tensor(ids["action_type_id"], dtype=torch.long),
            "task_family_id": torch.tensor(ids["task_family_id"], dtype=torch.long),
            "episode_index": torch.tensor(ei, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }
        if self.has_resistance:
            r = float(self.resistance_all[g])
            r = (r - robot_stats["resistance_mean"]) / robot_stats["resistance_std"]
            sample["resistance"] = torch.tensor([r], dtype=torch.float32)
        if self.feature_cache is not None:
            # Cached raw encoder features → no video decode this step.
            primary_feat, aux_feat = self.feature_cache.get(g)
            sample["primary_feat"] = primary_feat
            if aux_feat is not None:
                sample["aux_feat"] = aux_feat
        else:
            sample["image"] = self._load_image(g).float()
        return sample


def build_lerobot_vla_dataset(
    repo_id: str = C.DEFAULT_DATASET_REPO_ID,
    root: Optional[Path] = None,
    action_space: str = C.DEFAULT_ACTION_SPACE,
    chunk_size: int = C.CHUNK_SIZE,
    robot_id: Optional[str] = None,
    lab_id: str = C.DEFAULT_LAB_ID,
    embodiment: str = C.DEFAULT_EMBODIMENT,
    action_type: str = C.DEFAULT_ACTION_TYPE,
    task_family: str = C.DEFAULT_TASK_FAMILY,
) -> LeRobotVLADataset:
    """Load a LeRobot dataset and wrap it as a MicroVLA dataset.

    The metadata tokens (lab/embodiment/action_type/task_family) are constant for
    this single-rig dataset; language (the per-episode ``task``) is the real
    conditioning signal. ``robot_id`` defaults to the dataset's ``robot_type`` and
    must match the rollout adapter's ``robot_id`` so per-robot stats line up.
    A per-frame ``observation.resistance`` feature is used automatically if present.
    """
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    if root is None:
        root = HF_LEROBOT_HOME / repo_id
    ds = LeRobotDataset(repo_id, root=Path(root))

    robot_id = robot_id or (ds.meta.robot_type or C.DEFAULT_ROBOT_ID)

    # Bulk-read absolute state/action columns (cheap; does not decode images).
    states_all = np.asarray(ds.hf_dataset[C.LEROBOT_STATE_KEY], dtype=np.float32)
    actions_all = np.asarray(ds.hf_dataset[C.LEROBOT_ACTION_KEY], dtype=np.float32)

    resistance_all = None
    if C.LEROBOT_RESISTANCE_KEY in ds.hf_dataset.column_names:
        resistance_all = np.asarray(
            ds.hf_dataset[C.LEROBOT_RESISTANCE_KEY], dtype=np.float32
        ).reshape(-1)

    episodes: List[LeRobotEpisodeMeta] = []
    for row in ds.meta.episodes:
        tasks = row.get("tasks") or []
        instruction = str(tasks[0]) if len(tasks) else ""
        ei = int(row["episode_index"])
        episodes.append(LeRobotEpisodeMeta(
            episode_id=f"episode_{ei}",
            episode_index=ei,
            robot_id=robot_id,
            lab_id=lab_id,
            embodiment=embodiment,
            action_type=action_type,
            task_family=task_family,
            instruction=instruction,
            length=int(row["length"]),
            from_index=int(row["dataset_from_index"]),
            to_index=int(row["dataset_to_index"]),
        ))
    episodes.sort(key=lambda e: e.episode_index)

    vocabs = VocabBundle(
        robot_ids=_make_single_vocab(robot_id),
        lab_ids=_make_single_vocab(lab_id),
        embodiment_ids=_make_single_vocab(embodiment),
        action_type_ids=_make_single_vocab(action_type),
        task_family_ids=_make_single_vocab(task_family),
    )
    stats = compute_lerobot_norm_stats(
        states_all, actions_all, episodes, action_space, chunk_size, robot_id,
        resistance_all=resistance_all,
    )
    return LeRobotVLADataset(
        lerobot_ds=ds,
        states_all=states_all,
        actions_all=actions_all,
        episodes=episodes,
        stats=stats,
        vocabs=vocabs,
        action_space=action_space,
        chunk_size=chunk_size,
        resistance_all=resistance_all,
    )
