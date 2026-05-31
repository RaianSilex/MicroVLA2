"""Metadata-driven dataset for heterogeneous MicroVLA training.

Expected episode layout:

    dataset_vla/episodes/<episode_id>/
        metadata.json
        trajectory.csv
        frames/<camera_name>/frame_000000.png

Each episode declares its robot/task/lab metadata plus the state/action column
order used by its trajectory. States and actions are padded to
config.vla_config.MAX_*_DIM and accompanied by masks so single- and dual-arm
robots can train in the same batches.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import vla_config as C


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
    length: int


def _default_state_cols(dim: int) -> List[str]:
    base = [
        "current_x", "current_y", "current_z", "current_d",
        "current_x2", "current_y2", "current_z2", "current_d2",
    ]
    return base[:dim] if dim <= len(base) else [f"state_{i}" for i in range(dim)]


def _default_action_cols(dim: int) -> List[str]:
    base = [
        "target_x", "target_y", "target_z", "target_d",
        "target_x2", "target_y2", "target_z2", "target_d2",
    ]
    return base[:dim] if dim <= len(base) else [f"action_{i}" for i in range(dim)]


def _read_metadata(episode_dir: Path) -> dict:
    path = episode_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing episode metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_episode_dirs(root: Path = C.VLA_EPISODES_DIR) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"No VLA episodes directory found: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No episode directories found under {root}")
    return dirs


def load_episode(episode_dir: Path) -> VLAEpisode:
    meta = _read_metadata(episode_dir)
    episode_id = str(meta.get("episode_id", episode_dir.name))
    trajectory_name = str(meta.get("trajectory_file", "trajectory.csv"))
    csv_path = episode_dir / trajectory_name
    if not csv_path.exists():
        raise FileNotFoundError(f"{episode_id}: missing trajectory CSV: {csv_path}")

    state_dim = int(meta.get("state_dim", len(meta.get("state_cols", [])) or 0))
    action_dim = int(meta.get("action_dim", len(meta.get("action_cols", [])) or 0))
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError(f"{episode_id}: state_dim and action_dim must be positive")
    if state_dim > C.MAX_STATE_DIM:
        raise ValueError(f"{episode_id}: state_dim {state_dim} > MAX_STATE_DIM {C.MAX_STATE_DIM}")
    if action_dim > C.MAX_ACTION_DIM:
        raise ValueError(f"{episode_id}: action_dim {action_dim} > MAX_ACTION_DIM {C.MAX_ACTION_DIM}")

    state_cols = list(meta.get("state_cols") or _default_state_cols(state_dim))
    action_cols = list(meta.get("action_cols") or _default_action_cols(action_dim))
    if len(state_cols) != state_dim:
        raise ValueError(f"{episode_id}: state_cols length does not match state_dim")
    if len(action_cols) != action_dim:
        raise ValueError(f"{episode_id}: action_cols length does not match action_dim")

    df = pd.read_csv(csv_path)
    missing = [c for c in (*state_cols, *action_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"{episode_id}: missing trajectory columns: {missing}")

    image_col = str(meta.get("image_col", "image_path"))
    timestep_col = str(meta.get("timestep_col", "timestep"))
    raw_paths = (
        df[image_col].fillna("").astype(str).tolist()
        if image_col in df.columns
        else [""] * len(df)
    )

    return VLAEpisode(
        episode_dir=episode_dir,
        episode_id=episode_id,
        lab_id=str(meta.get("lab_id", C.DEFAULT_LAB_ID)),
        robot_id=str(meta.get("robot_id", C.DEFAULT_ROBOT_ID)),
        embodiment=str(meta.get("embodiment", C.DEFAULT_EMBODIMENT)),
        action_type=str(meta.get("action_type", C.DEFAULT_ACTION_TYPE)),
        task_family=str(meta.get("task_family", C.DEFAULT_TASK_FAMILY)),
        instruction=str(meta.get("instruction", "")),
        camera_names=list(meta.get("camera_names", ["cam_main"])),
        state_cols=state_cols,
        action_cols=action_cols,
        image_col=image_col,
        timestep_col=timestep_col,
        state_dim=state_dim,
        action_dim=action_dim,
        states=df[state_cols].to_numpy(dtype=np.float32),
        actions=df[action_cols].to_numpy(dtype=np.float32),
        image_paths=raw_paths,
        length=len(df),
    )


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
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))


def compute_vla_norm_stats(episodes: List[VLAEpisode]) -> dict:
    """Compute per-robot stats in padded MAX_*_DIM space."""
    by_robot = {}
    for robot_id in sorted({e.robot_id for e in episodes}):
        robot_eps = [e for e in episodes if e.robot_id == robot_id]
        state_rows = []
        action_rows = []
        state_masks = []
        action_masks = []
        for e in robot_eps:
            sp = np.zeros((e.length, C.MAX_STATE_DIM), dtype=np.float32)
            ap = np.zeros((e.length, C.MAX_ACTION_DIM), dtype=np.float32)
            sm = np.zeros((e.length, C.MAX_STATE_DIM), dtype=bool)
            am = np.zeros((e.length, C.MAX_ACTION_DIM), dtype=bool)
            sp[:, :e.state_dim] = e.states
            ap[:, :e.action_dim] = e.actions
            sm[:, :e.state_dim] = True
            am[:, :e.action_dim] = True
            state_rows.append(sp)
            action_rows.append(ap)
            state_masks.append(sm)
            action_masks.append(am)

        states = np.concatenate(state_rows, axis=0)
        actions = np.concatenate(action_rows, axis=0)
        state_masks_arr = np.concatenate(state_masks, axis=0)
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
        for dim in range(C.MAX_ACTION_DIM):
            valid = action_masks_arr[:, dim]
            if valid.any():
                vals = actions[valid, dim]
                action_mean[dim] = vals.mean()
                action_std[dim] = np.clip(vals.std(), 1e-2, None)

        by_robot[robot_id] = {
            "qpos_mean": state_mean,
            "qpos_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }

    return {
        "by_robot": by_robot,
        "image_mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "image_std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
    }


def _resolve_image_path(ep: VLAEpisode, raw: str, t: int, camera_name: str) -> Optional[Path]:
    raw = (raw or "").strip()
    fallback = ep.episode_dir / "frames" / camera_name / f"frame_{t:06d}.png"
    if not raw:
        return fallback if fallback.exists() else None

    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (ep.episode_dir, C.VLA_DATASET_ROOT, C.REPO_ROOT):
        q = base / p
        if q.exists():
            return q
    return fallback if fallback.exists() else None


def _load_image(path: Optional[Path], h: int, w: int) -> np.ndarray:
    if path is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.open(path).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class VLADataset(Dataset):
    """One sample per (episode, timestep), with padded heterogeneous actions."""

    def __init__(
        self,
        episodes: List[VLAEpisode],
        stats: dict,
        vocabs: VocabBundle,
        chunk_size: int = C.CHUNK_SIZE,
        image_hw: tuple = (C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
    ):
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
        self._warned_missing_image = False

    def __len__(self) -> int:
        return len(self.index)

    def _stats_for(self, robot_id: str) -> dict:
        try:
            return self.stats["by_robot"][robot_id]
        except KeyError as exc:
            raise KeyError(f"No normalization stats found for robot_id={robot_id!r}") from exc

    def __getitem__(self, i: int) -> dict:
        episode_idx, t = self.index[i]
        ep = self.episodes[episode_idx]
        robot_stats = self._stats_for(ep.robot_id)

        qpos = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
        qpos[:ep.state_dim] = ep.states[t]
        state_mask = np.zeros(C.MAX_STATE_DIM, dtype=bool)
        state_mask[:ep.state_dim] = True

        end = min(t + self.chunk_size, ep.length)
        avail = end - t
        action = np.zeros((self.chunk_size, C.MAX_ACTION_DIM), dtype=np.float32)
        action[:avail, :ep.action_dim] = ep.actions[t:end]
        action_mask = np.zeros(C.MAX_ACTION_DIM, dtype=bool)
        action_mask[:ep.action_dim] = True
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[avail:] = True

        camera_name = ep.camera_names[0] if ep.camera_names else "cam_main"
        raw = ep.image_paths[t] if t < len(ep.image_paths) else ""
        path = _resolve_image_path(ep, raw, t, camera_name)
        if path is None and not self._warned_missing_image:
            warnings.warn(
                f"{ep.episode_id}: image at t={t} unresolved (csv={raw!r}); "
                "returning zeros. Further warnings suppressed.",
                stacklevel=2,
            )
            self._warned_missing_image = True
        img = _load_image(path, self.image_h, self.image_w)
        img = img.astype(np.float32) / 255.0
        img = (img - self.stats["image_mean"]) / self.stats["image_std"]
        img = np.transpose(img, (2, 0, 1))[None]

        qpos_n = (qpos - robot_stats["qpos_mean"]) / robot_stats["qpos_std"]
        action_n = (action - robot_stats["action_mean"]) / robot_stats["action_std"]
        action_n[is_pad] = 0.0
        action_n[:, ~action_mask] = 0.0
        qpos_n[~state_mask] = 0.0

        return {
            "image": torch.from_numpy(img).float(),
            "qpos": torch.from_numpy(qpos_n).float(),
            "state_mask": torch.from_numpy(state_mask),
            "action": torch.from_numpy(action_n).float(),
            "action_mask": torch.from_numpy(action_mask),
            "is_pad": torch.from_numpy(is_pad),
            "instruction": ep.instruction,
            "robot_id": torch.tensor(_lookup(self.vocabs.robot_ids, ep.robot_id), dtype=torch.long),
            "lab_id": torch.tensor(_lookup(self.vocabs.lab_ids, ep.lab_id), dtype=torch.long),
            "embodiment_id": torch.tensor(
                _lookup(self.vocabs.embodiment_ids, ep.embodiment), dtype=torch.long
            ),
            "action_type_id": torch.tensor(
                _lookup(self.vocabs.action_type_ids, ep.action_type), dtype=torch.long
            ),
            "task_family_id": torch.tensor(
                _lookup(self.vocabs.task_family_ids, ep.task_family), dtype=torch.long
            ),
            "episode_index": torch.tensor(episode_idx, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }


def build_vla_dataset(
    episodes_dir: Path = C.VLA_EPISODES_DIR,
    stats_path: Path = C.VLA_STATS_PATH,
    recompute_stats: bool = False,
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

    return VLADataset(episodes, stats, vocabs)
