"""On-disk cache of frozen image-encoder features.

Both image encoders in a frozen backbone (e.g. DINOv2 + Cellpose-SAM) produce
the *same* feature maps for a given frame on every epoch, because they are
frozen and the training pipeline applies **no image augmentation**
(``LeRobotVLADataset.__getitem__`` returns a deterministic decode + resize).

Computing those features once and reading them back from a memory-mapped file
removes, from every training step:

  * the per-frame MP4 random-access decode (the dominant LeRobot v3 bottleneck),
  * the two frozen ViT forward passes.

Crucially we cache the **raw encoder outputs** (before ``input_proj`` /
``type_embed`` / ``pos_embed``). Those projection layers are *trainable* even
when the encoders are frozen, so they must keep running each step on the cached
features — only the expensive frozen part is skipped.

The cache is ONLY valid while the encoders stay frozen. With
``--unfreeze-backbone`` the encoder weights change during training, so the
cached features would be stale; callers must not use it in that case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


_PRIMARY_FILE = "primary.dat"
_AUX_FILE = "aux.dat"
_META_FILE = "meta.json"


class FeatureCache:
    """Memmap-backed store of raw encoder features keyed by global frame index."""

    def __init__(self, cache_dir, meta: dict, mode: str = "r"):
        self.dir = Path(cache_dir)
        self.meta = meta
        self.num_frames = int(meta["num_frames"])
        self.primary_shape = tuple(meta["primary_shape"])
        self.has_aux = bool(meta.get("has_aux", False))
        self.aux_shape = tuple(meta["aux_shape"]) if self.has_aux else None
        self.np_dtype = np.dtype(meta.get("dtype", "float16"))

        self._primary = np.memmap(
            self.dir / _PRIMARY_FILE,
            dtype=self.np_dtype,
            mode=mode,
            shape=(self.num_frames, *self.primary_shape),
        )
        self._aux = None
        if self.has_aux:
            self._aux = np.memmap(
                self.dir / _AUX_FILE,
                dtype=self.np_dtype,
                mode=mode,
                shape=(self.num_frames, *self.aux_shape),
            )

    # ------------------------------------------------------------------
    def get(self, g: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (primary_feat, aux_feat|None) as fp32 tensors for frame ``g``."""
        primary = torch.from_numpy(np.asarray(self._primary[g], dtype=np.float32))
        aux = None
        if self._aux is not None:
            aux = torch.from_numpy(np.asarray(self._aux[g], dtype=np.float32))
        return primary, aux

    # ------------------------------------------------------------------
    @staticmethod
    def _expected_ok(meta: dict, *, repo_id, backbone_name, image_hw, num_frames) -> bool:
        return (
            meta.get("complete", False)
            and meta.get("repo_id") == repo_id
            and meta.get("backbone_name") == backbone_name
            and tuple(meta.get("image_hw", ())) == tuple(image_hw)
            and int(meta.get("num_frames", -1)) == int(num_frames)
        )

    @classmethod
    def load_if_valid(
        cls,
        cache_dir,
        *,
        repo_id,
        backbone_name,
        image_hw,
        num_frames,
        log=print,
    ) -> Optional["FeatureCache"]:
        """Reuse an existing cache iff its metadata matches the current run."""
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / _META_FILE
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not cls._expected_ok(
            meta,
            repo_id=repo_id,
            backbone_name=backbone_name,
            image_hw=image_hw,
            num_frames=num_frames,
        ):
            log(f"[feature-cache] existing cache at {cache_dir} is stale/incomplete; rebuilding")
            return None
        log(f"[feature-cache] reusing valid cache at {cache_dir} ({num_frames} frames)")
        return cls(cache_dir, meta, mode="r")

    # ------------------------------------------------------------------
    @classmethod
    @torch.no_grad()
    def build(
        cls,
        cache_dir,
        full_ds,
        policy,
        device,
        *,
        repo_id,
        backbone_name,
        image_hw,
        batch_size: int = 32,
        log=print,
    ) -> "FeatureCache":
        """Run the frozen encoders over every frame once and memmap the result.

        ``full_ds`` must expose ``_load_image(g) -> (num_cam=1, 3, H, W)`` and
        ``states_all`` (one row per global frame). ``policy.model.backbone`` must
        provide ``encode_raw(x) -> (primary_feat, aux_feat|None)``.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        backbone = policy.model.backbone
        was_training = backbone.training
        backbone.eval()

        num_frames = int(full_ds.states_all.shape[0])
        if num_frames == 0:
            raise RuntimeError("feature cache: dataset has 0 frames")

        # Write an incomplete meta first so an interrupted build is detected as
        # stale on the next run (complete=False until the very end).
        meta: dict = {
            "repo_id": repo_id,
            "backbone_name": backbone_name,
            "image_hw": list(image_hw),
            "num_frames": num_frames,
            "dtype": "float16",
            "complete": False,
        }
        (cache_dir / _META_FILE).write_text(json.dumps(meta))

        primary_mm = None
        aux_mm = None
        has_aux = False

        log(f"[feature-cache] building at {cache_dir} for {num_frames} frames "
            f"(batch {batch_size})...")
        for start in range(0, num_frames, batch_size):
            gs = range(start, min(start + batch_size, num_frames))
            imgs = torch.stack([full_ds._load_image(g)[0] for g in gs]).to(device)
            primary_feat, aux_feat = backbone.encode_raw(imgs)

            primary_np = primary_feat.float().cpu().numpy().astype(np.float16)
            if primary_mm is None:
                primary_shape = tuple(primary_np.shape[1:])
                meta["primary_shape"] = list(primary_shape)
                primary_mm = np.memmap(
                    cache_dir / _PRIMARY_FILE, dtype=np.float16, mode="w+",
                    shape=(num_frames, *primary_shape),
                )
                has_aux = aux_feat is not None
                meta["has_aux"] = has_aux
                if has_aux:
                    aux_shape = tuple(aux_feat.shape[1:])
                    meta["aux_shape"] = list(aux_shape)
                    aux_mm = np.memmap(
                        cache_dir / _AUX_FILE, dtype=np.float16, mode="w+",
                        shape=(num_frames, *aux_shape),
                    )
            primary_mm[start:start + primary_np.shape[0]] = primary_np
            if has_aux:
                aux_mm[start:start + primary_np.shape[0]] = (
                    aux_feat.float().cpu().numpy().astype(np.float16)
                )

            done = start + primary_np.shape[0]
            if start == 0 or done == num_frames or (start // batch_size) % 25 == 0:
                log(f"[feature-cache]   {done}/{num_frames} frames")

        primary_mm.flush()
        if aux_mm is not None:
            aux_mm.flush()

        meta["complete"] = True
        (cache_dir / _META_FILE).write_text(json.dumps(meta))
        log(f"[feature-cache] done: primary{tuple(meta['primary_shape'])}"
            + (f" + aux{tuple(meta['aux_shape'])}" if has_aux else "")
            + f", fp16, {num_frames} frames")

        backbone.train(was_training)
        return cls(cache_dir, meta, mode="r")