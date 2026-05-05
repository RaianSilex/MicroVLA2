"""Helpers for finetuning a pretrained MicroVLA checkpoint on a new dataset.

Typical workflow:

    pretrained_ckpt = torch.load("vla_pretrained.pt", weights_only=False)
    pre_vocabs = VocabBundle(**pretrained_ckpt["vocabs"])
    pre_stats = pretrained_ckpt["stats"]

    new_ds = build_vla_dataset(...)                        # finetuner's data
    ext_vocabs = extend_vocabs(pre_vocabs, new_ds.episodes)
    merged = merge_stats(pre_stats, new_ds.stats)

    policy = build_vla_policy(stats=merged, vocabs=ext_vocabs, **build_kwargs)
    load_finetune_state_dict(policy, pretrained_ckpt["policy"], skip_patterns=("_table",))
    fill_robot_stats(policy, ext_vocabs, merged)
    apply_freeze_mode(policy, "trunk")
    apply_lora(policy, r=8, alpha=16.0)

The training script `train_vla.py` exposes this via `--finetune <ckpt>` plus
`--freeze-mode`, `--lora-r`, `--lora-alpha`, `--lora-targets`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import vla_config as C
from data.vla_dataset import VLAEpisode, VocabBundle


# ---------------------------------------------------------------------------
# Vocab + stats extension (preserve old IDs so old embedding rows remain valid)
# ---------------------------------------------------------------------------

def extend_vocab(old: Dict[str, int], new_values: Iterable[str]) -> Dict[str, int]:
    """Return a vocab dict that contains every old (name -> id) plus any
    previously-unseen names appended at fresh IDs (sorted for determinism).

    Old IDs are preserved verbatim, so an embedding row trained for
    'lab_a/sensapex_dual' keeps its meaning after extension.
    """
    out = dict(old)
    additions = sorted({str(v) for v in new_values if str(v) not in out})
    for value in additions:
        out[value] = len(out)
    return out


def extend_vocabs(old: VocabBundle, episodes: Sequence[VLAEpisode]) -> VocabBundle:
    """Extend all five vocab dicts with whatever new entries the new episodes use."""
    return VocabBundle(
        robot_ids=extend_vocab(old.robot_ids, (e.robot_id for e in episodes)),
        lab_ids=extend_vocab(old.lab_ids, (e.lab_id for e in episodes)),
        embodiment_ids=extend_vocab(old.embodiment_ids, (e.embodiment for e in episodes)),
        action_type_ids=extend_vocab(old.action_type_ids, (e.action_type for e in episodes)),
        task_family_ids=extend_vocab(old.task_family_ids, (e.task_family for e in episodes)),
    )


def merge_stats(old_stats: dict, new_stats: dict) -> dict:
    """Combine per-robot stats. New stats win where the same robot appears in both."""
    by_robot: Dict[str, dict] = dict(old_stats.get("by_robot", {}))
    by_robot.update(new_stats.get("by_robot", {}))
    return {
        "by_robot": by_robot,
        "image_mean": new_stats["image_mean"],
        "image_std": new_stats["image_std"],
    }


def fill_robot_stats(policy: nn.Module, vocabs: VocabBundle, stats: dict) -> None:
    """Overwrite per-robot rows of the policy's normalization buffers in-place.

    Used after `load_finetune_state_dict(skip_patterns=('_table',))` to make sure
    the merged stats end up in the buffers (rather than the pretrained-only ones).
    """
    by_robot = stats.get("by_robot", {})
    with torch.no_grad():
        for name, rid in vocabs.robot_ids.items():
            if name == C.UNKNOWN_TOKEN or name not in by_robot:
                continue
            rs = by_robot[name]
            policy.qpos_mean_table[rid] = torch.from_numpy(
                rs["qpos_mean"]).to(policy.qpos_mean_table.device)
            policy.qpos_std_table[rid] = torch.from_numpy(
                rs["qpos_std"]).to(policy.qpos_std_table.device)
            policy.action_mean_table[rid] = torch.from_numpy(
                rs["action_mean"]).to(policy.action_mean_table.device)
            policy.action_std_table[rid] = torch.from_numpy(
                rs["action_std"]).to(policy.action_std_table.device)


# ---------------------------------------------------------------------------
# Partial state-dict loader (handles grown embeddings via corner-copy)
# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    matched: List[str]
    partial: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]  # (key, ckpt_shape, model_shape)
    skipped: List[Tuple[str, str]]                               # (key, reason)

    def summary(self) -> str:
        return (f"matched={len(self.matched)} "
                f"partial-copied={len(self.partial)} "
                f"skipped={len(self.skipped)}")


def load_finetune_state_dict(
    model: nn.Module,
    ckpt_state_dict: dict,
    skip_patterns: Sequence[str] = (),
    verbose: bool = True,
) -> LoadReport:
    """Load `ckpt_state_dict` into `model` with three behaviors per key:

    * exact shape match  -> copy directly.
    * model tensor is element-wise >= ckpt tensor in every dim -> copy
      ckpt tensor into the leading corner; new rows/cols keep their init.
      This is the "embedding grew" case.
    * any substring in `skip_patterns` matches the key -> skip silently.
    * anything else -> skip with a logged reason.
    """
    own = model.state_dict()
    new_state = dict(own)
    matched: List[str] = []
    partial: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    skipped: List[Tuple[str, str]] = []

    for k, v in ckpt_state_dict.items():
        if any(pat in k for pat in skip_patterns):
            skipped.append((k, "skip-pattern"))
            continue
        if k not in own:
            skipped.append((k, "not in model"))
            continue
        target = own[k]
        if target.shape == v.shape:
            new_state[k] = v
            matched.append(k)
            continue
        if (target.dim() == v.dim()
                and all(t >= s for t, s in zip(target.shape, v.shape))):
            grown = target.clone()
            slices = tuple(slice(0, s) for s in v.shape)
            grown[slices] = v.to(grown.device, dtype=grown.dtype)
            new_state[k] = grown
            partial.append((k, tuple(v.shape), tuple(target.shape)))
            continue
        skipped.append((k, f"shape {tuple(v.shape)} vs model {tuple(target.shape)}"))

    model.load_state_dict(new_state, strict=True)

    if verbose:
        print(f"[finetune-load] {LoadReport(matched, partial, skipped).summary()}")
        for k, src, dst in partial:
            print(f"  partial-copy: {k}  ckpt {src} -> model {dst}")
        for k, reason in skipped[:20]:
            print(f"  skipped: {k}  ({reason})")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more skipped")

    return LoadReport(matched=matched, partial=partial, skipped=skipped)


# ---------------------------------------------------------------------------
# Selective freezing
# ---------------------------------------------------------------------------

def freeze_modules(*modules: nn.Module) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False


def apply_freeze_mode(policy: nn.Module, mode: str) -> None:
    """Set requires_grad on policy submodules based on the named mode.

    Modes:
      none       -> no extra freezing (image+language backbones are still
                    frozen by their own constructors when freeze=True).
      trunk      -> freeze the main transformer + style encoder. Embeddings
                    (including new vocab rows), projections, action head,
                    and any LoRA params remain trainable.
      head_only  -> freeze almost everything; only train metadata embeddings,
                    the action head, and LoRA params if present.
    """
    mode = str(mode).lower()
    inner = policy.model  # VLACVAE
    if mode == "none":
        return
    if mode == "trunk":
        freeze_modules(inner.transformer, inner.style_encoder)
        return
    if mode == "head_only":
        freeze_modules(
            inner.transformer,
            inner.style_encoder,
            inner.backbone,
            inner.language_encoder,
            inner.cls_embed,
            inner.style_qpos_proj,
            inner.style_action_proj,
            inner.style_pos_embed,
            inner.latent_proj,
            inner.latent_to_src,
            inner.qpos_to_src,
            inner.extra_src_pos,
            inner.query_embed,
        )
        return
    raise ValueError(f"Unknown freeze mode: {mode!r}")


# ---------------------------------------------------------------------------
# LoRA wrapping for nn.Linear (transformer FFN)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank trainable update.

    Forward computes `base(x) + (x @ A^T @ B^T) * (alpha / r)`. The base
    layer's weights and bias are frozen; A is initialized via Kaiming uniform
    and B is zero so the initial output equals the base layer's output.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = (self.dropout(x) @ self.A.T) @ self.B.T
        return out + update * self.scaling


def apply_lora(
    policy: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    targets: Sequence[str] = ("transformer", "style_encoder"),
    layer_name_substrings: Sequence[str] = ("linear1", "linear2"),
    dropout: float = 0.0,
    verbose: bool = True,
) -> int:
    """Wrap nn.Linear modules under `policy.model.<target>` with LoRALinear.

    Only modules whose attribute name contains any of `layer_name_substrings`
    are wrapped (default: transformer FFN linears). Returns the number of
    layers wrapped.

    LoRA on nn.MultiheadAttention's QKV projection is intentionally not
    handled here because MultiheadAttention uses a fused weight rather than
    nn.Linear; that would need a separate replacement module.
    """
    if r <= 0:
        return 0
    inner = policy.model
    swapped = 0
    for tname in targets:
        root = getattr(inner, tname, None)
        if root is None:
            continue
        for name, sub in list(root.named_modules()):
            if not isinstance(sub, nn.Linear):
                continue
            attr = name.rsplit(".", 1)[-1]
            if not any(s in attr for s in layer_name_substrings):
                continue
            parent_path, _, child = name.rpartition(".")
            parent = root if parent_path == "" else root.get_submodule(parent_path)
            setattr(parent, child, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout))
            swapped += 1
    if verbose:
        print(f"[finetune-lora] wrapped {swapped} Linear layers "
              f"(r={r}, alpha={alpha}, targets={list(targets)})")
    return swapped


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def parameter_summary(policy: nn.Module) -> str:
    """One-line trainable / total parameter count for the policy."""
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    return (f"trainable={trainable / 1e6:.2f}M / total={total / 1e6:.2f}M "
            f"({100.0 * trainable / max(total, 1):.2f}%)")
