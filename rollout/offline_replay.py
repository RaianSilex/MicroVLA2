"""Offline replay diagnostic for a MicroACT checkpoint.

Question answered: does this policy actually condition its output on the
recorded image + qpos, or has it collapsed to predicting the dataset-mean
action regardless of input?

No hardware/ROS needed. Feeds recorded TRAINING frames + qpos through
`policy.inference` (teacher-forced, using the exact training image loader) and
compares predictions to the logged ground-truth targets against a constant
"predict dataset mean" baseline.

Run from the repo root:
    python -m rollout.offline_replay \
        --checkpoint checkpoints/microact_resnet_66epi_200epoch/policy_best.pt \
        --stats-path checkpoints/microact_resnet_66epi_200epoch/dataset_stats.pkl \
        --backbone resnet18
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config as C
from data.dataset import (
    _load_image,
    _resolve_image_path,
    discover_trials,
    load_trial,
)
from model.act_policy import build_policy
from utils import load_checkpoint

AXES = ["x1", "y1", "z1", "d1", "x2", "y2", "z2", "d2"]
ACTIVE = [0, 1, 4, 5]  # x1, y1, x2, y2 — the axes that actually move in the demos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-trials", type=int, default=8)
    p.add_argument("--samples-per-trial", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_policy(args) -> tuple:
    with open(args.stats_path, "rb") as f:
        stats = pickle.load(f)
    policy = build_policy(
        stats=stats,
        pretrained_backbone=False,
        backbone_name=args.backbone,
        freeze_backbone=True,
    ).to(args.device)
    load_checkpoint(args.checkpoint, policy, map_location=args.device)
    policy.eval()
    return policy, stats


def main() -> None:
    args = parse_args()
    policy, stats = load_policy(args)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)

    trials = [load_trial(p) for p in discover_trials()]
    trials = [t for t in trials if t.length >= 5]
    # Pick trials spread across their TRUE endpoint x1 so we cover diverse targets.
    trials.sort(key=lambda t: float(t.actions[-1, 0]))
    pick = np.linspace(0, len(trials) - 1, min(args.num_trials, len(trials))).astype(int)
    chosen = [trials[i] for i in pick]

    H, W = C.IMAGE_HEIGHT, C.IMAGE_WIDTH
    rng = np.random.default_rng(args.seed)

    preds, gts = [], []
    endpoints = []  # (trial_id, true_end_8d, pred_end_8d)

    for tr in chosen:
        ts = rng.choice(tr.length, size=min(args.samples_per_trial, tr.length), replace=False)
        for t in sorted(ts):
            img = _load_image(_resolve_image_path(tr.image_paths[t], tr.trial_id, t), H, W)
            chunk = policy.inference(img, tr.states[t].astype(np.float32))  # (k, 8)
            preds.append(chunk[0])
            gts.append(tr.actions[t])

        # Where does the policy think this trial ends? Use an early frame and
        # read the last action of the predicted chunk.
        t0 = max(0, tr.length // 5)
        img = _load_image(_resolve_image_path(tr.image_paths[t0], tr.trial_id, t0), H, W)
        chunk = policy.inference(img, tr.states[t0].astype(np.float32))
        endpoints.append((tr.trial_id, tr.actions[-1], chunk[-1]))

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    err_policy = np.abs(preds - gts).mean(0)
    err_mean = np.abs(action_mean[None, :] - gts).mean(0)

    print(f"\nSamples: {len(preds)} from {len(chosen)} trials | device={args.device}")
    print("\n=== One-step action error (counts): policy vs 'predict dataset mean' ===")
    print(f"{'axis':>4} {'policy':>9} {'meanbase':>9} {'ratio':>6}")
    for i, a in enumerate(AXES):
        print(f"{a:>4} {err_policy[i]:9.1f} {err_mean[i]:9.1f} "
              f"{err_policy[i] / max(err_mean[i], 1e-6):6.2f}")
    print(f"{'ALL':>4} {err_policy.mean():9.1f} {err_mean.mean():9.1f} "
          f"{err_policy.mean() / max(err_mean.mean(), 1e-6):6.2f}")
    print("ratio ~1.0 => no better than predicting the mean (not conditioning);"
          " ratio << 1.0 => using the inputs")

    print("\n=== Predicted endpoint vs true endpoint (active axes) ===")
    print(f"{'trial':>6} {'x1_true':>8}{'x1_pred':>8} {'y1_true':>8}{'y1_pred':>8} "
          f"{'x2_true':>8}{'x2_pred':>8} {'y2_true':>8}{'y2_pred':>8}")
    te, pe = [], []
    for tid, tru, prd in endpoints:
        te.append(tru)
        pe.append(prd)
        print(f"{tid:>6} {tru[0]:8.0f}{prd[0]:8.0f} {tru[1]:8.0f}{prd[1]:8.0f} "
              f"{tru[4]:8.0f}{prd[4]:8.0f} {tru[5]:8.0f}{prd[5]:8.0f}")
    te = np.asarray(te)
    pe = np.asarray(pe)

    print("\ncorr(true_endpoint, pred_endpoint) on active axes:")
    for i in ACTIVE:
        if te[:, i].std() > 1e-6 and pe[:, i].std() > 1e-6:
            r = float(np.corrcoef(te[:, i], pe[:, i])[0, 1])
        else:
            r = float("nan")
        print(f"  {AXES[i]}: r={r:+.2f}  (pred std={pe[:, i].std():6.0f}, "
              f"true std={te[:, i].std():6.0f})")
    print("\nLow pred std + r~0 => policy emits ~the same endpoint regardless of trial"
          " => image ignored / mean collapse.")


if __name__ == "__main__":
    main()
