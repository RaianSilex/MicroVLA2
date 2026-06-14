"""Offline diagnostic for a MicroVLA checkpoint (no hardware / ROS).

Question answered: does this policy condition its output on the image + state +
language, or has it collapsed to predicting the dataset-mean action regardless of
input?

It runs recorded frames through the policy (``actions=None``, the deployment path)
and compares the predicted action chunk to the logged ground truth against a
"predict the dataset mean" baseline. Everything is in normalized space, where the
mean baseline is simply zero, so:

    ratio = mean|a_hat - a_gt| / mean|a_gt|

``ratio ~ 1.0`` => no better than predicting the mean (not conditioning on inputs);
``ratio << 1.0`` => the policy is using its inputs.

If the contact-point goal head is enabled, it also reports the correlation between
the predicted goal mean and the true contact point per active axis.

Run from the repo root:
    python -m rollout.offline_replay \
        --checkpoint checkpoints_vla/vla_policy_best.pt \
        --dataset-repo-id RaianSilex/microvla_ump_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import vla_config as C
from model.vla_policy import build_vla_policy


def _move(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(policy, dataset, device, num_samples: int = 64, seed: int = 0) -> dict:
    """Run the policy over sampled frames and measure conditioning vs mean-collapse.

    ``dataset`` yields the MicroVLA per-sample dict (LeRobotVLADataset or a mock).
    Returns a dict of metrics and prints a readable report.
    """
    policy.eval()
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_samples, n), replace=False)
    batch = _move(default_collate([dataset[int(i)] for i in idx]), device)

    image = batch.get("image")
    a_hat, goal_params, cell_params, _ = policy.model(
        image,
        batch["qpos"],
        batch["instruction"],
        batch["robot_id"],
        batch["lab_id"],
        batch["embodiment_id"],
        batch["action_type_id"],
        batch["task_family_id"],
        state_mask=batch.get("state_mask"),
        action_mask=batch.get("action_mask"),
        resistance=batch.get("resistance"),
        img_primary_feat=batch.get("primary_feat"),
        img_aux_feat=batch.get("aux_feat"),
    )

    gt = batch["action"]                                  # (B, k, A) normalized
    valid = (~batch["is_pad"]).unsqueeze(-1).float() * batch["action_mask"].unsqueeze(1).float()
    denom = valid.sum(dim=(0, 1)).clamp_min(1.0)          # (A,)
    err_policy = ((a_hat - gt).abs() * valid).sum(dim=(0, 1)) / denom
    err_mean = (gt.abs() * valid).sum(dim=(0, 1)) / denom  # baseline = predict 0 (the mean)
    ratio = err_policy / err_mean.clamp_min(1e-6)

    active = (batch["action_mask"][0] & (err_mean > 1e-4)).cpu().numpy()
    err_policy_np = err_policy.cpu().numpy()
    err_mean_np = err_mean.cpu().numpy()
    ratio_np = ratio.cpu().numpy()

    print(f"\nSamples: {len(idx)} | device={device}")
    print("=== normalized one-chunk action error: policy vs predict-mean baseline ===")
    print(f"{'axis':>4} {'policy':>9} {'meanbase':>9} {'ratio':>6}")
    for a in range(gt.shape[-1]):
        if not active[a]:
            continue
        print(f"{a:>4} {err_policy_np[a]:9.3f} {err_mean_np[a]:9.3f} {ratio_np[a]:6.2f}")
    active_ratio = float(ratio_np[active].mean()) if active.any() else float("nan")
    print(f"{'ALL':>4} {'':9} {'':9} {active_ratio:6.2f}   (mean over active axes)")
    print("ratio ~1.0 => mean-collapse (inputs ignored); ratio << 1.0 => conditioning on inputs")

    metrics = {"ratio_per_axis": ratio_np, "active_axes": active, "active_ratio": active_ratio}

    if goal_params is not None and "goal" in batch:
        goal_mu = goal_params[0].cpu().numpy()           # (B, A)
        goal_gt = batch["goal"].cpu().numpy()            # (B, A)
        print("\n=== contact-point goal: corr(true, predicted) per active axis ===")
        corrs = {}
        for a in range(goal_gt.shape[-1]):
            if not active[a]:
                continue
            if goal_gt[:, a].std() > 1e-6 and goal_mu[:, a].std() > 1e-6:
                r = float(np.corrcoef(goal_gt[:, a], goal_mu[:, a])[0, 1])
            else:
                r = float("nan")
            corrs[a] = r
            print(f"  axis {a}: r={r:+.2f}  (pred std={goal_mu[:, a].std():.3f}, "
                  f"true std={goal_gt[:, a].std():.3f})")
        metrics["goal_corr"] = corrs

    if cell_params is not None and "goal_pixel" in batch and "target_region" in batch:
        select_logits, cell_mu, _ = cell_params
        pred_region = select_logits.argmax(dim=-1)
        true_region = batch["target_region"]
        sel_acc = float((pred_region == true_region).float().mean())
        pix_err = float((cell_mu - batch["goal_pixel"]).abs().mean())
        print("\n=== cell-aware heads (Variant B) ===")
        print(f"  selection accuracy: {sel_acc:.2f}  (chance ~{1.0 / select_logits.shape[-1]:.2f})")
        print(f"  contact-point |pred-true| (normalized px): {pix_err:.3f}")
        metrics["cell_select_acc"] = sel_acc
        metrics["cell_pixel_err"] = pix_err

    return metrics


def load_policy(checkpoint: Path, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    policy = build_vla_policy(
        stats=ckpt["stats"],
        vocabs=ckpt["vocabs"],
        pretrained_backbone=False,
        backbone_name=cfg.get("backbone", C.DEFAULT_BACKBONE),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        language_backend=cfg.get("language_backend", C.LANGUAGE_BACKEND),
        text_model_name=cfg.get("text_model", C.DEFAULT_TEXT_MODEL),
        action_space=cfg.get("action_space", C.DEFAULT_ACTION_SPACE),
        chunk_size=int(cfg.get("chunk_size", C.CHUNK_SIZE)),
        goal_head=bool(cfg.get("goal_head", C.GOAL_HEAD)),
        use_resistance=bool(cfg.get("use_resistance", False)),
        cell_head=bool(cfg.get("cell_head", False)),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-repo-id", type=str, default=C.DEFAULT_DATASET_REPO_ID)
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from data.lerobot_vla_dataset import build_lerobot_vla_dataset

    policy, cfg = load_policy(args.checkpoint, args.device)
    dataset = build_lerobot_vla_dataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        action_space=cfg.get("action_space", C.DEFAULT_ACTION_SPACE),
        chunk_size=int(cfg.get("chunk_size", C.CHUNK_SIZE)),
    )
    evaluate(policy, dataset, args.device, num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
