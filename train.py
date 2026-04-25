"""Train MicroACT on the Sensapex dataset.

Examples:
    python train.py                                # defaults from config.config
    python train.py --epochs 200 --batch-size 16
    python train.py --resume checkpoints/policy_last.pt
    python train.py --no-pretrained                # skip ImageNet weight download
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from config import config as C
from data.dataset import build_dataset
from model.act_policy import build_policy
from utils import (
    AverageMeter,
    build_optimizer,
    format_meters,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=C.NUM_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=C.BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=C.LR)
    p.add_argument("--lr-backbone",  type=float, default=C.LR_BACKBONE)
    p.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    p.add_argument("--seed",         type=int,   default=C.SEED)
    p.add_argument("--device",       type=str,   default=C.DEVICE)
    p.add_argument("--val-split",    type=float, default=0.1)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--save-every",   type=int,   default=100,
                   help="Save a numbered checkpoint every N epochs (in addition to last/best).")
    p.add_argument("--ckpt-dir",     type=Path,  default=C.CKPT_DIR)
    p.add_argument("--resume",       type=Path,  default=None)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable ImageNet pretrained weights on the backbone.")
    return p.parse_args()


def run_epoch(policy, loader, optimizer, device, train: bool) -> dict:
    policy.train(train)
    meters: dict = defaultdict(AverageMeter)

    for batch in loader:
        image  = batch["image"].to(device, non_blocking=True)
        qpos   = batch["qpos"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)

        if train:
            loss_dict = policy(image, qpos, action, is_pad)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = policy(image, qpos, action, is_pad)

        bs = qpos.size(0)
        for k, v in loss_dict.items():
            meters[k].update(v.item(), bs)

    return meters


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    # ---- Data ----
    full_ds = build_dataset(recompute_stats=True)
    val_n = max(1, int(round(len(full_ds) * args.val_split)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"dataset: train={len(train_ds)} val={len(val_ds)}  (total {len(full_ds)})")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ---- Policy + optimizer ----
    policy = build_policy(
        stats=full_ds.norm_stats,
        pretrained_backbone=not args.no_pretrained,
    ).to(device)
    optimizer = build_optimizer(policy, args.lr, args.lr_backbone, args.weight_decay)

    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        start_epoch = load_checkpoint(args.resume, policy, optimizer, map_location=device)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # ---- Loop ----
    ckpt_last = args.ckpt_dir / "policy_last.pt"
    ckpt_best = args.ckpt_dir / "policy_best.pt"

    for epoch in range(start_epoch, args.epochs):
        tr = run_epoch(policy, train_loader, optimizer, device, train=True)
        vl = run_epoch(policy, val_loader,   optimizer, device, train=False)

        print(
            f"[epoch {epoch+1:4d}/{args.epochs}] "
            f"train {format_meters(tr)}  |  val {format_meters(vl)}"
        )

        save_checkpoint(ckpt_last, policy, optimizer, epoch + 1)
        if vl["loss"].avg < best_val:
            best_val = vl["loss"].avg
            save_checkpoint(ckpt_best, policy, optimizer, epoch + 1)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.ckpt_dir / f"policy_epoch{epoch+1}.pt",
                policy, optimizer, epoch + 1,
            )

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
