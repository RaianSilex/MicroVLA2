"""Train MicroVLA on metadata-driven heterogeneous demonstrations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config import vla_config as C
from data.vla_dataset import VocabBundle, build_vla_dataset
from model.finetune import (
    apply_freeze_mode,
    apply_lora,
    extend_vocabs,
    fill_robot_stats,
    load_finetune_state_dict,
    merge_stats,
    parameter_summary,
)
from model.vla_policy import build_vla_policy
from utils import AverageMeter, build_optimizer, format_meters, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MicroVLA on heterogeneous micromanipulation data.")
    p.add_argument("--episodes-dir", type=Path, default=C.VLA_EPISODES_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=C.VLA_CKPT_DIR)
    p.add_argument("--stats-path", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--lr-backbone", type=float, default=C.LR_BACKBONE)
    p.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--device", type=str, default=C.DEVICE)
    p.add_argument("--val-split", type=float, default=C.VAL_SPLIT)
    p.add_argument("--holdout-lab", type=str, default=None)
    p.add_argument("--holdout-robot", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--backbone",
        type=str,
        default=C.DEFAULT_BACKBONE,
        help="Image backbone, e.g. dinov2_vits14+cellpose4, dinov2_vits14+cellpose, or resnet18.",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet ResNet18 weights and Cellpose 4 cpsam weight loading "
             "(DINOv2 still uses torch.hub weights).",
    )
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--language-backend", choices=("hf", "simple"), default=C.LANGUAGE_BACKEND)
    p.add_argument("--text-model", type=str, default=C.DEFAULT_TEXT_MODEL)
    p.add_argument("--finetune", type=Path, default=None,
                   help="Pretrained MicroVLA checkpoint to start from. "
                        "Vocab and per-robot stats are extended to cover the "
                        "current dataset; old IDs are preserved. Mutually "
                        "exclusive with --resume in spirit (resume always wins "
                        "if both are given).")
    p.add_argument("--freeze-mode", choices=("none", "trunk", "head_only"),
                   default="none",
                   help="What to freeze on top of the always-frozen backbones. "
                        "'trunk' freezes the main transformer + style encoder; "
                        "'head_only' freezes everything except metadata "
                        "embeddings, action head, and any LoRA params.")
    p.add_argument("--lora-r", type=int, default=0,
                   help="LoRA rank applied to transformer FFN linears "
                        "(linear1/linear2). 0 disables LoRA.")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", type=str, default="transformer,style_encoder",
                   help="Comma-separated submodule names under policy.model "
                        "to wrap with LoRA (e.g. 'transformer,style_encoder').")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    return p.parse_args()


def _episode_split(full_ds, args):
    n_episodes = len(full_ds.episodes)
    all_episode_ids = set(range(n_episodes))

    if args.holdout_lab is not None:
        val_episode_ids = {
            i for i, ep in enumerate(full_ds.episodes) if ep.lab_id == args.holdout_lab
        }
        if not val_episode_ids:
            raise SystemExit(f"No episodes found for --holdout-lab {args.holdout_lab!r}")
    elif args.holdout_robot is not None:
        val_episode_ids = {
            i for i, ep in enumerate(full_ds.episodes) if ep.robot_id == args.holdout_robot
        }
        if not val_episode_ids:
            raise SystemExit(f"No episodes found for --holdout-robot {args.holdout_robot!r}")
    else:
        n_val = max(1, int(round(n_episodes * args.val_split)))
        if n_episodes - n_val < 1:
            raise SystemExit("VLA training needs at least 2 episodes for an episode-level split.")
        perm = torch.randperm(n_episodes, generator=torch.Generator().manual_seed(args.seed)).tolist()
        val_episode_ids = set(perm[:n_val])

    train_episode_ids = all_episode_ids - val_episode_ids
    if not train_episode_ids:
        raise SystemExit("No training episodes left after validation split.")

    train_idx = [i for i, (ei, _) in enumerate(full_ds.index) if ei in train_episode_ids]
    val_idx = [i for i, (ei, _) in enumerate(full_ds.index) if ei in val_episode_ids]
    return Subset(full_ds, train_idx), Subset(full_ds, val_idx), train_episode_ids, val_episode_ids


def run_epoch(policy, loader, optimizer, device, train: bool) -> dict:
    policy.train(train)
    meters: dict = defaultdict(AverageMeter)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        qpos = batch["qpos"].to(device, non_blocking=True)
        state_mask = batch["state_mask"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        robot_id = batch["robot_id"].to(device, non_blocking=True)
        lab_id = batch["lab_id"].to(device, non_blocking=True)
        embodiment_id = batch["embodiment_id"].to(device, non_blocking=True)
        action_type_id = batch["action_type_id"].to(device, non_blocking=True)
        task_family_id = batch["task_family_id"].to(device, non_blocking=True)
        instructions = batch["instruction"]

        if train:
            loss_dict = policy(
                image,
                qpos,
                instructions,
                robot_id,
                lab_id,
                embodiment_id,
                action_type_id,
                task_family_id,
                state_mask=state_mask,
                action_mask=action_mask,
                actions=action,
                is_pad=is_pad,
            )
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = policy(
                    image,
                    qpos,
                    instructions,
                    robot_id,
                    lab_id,
                    embodiment_id,
                    action_type_id,
                    task_family_id,
                    state_mask=state_mask,
                    action_mask=action_mask,
                    actions=action,
                    is_pad=is_pad,
                )

        bs = qpos.size(0)
        for k, v in loss_dict.items():
            meters[k].update(v.item(), bs)

    return meters


def save_vla_checkpoint(path: Path, policy, optimizer, epoch: int, best_val: float, full_ds, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "stats": full_ds.stats,
            "vocabs": full_ds.vocabs.as_dict(),
            "config": {
                "max_state_dim": C.MAX_STATE_DIM,
                "max_action_dim": C.MAX_ACTION_DIM,
                "chunk_size": C.CHUNK_SIZE,
                "image_height": C.IMAGE_HEIGHT,
                "image_width": C.IMAGE_WIDTH,
                "backbone": args.backbone,
                "language_backend": args.language_backend,
                "text_model": args.text_model,
                "pretrained_backbone": not args.no_pretrained,
                "freeze_backbone": not args.unfreeze_backbone,
                "freeze_mode": args.freeze_mode,
                "lora_r": int(args.lora_r),
                "lora_alpha": float(args.lora_alpha),
                "lora_targets": args.lora_targets,
                "lora_dropout": float(args.lora_dropout),
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    stats_path = args.stats_path or (args.ckpt_dir / "vla_stats.pkl")
    full_ds = build_vla_dataset(
        episodes_dir=args.episodes_dir,
        stats_path=stats_path,
        recompute_stats=True,
    )
    train_ds, val_ds, train_eps, val_eps = _episode_split(full_ds, args)
    val_names = [full_ds.episodes[i].episode_id for i in sorted(val_eps)]
    print(f"dataset: train={len(train_ds)} val={len(val_ds)} total={len(full_ds)}")
    print(f"validation episodes: {val_names}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ------------------------------------------------------------------
    # Policy construction. Three mutually-related modes:
    #   * --resume <ckpt>   : continue a previous (possibly LoRA/frozen) run.
    #                         Architecture (incl. freeze + LoRA) is rebuilt
    #                         from the resumed checkpoint's saved config.
    #   * --finetune <ckpt> : load a pretrained ckpt, extend vocabs+stats
    #                         to cover the new dataset, partial-load weights,
    #                         apply freezing + LoRA from CLI flags.
    #   * neither           : build from scratch on the new dataset.
    # If both --resume and --finetune are supplied, --resume wins.
    # ------------------------------------------------------------------
    resume_ckpt = None
    if args.resume is not None and args.resume.exists():
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

    if resume_ckpt is not None:
        # Resume rebuilds the exact prior architecture.
        rcfg = resume_ckpt.get("config", {})
        for attr in ("freeze_mode", "lora_r", "lora_alpha", "lora_targets", "lora_dropout"):
            if attr in rcfg:
                setattr(args, attr, rcfg[attr])
        full_ds.vocabs = VocabBundle(**resume_ckpt["vocabs"])
        full_ds.stats = resume_ckpt["stats"]
        policy = build_vla_policy(
            stats=full_ds.stats,
            vocabs=full_ds.vocabs,
            pretrained_backbone=not args.no_pretrained,
            backbone_name=args.backbone,
            freeze_backbone=not args.unfreeze_backbone,
            language_backend=args.language_backend,
            text_model_name=args.text_model,
        ).to(device)
        if args.freeze_mode != "none":
            apply_freeze_mode(policy, args.freeze_mode)
        if int(args.lora_r) > 0:
            targets = tuple(t.strip() for t in str(args.lora_targets).split(",") if t.strip())
            apply_lora(policy, r=int(args.lora_r), alpha=float(args.lora_alpha),
                       targets=targets, dropout=float(args.lora_dropout))
        policy.to(device)
        policy.load_state_dict(resume_ckpt["policy"])
        print(f"[resume] loaded {args.resume}  ({parameter_summary(policy)})")
    elif args.finetune is not None and args.finetune.exists():
        pre = torch.load(args.finetune, map_location=device, weights_only=False)
        pre_vocabs = VocabBundle(**pre["vocabs"]) if not isinstance(
            pre["vocabs"], VocabBundle) else pre["vocabs"]
        ext_vocabs = extend_vocabs(pre_vocabs, full_ds.episodes)
        merged = merge_stats(pre["stats"], full_ds.stats)
        # Swap the dataset's view so per-sample IDs come from the extended vocab
        # and per-robot normalization uses the merged stats.
        full_ds.vocabs = ext_vocabs
        full_ds.stats = merged
        policy = build_vla_policy(
            stats=merged,
            vocabs=ext_vocabs,
            pretrained_backbone=not args.no_pretrained,
            backbone_name=args.backbone,
            freeze_backbone=not args.unfreeze_backbone,
            language_backend=args.language_backend,
            text_model_name=args.text_model,
        ).to(device)
        load_finetune_state_dict(policy, pre["policy"], skip_patterns=("_table",))
        fill_robot_stats(policy, ext_vocabs, merged)
        if args.freeze_mode != "none":
            apply_freeze_mode(policy, args.freeze_mode)
        if int(args.lora_r) > 0:
            targets = tuple(t.strip() for t in str(args.lora_targets).split(",") if t.strip())
            apply_lora(policy, r=int(args.lora_r), alpha=float(args.lora_alpha),
                       targets=targets, dropout=float(args.lora_dropout))
        policy.to(device)
        new_robots = sorted(set(ext_vocabs.robot_ids) - set(pre_vocabs.robot_ids))
        print(f"[finetune] loaded {args.finetune}; new robots in this dataset: {new_robots}")
    else:
        policy = build_vla_policy(
            stats=full_ds.stats,
            vocabs=full_ds.vocabs,
            pretrained_backbone=not args.no_pretrained,
            backbone_name=args.backbone,
            freeze_backbone=not args.unfreeze_backbone,
            language_backend=args.language_backend,
            text_model_name=args.text_model,
        ).to(device)
    print(
        f"backbone: {args.backbone}  language={args.language_backend}:{args.text_model}  "
        f"freeze={args.freeze_mode}  lora_r={args.lora_r}  {parameter_summary(policy)}"
    )
    optimizer = build_optimizer(policy, args.lr, args.lr_backbone, args.weight_decay)

    ckpt_last = args.ckpt_dir / "vla_policy_last.pt"
    ckpt_best = args.ckpt_dir / "vla_policy_best.pt"
    start_epoch = 0
    best_val = float("inf")

    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            except (ValueError, KeyError) as e:
                # Param shapes / counts differ from saved optimizer state.
                # Falls back to a fresh optimizer.
                print(f"[resume] optimizer state ignored ({e}); starting fresh AdamW state")
        start_epoch = int(resume_ckpt.get("epoch") or 0)
        best_val = float(resume_ckpt.get("best_val", best_val))
        print(f"resumed at epoch {start_epoch}; best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        tr = run_epoch(policy, train_loader, optimizer, device, train=True)
        vl = run_epoch(policy, val_loader, optimizer, device, train=False)

        print(
            f"[epoch {epoch+1:4d}/{args.epochs}] "
            f"train {format_meters(tr)}  |  val {format_meters(vl)}"
        )

        val_loss = vl["loss"].avg
        if val_loss < best_val:
            best_val = val_loss
            save_vla_checkpoint(ckpt_best, policy, optimizer, epoch + 1, best_val, full_ds, args)
        save_vla_checkpoint(ckpt_last, policy, optimizer, epoch + 1, best_val, full_ds, args)
        if (epoch + 1) % args.save_every == 0:
            save_vla_checkpoint(
                args.ckpt_dir / f"vla_policy_epoch{epoch+1}.pt",
                policy,
                optimizer,
                epoch + 1,
                best_val,
                full_ds,
                args,
            )

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
