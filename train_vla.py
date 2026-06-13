"""Train MicroVLA on a LeRobot-format micromanipulation dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config import vla_config as C
from data.vocab import VocabBundle
from data.lerobot_vla_dataset import build_lerobot_vla_dataset
from data.feature_cache import FeatureCache
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
    p = argparse.ArgumentParser(description="Train MicroVLA on a LeRobot micromanipulation dataset.")
    p.add_argument("--dataset-repo-id", type=str, default=C.DEFAULT_DATASET_REPO_ID,
                   help="LeRobot dataset repo id (SmolVLA/OpenPI-style).")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="Local root for the LeRobot dataset. Default: HF_LEROBOT_HOME/<repo-id>.")
    p.add_argument("--action-space", choices=("delta", "absolute"), default=C.DEFAULT_ACTION_SPACE,
                   help="'delta' (relative to base state) is recommended; inference converts it "
                        "back to absolute. 'absolute' trains on raw targets.")
    p.add_argument("--ckpt-dir", type=Path, default=C.VLA_CKPT_DIR)
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
    p.add_argument("--chunk-size", type=int, default=C.CHUNK_SIZE,
                   help="Action chunk length. Read from the checkpoint on resume/finetune.")
    p.add_argument("--goal-weight", type=float, default=C.GOAL_LOSS_WEIGHT,
                   help="Weight on the contact-point Gaussian NLL. 0 disables that term.")
    p.add_argument("--no-goal-head", dest="goal_head", action="store_false",
                   help="Disable the contact-point Gaussian head entirely.")
    p.set_defaults(goal_head=C.GOAL_HEAD)
    p.add_argument(
        "--cache-features", action="store_true",
        help="Precompute and memmap the frozen image-encoder features once, then train on the "
             "cache. Removes the per-step video decode and the frozen ViT forward passes. Only "
             "valid with a frozen backbone (ignored if --unfreeze-backbone is set).",
    )
    p.add_argument("--feature-cache-dir", type=Path, default=None,
                   help="Where to store/read the feature cache. Default: "
                        "<ckpt-dir>/feat_cache_<backbone>.")
    p.add_argument("--precompute-batch", type=int, default=32,
                   help="Batch size for the one-time feature-cache precompute pass.")
    p.add_argument("--amp", action="store_true",
                   help="Run the forward pass under bf16 autocast (A100/H100). Loss/backward "
                        "stay in fp32.")
    p.add_argument("--backbone", type=str, default=C.DEFAULT_BACKBONE,
                   help="Image backbone, e.g. dinov2_vits14+cellpose4, dinov2_vits14+cellpose, "
                        "resnet18.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable ImageNet ResNet18 weights and Cellpose 4 cpsam weight loading "
                        "(DINOv2 still uses torch.hub weights).")
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="Fine-tune the image backbone end-to-end (recommended on a single rig).")
    p.add_argument("--language-backend", choices=("hf", "simple"), default=C.LANGUAGE_BACKEND)
    p.add_argument("--text-model", type=str, default=C.DEFAULT_TEXT_MODEL)
    p.add_argument("--finetune", type=Path, default=None,
                   help="Pretrained MicroVLA checkpoint to start from. Vocab and per-robot stats "
                        "are extended to cover the current dataset; old IDs are preserved. "
                        "--resume wins if both are given.")
    p.add_argument("--freeze-mode", choices=("none", "trunk", "head_only"), default="none",
                   help="What to freeze on top of the (optionally frozen) backbone.")
    p.add_argument("--lora-r", type=int, default=0,
                   help="LoRA rank on transformer FFN linears (linear1/linear2). 0 disables LoRA.")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", type=str, default="transformer,style_encoder",
                   help="Comma-separated submodules under policy.model to wrap with LoRA.")
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


def run_epoch(policy, loader, optimizer, device, train: bool, amp: bool = False) -> dict:
    policy.train(train)
    meters: dict = defaultdict(AverageMeter)

    use_amp = amp and str(device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext

    for batch in loader:
        qpos = batch["qpos"].to(device, non_blocking=True)
        state_mask = batch["state_mask"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        action_mask = batch["action_mask"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        goal = batch["goal"].to(device, non_blocking=True)
        robot_id = batch["robot_id"].to(device, non_blocking=True)
        lab_id = batch["lab_id"].to(device, non_blocking=True)
        embodiment_id = batch["embodiment_id"].to(device, non_blocking=True)
        action_type_id = batch["action_type_id"].to(device, non_blocking=True)
        task_family_id = batch["task_family_id"].to(device, non_blocking=True)
        instructions = batch["instruction"]
        resistance = batch["resistance"].to(device, non_blocking=True) if "resistance" in batch else None

        # Either raw frames (decode path) or cached frozen-encoder features.
        if "primary_feat" in batch:
            image = None
            img_primary_feat = batch["primary_feat"].to(device, non_blocking=True)
            img_aux_feat = (
                batch["aux_feat"].to(device, non_blocking=True) if "aux_feat" in batch else None
            )
        else:
            image = batch["image"].to(device, non_blocking=True)
            img_primary_feat = img_aux_feat = None

        grad_ctx = nullcontext() if train else torch.no_grad()
        with grad_ctx, amp_ctx():
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
                goal=goal,
                resistance=resistance,
                img_primary_feat=img_primary_feat,
                img_aux_feat=img_aux_feat,
            )

        if train:
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()

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
                "chunk_size": int(args.chunk_size),
                "image_height": C.IMAGE_HEIGHT,
                "image_width": C.IMAGE_WIDTH,
                "backbone": args.backbone,
                "language_backend": args.language_backend,
                "text_model": args.text_model,
                "action_space": args.action_space,
                "dataset_repo_id": args.dataset_repo_id,
                "pretrained_backbone": not args.no_pretrained,
                "freeze_backbone": not args.unfreeze_backbone,
                "goal_head": bool(args.goal_head),
                "goal_weight": float(args.goal_weight),
                "use_resistance": bool(policy.use_resistance),
                "freeze_mode": args.freeze_mode,
                "lora_r": int(args.lora_r),
                "lora_alpha": float(args.lora_alpha),
                "lora_targets": args.lora_targets,
                "lora_dropout": float(args.lora_dropout),
            },
        },
        path,
    )


def _build_policy(stats, vocabs, args, use_resistance):
    return build_vla_policy(
        stats=stats,
        vocabs=vocabs,
        pretrained_backbone=not args.no_pretrained,
        backbone_name=args.backbone,
        freeze_backbone=not args.unfreeze_backbone,
        language_backend=args.language_backend,
        text_model_name=args.text_model,
        action_space=args.action_space,
        goal_weight=args.goal_weight,
        goal_head=bool(args.goal_head),
        chunk_size=int(args.chunk_size),
        use_resistance=use_resistance,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to cpu")
        device = "cpu"

    # Load a resume checkpoint early so the dataset is built with the SAME
    # action space / chunk size / dataset the run was started with.
    resume_ckpt = None
    if args.resume is not None and args.resume.exists():
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        rcfg0 = resume_ckpt.get("config", {})
        for attr in ("action_space", "chunk_size", "backbone", "goal_head"):
            if attr in rcfg0:
                setattr(args, attr, rcfg0[attr])
        if rcfg0.get("dataset_repo_id"):
            args.dataset_repo_id = rcfg0["dataset_repo_id"]
    elif args.finetune is not None and args.finetune.exists():
        # Match the pretrained chunk size / action space so weights line up.
        fcfg = torch.load(args.finetune, map_location="cpu", weights_only=False).get("config", {})
        for attr in ("action_space", "chunk_size", "goal_head"):
            if attr in fcfg:
                setattr(args, attr, fcfg[attr])

    if not args.dataset_repo_id:
        raise SystemExit("--dataset-repo-id is required (MicroVLA trains from a LeRobot dataset).")

    full_ds = build_lerobot_vla_dataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        action_space=args.action_space,
        chunk_size=int(args.chunk_size),
    )
    print(f"loaded LeRobot dataset {args.dataset_repo_id} "
          f"(robot_id={full_ds.episodes[0].robot_id!r}, action_space={args.action_space}, "
          f"chunk_size={args.chunk_size}, resistance={full_ds.has_resistance})")

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
    # Policy construction:
    #   --resume   : rebuild the exact prior architecture from its config.
    #   --finetune : load pretrained ckpt, extend vocab+stats, partial-load.
    #   neither    : build from scratch on this dataset.
    # ------------------------------------------------------------------
    if resume_ckpt is not None:
        rcfg = resume_ckpt.get("config", {})
        for attr in ("freeze_mode", "lora_r", "lora_alpha", "lora_targets", "lora_dropout"):
            if attr in rcfg:
                setattr(args, attr, rcfg[attr])
        full_ds.vocabs = VocabBundle(**resume_ckpt["vocabs"])
        full_ds.stats = resume_ckpt["stats"]
        use_resistance = bool(rcfg.get("use_resistance", full_ds.has_resistance))
        policy = _build_policy(full_ds.stats, full_ds.vocabs, args, use_resistance).to(device)
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
        pre_cfg = pre.get("config", {})
        pre_vocabs = VocabBundle(**pre["vocabs"]) if not isinstance(
            pre["vocabs"], VocabBundle) else pre["vocabs"]
        ext_vocabs = extend_vocabs(pre_vocabs, full_ds.episodes)
        merged = merge_stats(pre["stats"], full_ds.stats)
        full_ds.vocabs = ext_vocabs
        full_ds.stats = merged
        # Keep resistance support if either the pretrained model or the new data has it.
        use_resistance = bool(pre_cfg.get("use_resistance", False) or full_ds.has_resistance)
        policy = _build_policy(merged, ext_vocabs, args, use_resistance).to(device)
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
        use_resistance = full_ds.has_resistance
        policy = _build_policy(full_ds.stats, full_ds.vocabs, args, use_resistance).to(device)
    print(
        f"backbone: {args.backbone}  language={args.language_backend}:{args.text_model}  "
        f"goal_head={args.goal_head}  resistance={policy.use_resistance}  "
        f"freeze={args.freeze_mode}  lora_r={args.lora_r}  {parameter_summary(policy)}"
    )

    # ------------------------------------------------------------------
    # Optional frozen-encoder feature cache (skips video decode + frozen ViT).
    # ------------------------------------------------------------------
    if args.cache_features:
        if args.unfreeze_backbone:
            print("[feature-cache] --unfreeze-backbone is set; encoder outputs change during "
                  "training, so the cache would be stale. Disabling.")
        else:
            cache_dir = args.feature_cache_dir or (
                args.ckpt_dir / f"feat_cache_{args.backbone.replace('+', '_').replace('/', '_')}"
            )
            image_hw = (C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
            num_frames = int(full_ds.states_all.shape[0])
            feat_cache = FeatureCache.load_if_valid(
                cache_dir, repo_id=args.dataset_repo_id, backbone_name=args.backbone,
                image_hw=image_hw, num_frames=num_frames,
            )
            if feat_cache is None:
                feat_cache = FeatureCache.build(
                    cache_dir, full_ds, policy, device,
                    repo_id=args.dataset_repo_id, backbone_name=args.backbone,
                    image_hw=image_hw, batch_size=args.precompute_batch,
                )
            full_ds.feature_cache = feat_cache

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
                print(f"[resume] optimizer state ignored ({e}); starting fresh AdamW state")
        start_epoch = int(resume_ckpt.get("epoch") or 0)
        best_val = float(resume_ckpt.get("best_val", best_val))
        print(f"resumed at epoch {start_epoch}; best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        tr = run_epoch(policy, train_loader, optimizer, device, train=True, amp=args.amp)
        vl = run_epoch(policy, val_loader, optimizer, device, train=False, amp=args.amp)

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
                policy, optimizer, epoch + 1, best_val, full_ds, args,
            )

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
