from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # ----- requirements.txt -----
    h1(story, "requirements.txt")
    h2(story, "Purpose")
    body(story, "Lists the Python runtime dependencies. Not executable logic — "
                "it just declares which third-party libraries the rest of the code "
                "assumes can be imported.")
    h2(story, "Object contract")
    bullets(story, [
        "Torch / torchvision give tensors, autograd, model layers, ResNet18, ONNX export.",
        "numpy / pandas / pillow handle CSV parsing and image I/O.",
        "transformers (added for VLA) provides DistilBERT/BERT for the HuggingFace text encoder.",
        "cellpose is optional; only needed for the <i>cellpose</i>/<i>cellpose4</i> backbones.",
    ])
    code_block(story, "requirements.txt:1-11", """\
torch>=2.0
torchvision>=0.15
numpy>=1.24,<2
pandas>=2.0
pillow>=10.0
transformers>=4.40
# Optional - only needed for backbones that include "cellpose" / "cellpose4".
# Cellpose 4 provides the Cellpose-SAM (`cpsam`) transformer used by the
# `cellpose4` backbone. The legacy `cellpose` backbone still imports only
# cellpose.resnet_torch.CPnet for the lighter Cellpose 3-style U-Net path.
cellpose>=4.0""")
    bullets(story, [
        "<b>transformers&gt;=4.40</b> is the new line versus the earlier ACT-only repo. "
        "It is required when <code>--language-backend hf</code> is selected (the default).",
        "Cellpose 4 is required for <code>cellpose4</code>. The legacy "
        "<code>cellpose</code> path still imports only <code>cellpose.resnet_torch</code> "
        "for the lighter U-Net-style feature extractor.",
    ])

    # ----- .gitignore -----
    h1(story, ".gitignore")
    h2(story, "Purpose")
    body(story, "Declares which paths git should ignore so generated artefacts "
                "(checkpoints, ONNX exports, large datasets, OS noise) are not committed.")
    code_block(story, ".gitignore:30-44 - ML artefacts and dataset", """\
# ML artifacts (too large for git; rebuild from requirements + dataset)
checkpoints/
onnx_exports/
torchviz_exports/
*.pt
*.pth
*.ckpt
*.pkl
*.onnx
*.onnx.data

# Dataset (use external storage / Git LFS, not git)
dataset/saved_frames/
dataset/saved_videos/""")
    bullets(story, [
        "<b>checkpoints/</b> and <b>*.pt / *.pth / *.ckpt</b> together exclude all "
        "PyTorch checkpoints. The new VLA pipeline writes to <b>checkpoints_vla/</b> "
        "which matches <code>checkpoints*</code> rules implicitly through the "
        "<code>*.pt</code> wildcards but is not specifically listed; the .gitignore "
        "currently relies on the file-suffix wildcards to keep VLA checkpoints out.",
        "Saved frames and videos under <b>dataset/saved_frames/</b> and "
        "<b>dataset/saved_videos/</b> are excluded as they are binary blobs.",
    ])

    # ----- utils.py -----
    h1(story, "utils.py")
    h2(story, "Purpose")
    body(story, "Reusable training utilities: deterministic seeding, AdamW optimizer "
                "with separate backbone parameter group, checkpoint save/load, and a "
                "running-mean meter. Both the ACT and the VLA training loops import "
                "these helpers.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Optimizer has two param groups: backbone parameters (lower LR) and everything else.",
        "Checkpoint dict keys: <code>policy</code> (state_dict), <code>epoch</code> (int), "
        "optional <code>best_val</code> (float), optional <code>optimizer</code> (state_dict).",
    ])
    code_block(story, "utils.py:13-18 - set_seed", """\
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)""")
    bullets(story, [
        "Seeds Python's <code>random</code>, NumPy, torch CPU, and torch CUDA generators "
        "in one call so a re-run produces the same data shuffling and weight init.",
    ])
    code_block(story, "utils.py:21-36 - build_optimizer", """\
def build_optimizer(
    policy: torch.nn.Module,
    lr: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    \"\"\"AdamW with two param groups: backbone (lower LR) vs. everything else.\"\"\"
    backbone_params = list(policy.model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in policy.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in other_params    if p.requires_grad], "lr": lr},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)""")
    bullets(story, [
        "<code>policy.model.backbone</code> works for both ACTPolicy.model = ACTCVAE and "
        "VLAPolicy.model = VLACVAE because both nest the image backbone under <code>.model.backbone</code>.",
        "Frozen backbone parameters (where <code>requires_grad=False</code>) are filtered "
        "out so AdamW does not try to track moments for them.",
        "Two LR groups let DINOv2/Cellpose features be fine-tuned at a tiny LR while the "
        "freshly-initialized transformer/heads train at the regular LR.",
    ])
    code_block(story, "utils.py:39-66 - save_checkpoint and load_checkpoint", """\
def save_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_val: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"policy": policy.state_dict(), "epoch": epoch}
    if best_val is not None:
        ckpt["best_val"] = float(best_val)
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> int:
    \"\"\"Loads weights (and optimizer state if provided). Returns the saved epoch.\"\"\"
    ckpt = torch.load(path, map_location=map_location)
    policy.load_state_dict(ckpt["policy"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch") or 0""")
    bullets(story, [
        "<code>policy.state_dict()</code> contains both trainable parameters and buffers. "
        "For ACTPolicy this includes <code>qpos_mean (8,)</code>, <code>image_mean (3,1,1)</code> "
        "and so on, so a saved checkpoint is self-contained even without the stats pickle.",
        "<code>load_checkpoint</code> tolerates missing optimizer state — useful when "
        "starting a fresh fine-tune from a pretrained checkpoint.",
        "Returning the saved epoch lets <code>train.py</code> resume the loop counter.",
    ])
    code_block(story, "utils.py:69-90 - AverageMeter and format_meters", """\
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def format_meters(meters: dict) -> str:
    return "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())""")
    bullets(story, [
        "Sample-weighted average — call <code>update(value, batch_size)</code> per batch "
        "and the meter weights each batch by its size, so the final average is the same as "
        "if you had averaged over individual samples.",
        "<code>format_meters</code> turns a <code>{name: AverageMeter}</code> dict into "
        "<code>loss=0.1234  l1=0.0987  kl=0.0247</code> for one-line console logging.",
    ])

    # ----- train.py -----
    h1(story, "train.py")
    h2(story, "Purpose")
    body(story, "Top-level training script for the homogeneous MicroACT pipeline. "
                "Builds the dataset and policy from <code>config.config</code>, splits "
                "train/val by timestep (default) or by trial, runs a standard PyTorch "
                "loop, and writes <code>policy_last.pt</code>, <code>policy_best.pt</code>, "
                "and periodic numbered checkpoints into <code>--ckpt-dir</code>.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Each batch from the DataLoader: <b>image (B,1,3,240,320)</b>, "
        "<b>qpos (B,8)</b>, <b>action (B,100,8)</b>, <b>is_pad (B,100)</b>.",
        "<code>policy(image, qpos, action, is_pad)</code> returns a "
        "<code>{loss, l1, kl}</code> dict where each value is a 0-D tensor.",
        "Checkpoint dict keys when written here: <code>policy</code> (state_dict), "
        "<code>optimizer</code>, <code>epoch</code> (int), <code>best_val</code> (float).",
    ])
    code_block(story, "train.py:32-58 - parse_args (CLI surface)", """\
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
    p.add_argument("--val-by-trial", action="store_true", ...)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--save-every",   type=int,   default=100, ...)
    p.add_argument("--ckpt-dir",     type=Path,  default=C.CKPT_DIR)
    p.add_argument("--resume",       type=Path,  default=None)
    p.add_argument("--no-pretrained", action="store_true", ...)
    p.add_argument("--backbone", type=str, default=C.BACKBONE, ...)
    p.add_argument("--unfreeze-backbone", action="store_true", ...)
    return p.parse_args()""")
    bullets(story, [
        "Every flag has a default sourced from <code>config.config</code>, so "
        "<code>python train.py</code> with no args runs a fully reproducible default "
        "training session.",
        "<b>--val-by-trial</b> avoids leaking adjacent timesteps into val by holding "
        "out whole trials. The default (random per-timestep split) is statistically "
        "more efficient with a small trial count but does leak nearby frames.",
        "<b>--no-pretrained</b> disables ImageNet weight downloads — needed when "
        "running offline. <b>--unfreeze-backbone</b> turns frozen DINOv2/Cellpose "
        "encoders into trainable ones (use <code>--lr-backbone</code> to set their LR).",
    ])
    code_block(story, "train.py:61-84 - run_epoch", """\
def run_epoch(policy, loader, optimizer, device, train: bool) -> dict:
    policy.train(train)
    meters: dict = defaultdict(AverageMeter)

    for batch in loader:
        image  = batch["image"].to(device, non_blocking=True)   # (B,1,3,240,320)
        qpos   = batch["qpos"].to(device, non_blocking=True)    # (B,8)
        action = batch["action"].to(device, non_blocking=True)  # (B,100,8)
        is_pad = batch["is_pad"].to(device, non_blocking=True)  # (B,100)

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

    return meters""")
    bullets(story, [
        "<code>policy.train(train)</code> flips both modules and BatchNorm/Dropout "
        "behaviour. Frozen sub-modules (DINOv2, Cellpose) override this in their own "
        "<code>train()</code> methods to stay in eval.",
        "Same code path for train and val — only difference is "
        "<code>torch.no_grad()</code> + no optimizer step. Loss dict structure is "
        "identical so meters can be aggregated the same way.",
        "<code>m.update(v.item(), bs)</code> weights each batch by its size so the "
        "average is over samples, not over batches. Important when the last batch is "
        "smaller than the rest.",
    ])
    code_block(story, "train.py:96-134 - dataset and split", """\
stats_path = args.ckpt_dir / "dataset_stats.pkl"
full_ds = build_dataset(stats_path=stats_path, recompute_stats=True)

if args.val_by_trial:
    n_trials = len(full_ds.trials)
    n_val_trials = max(1, int(round(n_trials * args.val_split)))
    perm = torch.randperm(
        n_trials, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    val_trial_set = set(perm[:n_val_trials])
    train_idx = [i for i, (ti, _) in enumerate(full_ds.index) if ti not in val_trial_set]
    val_idx   = [i for i, (ti, _) in enumerate(full_ds.index) if ti in val_trial_set]
    train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds, val_idx)
else:
    val_n = max(1, int(round(len(full_ds) * args.val_split)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed),
    )

loader_kwargs = dict(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=(device.startswith("cuda")),
    persistent_workers=args.num_workers > 0,
)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)""")
    bullets(story, [
        "<code>recompute_stats=True</code> always rewrites the per-axis "
        "<code>qpos_mean/std</code> and <code>action_mean/std</code> arrays so adding "
        "new trials does not silently keep stale normalization.",
        "<b>val-by-trial</b> walks <code>full_ds.index</code> (a flat list of "
        "<code>(trial_idx, timestep)</code> pairs) and partitions sample IDs based on "
        "which trial each sample belongs to. <code>Subset</code> wraps the original "
        "dataset with the chosen index list.",
        "<b>random_split</b> uses a seeded generator so train/val are reproducible "
        "across restarts. <code>persistent_workers=True</code> avoids re-spawning "
        "DataLoader workers each epoch (the small dataset fits the worker IPC budget).",
    ])
    code_block(story, "train.py:136-146 - policy + optimizer", """\
policy = build_policy(
    stats=full_ds.norm_stats,
    pretrained_backbone=not args.no_pretrained,
    backbone_name=args.backbone,
    freeze_backbone=not args.unfreeze_backbone,
).to(device)
optimizer = build_optimizer(policy, args.lr, args.lr_backbone, args.weight_decay)""")
    bullets(story, [
        "<code>build_policy</code> wires the freshly computed stats straight into the "
        "policy's buffers (<code>qpos_mean</code>, <code>image_std</code>, ...) so the "
        "saved checkpoint is self-contained at inference time.",
        "<code>build_optimizer</code> creates two AdamW param groups: backbone "
        "parameters at <code>lr_backbone</code>, everything else at <code>lr</code>. "
        "Frozen backbone parameters are filtered out so AdamW never tracks moments "
        "for them.",
    ])
    code_block(story, "train.py:148-189 - resume + main loop", """\
ckpt_last = args.ckpt_dir / "policy_last.pt"
ckpt_best = args.ckpt_dir / "policy_best.pt"

start_epoch = 0
best_val = float("inf")
if args.resume is not None and args.resume.exists():
    resume_ckpt = torch.load(args.resume, map_location=device)
    start_epoch = load_checkpoint(args.resume, policy, optimizer, map_location=device)
    best_val = resume_ckpt.get("best_val", best_val)
    if best_val == float("inf") and ckpt_best.exists():
        load_checkpoint(ckpt_best, policy, map_location=device)
        best_val = run_epoch(policy, val_loader, optimizer, device, train=False)["loss"].avg
        load_checkpoint(args.resume, policy, optimizer, map_location=device)
    elif best_val == float("inf"):
        best_val = run_epoch(policy, val_loader, optimizer, device, train=False)["loss"].avg

for epoch in range(start_epoch, args.epochs):
    tr = run_epoch(policy, train_loader, optimizer, device, train=True)
    vl = run_epoch(policy, val_loader,   optimizer, device, train=False)
    print(f"[epoch {epoch+1:4d}/{args.epochs}] train {format_meters(tr)}  |  val {format_meters(vl)}")

    val_loss = vl["loss"].avg
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(ckpt_best, policy, optimizer, epoch + 1, best_val=best_val)

    save_checkpoint(ckpt_last, policy, optimizer, epoch + 1, best_val=best_val)

    if (epoch + 1) % args.save_every == 0:
        save_checkpoint(
            args.ckpt_dir / f"policy_epoch{epoch+1}.pt",
            policy, optimizer, epoch + 1, best_val=best_val,
        )""")
    bullets(story, [
        "<b>Resume</b> handles three cases: (1) <code>best_val</code> present in the "
        "resumed checkpoint — used directly; (2) absent, but <code>policy_best.pt</code> "
        "exists — load it, evaluate, then reload the resume checkpoint to restore "
        "training state; (3) neither — evaluate the resumed model itself.",
        "Each epoch writes <b>policy_last.pt</b> always, <b>policy_best.pt</b> if val "
        "loss improved, and <b>policy_epoch{N}.pt</b> every <code>save_every</code> "
        "epochs (default 100). Three independent files give you both the latest state "
        "and your best-so-far without races.",
    ])

    # ----- train_vla.py -----
    h1(story, "train_vla.py (NEW)")
    h2(story, "Purpose")
    body(story, "Training script for MicroVLA. Same skeleton as train.py but plugs in "
                "the heterogeneous dataset, episode-level splits, and saves richer "
                "checkpoints with stats + vocabs + the backbone/language config used.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Each batch: <b>image (B,1,3,240,320)</b>, <b>qpos (B,16)</b>, "
        "<b>state_mask (B,16)</b>, <b>action (B,100,16)</b>, "
        "<b>action_mask (B,16)</b>, <b>is_pad (B,100)</b>, plus 5 ID tensors "
        "<b>(B,)</b> long, plus a Python list of B instruction strings.",
        "Checkpoint dict keys: <code>policy, optimizer, epoch, best_val, stats, "
        "vocabs, config</code>. <code>config</code> is a dict snapshot of the "
        "backbone / language / max-dim / chunk settings used to build the policy.",
    ])
    code_block(story, "train_vla.py:18-41 - parse_args (VLA-specific flags)", """\
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(...)
    p.add_argument("--episodes-dir", type=Path, default=C.VLA_EPISODES_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=C.VLA_CKPT_DIR)
    p.add_argument("--stats-path", type=Path, default=None)
    p.add_argument("--epochs", ...); p.add_argument("--batch-size", ...)
    p.add_argument("--lr", ...); p.add_argument("--lr-backbone", ...)
    p.add_argument("--val-split", type=float, default=C.VAL_SPLIT)
    p.add_argument("--holdout-lab", type=str, default=None)
    p.add_argument("--holdout-robot", type=str, default=None)
    p.add_argument("--backbone", type=str, default=C.DEFAULT_BACKBONE)
    p.add_argument("--language-backend", choices=("hf", "simple"),
                   default=C.LANGUAGE_BACKEND)
    p.add_argument("--text-model", type=str, default=C.DEFAULT_TEXT_MODEL)
    return p.parse_args()""")
    bullets(story, [
        "<b>--holdout-lab / --holdout-robot</b> are the cross-domain generalization "
        "knobs: pass <code>--holdout-lab lab_b</code> and every episode tagged with "
        "<code>lab_id == \"lab_b\"</code> goes into the val set, the rest into train.",
        "<b>--language-backend hf</b> uses HuggingFace DistilBERT (frozen, ~66M params + "
        "trainable 768→512 projection). <b>simple</b> swaps to a hash-based tokenizer "
        "for offline smoke tests.",
    ])
    code_block(story, "train_vla.py:44-73 - episode-level split", """\
def _episode_split(full_ds, args):
    n_episodes = len(full_ds.episodes)
    all_episode_ids = set(range(n_episodes))

    if args.holdout_lab is not None:
        val_episode_ids = {i for i, ep in enumerate(full_ds.episodes)
                           if ep.lab_id == args.holdout_lab}
    elif args.holdout_robot is not None:
        val_episode_ids = {i for i, ep in enumerate(full_ds.episodes)
                           if ep.robot_id == args.holdout_robot}
    else:
        n_val = max(1, int(round(n_episodes * args.val_split)))
        perm = torch.randperm(n_episodes,
                              generator=torch.Generator().manual_seed(args.seed)).tolist()
        val_episode_ids = set(perm[:n_val])

    train_episode_ids = all_episode_ids - val_episode_ids
    train_idx = [i for i, (ei, _) in enumerate(full_ds.index) if ei in train_episode_ids]
    val_idx = [i for i, (ei, _) in enumerate(full_ds.index) if ei in val_episode_ids]
    return Subset(full_ds, train_idx), Subset(full_ds, val_idx), train_episode_ids, val_episode_ids""")
    bullets(story, [
        "Splits at the <i>episode</i> level, never the timestep level. This is the "
        "default for VLA because adjacent timesteps inside one episode are highly "
        "correlated; sampling some into train and others into val would inflate the "
        "val score.",
        "<code>full_ds.index</code> is a flat list of <code>(episode_idx, t)</code> "
        "pairs of length <code>sum(ep.length)</code>. Filtering by membership of "
        "<code>episode_idx</code> in the train / val sets carves it into two index "
        "lists for <code>Subset</code>.",
    ])
    code_block(story, "train_vla.py:76-133 - run_epoch (passes everything through)", """\
def run_epoch(policy, loader, optimizer, device, train: bool) -> dict:
    policy.train(train)
    meters: dict = defaultdict(AverageMeter)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)             # (B,1,3,240,320)
        qpos = batch["qpos"].to(device, non_blocking=True)               # (B,16)
        state_mask = batch["state_mask"].to(device, non_blocking=True)   # (B,16) bool
        action = batch["action"].to(device, non_blocking=True)           # (B,100,16)
        action_mask = batch["action_mask"].to(device, non_blocking=True) # (B,16) bool
        is_pad = batch["is_pad"].to(device, non_blocking=True)           # (B,100) bool
        robot_id = batch["robot_id"].to(device, non_blocking=True)       # (B,) long
        lab_id = batch["lab_id"].to(device, non_blocking=True)           # (B,) long
        embodiment_id = batch["embodiment_id"].to(...)                   # (B,) long
        action_type_id = batch["action_type_id"].to(...)                 # (B,) long
        task_family_id = batch["task_family_id"].to(...)                 # (B,) long
        instructions = batch["instruction"]                              # list[str] of length B

        if train:
            loss_dict = policy(image, qpos, instructions,
                               robot_id, lab_id, embodiment_id, action_type_id, task_family_id,
                               state_mask=state_mask, action_mask=action_mask,
                               actions=action, is_pad=is_pad)
            optimizer.zero_grad(); loss_dict["loss"].backward(); optimizer.step()
        else:
            with torch.no_grad():
                loss_dict = policy(image, qpos, instructions, ...,
                                   state_mask=..., action_mask=..., actions=..., is_pad=...)

        bs = qpos.size(0)
        for k, v in loss_dict.items():
            meters[k].update(v.item(), bs)
    return meters""")
    bullets(story, [
        "<b>instructions</b> stays on the CPU as a Python list[str] — the language "
        "encoder owns its own tokenizer that handles device placement after tokenizing.",
        "All five categorical IDs are <code>(B,)</code> long tensors ready to feed "
        "<code>nn.Embedding</code> lookups inside <code>EmbodimentConditioner</code>.",
    ])
    code_block(story, "train_vla.py:136-160 - save_vla_checkpoint (rich metadata)", """\
def save_vla_checkpoint(path: Path, policy, optimizer, epoch: int,
                        best_val: float, full_ds, args) -> None:
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
            },
        },
        path,
    )""")
    bullets(story, [
        "Beyond the standard <code>policy/optimizer/epoch/best_val</code>, the VLA "
        "checkpoint embeds <b>stats</b> and <b>vocabs</b> directly. The rollout script "
        "rebuilds the policy from these without needing the dataset on disk.",
        "<b>config</b> snapshots the backbone choice, language backend, text model, "
        "and freeze flags so the rollout can build a structurally identical model "
        "before <code>load_state_dict</code>. Without this metadata you would have to "
        "remember every CLI flag from training.",
    ])

    # ----- export_onnx.py -----
    h1(story, "export_onnx.py")
    h2(story, "Purpose")
    body(story, "Exports the ACT model to two ONNX files (inference path and training "
                "path) so they can be inspected in Netron. Not a deployment tool — "
                "purely for visualization / architecture review.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Inputs (batch=1 dummy): <b>image (1,1,3,240,320)</b>, <b>qpos (1,8)</b>, "
        "<b>actions (1,100,8)</b>, <b>is_pad (1,100)</b> bool.",
        "Inference graph outputs: <b>action_chunk (B,100,8)</b>.",
        "Training graph outputs: <b>action_chunk (B,100,8)</b>, <b>mu (B,32)</b>, "
        "<b>logvar (B,32)</b>.",
    ])
    code_block(story, "export_onnx.py:24-51 - wrappers", """\
class _InferenceWrapper(nn.Module):
    def __init__(self, cvae: nn.Module):
        super().__init__()
        self.cvae = cvae

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        a_hat, _ = self.cvae(image, qpos, actions=None, is_pad=None)
        return a_hat


class _TrainingWrapper(nn.Module):
    def __init__(self, cvae: nn.Module):
        super().__init__()
        self.cvae = cvae

    def forward(self, image, qpos, actions, is_pad):
        a_hat, (mu, logvar) = self.cvae(image, qpos, actions=actions, is_pad=is_pad)
        return a_hat, mu, logvar""")
    bullets(story, [
        "ONNX export tracing rejects <code>None</code> arguments and tuple-of-tuple "
        "outputs. The two wrappers expose flat positional signatures so the exporter "
        "can trace cleanly.",
        "<b>Inference wrapper</b> mirrors what the rollout sees: image + qpos in, "
        "action chunk out. <b>Training wrapper</b> additionally returns "
        "<code>mu</code> and <code>logvar</code> so the style encoder branch shows "
        "up in the exported graph.",
    ])
    code_block(story, "export_onnx.py:54-108 - main: build dummies and export", """\
def main(out_dir=Path("onnx_exports"), opset=18, backbone=None, freeze_backbone=True):
    cvae = build_cvae(backbone_name=backbone, freeze_backbone=freeze_backbone).eval()
    backbone_name = cvae.backbone.backbone_name
    out_dir = out_dir / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
    qpos = torch.zeros(1, C.STATE_DIM)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    torch.onnx.export(
        _InferenceWrapper(cvae),
        (image, qpos),
        (out_dir / "act_inference.onnx").as_posix(),
        input_names=["image", "qpos"],
        output_names=["action_chunk"],
        opset_version=opset,
        dynamic_axes={"image": {0: "batch"}, "qpos": {0: "batch"},
                      "action_chunk": {0: "batch"}},
    )

    torch.onnx.export(
        _TrainingWrapper(cvae),
        (image, qpos, actions, is_pad),
        (out_dir / "act_training.onnx").as_posix(),
        input_names=["image", "qpos", "actions", "is_pad"],
        output_names=["action_chunk", "mu", "logvar"],
        opset_version=opset,
        dynamic_axes={...},
    )""")
    bullets(story, [
        "<b>Per-backbone subdirs</b>: outputs land under "
        "<code>onnx_exports/&lt;backbone_name&gt;/</code> so different backbones do "
        "not overwrite each other's graphs.",
        "<b>dynamic_axes</b> declares axis 0 as variable (\"batch\"). Without this, "
        "the exported graph would hard-code batch=1 and Netron would refuse different "
        "batch sizes at runtime.",
        "<b>opset 18</b> is required because <code>torch.nn.MultiheadAttention</code> "
        "uses ONNX ops introduced in opset 14+ (and tooling expects 18 to be safe with "
        "DINOv2's vision-transformer ops).",
    ])

    # ----- evaluate.py (empty) -----
    h1(story, "evaluate.py")
    h2(story, "Purpose")
    body(story, "Empty placeholder. The repo currently does not ship a standalone "
                "offline evaluator — validation runs inside <code>train.py</code> / "
                "<code>train_vla.py</code> at every epoch, and policy quality is "
                "measured at the rig via the rollout scripts.")
    body(story, "File length: 0 bytes. No imports, no symbols.")
