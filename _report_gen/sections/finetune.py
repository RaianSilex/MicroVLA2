from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    h1(story, "model/finetune.py (NEW)")
    h2(story, "Purpose")
    body(story, "Helpers that turn a pretrained MicroVLA checkpoint into a "
                "starting point for end-user finetuning on a different rig or "
                "task family. Five concerns: extending the vocab without "
                "shifting old IDs, merging per-robot normalization stats, "
                "loading old weights into a now-larger architecture (corner-copy "
                "for grown embedding tables), selectively freezing parts of the "
                "policy, and wrapping FFN linears with LoRA for parameter-cheap "
                "finetuning.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>extend_vocab(old, new) -&gt; dict</code>: preserves old "
        "<code>name -&gt; id</code> entries; appends new names at fresh IDs in "
        "sorted order. Determinism matters because the embedding row at ID i "
        "was trained for whatever name was at i originally.",
        "<code>extend_vocabs(old: VocabBundle, episodes) -&gt; VocabBundle</code>: "
        "applies <code>extend_vocab</code> to all five fields.",
        "<code>merge_stats(old, new) -&gt; dict</code>: per-robot stats; new "
        "stats win when the same robot appears in both. Image stats taken from new.",
        "<code>fill_robot_stats(policy, vocabs, stats)</code>: in-place writes "
        "to <code>policy.qpos_mean_table[rid]</code> etc., one row per robot in "
        "<code>stats[\"by_robot\"]</code>.",
        "<code>load_finetune_state_dict(model, ckpt_state_dict, skip_patterns)"
        "</code> &rarr; <code>LoadReport(matched, partial, skipped)</code>. "
        "Three behaviors: exact match copies; element-wise larger model tensor "
        "gets the ckpt tensor copied into its leading corner (handles grown "
        "embeddings); anything else is skipped with a logged reason.",
        "<code>apply_freeze_mode(policy, mode)</code>: 'none' / 'trunk' / "
        "'head_only'. Sets <code>requires_grad=False</code> on the listed "
        "submodules.",
        "<code>LoRALinear(base: nn.Linear, r=8, alpha=16.0)</code>: wraps a "
        "frozen Linear with a low-rank update <code>(x @ A^T @ B^T) * alpha/r</code>. "
        "<code>A: (r, in_features)</code> Kaiming-init, <code>B: (out_features, "
        "r)</code> zero-init so initial output equals the base.",
        "<code>apply_lora(policy, r, alpha, targets, layer_name_substrings)"
        "</code>: walks <code>policy.model.&lt;target&gt;</code> and replaces "
        "matching nn.Linear modules with LoRALinear in place. Default targets: "
        "<code>('transformer', 'style_encoder')</code>; default substrings: "
        "<code>('linear1', 'linear2')</code> &mdash; the FFN linears.",
    ])

    code_block(story, "model/finetune.py:42-65 - extend_vocab and extend_vocabs", """\
def extend_vocab(old: Dict[str, int], new_values: Iterable[str]) -> Dict[str, int]:
    out = dict(old)
    additions = sorted({str(v) for v in new_values if str(v) not in out})
    for value in additions:
        out[value] = len(out)
    return out


def extend_vocabs(old: VocabBundle, episodes: Sequence[VLAEpisode]) -> VocabBundle:
    return VocabBundle(
        robot_ids=extend_vocab(old.robot_ids, (e.robot_id for e in episodes)),
        lab_ids=extend_vocab(old.lab_ids, (e.lab_id for e in episodes)),
        embodiment_ids=extend_vocab(old.embodiment_ids, (e.embodiment for e in episodes)),
        action_type_ids=extend_vocab(old.action_type_ids, (e.action_type for e in episodes)),
        task_family_ids=extend_vocab(old.task_family_ids, (e.task_family for e in episodes)),
    )""")
    bullets(story, [
        "<b>Why preserve old IDs</b>: an embedding row at ID 3 was trained as the "
        "representation for whatever name was at ID 3 originally. Renumbering "
        "would silently swap which row corresponds to which robot &mdash; the "
        "model would still load but its conditioning would be scrambled.",
        "<b>Sorted additions</b> make the extension deterministic: the same "
        "(old vocab, new dataset) pair always produces the same final vocab, "
        "regardless of episode order on disk.",
    ])

    code_block(story, "model/finetune.py:68-77 - merge_stats", """\
def merge_stats(old_stats: dict, new_stats: dict) -> dict:
    by_robot: Dict[str, dict] = dict(old_stats.get("by_robot", {}))
    by_robot.update(new_stats.get("by_robot", {}))
    return {
        "by_robot": by_robot,
        "image_mean": new_stats["image_mean"],
        "image_std": new_stats["image_std"],
    }""")
    bullets(story, [
        "<b>New wins</b> for shared robots: the finetuner's own data probably "
        "reflects current rig calibration better than the pretrainer's. Robots "
        "only seen during pretraining keep their old stats so inference on them "
        "still works.",
        "Image stats are ImageNet constants in this codebase, so taking them "
        "from new_stats is purely cosmetic but consistent.",
    ])

    code_block(story, "model/finetune.py:80-94 - fill_robot_stats", """\
def fill_robot_stats(policy: nn.Module, vocabs: VocabBundle, stats: dict) -> None:
    by_robot = stats.get("by_robot", {})
    with torch.no_grad():
        for name, rid in vocabs.robot_ids.items():
            if name == C.UNKNOWN_TOKEN or name not in by_robot:
                continue
            rs = by_robot[name]
            policy.qpos_mean_table[rid]   = torch.from_numpy(rs["qpos_mean"]).to(...)
            policy.qpos_std_table[rid]    = torch.from_numpy(rs["qpos_std"]).to(...)
            policy.action_mean_table[rid] = torch.from_numpy(rs["action_mean"]).to(...)
            policy.action_std_table[rid]  = torch.from_numpy(rs["action_std"]).to(...)""")
    bullets(story, [
        "<b>Used after partial loading</b>. The pattern is: build new policy "
        "with merged_stats (which fills tables correctly), partial-load with "
        "<code>skip_patterns=(\"_table\",)</code> so the loader does NOT touch "
        "those buffers, then there is nothing to undo. The function exists for "
        "the symmetric case of building a policy and explicitly re-filling "
        "later, e.g. after extending vocab post-construction.",
        "<b>Buffer assignment is in-place</b>: <code>policy.qpos_mean_table[rid] "
        "= ...</code> writes into the existing buffer. <code>register_buffer</code> "
        "would create a new attribute and the optimizer / state_dict would lose "
        "the old reference.",
    ])

    code_block(story, "model/finetune.py:114-156 - load_finetune_state_dict", """\
def load_finetune_state_dict(model, ckpt_state_dict, skip_patterns=(), verbose=True):
    own = model.state_dict()
    new_state = dict(own)
    matched, partial, skipped = [], [], []

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
    return LoadReport(matched=matched, partial=partial, skipped=skipped)""")
    bullets(story, [
        "<b>Three branches per key</b>: skip-pattern, exact-shape, corner-copy. "
        "The corner-copy branch is what makes grown embeddings work: model "
        "buffer <code>(V_new, 512)</code> with V_new &gt; V_old gets ckpt buffer "
        "<code>(V_old, 512)</code> copied into rows <code>[0, V_old)</code>; "
        "rows <code>[V_old, V_new)</code> keep their fresh init.",
        "<b>Same logic handles other grown shapes</b>: if a finetuner increases "
        "<code>max_language_tokens</code>, <code>extra_src_pos.weight</code> "
        "grows from <code>(39, 512)</code> to e.g. <code>(71, 512)</code>; the "
        "first 39 positions copy across, new ones init fresh.",
        "<b>strict=True at the final load</b>: even though we built "
        "<code>new_state</code> from <code>own.copy()</code> plus selective "
        "overrides, every key is present, so strict load just sanity-checks no "
        "key was forgotten.",
        "<b>Returns a <code>LoadReport</code></b> dataclass with three lists so "
        "callers can introspect: matched (exact), partial (corner-copied with "
        "old + new shapes), skipped (with reasons).",
    ])

    code_block(story, "model/finetune.py:165-200 - apply_freeze_mode", """\
def apply_freeze_mode(policy: nn.Module, mode: str) -> None:
    inner = policy.model  # VLACVAE
    if mode == "none":
        return
    if mode == "trunk":
        freeze_modules(inner.transformer, inner.style_encoder)
        return
    if mode == "head_only":
        freeze_modules(
            inner.transformer, inner.style_encoder,
            inner.backbone, inner.language_encoder,
            inner.cls_embed,
            inner.style_qpos_proj, inner.style_action_proj, inner.style_pos_embed,
            inner.latent_proj,
            inner.latent_to_src, inner.qpos_to_src, inner.extra_src_pos,
            inner.query_embed,
        )
        return
    raise ValueError(f"Unknown freeze mode: {mode!r}")""")
    bullets(story, [
        "<b>'trunk'</b>: the most common finetune setting. The transformer "
        "(~17M + ~17M + ~44M = ~78M trainable in the default architecture) is "
        "frozen; the metadata embeddings (which include any new vocab rows), "
        "the input/output projections, the action head, and any LoRA params "
        "stay trainable. Typical &lt;5M trainable params after this.",
        "<b>'head_only'</b>: nuclear option. Only metadata embeddings + action "
        "head + LoRA train. Useful when you literally only want to teach the "
        "model a new robot's action mapping without changing what features get "
        "extracted upstream.",
        "<b>Backbones are still controlled by --unfreeze-backbone</b> at the "
        "VLACVAE constructor level. <code>head_only</code> additionally freezes "
        "them (in case the user passed <code>--unfreeze-backbone</code> for "
        "pretraining and now wants them frozen for finetuning).",
    ])

    code_block(story, "model/finetune.py:208-240 - LoRALinear", """\
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                        # freeze the base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = (self.dropout(x) @ self.A.T) @ self.B.T
        return out + update * self.scaling""")
    bullets(story, [
        "<b>Math</b>: <code>y = base(x) + (x A^T B^T) * alpha/r</code>. The "
        "rank-r update has only <code>r * (in + out)</code> trainable params "
        "instead of <code>in * out</code>. For <code>r=8</code> on a "
        "<code>3200-&gt;512</code> Linear: 8*(3200+512) = 29 696 params vs the "
        "1.6M of the dense weight.",
        "<b>B init to zero</b>: at step 0 the LoRA contribution is exactly zero, "
        "so the model behaves identically to the base. Training only departs "
        "from baseline once gradients flow into A/B. This makes LoRA insertion "
        "safe to apply mid-training without spiking the loss.",
        "<b>scaling = alpha / r</b>: the effective magnitude of the update. "
        "Convention is <code>alpha = 2 * r</code> (here 16 for r=8); halving r "
        "doubles the per-parameter update strength to compensate for the lower "
        "capacity, keeping behavior roughly comparable across ranks.",
        "<b>Shape walk</b>: <code>x: (..., in_features); A.T: (in_features, r); "
        "x @ A.T: (..., r); B.T: (r, out_features); result: (..., out_features)</code>. "
        "Same shape as the base output so the residual add works.",
    ])

    code_block(story, "model/finetune.py:243-280 - apply_lora", """\
def apply_lora(policy: nn.Module, r: int = 8, alpha: float = 16.0,
               targets: Sequence[str] = ("transformer", "style_encoder"),
               layer_name_substrings: Sequence[str] = ("linear1", "linear2"),
               dropout: float = 0.0, verbose: bool = True) -> int:
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
    return swapped""")
    bullets(story, [
        "<b>FFN-only by design</b>: <code>linear1</code> and <code>linear2</code> "
        "are the FFN inside each TransformerEncoderLayer / DecoderLayer "
        "(<code>512 -&gt; 3200</code> and <code>3200 -&gt; 512</code>). LoRA on "
        "<code>nn.MultiheadAttention</code>'s QKV projection would need a "
        "different replacement module because MHA uses a fused weight, not an "
        "<code>nn.Linear</code>; that surgery is not done here.",
        "<b>For the default ACT/VLA architecture</b> with 4 encoder layers + "
        "7 decoder layers (= 11 layers in main transformer + 4 in style "
        "encoder = 15 total), each containing 2 FFN linears, "
        "<code>apply_lora</code> wraps 30 modules.",
        "<b>setattr replaces in-place</b>: the parent module's attribute now "
        "points to the LoRALinear, which holds the original Linear as "
        "<code>self.base</code>. The old reference is gone from the module "
        "tree so <code>state_dict()</code> shows LoRALinear's params instead.",
    ])

    h1(story, "train_vla.py - finetune integration")
    h2(story, "New CLI flags")
    bullets(story, [
        "<code>--finetune &lt;ckpt&gt;</code>: pretrained MicroVLA checkpoint "
        "to start from. Vocab and per-robot stats are extended; old IDs are "
        "preserved.",
        "<code>--freeze-mode {none,trunk,head_only}</code>: extra freezing on "
        "top of the always-frozen image+language backbones.",
        "<code>--lora-r &lt;int&gt;</code> (default 0 = disabled), "
        "<code>--lora-alpha &lt;float&gt;</code> (default 16), "
        "<code>--lora-targets &lt;csv&gt;</code> (default "
        "'transformer,style_encoder'), <code>--lora-dropout &lt;float&gt;</code>.",
    ])
    h2(story, "Three policy-construction modes in main()")
    bullets(story, [
        "<b>--resume &lt;ckpt&gt;</b> wins if both --resume and --finetune are "
        "given. Resume rebuilds the exact prior architecture from the saved "
        "config (including freeze_mode + lora_r), so LoRA state is reloaded "
        "into LoRALinear modules instead of trying to load LoRA weights into a "
        "plain nn.Linear.",
        "<b>--finetune &lt;ckpt&gt;</b>: load pretrained ckpt, extend vocabs "
        "with the new dataset, merge stats (new wins), build policy at "
        "extended sizes, partial-load weights skipping <code>_table</code> "
        "buffers, fill stats explicitly, then apply freeze + LoRA from CLI flags.",
        "<b>Neither</b>: original from-scratch training path.",
    ])
    h2(story, "Checkpoint contract additions")
    bullets(story, [
        "<code>save_vla_checkpoint</code> now records "
        "<code>freeze_mode, lora_r, lora_alpha, lora_targets, lora_dropout</code> "
        "inside the <code>config</code> sub-dict. The resume branch reads these "
        "back and re-applies them BEFORE <code>load_state_dict</code> so the "
        "checkpoint's LoRA params land in actual LoRALinear modules.",
    ])
    h2(story, "Optimizer caveat")
    bullets(story, [
        "AdamW state from a pretrained checkpoint refers to the OLD parameter "
        "set. After finetune-load + freeze + LoRA, the trainable parameter "
        "list is different, so the optimizer is built fresh at finetune time. "
        "On <code>--resume</code> we try the saved optimizer state and fall "
        "back to fresh AdamW state if the shapes don't line up.",
    ])

    h1(story, "Recommended finetune recipes")
    h2(story, "Lightweight on-rig finetune (50-200 demos)")
    code_block(story, "Recommended for end-users adapting to a new lab/rig", """\
python train_vla.py \\
    --episodes-dir /path/to/lab_b/episodes \\
    --finetune checkpoints_vla/pretrained_5000.pt \\
    --freeze-mode trunk \\
    --lora-r 8 --lora-alpha 16 \\
    --epochs 200 --batch-size 8 --lr 5e-5""")
    bullets(story, [
        "Trunk frozen + LoRA on FFN: trainable surface is a few-hundred-K "
        "parameters, which suits the small dataset. The pretrained transformer "
        "trunk does the heavy lifting; LoRA adapts the FFN behavior to the new "
        "rig.",
        "<b>--lr 5e-5</b> is higher than pretraining (1e-5) because we now "
        "have far fewer parameters reacting to the same gradient signal &mdash; "
        "the pretrained trunk acts as a strong prior so a higher LR on the "
        "small head doesn't blow up.",
    ])
    h2(story, "Heavier finetune (500-2000 demos)")
    code_block(story, "Same robot family, more data", """\
python train_vla.py \\
    --episodes-dir /path/to/lab_b/episodes \\
    --finetune checkpoints_vla/pretrained_5000.pt \\
    --freeze-mode none \\
    --epochs 500 --batch-size 8 --lr 1e-5""")
    bullets(story, [
        "With more data, full-trunk training pays off. Backbones are still "
        "frozen by their constructors; everything above them adapts. Use the "
        "pretrained learning rate.",
    ])
