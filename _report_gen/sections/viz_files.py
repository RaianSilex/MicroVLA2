from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # =====================================================================
    # viz_summary.py
    # =====================================================================
    h1(story, "viz_summary.py")
    h2(story, "Purpose")
    body(story, "Prints a Keras-style layer summary of the ACT model: nn.Module "
                "hierarchy first, then a <code>torchinfo.summary</code> table for "
                "both the inference path (no actions) and the training path (with "
                "the style encoder branch). Output goes to stdout — pipe it to a "
                "file to keep a copy.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Inputs (batch=1 dummies): <code>image (1,1,3,240,320)</code>, "
        "<code>qpos (1,8)</code>, <code>actions (1,100,8)</code>, "
        "<code>is_pad (1,100)</code>.",
        "Output: pretty-printed text on stdout. No files written.",
    ])

    code_block(story, "viz_summary.py:25-62 - main", """\
def main(depth: int = 1, backbone: str = None) -> None:
    cvae = build_cvae(backbone_name=backbone).eval()
    print(f"backbone: {cvae.backbone.backbone_name}")

    print("=" * 78)
    print("nn.Module hierarchy (names only)")
    print("=" * 78)
    print(cvae)                                           # repr -> nested module names
    print()

    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
    qpos = torch.zeros(1, C.STATE_DIM)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    print("=" * 78)
    print(f"torchinfo.summary - INFERENCE path (depth={depth})")
    print("=" * 78)
    summary(cvae, input_data=(image, qpos), depth=depth,
            col_names=("input_size", "output_size", "num_params"), verbose=1)

    print("=" * 78)
    print(f"torchinfo.summary - TRAINING path (depth={depth})")
    print("=" * 78)
    summary(cvae, input_data=(image, qpos, actions, is_pad), depth=depth,
            col_names=("input_size", "output_size", "num_params"), verbose=1)""")
    bullets(story, [
        "<b>Two summaries</b>: the inference path skips the style encoder branch "
        "(actions=None &rarr; z=0 directly), so its parameter count is smaller. "
        "The training path includes the style encoder.",
        "<b>depth controls torchinfo's nesting</b>. Default 1 prints only "
        "top-level submodules (<code>backbone, transformer, style_encoder, ...</code>); "
        "<code>--depth 4</code> drills into individual attention sub-layers.",
        "<b>print(cvae)</b> uses PyTorch's default <code>nn.Module.__repr__</code> "
        "which walks the module tree and prints submodule names. No tensor flow info — "
        "just structure.",
    ])

    code_block(story, "viz_summary.py:65-73 - CLI", """\
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None,
                   help="resnet18 | dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14. "
                        "Defaults to config.BACKBONE.")
    p.add_argument("--depth", type=int, default=1,
                   help="torchinfo nesting depth. 1 = top-level only, 4+ = inner attention.")
    args = p.parse_args()
    main(depth=args.depth, backbone=args.backbone)""")

    # =====================================================================
    # viz_torchviz.py
    # =====================================================================
    h1(story, "viz_torchviz.py")
    h2(story, "Purpose")
    body(story, "Renders SVG diagrams of the model's <i>autograd</i> graph using "
                "<code>torchviz.make_dot</code>. Where ONNX/Netron shows the clean "
                "module-level architecture, this shows every backward op (matmul, "
                "softmax, layernorm, ...). Four sub-graphs are produced per backbone, "
                "saved under <code>torchviz_exports/&lt;backbone&gt;/</code>.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Inputs (batch=1 dummies, all <code>requires_grad=True</code> for graph "
        "tracing): <code>image (1,1,3,240,320), qpos (1,8), actions (1,100,8), "
        "is_pad (1,100)</code>.",
        "Outputs: 4 SVG files numbered 01..04 under "
        "<code>torchviz_exports/&lt;backbone_name&gt;/</code>.",
    ])

    code_block(story, "viz_torchviz.py:31-35 - render", """\
def render(out: Path, output_tensor: torch.Tensor, params: dict, label: str) -> None:
    print(f"  rendering {out.name}.svg  ({label})")
    dot = make_dot(output_tensor, params=params, show_attrs=False, show_saved=False)
    dot.format = "svg"
    dot.render(out.as_posix(), cleanup=True)""")
    bullets(story, [
        "<b>make_dot(scalar, params=...)</b>: walks the autograd graph backward from "
        "<code>scalar</code>, naming nodes that match an entry in <code>params</code> "
        "(typically <code>cvae.named_parameters()</code>).",
        "<b>show_attrs=False, show_saved=False</b>: declutter — skip per-op attribute "
        "dumps and saved-tensor metadata. The graph is already huge.",
        "<b>cleanup=True</b>: removes the intermediate Graphviz <code>.dot</code> "
        "file after rendering, keeping only the SVG.",
    ])

    code_block(story, "viz_torchviz.py:38-65 - main: dummies + per-section graphs", """\
def main(out_dir=Path("torchviz_exports"), backbone=None, freeze_backbone=True):
    cvae = build_cvae(backbone_name=backbone, freeze_backbone=freeze_backbone).eval()
    backbone_name = cvae.backbone.backbone_name
    out_dir = out_dir / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH,
                        requires_grad=True)
    qpos = torch.zeros(1, C.STATE_DIM, requires_grad=True)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM, requires_grad=True)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    # --- 1. Backbone only ---
    feat, pos = cvae.backbone(image[:, 0])               # (B, 3, H, W)
    render(out_dir / "01_backbone",
           feat.sum(),                                    # scalar root
           dict(cvae.backbone.named_parameters()),
           f"{backbone_name} + 1x1 projection + 2D sine pos embed")""")
    bullets(story, [
        "<b>requires_grad=True on the inputs</b>: needed because <code>make_dot</code> "
        "walks the autograd graph from outputs back through inputs. Frozen tensors "
        "would not be traced.",
        "<b>image[:, 0]</b> drops the camera axis &rarr; <code>(B, 3, H, W)</code>, "
        "matching what the backbone's <code>forward</code> expects in its "
        "single-encoder mode.",
        "<b>feat.sum()</b> reduces the spatial feature map to a scalar so "
        "<code>make_dot</code> has a single root. Doesn't matter that the value is "
        "meaningless — only the structure of the graph is being visualized.",
    ])

    code_block(story, "viz_torchviz.py:67-78 - 02_style_encoder graph", """\
# --- 2. Style encoder only (CVAE branch, qpos+actions -> mu/logvar) ---
mu, logvar = cvae._encode_style(qpos, actions, is_pad)
style_params = {}
for name, p in cvae.named_parameters():
    if any(k in name for k in ("style_", "cls_embed", "latent_proj")):
        style_params[name] = p
render(out_dir / "02_style_encoder",
       (mu.sum() + logvar.sum()),                      # scalar root combining both
       style_params,
       "CVAE style encoder: (qpos, actions) -> (mu, logvar)")""")
    bullets(story, [
        "<b>style_params filter</b>: only parameters whose name contains "
        "<code>style_</code>, <code>cls_embed</code>, or <code>latent_proj</code> "
        "are passed to <code>make_dot</code>. The rendered graph still shows the "
        "full autograd backward, but only the listed params get human-readable names.",
        "<b>mu.sum() + logvar.sum()</b>: combine the two outputs into a single "
        "scalar so <code>make_dot</code> can render one connected graph.",
    ])

    code_block(story, "viz_torchviz.py:80-97 - 03_inference and 04_training full graphs", """\
# --- 3. Full inference forward ---
a_hat, _ = cvae(image, qpos, actions=None, is_pad=None)
render(out_dir / "03_inference_full",
       a_hat.sum(),
       dict(cvae.named_parameters()),
       "Full inference path (backbone + main encoder + decoder + head)")

# --- 4. Full training forward ---
a_hat_t, (mu_t, logvar_t) = cvae(image, qpos, actions=actions, is_pad=is_pad)
loss_proxy = a_hat_t.sum() + mu_t.sum() + logvar_t.sum()
render(out_dir / "04_training_full",
       loss_proxy,
       dict(cvae.named_parameters()),
       "Full training path (inference + style encoder branch)")""")
    bullets(story, [
        "<b>03_inference_full</b>: skips the style encoder, so the graph is "
        "<code>backbone &rarr; main encoder &rarr; decoder &rarr; action_head</code>.",
        "<b>04_training_full</b>: includes everything. The <code>loss_proxy</code> "
        "sums all three outputs (action chunk, mu, logvar) so the graph captures "
        "both branches in one render.",
        "<b>Frozen backbone, frozen graph</b>: when DINOv2 / Cellpose are frozen "
        "(default), the autograd graph stops at the backbone output — "
        "<code>requires_grad=False</code> tensors don't contribute backward edges. "
        "Pass <code>--unfreeze-backbone</code> to render the inner ops.",
    ])

    code_block(story, "viz_torchviz.py:99-115 - CLI + listing", """\
print()
print("done. open in any browser (SVG is vector):")
for p in sorted(out_dir.glob("*.svg")):
    size_kb = p.stat().st_size // 1024
    print(f"  {p}  ({size_kb} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None, ...)
    p.add_argument("--unfreeze-backbone", action="store_true", ...)
    args = p.parse_args()
    main(backbone=args.backbone, freeze_backbone=not args.unfreeze_backbone)""")
    bullets(story, [
        "<b>Per-backbone subdirectory</b>: <code>torchviz_exports/&lt;backbone&gt;/</code>. "
        "Same convention as <code>export_onnx.py</code> — different backbones never "
        "overwrite each other's diagrams.",
        "<b>SVG is vector</b>: a 5000-node graph stays sharp at any zoom in the "
        "browser. PNG / PDF would either blow up file size or lose detail.",
    ])
