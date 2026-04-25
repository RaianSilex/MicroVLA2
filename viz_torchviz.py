"""Render the model's autograd graph with torchviz.

Produces SVG files in `torchviz_exports/<backbone>/`. Open them in any
browser (SVG is vector — zoom freely).

Difference from `export_onnx.py`:
  - ONNX/Netron shows the **architecture** (clean, module-level boxes).
  - torchviz shows the **autograd graph** (every add, matmul, layernorm,
    softmax — the operations PyTorch will run during backward).

The full-network graph is huge (thousands of ops). The script also renders
focused sub-graphs (backbone, transformer, style encoder) which are far
easier to read.

For frozen backbones (DINOv2 default), the autograd graph stops at the
backbone output — that's actually informative, since it shows exactly
what's *trainable*. Pass `--unfreeze-backbone` to also see the inner
backbone ops.
"""

import argparse
from pathlib import Path

import torch
from torchviz import make_dot

from config import config as C
from model.cvae import build_cvae


def render(out: Path, output_tensor: torch.Tensor, params: dict, label: str) -> None:
    print(f"  rendering {out.name}.svg  ({label})")
    dot = make_dot(output_tensor, params=params, show_attrs=False, show_saved=False)
    dot.format = "svg"
    dot.render(out.as_posix(), cleanup=True)


def main(
    out_dir: Path = Path("torchviz_exports"),
    backbone: str = None,
    freeze_backbone: bool = True,
) -> None:
    cvae = build_cvae(backbone_name=backbone, freeze_backbone=freeze_backbone).eval()
    backbone_name = cvae.backbone.backbone_name
    out_dir = out_dir / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"built cvae: {sum(p.numel() for p in cvae.parameters()) / 1e6:.1f} M params  "
          f"(backbone={backbone_name})")

    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH, requires_grad=True)
    qpos = torch.zeros(1, C.STATE_DIM, requires_grad=True)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM, requires_grad=True)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    print("rendering graphs ...")

    # --- 1. Backbone only (smallest, most readable) ---
    feat, pos = cvae.backbone(image[:, 0])  # (B, 3, H, W)
    render(
        out_dir / "01_backbone",
        feat.sum(),  # scalar so make_dot has a single root
        dict(cvae.backbone.named_parameters()),
        f"{backbone_name} + 1x1 projection + 2D sine pos embed",
    )

    # --- 2. Style encoder only (CVAE branch, qpos+actions -> mu/logvar) ---
    mu, logvar = cvae._encode_style(qpos, actions, is_pad)
    style_params = {}
    for name, p in cvae.named_parameters():
        if any(k in name for k in ("style_", "cls_embed", "latent_proj")):
            style_params[name] = p
    render(
        out_dir / "02_style_encoder",
        (mu.sum() + logvar.sum()),
        style_params,
        "CVAE style encoder: (qpos, actions) -> (mu, logvar)",
    )

    # --- 3. Full inference forward (huge — every op in the deployment path) ---
    a_hat, _ = cvae(image, qpos, actions=None, is_pad=None)
    render(
        out_dir / "03_inference_full",
        a_hat.sum(),
        dict(cvae.named_parameters()),
        "Full inference path (backbone + main encoder + decoder + head)",
    )

    # --- 4. Full training forward (inference + style encoder branch) ---
    a_hat_t, (mu_t, logvar_t) = cvae(image, qpos, actions=actions, is_pad=is_pad)
    loss_proxy = a_hat_t.sum() + mu_t.sum() + logvar_t.sum()
    render(
        out_dir / "04_training_full",
        loss_proxy,
        dict(cvae.named_parameters()),
        "Full training path (inference + style encoder branch)",
    )

    print()
    print("done. open in any browser (SVG is vector):")
    for p in sorted(out_dir.glob("*.svg")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None,
                   help="resnet18 | dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14. "
                        "Defaults to config.BACKBONE.")
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="DINOv2 backbones are frozen by default; pass this to render "
                        "their internal autograd graph too.")
    args = p.parse_args()
    main(backbone=args.backbone, freeze_backbone=not args.unfreeze_backbone)
