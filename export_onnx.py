"""Export the ACT model to ONNX for visualization in Netron.

Produces two files under `onnx_exports/<backbone>/`:
  - act_inference.onnx  : forward path used at deployment (image + qpos -> action chunk).
                         No CVAE style encoder.
  - act_training.onnx   : forward path used during training (image + qpos + actions + is_pad
                         -> action chunk + mu + logvar). Includes the style encoder.

After running, open https://netron.app in your browser and drag the .onnx files in.

Use `--backbone` to switch between ResNet18, DINOv2, Cellpose, and dual variants.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import config as C
from model.cvae import build_cvae


class _InferenceWrapper(nn.Module):
    """Inference path: no actions, no is_pad. Returns predicted action chunk."""

    def __init__(self, cvae: nn.Module):
        super().__init__()
        self.cvae = cvae

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        a_hat, _ = self.cvae(image, qpos, actions=None, is_pad=None)
        return a_hat


class _TrainingWrapper(nn.Module):
    """Training path: actions + is_pad provided. Returns chunk + style distribution."""

    def __init__(self, cvae: nn.Module):
        super().__init__()
        self.cvae = cvae

    def forward(
        self,
        image: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ):
        a_hat, (mu, logvar) = self.cvae(image, qpos, actions=actions, is_pad=is_pad)
        return a_hat, mu, logvar


def main(
    out_dir: Path = Path("onnx_exports"),
    opset: int = 18,
    backbone: str = None,
    freeze_backbone: bool = True,
) -> None:
    cvae = build_cvae(backbone_name=backbone, freeze_backbone=freeze_backbone).eval()
    backbone_name = cvae.backbone.backbone_name
    out_dir = out_dir / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"built cvae: {sum(p.numel() for p in cvae.parameters()) / 1e6:.1f} M params  "
          f"(backbone={backbone_name})")

    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
    qpos = torch.zeros(1, C.STATE_DIM)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    inference_path = out_dir / "act_inference.onnx"
    training_path = out_dir / "act_training.onnx"

    print(f"exporting inference graph -> {inference_path}")
    torch.onnx.export(
        _InferenceWrapper(cvae),
        (image, qpos),
        inference_path.as_posix(),
        input_names=["image", "qpos"],
        output_names=["action_chunk"],
        opset_version=opset,
        dynamic_axes={
            "image": {0: "batch"},
            "qpos": {0: "batch"},
            "action_chunk": {0: "batch"},
        },
    )

    print(f"exporting training graph  -> {training_path}")
    torch.onnx.export(
        _TrainingWrapper(cvae),
        (image, qpos, actions, is_pad),
        training_path.as_posix(),
        input_names=["image", "qpos", "actions", "is_pad"],
        output_names=["action_chunk", "mu", "logvar"],
        opset_version=opset,
        dynamic_axes={
            "image":   {0: "batch"},
            "qpos":    {0: "batch"},
            "actions": {0: "batch"},
            "is_pad":  {0: "batch"},
            "action_chunk": {0: "batch"},
            "mu":      {0: "batch"},
            "logvar":  {0: "batch"},
        },
    )

    print("done.")
    print(f"open https://netron.app and drag in:\n  {inference_path}\n  {training_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None,
                   help="resnet18 | dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | "
                        "cellpose | cellpose4 | <primary>+cellpose[4]. "
                        "Defaults to config.BACKBONE.")
    p.add_argument("--unfreeze-backbone", action="store_true",
                   help="DINOv2 backbones are frozen by default; pass this to include "
                        "their internal ops in the exported graph.")
    args = p.parse_args()
    main(backbone=args.backbone, freeze_backbone=not args.unfreeze_backbone)
