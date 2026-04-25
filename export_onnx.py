"""Export the ACT model to ONNX for visualization in Netron.

Produces two files:
  - act_inference.onnx  : forward path used at deployment (image + qpos -> action chunk).
                         No CVAE style encoder.
  - act_training.onnx   : forward path used during training (image + qpos + actions + is_pad
                         -> action chunk + mu + logvar). Includes the style encoder.

After running, open https://netron.app in your browser and drag the .onnx files in.
"""

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


def main(out_dir: Path = Path("onnx_exports"), opset: int = 18) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cvae = build_cvae().eval()
    print(f"built cvae: {sum(p.numel() for p in cvae.parameters()) / 1e6:.1f} M params")

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
    main()
