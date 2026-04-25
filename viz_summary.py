"""High-level layer summary of the ACT model.

Prints a Keras-style table — every named layer, output shape, and param
count, with no math operations. The right granularity for "what is this
model made of?" without drowning in autograd ops.

Two views are printed:
  1. nn.Module hierarchy (just names, no shapes)
  2. torchinfo.summary table (with shapes + params), at three depths

Output goes to stdout. Pipe to a file if you want to keep it:
    python viz_summary.py > architecture.txt
"""

import torch
from torchinfo import summary

from config import config as C
from model.cvae import build_cvae


def main(depth: int = 4) -> None:
    cvae = build_cvae().eval()

    print("=" * 78)
    print("nn.Module hierarchy (names only)")
    print("=" * 78)
    print(cvae)
    print()

    # Two forward calls — inference path (no actions) and training path (with).
    image = torch.zeros(1, C.NUM_CAMERAS, 3, C.IMAGE_HEIGHT, C.IMAGE_WIDTH)
    qpos = torch.zeros(1, C.STATE_DIM)
    actions = torch.zeros(1, C.CHUNK_SIZE, C.ACTION_DIM)
    is_pad = torch.zeros(1, C.CHUNK_SIZE, dtype=torch.bool)

    print("=" * 78)
    print(f"torchinfo.summary — INFERENCE path (depth={depth})")
    print("=" * 78)
    summary(
        cvae,
        input_data=(image, qpos),
        depth=depth,
        col_names=("input_size", "output_size", "num_params"),
        verbose=1,
    )
    print()

    print("=" * 78)
    print(f"torchinfo.summary — TRAINING path (depth={depth})")
    print("=" * 78)
    summary(
        cvae,
        input_data=(image, qpos, actions, is_pad),
        depth=depth,
        col_names=("input_size", "output_size", "num_params"),
        verbose=1,
    )


if __name__ == "__main__":
    main()
