"""Image backbone for MicroACT.

ResNet18 with frozen batch norm → 1x1 projection to `hidden_dim` → 2D
sinusoidal position embedding. One call takes a single image tensor and
returns (features, pos_embed) matching in shape, ready to be flattened into
a token sequence for the transformer encoder.

Multi-camera handling (concatenation along spatial dim, per-camera pos
offsets, etc.) lives in the CVAE module, not here.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from config import config as C


# ---------------------------------------------------------------------------
# Frozen BatchNorm2d (stable under small batch sizes, matches DETR / ACT ref)
# ---------------------------------------------------------------------------

class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with running stats and affine parameters frozen as buffers."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # torchvision's BN writes num_batches_tracked; drop it to avoid key errors.
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ---------------------------------------------------------------------------
# ResNet18 feature extractor (outputs layer4, 1/32 resolution, 512 channels)
# ---------------------------------------------------------------------------

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet = torchvision.models.resnet18(
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.body = IntermediateLayerGetter(resnet, return_layers={"layer4": "feat"})
        self.num_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 512, H/32, W/32)
        return self.body(x)["feat"]


# ---------------------------------------------------------------------------
# 2D sinusoidal position embedding (DETR-style)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine2D(nn.Module):
    """Returns a (B, 2*num_pos_feats, H, W) position embedding for a feature map."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        device = feat.device

        # All positions are "valid" (no padding mask in our image inputs).
        ones = torch.ones((b, h, w), device=device)
        y_embed = ones.cumsum(1, dtype=torch.float32)
        x_embed = ones.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Combined backbone: features + projection + pos embed
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = C.HIDDEN_DIM,
        pretrained: bool = C.BACKBONE_PRETRAINED,
    ):
        super().__init__()
        self.resnet = ResNet18Backbone(pretrained=pretrained)
        self.input_proj = nn.Conv2d(
            self.resnet.num_channels, hidden_dim, kernel_size=1
        )
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) image batch (ImageNet-normalized).
        Returns:
            feat: (B, hidden_dim, H', W')
            pos:  (B, hidden_dim, H', W')
        """
        feat = self.resnet(x)
        feat = self.input_proj(feat)
        pos = self.pos_embed(feat)
        return feat, pos


def build_backbone(
    hidden_dim: int = C.HIDDEN_DIM,
    pretrained: bool = C.BACKBONE_PRETRAINED,
) -> Backbone:
    return Backbone(hidden_dim=hidden_dim, pretrained=pretrained)
