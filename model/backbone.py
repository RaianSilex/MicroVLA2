"""Image backbone for MicroACT.

Two backbone families are supported and selected by the `backbone_name`
argument (or the `BACKBONE` config constant):

  - `"resnet18"`           — ImageNet-pretrained ResNet18 with frozen BN.
                             Default. ~11M backbone params.
  - `"dinov2_vits14"`      — DINOv2 ViT-S/14, frozen by default. ~21M.
  - `"dinov2_vitb14"`      — DINOv2 ViT-B/14, frozen by default. ~86M.
  - `"dinov2_vitl14"`      — DINOv2 ViT-L/14, frozen by default. ~300M.

Both paths return spatial features `(B, num_channels, Hp, Wp)`. The wrapper
`Backbone` then projects to `hidden_dim` with a 1x1 conv and emits a 2D
sinusoidal position embedding of matching shape, so the rest of the model
sees an identical interface regardless of backbone.

Multi-camera handling (concatenation along spatial dim, per-camera pos
offsets, etc.) lives in the CVAE module, not here.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from config import config as C


_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
_DINOV2_PATCH_SIZE = 14


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
# DINOv2 feature extractor (outputs patch tokens reshaped to spatial grid)
# ---------------------------------------------------------------------------

class DinoV2Backbone(nn.Module):
    """DINOv2 ViT backbone, frozen by default.

    The model expects spatial dims divisible by the patch size (14). We
    bilinearly resize on the fly, so the rest of the pipeline is free to
    use any image shape (we currently use 240x320 → resized to 238x322 =
    17x23 patches).
    """

    def __init__(self, name: str = "dinov2_vits14", freeze: bool = True):
        super().__init__()
        if name not in _DINOV2_EMBED_DIMS:
            raise ValueError(
                f"Unknown DINOv2 variant {name!r}. "
                f"Choose from {list(_DINOV2_EMBED_DIMS)}."
            )
        self.name = name
        self.num_channels = _DINOV2_EMBED_DIMS[name]
        self.patch_size = _DINOV2_PATCH_SIZE

        # First call downloads ~85 MB (ViT-S) → ~1 GB (ViT-L) into ~/.cache/torch/hub.
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", name, verbose=False)

        self.frozen = freeze
        if freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        # Frozen DINOv2 must stay in eval mode to disable any train-only ops.
        super().train(mode)
        if self.frozen:
            self.dinov2.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W). Resize to nearest multiple of patch_size (round down).
        B, _, H, W = x.shape
        Hp_full = max(self.patch_size, (H // self.patch_size) * self.patch_size)
        Wp_full = max(self.patch_size, (W // self.patch_size) * self.patch_size)
        if (Hp_full, Wp_full) != (H, W):
            x = F.interpolate(x, size=(Hp_full, Wp_full), mode="bilinear", align_corners=False)
        Hp, Wp = Hp_full // self.patch_size, Wp_full // self.patch_size

        # forward_features returns a dict with 'x_norm_patchtokens' shape (B, N, D)
        # where N = Hp * Wp. Skip the CLS token (we use the spatial grid directly).
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.dinov2.forward_features(x)
        tokens = out["x_norm_patchtokens"]                            # (B, N, D)
        feat = tokens.transpose(1, 2).reshape(B, self.num_channels, Hp, Wp)
        return feat


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
        backbone_name: str = None,
        freeze: bool = True,
    ):
        super().__init__()
        backbone_name = backbone_name or getattr(C, "BACKBONE", "resnet18")
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            self.resnet = ResNet18Backbone(pretrained=pretrained)
            num_channels = self.resnet.num_channels
        elif backbone_name in _DINOV2_EMBED_DIMS:
            self.dinov2 = DinoV2Backbone(name=backbone_name, freeze=freeze)
            num_channels = self.dinov2.num_channels
        else:
            raise ValueError(
                f"Unknown backbone {backbone_name!r}. "
                f"Supported: 'resnet18', {list(_DINOV2_EMBED_DIMS)}"
            )

        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
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
        if self.backbone_name == "resnet18":
            feat = self.resnet(x)
        else:
            feat = self.dinov2(x)
        feat = self.input_proj(feat)
        pos = self.pos_embed(feat)
        return feat, pos


def build_backbone(
    hidden_dim: int = C.HIDDEN_DIM,
    pretrained: bool = C.BACKBONE_PRETRAINED,
    backbone_name: str = None,
    freeze: bool = True,
) -> Backbone:
    return Backbone(
        hidden_dim=hidden_dim,
        pretrained=pretrained,
        backbone_name=backbone_name,
        freeze=freeze,
    )
