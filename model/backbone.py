"""Image backbone for MicroACT.

Three backbone families are supported, selected by the `backbone_name`
argument (or the `BACKBONE` config constant):

  - `"resnet18"`           — ImageNet-pretrained ResNet18 with frozen BN.
                             Default. ~11M backbone params.
  - `"dinov2_vits14"`      — DINOv2 ViT-S/14, frozen by default. ~22M.
  - `"dinov2_vitb14"`      — DINOv2 ViT-B/14, frozen by default. ~87M.
  - `"dinov2_vitl14"`      — DINOv2 ViT-L/14, frozen by default. ~304M.

Plus a **dual encoder** mode for hybrid generalist + domain-specialist
features:

  - `"cellpose4"`              — Cellpose 4 / Cellpose-SAM transformer
                                  features, using the CP-SAM neck plus
                                  averaged flow/cellprob readout channels.
  - `"dinov2_vits14+cellpose"` — DINOv2 ViT-S as the primary encoder
                                  for general scene understanding +
                                  Cellpose 3 cyto3 U-Net encoder as a
                                  cell-aware specialist. Both frozen.
  - `"dinov2_vits14+cellpose4"` — same dual idea with Cellpose 4 / CP-SAM
                                  as the specialist stream.
  - `"resnet18+cellpose"`      — same idea with ResNet18 as primary.
  - `"resnet18+cellpose4"`     — ResNet18 primary + Cellpose 4 specialist.

Single-encoder paths return spatial features `(B, num_channels, Hp, Wp)`,
which the `Backbone` wrapper projects to `hidden_dim`, position-embeds,
and flattens to a token sequence.

Dual-encoder paths run both encoders independently, project each to
`hidden_dim`, add a per-encoder type embedding, position-embed each in
its own normalized coordinate frame, then concatenate along the token
sequence. High-resolution auxiliary features are 2x2-avg-pooled before
flattening to keep the token budget reasonable (240x320 -> 30x40 ->
15x20 = 300 Cellpose tokens, vs 238x308 -> 17x22 = 374 DINOv2 tokens).

Multi-camera handling (concatenation along spatial dim, per-camera pos
offsets, etc.) lives in the CVAE module, not here.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import math
import os
from pathlib import Path

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
_CELLPOSE4_NAMES = {"cellpose4", "cpsam"}
_CELLPOSE4_WEIGHTS_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
    use any image shape (we currently use 240x320 -> resized to 238x308 =
    17x22 patches).
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
# Cellpose 3 cyto3 U-Net encoder (frozen domain-specialist)
# ---------------------------------------------------------------------------

_CELLPOSE_WEIGHTS_URL = "https://www.cellpose.org/models/cyto3"
_CELLPOSE_NBASE = [2, 32, 64, 128, 256]   # cyto3 default U-Net channel widths


class CellposeBackbone(nn.Module):
    """Cellpose 3 cyto3 encoder as a frozen feature extractor.

    Discards the U-Net's decoder + segmentation heads; uses only the
    deepest encoder feature map at 1/8 resolution, 256 channels. For
    240x320 RGB input → output is `(B, 256, 30, 40)`.

    The model expects 2-channel input (cyto, nuclei). We map RGB to a
    single luminance channel and pad the second channel with zeros —
    fine for microscope footage with no nuclear stain.

    First instantiation downloads ~25 MB of cyto3 weights into
    `~/.cellpose/models/cyto3`.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        # Lazy import: cellpose's package __init__ pulls numba via
        # cellpose.dynamics, which has a known coverage-package conflict.
        # The resnet_torch module is pure PyTorch — import only that.
        from cellpose.resnet_torch import CPnet

        self.net = CPnet(nbase=_CELLPOSE_NBASE, nout=3, sz=3, mkldnn=False)
        self._load_pretrained()
        self.num_channels = _CELLPOSE_NBASE[-1]   # 256

        self.frozen = freeze
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False

        # ImageNet luminance weights, registered as buffers so .to(device)
        # moves them with the model.
        self.register_buffer(
            "_luma_w",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _load_pretrained(self):
        import pathlib
        cache = pathlib.Path.home() / ".cellpose" / "models" / "cyto3"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            torch.hub.download_url_to_file(
                _CELLPOSE_WEIGHTS_URL, str(cache), progress=False
            )
        state = torch.load(cache, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=False)

    def train(self, mode: bool = True):
        # Frozen Cellpose stays in eval mode (BN running stats fixed, etc.)
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def _to_2chan(self, x: torch.Tensor) -> torch.Tensor:
        """RGB (B,3,H,W) → 2-channel (luminance, zero) for Cellpose."""
        gray = (x * self._luma_w).sum(dim=1, keepdim=True)        # (B, 1, H, W)
        zero = torch.zeros_like(gray)
        return torch.cat([gray, zero], dim=1)                      # (B, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) ImageNet-normalized RGB.
        x = self._to_2chan(x)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats = self.net.downsample(x)
        return feats[-1]                                            # (B, 256, H/8, W/8)


# ---------------------------------------------------------------------------
# Cellpose 4 / Cellpose-SAM transformer feature extractor
# ---------------------------------------------------------------------------

class Cellpose4Backbone(nn.Module):
    """Cellpose 4 / Cellpose-SAM as a frozen feature extractor.

    Cellpose 4's public API is segmentation-oriented (`CellposeModel.eval`),
    but the underlying `vit_sam.Transformer` exposes a SAM image encoder neck
    with 256 spatial channels. This backbone uses that neck feature map and,
    by default, concatenates three patch-level readout channels averaged from
    the CP-SAM flow/cellprob head. The result is a compact cell-aware feature
    grid rather than full mask post-processing inside the training loop.

    The default diameter follows the raw-image Cellpose 4 setting that worked
    well for this rig. It rescales the image the same way Cellpose eval does:
    objects of `diameter` pixels are presented to CP-SAM at the canonical
    30-pixel scale. `cellprob_threshold` and `flow_threshold` are kept as
    metadata for parity with the tested segmentation recipe; they only affect
    mask post-processing, which this feature extractor intentionally skips.
    """

    def __init__(
        self,
        freeze: bool = True,
        pretrained: bool = True,
        diameter: float = None,
        include_readout: bool = None,
        use_bfloat16: bool = False,
    ):
        super().__init__()
        self.diameter = float(
            getattr(C, "CELLPOSE4_DIAMETER", 180.0) if diameter is None else diameter
        )
        self.cellprob_threshold = float(getattr(C, "CELLPOSE4_CELLPROB_THRESHOLD", -2.0))
        self.flow_threshold = float(getattr(C, "CELLPOSE4_FLOW_THRESHOLD", 1.5))
        self.include_readout = bool(
            getattr(C, "CELLPOSE4_INCLUDE_READOUT", True)
            if include_readout is None
            else include_readout
        )

        self.net = self._build_net(pretrained=pretrained, use_bfloat16=use_bfloat16)
        self.num_channels = 256 + (int(getattr(self.net, "nout", 3)) if self.include_readout else 0)

        self.frozen = freeze
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False

        self.register_buffer(
            "_image_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _build_net(self, pretrained: bool, use_bfloat16: bool):
        try:
            from cellpose.vit_sam import Transformer
        except ImportError as exc:
            try:
                version = importlib_metadata.version("cellpose")
            except importlib_metadata.PackageNotFoundError:
                version = "not installed"
            raise ImportError(
                "The 'cellpose4' backbone requires Cellpose >= 4.0 with "
                "Cellpose-SAM support. Install/upgrade with "
                "`python3 -m pip install 'cellpose>=4.0'`. "
                f"Current cellpose version: {version}."
            ) from exc

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        net = Transformer(dtype=dtype)
        if pretrained:
            weights = self._cache_pretrained()
            net.load_model(str(weights), device=torch.device("cpu"))
        return net

    @staticmethod
    def _cache_pretrained() -> Path:
        model_dir_env = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
        model_dir = Path(model_dir_env) if model_dir_env else Path.home() / ".cellpose" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "cpsam"
        if not path.exists():
            torch.hub.download_url_to_file(_CELLPOSE4_WEIGHTS_URL, str(path), progress=False)
        return path

    def train(self, mode: bool = True):
        # Frozen Cellpose-SAM stays in eval mode (no random layer dropping).
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def _to_cellpose_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized RGB -> Cellpose-style per-image normalized RGB."""
        x = (x * self._image_std.to(x.dtype) + self._image_mean.to(x.dtype)).clamp(0.0, 1.0)
        return self._percentile_normalize(x)

    @staticmethod
    def _percentile_normalize(x: torch.Tensor) -> torch.Tensor:
        # Cellpose eval normalizes channels by low/high image percentiles. A
        # torch implementation keeps this backbone batched and device-local.
        orig_dtype = x.dtype
        b, c, h, w = x.shape
        flat = x.float().flatten(2)
        lo = torch.quantile(flat, 0.01, dim=-1, keepdim=True)
        hi = torch.quantile(flat, 0.99, dim=-1, keepdim=True)
        flat = (flat - lo) / (hi - lo).clamp_min(1e-6)
        return flat.clamp(0.0, 1.0).view(b, c, h, w).to(orig_dtype)

    def _scale_for_diameter(self, x: torch.Tensor) -> torch.Tensor:
        if self.diameter <= 0:
            return x
        scale = 30.0 / self.diameter
        if abs(scale - 1.0) < 1e-6:
            return x

        _, _, h, w = x.shape
        ps = int(getattr(self.net, "ps", 8))
        h2 = max(ps, int(math.ceil((h * scale) / ps) * ps))
        w2 = max(ps, int(math.ceil((w * scale) / ps) * ps))
        if (h2, w2) == (h, w):
            return x
        return F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)

    @staticmethod
    def _resize_pos_embed(pos: torch.Tensor, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        if tuple(pos.shape[1:3]) == (h, w):
            return pos.to(dtype=dtype)
        pos_nchw = pos.permute(0, 3, 1, 2).float()
        pos_nchw = F.interpolate(pos_nchw, size=(h, w), mode="bicubic", align_corners=False)
        return pos_nchw.permute(0, 2, 3, 1).to(dtype=dtype)

    def _forward_neck(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        net_dtype = next(self.net.parameters()).dtype
        x = x.to(dtype=net_dtype)

        tokens = self.net.encoder.patch_embed(x)                 # (B, Hp, Wp, D)
        if self.net.encoder.pos_embed is not None:
            pos = self._resize_pos_embed(
                self.net.encoder.pos_embed,
                tokens.shape[1],
                tokens.shape[2],
                tokens.dtype,
            )
            tokens = tokens + pos

        if self.net.training and getattr(self.net, "rdrop", 0.0) > 0:
            nlay = len(self.net.encoder.blocks)
            probs = torch.linspace(0, self.net.rdrop, nlay, device=tokens.device)
            drop = (torch.rand((tokens.shape[0], nlay), device=tokens.device) < probs).to(tokens.dtype)
            for i, block in enumerate(self.net.encoder.blocks):
                mask = drop[:, i].view(-1, 1, 1, 1)
                tokens = tokens * mask + block(tokens) * (1 - mask)
        else:
            for block in self.net.encoder.blocks:
                tokens = block(tokens)

        feat = self.net.encoder.neck(tokens.permute(0, 3, 1, 2))  # (B, 256, Hp, Wp)
        if self.include_readout:
            readout = self.net.out(feat)                          # (B, nout*ps^2, Hp, Wp)
            b, _, hp, wp = readout.shape
            nout = int(getattr(self.net, "nout", 3))
            ps = int(getattr(self.net, "ps", 8))
            readout = readout.view(b, nout, ps * ps, hp, wp).mean(dim=2)
            feat = torch.cat([feat, readout.to(dtype=feat.dtype)], dim=1)
        return feat.to(dtype=orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) ImageNet-normalized RGB.
        x = self._to_cellpose_rgb(x)
        x = self._scale_for_diameter(x)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            return self._forward_neck(x)


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

def _build_single_encoder(name: str, pretrained: bool, freeze: bool):
    """Returns (module, num_channels) for a single backbone name."""
    if name == "resnet18":
        m = ResNet18Backbone(pretrained=pretrained)
        return m, m.num_channels
    if name in _DINOV2_EMBED_DIMS:
        m = DinoV2Backbone(name=name, freeze=freeze)
        return m, m.num_channels
    if name == "cellpose":
        m = CellposeBackbone(freeze=freeze)
        return m, m.num_channels
    if name in _CELLPOSE4_NAMES:
        m = Cellpose4Backbone(freeze=freeze, pretrained=pretrained)
        return m, m.num_channels
    raise ValueError(
        f"Unknown backbone {name!r}. Supported: 'resnet18', "
        f"{list(_DINOV2_EMBED_DIMS)}, 'cellpose', 'cellpose4'."
    )


class Backbone(nn.Module):
    """Wraps one or two image feature extractors + projection + pos embed.

    Single-encoder mode (e.g. `backbone_name='resnet18'` or
    `'dinov2_vits14'`):
        forward(x) returns (feat, pos) — both shape (B, hidden_dim, Hp, Wp).
        Backward-compatible with previous Backbone.

    Dual-encoder mode (e.g. `backbone_name='dinov2_vits14+cellpose4'`):
        forward(x) returns (tokens, pos_tokens) — both shape (S, B, hidden_dim).
        Tokens from the two encoders are concatenated along the sequence
        dim, each tagged with a learned type embedding. Large auxiliary
        feature grids are 2x2-avg-pooled before flattening to keep the token
        budget down.
    """

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
        self.hidden_dim = hidden_dim

        parts = backbone_name.split("+")
        self.is_dual = len(parts) > 1
        if self.is_dual and len(parts) != 2:
            raise ValueError(
                f"Only 2-encoder fusion supported, got {len(parts)} parts in "
                f"{backbone_name!r}"
            )

        # ---- Build encoder(s) ----
        primary_name = parts[0]
        primary, primary_chan = _build_single_encoder(primary_name, pretrained, freeze)
        self._set_encoder(primary_name, primary)

        if self.is_dual:
            aux_name = parts[1]
            aux, aux_chan = _build_single_encoder(aux_name, pretrained, freeze)
            self._set_encoder(aux_name, aux)
            self.input_proj_aux = nn.Conv2d(aux_chan, hidden_dim, kernel_size=1)
            # 2x2 average pool to halve large aux spatial extents (Cellpose 3
            # is at 1/8 native resolution; this brings it near DINOv2's 1/14
            # patch grid). Very small Cellpose 4 grids skip this in forward().
            self.aux_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            # Token-type embeddings: 0 = primary, 1 = auxiliary.
            self.type_embed = nn.Embedding(2, hidden_dim)
            self.primary_name = primary_name
            self.aux_name = aux_name

        self.input_proj = nn.Conv2d(primary_chan, hidden_dim, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)

    def _set_encoder(self, name: str, module: nn.Module) -> None:
        # Use a name-specific attribute so existing ResNet/DINOv2 checkpoints
        # still load (their state_dict keys reference 'resnet'/'dinov2').
        if name == "resnet18":
            self.resnet = module
        elif name in _DINOV2_EMBED_DIMS:
            self.dinov2 = module
        elif name == "cellpose":
            self.cellpose = module
        elif name in _CELLPOSE4_NAMES:
            self.cellpose4 = module
        else:
            raise ValueError(f"unreachable: {name!r}")

    def _encoder_feat(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "resnet18":
            return self.resnet(x)
        if name in _DINOV2_EMBED_DIMS:
            return self.dinov2(x)
        if name == "cellpose":
            return self.cellpose(x)
        if name in _CELLPOSE4_NAMES:
            return self.cellpose4(x)
        raise ValueError(f"unreachable: {name!r}")

    def _primary_feat(self, x: torch.Tensor) -> torch.Tensor:
        primary_name = self.primary_name if self.is_dual else self.backbone_name
        return self._encoder_feat(primary_name, x)

    def forward(self, x: torch.Tensor):
        """
        Single mode → returns (feat, pos), both (B, hidden_dim, Hp, Wp).
        Dual   mode → returns (tokens, pos_tokens), both (S, B, hidden_dim).
        """
        # Primary encoder
        feat_p = self._primary_feat(x)
        feat_p = self.input_proj(feat_p)
        pos_p = self.pos_embed(feat_p)

        if not self.is_dual:
            return feat_p, pos_p

        # Auxiliary encoder. Pool only if the feature grid is large enough;
        # Cellpose 4 with diameter scaling can already be very compact.
        feat_a = self._encoder_feat(self.aux_name, x)
        if min(feat_a.shape[-2:]) >= 16:
            feat_a = self.aux_pool(feat_a)
        feat_a = self.input_proj_aux(feat_a)                  # (B, D, h, w)
        pos_a = self.pos_embed(feat_a)

        # Add type embeddings (broadcast over spatial dims).
        D = self.hidden_dim
        type_p = self.type_embed.weight[0].view(1, D, 1, 1)
        type_a = self.type_embed.weight[1].view(1, D, 1, 1)
        feat_p = feat_p + type_p
        feat_a = feat_a + type_a

        # Flatten each to (S, B, D) and concat along token dim.
        f_p = feat_p.flatten(2).permute(2, 0, 1)              # (S_p, B, D)
        f_a = feat_a.flatten(2).permute(2, 0, 1)              # (S_a, B, D)
        p_p = pos_p.flatten(2).permute(2, 0, 1)
        p_a = pos_a.flatten(2).permute(2, 0, 1)

        tokens = torch.cat([f_p, f_a], dim=0)                 # (S_p + S_a, B, D)
        pos = torch.cat([p_p, p_a], dim=0)
        return tokens, pos


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
