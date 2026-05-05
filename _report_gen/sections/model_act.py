from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # =====================================================================
    # model/transformer.py
    # =====================================================================
    h1(story, "model/transformer.py")
    h2(story, "Purpose")
    body(story, "DETR-style transformer primitives: a <code>TransformerEncoderLayer</code>, "
                "a <code>TransformerDecoderLayer</code>, plus stack and full "
                "encoder-decoder wrappers. Used by both the main ACT encoder-decoder "
                "(image+qpos+latent &rarr; action chunk) and the CVAE style encoder. "
                "Tensors are <b>sequence-first (L, B, D)</b> throughout, matching "
                "<code>torch.nn.MultiheadAttention</code>.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Encoder layer: <code>src (L,B,D)</code> &rarr; <code>(L,B,D)</code> with "
        "self-attention + FFN.",
        "Decoder layer: <code>tgt (Q,B,D)</code>, <code>memory (L,B,D)</code> &rarr; "
        "<code>(Q,B,D)</code>, with masked self-attn on tgt then cross-attn into memory.",
        "Full <code>Transformer.forward</code>: <code>src (L,B,D), pos (L,B,D), "
        "query_embed (Q,D)</code> &rarr; <code>(Q,B,D)</code>.",
    ])

    code_block(story, "model/transformer.py:23-36 - tiny utilities", """\
def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def _activation(name: str):
    if name == "relu": return F.relu
    if name == "gelu": return F.gelu
    if name == "glu":  return F.glu
    raise ValueError(f"unsupported activation: {name}")

def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos is None else x + pos""")
    bullets(story, [
        "<code>_clones</code> deepcopies a layer N times — each clone gets independent "
        "weights, but the constructor only has to be written once.",
        "<code>_with_pos</code> implements the DETR convention of <i>adding</i> the "
        "position embedding into the queries and keys at every layer (NOT pre-baked "
        "once at the input). The values do not get the position embedding — only Q/K do.",
    ])

    code_block(story, "model/transformer.py:42-73 - TransformerEncoderLayer", """\
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)   # 512 -> 3200
        self.linear2 = nn.Linear(dim_feedforward, d_model)   # 3200 -> 512
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        q = k = _with_pos(src, pos)                          # (L,B,D)
        src2 = self.self_attn(q, k, src,
                              key_padding_mask=src_key_padding_mask)[0]   # (L,B,D)
        src = self.norm1(src + self.dropout1(src2))          # post-norm residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))          # (L,B,D)
        return src""")
    bullets(story, [
        "<b>D = 512</b> hidden width, <b>nhead = 8</b>, <b>dim_feedforward = 3200</b>. "
        "The FFN layer is ~6x the hidden width — heavy on FFN, light on attention.",
        "<b>Q = K = src + pos</b>; <b>V = src</b> (no positional offset). This is "
        "DETR's convention: position is a relational bias on attention, not part of "
        "the value being aggregated.",
        "<b>Post-LN</b>: residual is added <i>before</i> LayerNorm. The two norms "
        "after each sub-block (attention, FFN) match the original Transformer paper.",
        "<code>src_key_padding_mask</code> shape is <b>(B, L)</b> bool. True positions "
        "are masked OUT (no attention weight). For ACT this is non-trivial only for "
        "the action tokens in the style encoder (padded actions are masked).",
    ])

    code_block(story, "model/transformer.py:76-98 - TransformerEncoder stack", """\
class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, pos=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out""")
    bullets(story, [
        "Just iterates the cloned layers. The optional final <code>LayerNorm</code> "
        "is set in <code>build_encoder</code> (the CVAE style encoder uses one); the "
        "main ACT encoder inside <code>Transformer.__init__</code> does NOT (it is "
        "<code>norm=None</code>).",
    ])

    code_block(story, "model/transformer.py:105-150 - TransformerDecoderLayer", """\
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, tgt, memory,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        # 1) Self-attention over decoder queries (Q queries attend to each other).
        q = k = _with_pos(tgt, query_pos)                    # (Q,B,D)
        tgt2 = self.self_attn(q, k, tgt)[0]                  # (Q,B,D)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2) Cross-attention: decoder queries attend over encoder memory.
        tgt2 = self.multihead_attn(
            query=_with_pos(tgt, query_pos),                 # (Q,B,D)
            key=_with_pos(memory, pos),                      # (L,B,D)
            value=memory,                                    # (L,B,D)
            key_padding_mask=memory_key_padding_mask,
        )[0]                                                 # (Q,B,D)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3) FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt""")
    bullets(story, [
        "Three sub-blocks per decoder layer: <b>self-attn</b> (queries among "
        "themselves), <b>cross-attn</b> (queries attending to encoder memory), "
        "<b>FFN</b>. Each gets its own LayerNorm + dropout + residual.",
        "<b>query_pos</b> is the learned per-query embedding from "
        "<code>cvae.query_embed</code> (shape <code>(100, D)</code>). It is added to "
        "the running <code>tgt</code> on every layer, mirroring DETR — this is what "
        "differentiates the 100 query positions and tells each query \"you are "
        "predicting timestep i\".",
        "<b>memory_key_padding_mask (B, L)</b> is forwarded into cross-attn. For ACT "
        "the source sequence is <code>[latent, qpos, image_tokens]</code> with no "
        "padding, so this is <code>None</code>; for VLA, language tokens may be padded.",
    ])

    code_block(story, "model/transformer.py:153-183 - TransformerDecoder stack", """\
class TransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers, norm=None):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = norm

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)
        if self.norm is not None:
            out = self.norm(out)
        return out""")
    bullets(story, [
        "ACT's decoder has <b>DEC_LAYERS = 7</b> stacked decoder blocks plus a "
        "final <code>LayerNorm</code> at the top. Output shape is preserved across "
        "the stack: <code>(100, B, 512)</code> in, <code>(100, B, 512)</code> out.",
    ])

    code_block(story, "model/transformer.py:190-245 - Transformer (full enc-dec)", """\
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=7, dim_feedforward=3200, dropout=0.1,
                 activation="relu"):
        super().__init__()
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(
            dec_layer, num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        self.d_model = d_model; self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed, query_embed, src_key_padding_mask=None):
        # src:         (S, B, D)   - encoder input tokens
        # pos_embed:   (S, B, D)   - same shape, added to Q/K at every layer
        # query_embed: (Q, D) or (Q, B, D) - decoder query positional embeddings
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask, pos=pos_embed)
        # memory: (S, B, D)

        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).expand(-1, src.size(1), -1)  # (Q, B, D)
        tgt = torch.zeros_like(query_embed)                  # (Q, B, D) initial decoder input

        return self.decoder(
            tgt, memory,
            memory_key_padding_mask=src_key_padding_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )                                                    # (Q, B, D)""")
    bullets(story, [
        "<b>Defaults match config.config</b>: 4 encoder layers, 7 decoder layers, "
        "8 heads, hidden 512, FFN 3200, dropout 0.1.",
        "<b>Xavier init</b> on every weight tensor with dim &gt; 1. Biases and "
        "1-D LayerNorm/Embedding params keep their default init.",
        "<b>tgt = zeros(Q,B,D)</b>: the decoder always starts from zero queries; the "
        "learned <code>query_embed</code> is what differentiates the 100 query slots, "
        "added at every layer via <code>query_pos</code>. This matches DETR exactly.",
        "<b>memory_key_padding_mask = src_key_padding_mask</b>: the same padding mask "
        "is reused for both encoder self-attention and decoder cross-attention.",
    ])

    code_block(story, "model/transformer.py:248-264 - builders", """\
def build_transformer(**kwargs) -> Transformer:
    return Transformer(**kwargs)


def build_encoder(d_model=C.HIDDEN_DIM, nhead=C.NHEAD, num_layers=C.ENC_LAYERS,
                  dim_feedforward=C.DIM_FEEDFORWARD, dropout=C.DROPOUT, activation="relu"
                  ) -> TransformerEncoder:
    layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    return TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model))""")
    bullets(story, [
        "<code>build_encoder</code> is the encoder-only stack used as the CVAE style "
        "encoder. It comes with a final <code>LayerNorm</code> (the main "
        "<code>Transformer</code>'s encoder does not).",
    ])

    # =====================================================================
    # model/backbone.py
    # =====================================================================
    h1(story, "model/backbone.py")
    h2(story, "Purpose")
    body(story, "Image feature extractors plus a 2D sinusoidal position embedding and "
                "a 1x1 projection to <b>hidden_dim = 512</b>. Three single-encoder "
                "options (ResNet18, DINOv2 ViT-S/B/L, Cellpose 3 cyto3) plus a "
                "dual-encoder fusion mode that concatenates two encoders' tokens "
                "with a learned per-encoder type embedding.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Single-encoder mode: <code>forward(x)</code> returns "
        "<b>(feat, pos)</b> both <b>(B, 512, Hp, Wp)</b>.",
        "Dual-encoder mode: <code>forward(x)</code> returns "
        "<b>(tokens, pos_tokens)</b> both <b>(S_p+S_a, B, 512)</b>, already "
        "flattened to sequence-first.",
        "ResNet18 with 240x320 input: feat <b>(B,512,7,10)</b> at 1/32 resolution.",
        "DINOv2 ViT-S/14 with 240x320 input: resized to 238x308 &rarr; "
        "<b>(B, 384, 17, 22)</b> patches &rarr; projected to <b>(B, 512, 17, 22)</b>.",
        "Cellpose at 240x320 input: <b>(B, 256, 30, 40)</b> at 1/8 resolution &rarr; "
        "AvgPool2d(2) &rarr; <b>(B, 256, 15, 20)</b> &rarr; projected to <b>(B, 512, 15, 20)</b>.",
        "DINOv2+Cellpose combined token count: 17*22 + 15*20 = 374 + 300 = "
        "<b>674 tokens</b>.",
    ])

    code_block(story, "model/backbone.py:51-57 - DINOv2 dim table", """\
_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
_DINOV2_PATCH_SIZE = 14""")
    bullets(story, [
        "Native DINOv2 embedding dims. The 1x1 <code>input_proj</code> always maps "
        "down/up to <code>hidden_dim = 512</code> regardless of which variant is used.",
    ])

    code_block(story, "model/backbone.py:64-87 - FrozenBatchNorm2d", """\
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias""")
    bullets(story, [
        "Drop-in replacement for <code>nn.BatchNorm2d</code> where everything is a "
        "buffer (no <code>nn.Parameter</code>). Used inside ResNet18 because ACT "
        "trains with very small batches — running stats from a fresh 8-sample batch "
        "would be too noisy.",
        "<b>Math</b>: <code>scale = weight / sqrt(running_var + eps)</code>; "
        "<code>bias = bias - running_mean * scale</code>; output = "
        "<code>x * scale + bias</code>. Identical to BN's inference-mode formula.",
    ])

    code_block(story, "model/backbone.py:94-109 - ResNet18Backbone", """\
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                   if pretrained else None)
        resnet = torchvision.models.resnet18(
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.body = IntermediateLayerGetter(resnet, return_layers={"layer4": "feat"})
        self.num_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 512, H/32, W/32)
        return self.body(x)["feat"]""")
    bullets(story, [
        "<code>IntermediateLayerGetter</code> is a torchvision helper that returns "
        "any named intermediate output instead of the original final classifier. "
        "Here we grab <b>layer4</b>, the last residual stage at 1/32 resolution.",
        "For 240x320 input: <code>(B, 3, 240, 320) &rarr; (B, 512, 7, 10)</code> "
        "(rounding because 240/32 = 7.5, 320/32 = 10). 70 spatial tokens.",
    ])

    code_block(story, "model/backbone.py:116-167 - DinoV2Backbone", """\
class DinoV2Backbone(nn.Module):
    def __init__(self, name="dinov2_vits14", freeze=True):
        super().__init__()
        self.name = name
        self.num_channels = _DINOV2_EMBED_DIMS[name]   # 384 for vits14
        self.patch_size = _DINOV2_PATCH_SIZE           # 14
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", name, verbose=False)

        self.frozen = freeze
        if freeze:
            for p in self.dinov2.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
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
            x = F.interpolate(x, size=(Hp_full, Wp_full),
                              mode="bilinear", align_corners=False)
        Hp, Wp = Hp_full // self.patch_size, Wp_full // self.patch_size

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.dinov2.forward_features(x)
        tokens = out["x_norm_patchtokens"]                   # (B, N, D), N = Hp*Wp
        feat = tokens.transpose(1, 2).reshape(B, self.num_channels, Hp, Wp)
        return feat""")
    bullets(story, [
        "<b>Patch-size resize</b>: 240/14 = 17.14, so we round down to 17 patches and "
        "interpolate the input from 240x320 to 17*14 x 22*14 = 238x308. The DataLoader "
        "feeds 240x320; the backbone silently resizes.",
        "<b>forward_features</b>'s <code>x_norm_patchtokens</code> drops the CLS token "
        "and returns <code>(B, N, D)</code> where N = Hp*Wp = 17*22 = 374 tokens. "
        "<code>.transpose(1,2).reshape(B, D, Hp, Wp)</code> turns the token sequence "
        "back into a spatial feature map so the rest of the pipeline can use the same "
        "code path as ResNet.",
        "<b>train()</b> is overridden so <code>policy.train()</code> never flips the "
        "frozen DINOv2 into train mode — important because Dropout / LayerNorm running "
        "stats would otherwise drift even with <code>requires_grad=False</code>.",
    ])

    code_block(story, "model/backbone.py:178-247 - CellposeBackbone", """\
_CELLPOSE_NBASE = [2, 32, 64, 128, 256]   # cyto3 default U-Net channel widths

class CellposeBackbone(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        from cellpose.resnet_torch import CPnet
        self.net = CPnet(nbase=_CELLPOSE_NBASE, nout=3, sz=3, mkldnn=False)
        self._load_pretrained()
        self.num_channels = _CELLPOSE_NBASE[-1]   # 256
        self.frozen = freeze
        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False
        self.register_buffer("_luma_w",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False)

    def _load_pretrained(self):
        cache = pathlib.Path.home() / ".cellpose" / "models" / "cyto3"
        if not cache.exists():
            torch.hub.download_url_to_file(_CELLPOSE_WEIGHTS_URL, str(cache), progress=False)
        state = torch.load(cache, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=False)

    def _to_2chan(self, x):
        # RGB (B,3,H,W) -> 2-channel (luminance, zero) for Cellpose.
        gray = (x * self._luma_w).sum(dim=1, keepdim=True)   # (B, 1, H, W)
        zero = torch.zeros_like(gray)
        return torch.cat([gray, zero], dim=1)                 # (B, 2, H, W)

    def forward(self, x):
        x = self._to_2chan(x)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats = self.net.downsample(x)
        return feats[-1]                                      # (B, 256, H/8, W/8)""")
    bullets(story, [
        "<b>Cellpose is 2-channel</b> (cyto + nuclei). Microscope footage here only "
        "has cyto stain, so we collapse RGB to luminance with the standard ITU-R "
        "Y'601 weights (0.299, 0.587, 0.114) and zero-pad the second channel.",
        "<b>net.downsample(x)</b> returns the U-Net's encoder feature pyramid; "
        "<code>feats[-1]</code> is the deepest (smallest, most abstract) map. "
        "Decoder + segmentation head are discarded.",
        "Output spatial size is <b>1/8</b> of the input (Cellpose downsamples 3 times "
        "by 2x). For 240x320: <code>(B, 256, 30, 40)</code> = 1200 raw tokens before "
        "the dual-encoder pipeline pools 2x2 to bring it down to 300.",
        "<b>Lazy import</b>: <code>from cellpose.resnet_torch import CPnet</code> "
        "skips <code>cellpose/__init__.py</code>'s numba-pulling chain. The full "
        "<code>cellpose.models</code> API is never touched.",
    ])

    code_block(story, "model/backbone.py:254-283 - PositionEmbeddingSine2D", """\
class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        ones = torch.ones((b, h, w), device=feat.device)
        y_embed = ones.cumsum(1, dtype=torch.float32)        # (B, h, w)
        x_embed = ones.cumsum(2, dtype=torch.float32)        # (B, h, w)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=feat.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t               # (B, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t               # (B, h, w, num_pos_feats)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # output: (B, 2*num_pos_feats, h, w) = (B, 256, h, w)""")
    bullets(story, [
        "<b>2D sinusoidal embedding, DETR-style</b>. Each row gets a y-encoded sine "
        "vector of length <code>num_pos_feats=128</code>, each column gets an "
        "x-encoded one of length 128, and they are concatenated to "
        "<b>2*num_pos_feats = 256 = HIDDEN_DIM/2</b>... wait — actually "
        "<code>num_pos_feats=hidden_dim//2 = 256</code>, producing "
        "<b>(B, 512, h, w)</b> matching D.",
        "<code>cumsum</code> + normalization yields y_embed in [0, 2*pi] per row and "
        "x_embed in [0, 2*pi] per column.",
        "<code>dim_t = temperature**(2 * (i // 2) / num_pos_feats)</code> creates "
        "the geometric frequency ladder; "
        "<code>stack(sin(...), cos(...)).flatten</code> interleaves sin and cos.",
    ])

    code_block(story, "model/backbone.py:290-304 - _build_single_encoder", """\
def _build_single_encoder(name: str, pretrained: bool, freeze: bool):
    if name == "resnet18":
        m = ResNet18Backbone(pretrained=pretrained)
        return m, m.num_channels
    if name in _DINOV2_EMBED_DIMS:
        m = DinoV2Backbone(name=name, freeze=freeze)
        return m, m.num_channels
    if name == "cellpose":
        m = CellposeBackbone(freeze=freeze)
        return m, m.num_channels
    raise ValueError(...)""")

    code_block(story, "model/backbone.py:307-362 - Backbone (single + dual mode)", """\
class Backbone(nn.Module):
    def __init__(self, hidden_dim=C.HIDDEN_DIM, pretrained=C.BACKBONE_PRETRAINED,
                 backbone_name=None, freeze=True):
        super().__init__()
        backbone_name = backbone_name or getattr(C, "BACKBONE", "resnet18")
        self.backbone_name = backbone_name
        self.hidden_dim = hidden_dim

        parts = backbone_name.split("+")
        self.is_dual = len(parts) > 1
        if self.is_dual and len(parts) != 2:
            raise ValueError("Only 2-encoder fusion supported")

        primary_name = parts[0]
        primary, primary_chan = _build_single_encoder(primary_name, pretrained, freeze)
        self._set_encoder(primary_name, primary)

        if self.is_dual:
            aux_name = parts[1]
            aux, aux_chan = _build_single_encoder(aux_name, pretrained, freeze)
            self._set_encoder(aux_name, aux)
            self.input_proj_aux = nn.Conv2d(aux_chan, hidden_dim, kernel_size=1)
            self.aux_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.type_embed = nn.Embedding(2, hidden_dim)        # primary=0, aux=1
            self.primary_name = primary_name
            self.aux_name = aux_name

        self.input_proj = nn.Conv2d(primary_chan, hidden_dim, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)""")
    bullets(story, [
        "<b>backbone_name = \"a+b\"</b> selects the dual-encoder mode. The repo only "
        "uses <code>dinov2_vits14+cellpose</code> and <code>resnet18+cellpose</code>; "
        "the code rejects more than 2 parts.",
        "<b>1x1 Conv2d</b> is the projection that maps native channels (512 for "
        "ResNet, 384 for DINOv2-S, 256 for Cellpose) to <code>hidden_dim = 512</code>. "
        "It is just a per-spatial-location linear layer.",
        "<b>type_embed = nn.Embedding(2, 512)</b>: a two-row lookup giving each "
        "encoder a learned 512-D \"this token came from primary/aux\" tag. Added to "
        "tokens before concatenation so attention can route.",
        "<b>aux_pool = AvgPool2d(2,2)</b> halves the Cellpose grid — 30x40 &rarr; "
        "15x20 — which keeps the dual-encoder token total around 674.",
    ])

    code_block(story, "model/backbone.py:386-420 - Backbone.forward", """\
def forward(self, x: torch.Tensor):
    feat_p = self._primary_feat(x)                    # (B, primary_chan, Hp, Wp)
    feat_p = self.input_proj(feat_p)                  # (B, D, Hp, Wp)
    pos_p = self.pos_embed(feat_p)                    # (B, D, Hp, Wp)

    if not self.is_dual:
        return feat_p, pos_p                          # 4D pair

    feat_a = self.cellpose(x)                         # (B, 256, H/8, W/8) e.g. (B,256,30,40)
    feat_a = self.aux_pool(feat_a)                    # (B, 256, H/16, W/16) = (B,256,15,20)
    feat_a = self.input_proj_aux(feat_a)              # (B, D, 15, 20)
    pos_a = self.pos_embed(feat_a)                    # (B, D, 15, 20)

    D = self.hidden_dim
    type_p = self.type_embed.weight[0].view(1, D, 1, 1)
    type_a = self.type_embed.weight[1].view(1, D, 1, 1)
    feat_p = feat_p + type_p                          # broadcast over (Hp,Wp)
    feat_a = feat_a + type_a

    f_p = feat_p.flatten(2).permute(2, 0, 1)          # (Hp*Wp, B, D)
    f_a = feat_a.flatten(2).permute(2, 0, 1)          # (15*20, B, D)
    p_p = pos_p.flatten(2).permute(2, 0, 1)
    p_a = pos_a.flatten(2).permute(2, 0, 1)

    tokens = torch.cat([f_p, f_a], dim=0)             # (Hp*Wp + 300, B, D)
    pos = torch.cat([p_p, p_a], dim=0)
    return tokens, pos""")
    bullets(story, [
        "<b>Single-encoder mode</b> returns 4-D <code>(B, D, Hp, Wp)</code> tensors — "
        "the CVAE then flattens these to sequence-first.",
        "<b>Dual-encoder mode</b> returns sequence-first <code>(S, B, D)</code> "
        "directly because the primary and aux grids have different spatial sizes "
        "(17x22 vs 15x20) and cannot be packed into one rectangular tensor.",
        "<b>Token type addition</b>: the per-encoder type embedding is broadcast over "
        "spatial dims by reshaping to <code>(1, D, 1, 1)</code> — every spatial "
        "position from the primary encoder gets the same row 0 added; every Cellpose "
        "position gets row 1.",
        "<b>flatten(2).permute(2,0,1)</b>: <code>(B,D,Hp,Wp) &rarr; (B,D,Hp*Wp) &rarr; "
        "(Hp*Wp, B, D)</code>.",
        "<b>Concat along dim=0</b> stacks primary tokens then aux tokens. Final "
        "shape: <code>(S_p + S_a, B, D)</code>. For DINOv2-S+Cellpose at 240x320: "
        "<code>(374 + 300, B, 512) = (674, B, 512)</code>.",
    ])

    # =====================================================================
    # model/cvae.py
    # =====================================================================
    h1(story, "model/cvae.py")
    h2(story, "Purpose")
    body(story, "The action-chunking CVAE that defines the ACT model. Two transformer "
                "stacks: a <b>style encoder</b> that maps demonstration "
                "<code>(qpos, actions)</code> to a 32-D Gaussian latent, and a "
                "<b>main encoder-decoder</b> that conditions on "
                "<code>(latent, qpos, image_tokens)</code> to predict 100 future "
                "actions. At inference the style encoder is skipped and "
                "<code>z = 0</code> (the prior mean).")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>_encode_style(qpos, actions, is_pad)</code>: "
        "<code>qpos (B,8), actions (B,100,8), is_pad (B,100)</code> &rarr; "
        "<code>(mu (B,32), logvar (B,32))</code>.",
        "<code>_encode_image(image)</code>: <code>image (B,1,3,240,320)</code> &rarr; "
        "<code>(feat (S_img,B,512), pos (S_img,B,512))</code>.",
        "<code>forward(image, qpos, actions, is_pad)</code> &rarr; "
        "<code>a_hat (B,100,8), (mu (B,32), logvar (B,32))</code>.",
    ])

    code_block(story, "model/cvae.py:28-31 - reparameterize", """\
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    return mu + torch.randn_like(std) * std""")
    bullets(story, [
        "Standard VAE reparameterization trick. <code>logvar = log(sigma^2)</code>, "
        "so <code>(0.5 * logvar).exp() = sigma</code>. Output shape matches input: "
        "<b>(B, 32)</b>.",
    ])

    code_block(story, "model/cvae.py:33-78 - ACTCVAE.__init__", """\
class ACTCVAE(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 chunk_size=100, num_cameras=1, pretrained_backbone=True,
                 backbone_name=None, freeze_backbone=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras

        # ---- Backbones ----
        self.backbone = build_backbone(
            hidden_dim=hidden_dim, pretrained=pretrained_backbone,
            backbone_name=backbone_name, freeze=freeze_backbone,
        )
        self.transformer = build_transformer(d_model=hidden_dim)
        self.style_encoder = build_encoder(d_model=hidden_dim)

        # ---- Style-encoder IO ----
        self.cls_embed = nn.Embedding(1, hidden_dim)         # the [CLS] row
        self.style_qpos_proj = nn.Linear(state_dim, hidden_dim)        # 8 -> 512
        self.style_action_proj = nn.Linear(action_dim, hidden_dim)     # 8 -> 512
        self.style_pos_embed = nn.Embedding(1 + 1 + chunk_size, hidden_dim)  # 102 rows
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)       # 512 -> 64 (mu | logvar)

        # ---- Main-encoder non-image tokens ----
        self.latent_to_src = nn.Linear(latent_dim, hidden_dim)         # 32 -> 512
        self.qpos_to_src = nn.Linear(state_dim, hidden_dim)            # 8  -> 512
        self.extra_src_pos = nn.Embedding(2, hidden_dim)               # for [latent, qpos]

        # ---- Decoder queries + action head ----
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)        # 100 learnable queries
        self.action_head = nn.Linear(hidden_dim, action_dim)           # 512 -> 8""")
    bullets(story, [
        "<b>cls_embed = nn.Embedding(1, 512)</b> is a single learned 512-D row used "
        "as the [CLS] token in the style encoder. Sitting at sequence position 0, its "
        "post-encoder output is what gets projected to <code>(mu, logvar)</code>.",
        "<b>style_pos_embed = nn.Embedding(102, 512)</b>: 1 CLS + 1 qpos + 100 actions = "
        "102 sequence positions, each with its own learned 512-D positional vector.",
        "<b>latent_proj: 512 -&gt; 64</b>. The output is split into two halves with "
        "<code>chunk(2)</code>: first 32 are mu, last 32 are logvar.",
        "<b>extra_src_pos = nn.Embedding(2, 512)</b>: positions 0 and 1 of the main "
        "encoder source — for the latent token and the qpos token. Image tokens get "
        "the 2D sinusoidal pos from the backbone.",
        "<b>query_embed = nn.Embedding(100, 512)</b>: the 100 decoder query positional "
        "embeddings. Each row is a unique \"predict action at offset i\" tag.",
        "<b>action_head: 512 -&gt; 8</b>. Maps each decoder output token to a single "
        "action vector; same head shared across all 100 query positions.",
    ])

    code_block(story, "model/cvae.py:82-105 - _encode_style", """\
def _encode_style(self, qpos, actions, is_pad):
    B = qpos.size(0); device = qpos.device

    cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)        # (B, 1, 512)
    qpos_tok = self.style_qpos_proj(qpos).unsqueeze(1)                # (B, 1, 512)
    act_toks = self.style_action_proj(actions)                        # (B, 100, 512)
    seq = torch.cat([cls, qpos_tok, act_toks], dim=1)                 # (B, 102, 512)
    seq = seq.permute(1, 0, 2).contiguous()                           # (102, B, 512)

    pos = self.style_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (102, B, 512)

    always_valid = torch.zeros(B, 2, dtype=torch.bool, device=device)
    pad_mask = torch.cat([always_valid, is_pad], dim=1)               # (B, 102)

    out = self.style_encoder(seq, src_key_padding_mask=pad_mask, pos=pos)
    cls_out = out[0]                                                   # (B, 512)
    mu, logvar = self.latent_proj(cls_out).chunk(2, dim=-1)           # 2x (B, 32)
    return mu, logvar""")
    bullets(story, [
        "<b>cls.weight</b> shape is <code>(1, 512)</code>; "
        "<code>.unsqueeze(0)</code> &rarr; <code>(1, 1, 512)</code>; "
        "<code>.expand(B, -1, -1)</code> &rarr; <code>(B, 1, 512)</code> — broadcast "
        "across the batch without copying memory.",
        "<b>style_qpos_proj(qpos)</b>: <code>(B, 8) @ (8, 512) = (B, 512)</code>; "
        "<code>.unsqueeze(1)</code> &rarr; <code>(B, 1, 512)</code>.",
        "<b>style_action_proj(actions)</b>: <code>(B, 100, 8) @ (8, 512) = "
        "(B, 100, 512)</code>. nn.Linear broadcasts over the timestep axis.",
        "<b>cat dim=1</b>: <code>(B,1,512) + (B,1,512) + (B,100,512) = (B,102,512)</code>.",
        "<b>permute(1,0,2)</b>: <code>(B,102,512) &rarr; (102,B,512)</code> for "
        "torch.nn.MultiheadAttention's sequence-first convention.",
        "<b>pad_mask</b>: 2 always-valid rows for [CLS, qpos] (False = unmasked) "
        "concatenated with the action <code>is_pad</code> from the dataset. Shape "
        "<code>(B, 102)</code>. True positions are masked out of attention.",
        "<b>cls_out = out[0]</b> grabs sequence position 0 (the CLS row). "
        "<code>latent_proj(cls_out)</code>: <code>(B, 512) @ (512, 64) = (B, 64)</code>; "
        "<code>.chunk(2, dim=-1)</code> splits to <code>mu (B, 32), logvar (B, 32)</code>.",
    ])

    code_block(story, "model/cvae.py:107-127 - _encode_image", """\
def _encode_image(self, image):
    # image: (B, num_cam, 3, H, W)
    B, N = image.shape[:2]
    flat = image.flatten(0, 1)                                         # (B*N, 3, H, W)
    feat, pos = self.backbone(flat)
    if feat.dim() == 4:
        # Single-encoder backbone: (B*N, D, Hp, Wp)
        D, Hp, Wp = feat.shape[1:]
        feat = feat.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
        # -> (B, D, N, Hp, Wp) -> (B, D, N*Hp*Wp)
        pos = pos.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
        feat = feat.permute(2, 0, 1).contiguous()                      # (N*Hp*Wp, B, D)
        pos = pos.permute(2, 0, 1).contiguous()
    else:
        # Dual-encoder backbone: feat is already (S, B*N, D)
        S, BN, D = feat.shape
        if N > 1:
            feat = feat.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
            pos = pos.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
    return feat, pos""")
    bullets(story, [
        "<b>image.flatten(0, 1)</b>: collapses batch and camera axes so the backbone "
        "sees one batch of <code>B*N</code> images. With <code>N = 1</code> camera "
        "today this is functionally a no-op but keeps the multi-camera path open.",
        "<b>Single-encoder reshape (4D output)</b>: backbone gives "
        "<code>(B*N, D, Hp, Wp)</code>. We split B and N back out, permute to put "
        "channels first, then flatten cams + spatial dims. Final permute makes it "
        "sequence-first <code>(S_img, B, D)</code> where <code>S_img = N*Hp*Wp</code>.",
        "<b>Dual-encoder reshape (3D output)</b>: backbone already returns "
        "<code>(S, B*N, D)</code>. With one camera we just rename the second axis "
        "to B; with multiple cameras we stack each camera's tokens along the sequence.",
        "<b>For ResNet18, B=8, N=1, 240x320</b>: <code>feat (8,512,7,10) &rarr; "
        "(8,512,1,7,10) &rarr; (8,512,1,7,10) [permuted] &rarr; (8,512,70) &rarr; "
        "(70, 8, 512)</code>. S_img = 70.",
        "<b>For DINOv2-S+Cellpose, B=8</b>: backbone returns <code>(674, 8, 512)</code> "
        "directly; with N=1 the else-branch is a no-op.",
    ])

    code_block(story, "model/cvae.py:131-159 - forward", """\
def forward(self, image, qpos, actions=None, is_pad=None):
    B = qpos.size(0); device = qpos.device

    if actions is not None:
        mu, logvar = self._encode_style(qpos, actions, is_pad)         # (B,32), (B,32)
        z = reparameterize(mu, logvar)                                 # (B, 32)
    else:
        mu = torch.zeros(B, self.latent_dim, device=device)            # inference: prior mean
        logvar = torch.zeros(B, self.latent_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

    img_feat, img_pos = self._encode_image(image)                      # (S_img, B, 512)
    latent_tok = self.latent_to_src(z).unsqueeze(0)                    # (1, B, 512)
    qpos_tok = self.qpos_to_src(qpos).unsqueeze(0)                     # (1, B, 512)
    extra_pos = self.extra_src_pos.weight.unsqueeze(1).expand(-1, B, -1)  # (2, B, 512)

    src = torch.cat([latent_tok, qpos_tok, img_feat], dim=0)           # (2 + S_img, B, 512)
    src_pos = torch.cat([extra_pos, img_pos], dim=0)                   # (2 + S_img, B, 512)

    hs = self.transformer(src, src_pos, self.query_embed.weight)       # (100, B, 512)
    a_hat = self.action_head(hs.transpose(0, 1))                       # (B, 100, 8)
    return a_hat, (mu, logvar)""")
    bullets(story, [
        "<b>Inference branch</b> (actions is None): z = 0 instead of sampling. mu / "
        "logvar are zeros so the loss-time KL would be zero — the rollout never "
        "computes loss, so this is just placeholder.",
        "<b>latent_to_src(z)</b>: <code>(B, 32) @ (32, 512) = (B, 512)</code>; "
        "<code>.unsqueeze(0)</code> &rarr; <code>(1, B, 512)</code>.",
        "<b>qpos_to_src(qpos)</b>: <code>(B, 8) @ (8, 512) = (B, 512)</code>; "
        "<code>.unsqueeze(0)</code> &rarr; <code>(1, B, 512)</code>. Notice the "
        "main encoder uses a <i>separate</i> Linear from the style encoder's "
        "<code>style_qpos_proj</code>. Same input, different mapping.",
        "<b>cat along dim=0 (sequence axis)</b>: <code>(1,B,512) + (1,B,512) + "
        "(S_img,B,512) = (2+S_img, B, 512)</code>. For ResNet18 S_img=70 &rarr; total "
        "72; for DINOv2+Cellpose S_img=674 &rarr; total 676.",
        "<b>query_embed.weight</b> shape <code>(100, 512)</code> is passed straight "
        "to the transformer; it gets broadcast to <code>(100, B, 512)</code> inside.",
        "<b>hs.transpose(0, 1)</b>: <code>(100, B, 512) &rarr; (B, 100, 512)</code> "
        "before the linear head, so the head's matmul stays a clean batched "
        "<code>(B, 100, 512) @ (512, 8) = (B, 100, 8)</code>.",
    ])

    code_block(story, "model/cvae.py:162-163 - build_cvae", """\
def build_cvae(**kwargs) -> ACTCVAE:
    return ACTCVAE(**kwargs)""")
    bullets(story, [
        "Trivial factory; primarily exists so visualization / export scripts can call "
        "<code>build_cvae(backbone_name=...)</code> without importing the class itself.",
    ])

    # =====================================================================
    # model/act_policy.py
    # =====================================================================
    h1(story, "model/act_policy.py")
    h2(story, "Purpose")
    body(story, "Wraps <code>ACTCVAE</code> with: (1) the dataset normalization stats "
                "registered as buffers so a checkpoint is self-contained, (2) the "
                "training-time loss (masked L1 + KL), and (3) a numpy-in/numpy-out "
                "<code>inference()</code> entry point for the rollout loop.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Buffers: <code>qpos_mean (8,), qpos_std (8,), action_mean (8,), action_std "
        "(8,), image_mean (3,1,1), image_std (3,1,1)</code>.",
        "<code>forward(image, qpos, action, is_pad)</code> at training: "
        "<code>image (B,1,3,240,320), qpos (B,8), action (B,100,8), is_pad (B,100)</code> "
        "&rarr; <code>{loss, l1, kl}</code> dict of 0-D tensors.",
        "<code>forward(image, qpos)</code> at eval (no actions): returns a_hat "
        "<code>(B, 100, 8)</code>.",
        "<code>inference(image_np, qpos_np)</code>: <code>image_np (H,W,3) uint8</code>, "
        "<code>qpos_np (8,) float</code> &rarr; numpy <code>(100, 8)</code> in raw "
        "Sensapex counts.",
    ])

    code_block(story, "model/act_policy.py:26-46 - ACTPolicy.__init__", """\
class ACTPolicy(nn.Module):
    def __init__(self, stats: dict, kl_weight: float = C.KL_WEIGHT, **cvae_kwargs):
        super().__init__()
        self.model = ACTCVAE(**cvae_kwargs)
        self.kl_weight = kl_weight

        self.register_buffer("qpos_mean",   torch.from_numpy(stats["qpos_mean"]))    # (8,)
        self.register_buffer("qpos_std",    torch.from_numpy(stats["qpos_std"]))     # (8,)
        self.register_buffer("action_mean", torch.from_numpy(stats["action_mean"]))  # (8,)
        self.register_buffer("action_std",  torch.from_numpy(stats["action_std"]))   # (8,)
        self.register_buffer(
            "image_mean", torch.from_numpy(stats["image_mean"]).view(3, 1, 1)        # (3,1,1)
        )
        self.register_buffer(
            "image_std", torch.from_numpy(stats["image_std"]).view(3, 1, 1)          # (3,1,1)
        )""")
    bullets(story, [
        "<b>register_buffer</b>: stores as part of <code>state_dict()</code> AND moves "
        "with <code>.to(device)</code>, but is NOT trainable. Perfect for "
        "normalization stats.",
        "<b>image_mean / image_std reshaped to (3, 1, 1)</b>: broadcast-ready against "
        "a <code>(3, H, W)</code> CHW tensor in <code>_preprocess_image</code>.",
        "<b>self.model = ACTCVAE(**cvae_kwargs)</b>: the policy delegates the actual "
        "forward to ACTCVAE. <code>build_optimizer</code> reaches the backbone via "
        "<code>policy.model.backbone</code>.",
    ])

    code_block(story, "model/act_policy.py:52-65 - forward", """\
def forward(self, image, qpos, actions=None, is_pad=None):
    if actions is not None:
        assert is_pad is not None, "is_pad required when actions is given"
        a_hat, (mu, logvar) = self.model(image, qpos, actions, is_pad)
        return self._compute_loss(a_hat, actions, is_pad, mu, logvar)
    a_hat, _ = self.model(image, qpos)
    return a_hat""")
    bullets(story, [
        "Single forward serves both training and eval. Training callers pass actions; "
        "they get back a loss dict. Eval callers omit actions; they get the predicted "
        "chunk tensor directly.",
    ])

    code_block(story, "model/act_policy.py:67-82 - _compute_loss (masked L1 + KL)", """\
def _compute_loss(self, a_hat, actions, is_pad, mu, logvar):
    l1 = F.l1_loss(a_hat, actions, reduction="none")           # (B, 100, 8)
    valid = (~is_pad).unsqueeze(-1).float()                    # (B, 100, 1)
    l1 = (l1 * valid).sum() / (valid.sum().clamp_min(1.0) * actions.size(-1))

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    total = l1 + self.kl_weight * kl
    return {"loss": total, "l1": l1.detach(), "kl": kl.detach()}""")
    bullets(story, [
        "<b>L1 reduction='none'</b> keeps the per-element absolute error so we can "
        "weight padded rows to zero. Shape <code>(B, 100, 8)</code>.",
        "<b>valid = (~is_pad).unsqueeze(-1).float()</b>: <code>(B, 100, 1)</code> "
        "with 1.0 for real timesteps and 0.0 for padded ones. Broadcast multiplies "
        "to <code>(B, 100, 8)</code>.",
        "<b>Denominator</b>: <code>valid.sum() * actions.size(-1)</code> = "
        "(number of real timesteps across the batch) * 8 = total real action "
        "elements. <code>clamp_min(1.0)</code> guards a degenerate batch.",
        "<b>KL closed form</b>: <code>-0.5 * sum(1 + logvar - mu^2 - exp(logvar))</code> "
        "summed over the latent dim then averaged over batch &rarr; 0-D scalar. "
        "Standard <code>KL(N(mu, sigma^2) || N(0, I))</code>.",
        "<b>kl_weight = 10.0</b> (the ACT paper default). Without it the L1 dominates "
        "and the latent collapses (mu &rarr; 0, sigma &rarr; 1) so the model never "
        "learns multimodal styles.",
        "<code>l1.detach()</code> and <code>kl.detach()</code> are returned for "
        "logging — only <code>total</code> tracks gradients.",
    ])

    code_block(story, "model/act_policy.py:88-120 - inference + _preprocess_image", """\
@torch.no_grad()
def inference(self, image_np: np.ndarray, qpos_np: np.ndarray) -> np.ndarray:
    self.eval()
    device = self.qpos_mean.device

    img = self._preprocess_image(image_np).to(device).unsqueeze(0)    # (1, num_cam, 3, H, W)
    qpos = torch.from_numpy(qpos_np.astype(np.float32)).to(device)
    qpos = ((qpos - self.qpos_mean) / self.qpos_std).unsqueeze(0)     # (1, 8)

    a_hat, _ = self.model(img, qpos)                                  # (1, 100, 8)
    a = a_hat[0] * self.action_std + self.action_mean                 # de-normalize -> (100, 8)
    return a.cpu().numpy().astype(np.float32)


def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
    h, w = img_np.shape[:2]
    if (h, w) != (C.IMAGE_HEIGHT, C.IMAGE_WIDTH):
        pil = Image.fromarray(img_np).resize((C.IMAGE_WIDTH, C.IMAGE_HEIGHT),
                                              Image.BILINEAR)
        img_np = np.array(pil)                          # copy so torch can own it
    x = torch.from_numpy(img_np).float() / 255.0        # (H, W, 3)
    x = x.permute(2, 0, 1)                              # (3, H, W)
    x = (x - self.image_mean) / self.image_std
    return x.unsqueeze(0)                               # (num_cam=1, 3, H, W)""")
    bullets(story, [
        "<b>Single-sample numpy in / numpy out</b>: rollout passes a raw uint8 RGB "
        "frame and an 8-D state in centered Sensapex counts. Inference returns the "
        "100-action chunk in raw units (de-normalized).",
        "<b>img after preprocessing</b>: <code>(3, 240, 320) &rarr; "
        "(num_cam=1, 3, 240, 320)</code> after <code>.unsqueeze(0)</code>; the call "
        "site adds another <code>.unsqueeze(0)</code> for batch &rarr; "
        "<code>(1, 1, 3, 240, 320)</code>.",
        "<b>qpos normalization</b>: subtract per-axis mean, divide by per-axis std. "
        "Buffers were registered as <code>(8,)</code> so the broadcast against "
        "<code>(8,)</code> qpos is element-wise; <code>.unsqueeze(0)</code> adds the "
        "batch dim before passing to the model.",
        "<b>De-normalization</b>: <code>a_hat[0]</code> shape "
        "<code>(100, 8)</code>; multiply by <code>action_std (8,)</code> "
        "(broadcasts) and add <code>action_mean (8,)</code> &rarr; raw absolute "
        "Sensapex targets.",
        "<b>BGR vs RGB warning</b>: the docstring reminds callers using OpenCV to "
        "convert BGR&rarr;RGB first. Training data was loaded with PIL (RGB) so "
        "feeding BGR at rollout would scramble channel-wise normalization.",
    ])

    code_block(story, "model/act_policy.py:125-134 - build_policy", """\
def build_policy(stats: Optional[dict] = None,
                 stats_path: Path = C.STATS_PATH,
                 kl_weight: float = C.KL_WEIGHT,
                 **cvae_kwargs) -> ACTPolicy:
    if stats is None:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
    return ACTPolicy(stats=stats, kl_weight=kl_weight, **cvae_kwargs)""")
    bullets(story, [
        "Two ways to construct: pass <code>stats</code> directly (training, where the "
        "dataset already computed them) or let the function load them from "
        "<code>STATS_PATH</code> (rollout, where the dataset is not on disk).",
    ])
