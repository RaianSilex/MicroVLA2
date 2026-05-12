from build_report import h1, h2, h3, body, bullets, code_block


def add(story):
    # =====================================================================
    # model/embodiment.py
    # =====================================================================
    h1(story, "model/embodiment.py (NEW)")
    h2(story, "Purpose")
    body(story, "Turns the five categorical dataset metadata fields "
                "(robot_id, lab_id, embodiment, action_type, task_family) into "
                "5 source tokens for the VLA transformer. Each field has its own "
                "<code>nn.Embedding</code> table indexed by the integer ID assigned "
                "during dataset vocab construction.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Inputs: 5 long tensors each of shape <b>(B,)</b>, holding the per-sample "
        "vocab ID for each field.",
        "Output: <b>(5, B, 512)</b>, sequence-first so it concatenates directly "
        "into the main encoder source.",
        "<code>num_tokens = 5</code> is published as an attribute so VLACVAE can "
        "reserve the right number of source positions in <code>extra_src_pos</code>.",
    ])

    code_block(story, "model/embodiment.py:11-33 - EmbodimentConditioner.__init__", """\
class EmbodimentConditioner(nn.Module):
    def __init__(
        self,
        hidden_dim: int = C.HIDDEN_DIM,
        num_robot_ids: int = C.NUM_ROBOT_IDS_FALLBACK,
        num_lab_ids: int = C.NUM_LAB_IDS_FALLBACK,
        num_embodiment_ids: int = C.NUM_EMBODIMENT_IDS_FALLBACK,
        num_action_type_ids: int = C.NUM_ACTION_TYPE_IDS_FALLBACK,
        num_task_family_ids: int = C.NUM_TASK_FAMILY_IDS_FALLBACK,
    ):
        super().__init__()
        self.robot_embed = nn.Embedding(num_robot_ids, hidden_dim)             # (V_robot, 512)
        self.lab_embed = nn.Embedding(num_lab_ids, hidden_dim)                 # (V_lab, 512)
        self.embodiment_embed = nn.Embedding(num_embodiment_ids, hidden_dim)
        self.action_type_embed = nn.Embedding(num_action_type_ids, hidden_dim)
        self.task_family_embed = nn.Embedding(num_task_family_ids, hidden_dim)
        self.num_tokens = 5""")
    bullets(story, [
        "Each <code>nn.Embedding(V, 512)</code> is a learnable lookup table of shape "
        "<code>(V, 512)</code>. Indexing it with a long tensor of IDs returns the "
        "matching row(s).",
        "<b>Vocab sizes</b>: <code>VLAPolicy</code> overrides the fallbacks "
        "(64/64/32/32/64) at construction time with the actual vocab counts derived "
        "from the dataset (e.g. 5 robots, 3 labs, ...). The fallbacks ensure the "
        "module is well-defined even when imported standalone for unit tests.",
    ])

    code_block(story, "model/embodiment.py:35-53 - forward", """\
def forward(
    self,
    robot_id: torch.Tensor,
    lab_id: torch.Tensor,
    embodiment_id: torch.Tensor,
    action_type_id: torch.Tensor,
    task_family_id: torch.Tensor,
) -> torch.Tensor:
    tokens = torch.stack(
        [
            self.robot_embed(robot_id),               # (B, 512)
            self.lab_embed(lab_id),                   # (B, 512)
            self.embodiment_embed(embodiment_id),     # (B, 512)
            self.action_type_embed(action_type_id),   # (B, 512)
            self.task_family_embed(task_family_id),   # (B, 512)
        ],
        dim=0,
    )
    return tokens                                     # (5, B, 512)""")
    bullets(story, [
        "Each embedding lookup with a <code>(B,)</code> ID tensor returns "
        "<code>(B, 512)</code>. <code>torch.stack(..., dim=0)</code> stacks the five "
        "results into a new leading axis &rarr; <code>(5, B, 512)</code>.",
        "<b>Sequence-first by construction</b>: dim 0 is the token / sequence axis "
        "as required by <code>torch.nn.MultiheadAttention</code>, so VLACVAE can "
        "concatenate this output straight into the main encoder source.",
        "Token order is fixed: <code>[robot, lab, embodiment, action_type, "
        "task_family]</code>. The transformer is order-invariant up to position "
        "embeddings, but the order is also used to slot specific entries in "
        "<code>extra_src_pos</code>.",
    ])

    # =====================================================================
    # model/language_encoder.py
    # =====================================================================
    h1(story, "model/language_encoder.py (NEW)")
    h2(story, "Purpose")
    body(story, "Two interchangeable language encoders for VLA: "
                "<b>HuggingFaceTextEncoder</b> (frozen DistilBERT/BERT + trainable "
                "linear projection to 512) and <b>SimpleHashTextEncoder</b> (offline "
                "hash-based fallback for smoke tests). Both produce sequence-first "
                "<code>(L, B, 512)</code> token tensors plus a <code>(B, L)</code> "
                "padding mask.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Forward input: <code>instructions</code>, either a single string or a list "
        "of strings of length B.",
        "Forward output: <b>(tokens (L, B, 512), pad_mask (B, L))</b> where "
        "<code>L = MAX_LANGUAGE_TOKENS = 32</code>.",
        "<code>pad_mask</code> is True at PAD positions (matching MultiheadAttention's "
        "<code>key_padding_mask</code> convention).",
    ])

    code_block(story, "model/language_encoder.py:20-23 - _as_list helper", """\
def _as_list(instructions: Sequence[str] | str) -> List[str]:
    if isinstance(instructions, str):
        return [instructions]
    return [str(x) for x in instructions]""")
    bullets(story, [
        "Allows both styles of input: training passes a Python list of B strings "
        "(DataLoader's default collate_fn), inference passes a single string.",
    ])

    code_block(story, "model/language_encoder.py:26-58 - HuggingFaceTextEncoder.__init__", """\
class HuggingFaceTextEncoder(nn.Module):
    def __init__(self,
                 model_name: str = C.DEFAULT_TEXT_MODEL,    # "distilbert-base-uncased"
                 hidden_dim: int = C.HIDDEN_DIM,            # 512
                 max_tokens: int = C.MAX_LANGUAGE_TOKENS):  # 32
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()

        model_dim = int(self.text_model.config.hidden_size)   # 768 for DistilBERT
        self.proj = nn.Linear(model_dim, hidden_dim)          # 768 -> 512

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()    # frozen text model never goes back to train mode
        return self""")
    bullets(story, [
        "<b>Frozen text encoder + trainable projection</b>: ~66M frozen DistilBERT "
        "params (no gradients) plus a single <code>768 &rarr; 512</code> Linear "
        "with ~393K trainable params. Cheap to train, retains the pretraining "
        "knowledge.",
        "<b>train() override</b> mirrors the DINOv2 / Cellpose pattern — "
        "<code>policy.train()</code> propagates here but the text model is forced "
        "back into eval so dropout / LayerNorm running stats stay frozen.",
        "<code>model_dim</code> is read from the loaded model's config so swapping "
        "to BERT-base (768), BERT-large (1024), etc. needs no code change.",
    ])

    code_block(story, "model/language_encoder.py:60-75 - HuggingFaceTextEncoder.forward", """\
def forward(self, instructions):
    texts = _as_list(instructions)
    device = self.proj.weight.device
    enc = self.tokenizer(
        texts,
        padding="max_length",      # pad every sequence up to max_tokens (32)
        truncation=True,           # truncate longer sequences
        max_length=self.max_tokens,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = self.text_model(**enc).last_hidden_state                  # (B, 32, 768)
    tokens = self.proj(out).permute(1, 0, 2).contiguous()               # (32, B, 512)
    pad_mask = ~enc["attention_mask"].bool()                            # (B, 32) True where PAD
    return tokens, pad_mask""")
    bullets(story, [
        "<b>Tokenizer output</b>: <code>enc[\"input_ids\"] (B, 32)</code> token IDs, "
        "<code>enc[\"attention_mask\"] (B, 32)</code> with 1 for real tokens and 0 "
        "for padding.",
        "<b>last_hidden_state shape</b>: <code>(B, 32, 768)</code> for DistilBERT — "
        "one 768-D vector per token position.",
        "<b>self.proj(out)</b>: <code>(B, 32, 768) @ (768, 512) = (B, 32, 512)</code>; "
        "<code>.permute(1, 0, 2)</code> &rarr; <code>(32, B, 512)</code> for "
        "sequence-first.",
        "<b>pad_mask</b>: <code>~attention_mask.bool()</code> inverts so True marks "
        "PAD positions (which the transformer's <code>key_padding_mask</code> will "
        "ignore in attention).",
    ])

    code_block(story, "model/language_encoder.py:78-113 - SimpleHashTextEncoder", """\
class SimpleHashTextEncoder(nn.Module):
    _word_re = re.compile(r"[A-Za-z0-9_]+|[^\\sA-Za-z0-9_]")

    def __init__(self, hidden_dim=C.HIDDEN_DIM,
                 max_tokens=C.MAX_LANGUAGE_TOKENS, vocab_size=C.SIMPLE_TEXT_VOCAB_SIZE):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_tokens = int(max_tokens)
        self.vocab_size = int(vocab_size)
        self.embed = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(self.max_tokens, hidden_dim)

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % (self.vocab_size - 1) + 1

    def _encode_one(self, text: str) -> List[int]:
        pieces = self._word_re.findall(text)[: self.max_tokens]
        ids = [self._token_id(p) for p in pieces]
        ids.extend([0] * (self.max_tokens - len(ids)))
        return ids

    def forward(self, instructions):
        texts = _as_list(instructions)
        device = self.embed.weight.device
        ids = torch.tensor([self._encode_one(t) for t in texts],
                           dtype=torch.long, device=device)            # (B, 32)
        positions = torch.arange(self.max_tokens, dtype=torch.long, device=device)
        tokens = self.embed(ids) + self.pos_embed(positions).unsqueeze(0)  # (B, 32, 512)
        pad_mask = ids.eq(0)                                               # (B, 32)
        return tokens.permute(1, 0, 2).contiguous(), pad_mask""")
    bullets(story, [
        "<b>Why a hash tokenizer</b>: the HF backend pulls ~250 MB of weights and "
        "expects internet on first run. The simple backend is fully deterministic, "
        "offline, and lets the unit tests / ONNX export pipeline run without optional "
        "dependencies. <i>Not</i> a substitute for real language pretraining.",
        "<b>BLAKE2b 4-byte digest</b> hashed mod (vocab_size - 1) + 1 keeps IDs in "
        "<code>[1, vocab_size)</code>; ID 0 is reserved as PAD (matches "
        "<code>padding_idx=0</code> in the embedding).",
        "<b>Word regex</b>: <code>[A-Za-z0-9_]+|[^\\sA-Za-z0-9_]</code> splits into "
        "alphanumeric runs plus single non-alphanumeric symbols.",
        "<b>Position embeddings are learned</b>, not sinusoidal — same as DistilBERT. "
        "Added to the token embeddings before permuting to sequence-first.",
        "<b>pad_mask = ids.eq(0)</b>: True where the token is the PAD slot. Same "
        "semantics as the HF encoder's pad_mask.",
    ])

    code_block(story, "model/language_encoder.py:116-127 - build_language_encoder", """\
def build_language_encoder(
    backend: str = C.LANGUAGE_BACKEND,
    model_name: str = C.DEFAULT_TEXT_MODEL,
    hidden_dim: int = C.HIDDEN_DIM,
    max_tokens: int = C.MAX_LANGUAGE_TOKENS,
) -> nn.Module:
    backend = str(backend).lower()
    if backend == "hf":
        return HuggingFaceTextEncoder(model_name=model_name,
                                      hidden_dim=hidden_dim, max_tokens=max_tokens)
    if backend == "simple":
        return SimpleHashTextEncoder(hidden_dim=hidden_dim, max_tokens=max_tokens)
    raise ValueError(f"Unsupported language backend: {backend!r}")""")

    # =====================================================================
    # model/vla_cvae.py
    # =====================================================================
    h1(story, "model/vla_cvae.py (NEW)")
    h2(story, "Purpose")
    body(story, "Vision-language-action CVAE. Keeps the ACT decoder + 100-step "
                "action chunking, but conditions the encoder on "
                "<b>image tokens + instruction tokens + padded robot state + "
                "embodiment metadata</b>. Action output is always sized to "
                "<code>MAX_ACTION_DIM = 16</code>; the loss masks invalid dims.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "<code>state_dim = MAX_STATE_DIM = 16</code>, "
        "<code>action_dim = MAX_ACTION_DIM = 16</code> by default.",
        "<code>num_non_image_tokens = 2 + 5 + 32 = 39</code>: "
        "[latent, qpos] + 5 metadata + 32 language tokens.",
        "<code>extra_src_pos = nn.Embedding(39, 512)</code> covers all non-image "
        "positions; image tokens get the 2D sinusoidal pos from the backbone.",
        "<code>forward(...)</code> returns <code>a_hat (B, 100, 16), "
        "(mu (B, 32), logvar (B, 32))</code>.",
    ])

    code_block(story, "model/vla_cvae.py:27-94 - VLACVAE.__init__", """\
class VLACVAE(nn.Module):
    def __init__(self, state_dim=16, action_dim=16, hidden_dim=512, latent_dim=32,
                 chunk_size=100, num_cameras=1, pretrained_backbone=True,
                 backbone_name=C.DEFAULT_BACKBONE, freeze_backbone=True,
                 language_backend=C.LANGUAGE_BACKEND, text_model_name=C.DEFAULT_TEXT_MODEL,
                 max_language_tokens=C.MAX_LANGUAGE_TOKENS,
                 num_robot_ids=..., num_lab_ids=..., num_embodiment_ids=...,
                 num_action_type_ids=..., num_task_family_ids=...):
        super().__init__()
        self.state_dim = int(state_dim)            # 16
        self.action_dim = int(action_dim)          # 16
        self.hidden_dim = int(hidden_dim)          # 512
        self.latent_dim = int(latent_dim)          # 32
        self.chunk_size = int(chunk_size)          # 100
        self.num_cameras = int(num_cameras)        # 1
        self.max_language_tokens = int(max_language_tokens)   # 32

        self.backbone = build_backbone(hidden_dim=hidden_dim, ...)
        self.language_encoder = build_language_encoder(
            backend=language_backend, model_name=text_model_name,
            hidden_dim=hidden_dim, max_tokens=max_language_tokens)
        self.embodiment = EmbodimentConditioner(hidden_dim=hidden_dim, ...)

        self.transformer = build_transformer(d_model=hidden_dim)
        self.style_encoder = build_encoder(d_model=hidden_dim)

        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.style_qpos_proj = nn.Linear(state_dim, hidden_dim)            # 16 -> 512
        self.style_action_proj = nn.Linear(action_dim, hidden_dim)         # 16 -> 512
        self.style_pos_embed = nn.Embedding(1 + 1 + chunk_size, hidden_dim)  # 102 rows
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)           # 512 -> 64

        self.latent_to_src = nn.Linear(latent_dim, hidden_dim)             # 32 -> 512
        self.qpos_to_src = nn.Linear(state_dim, hidden_dim)                # 16 -> 512

        self.num_non_image_tokens = 2 + self.embodiment.num_tokens + self.max_language_tokens
        # = 2 + 5 + 32 = 39
        self.extra_src_pos = nn.Embedding(self.num_non_image_tokens, hidden_dim)

        self.query_embed = nn.Embedding(chunk_size, hidden_dim)            # 100 queries
        self.action_head = nn.Linear(hidden_dim, action_dim)               # 512 -> 16""")
    bullets(story, [
        "Mirrors <code>ACTCVAE.__init__</code> almost line-for-line. The notable "
        "differences are: state/action dims are 16 (padded), three new sub-modules "
        "(<code>language_encoder</code>, <code>embodiment</code>) and "
        "<code>extra_src_pos</code> covers 39 positions instead of 2.",
        "Vocab sizes (num_robot_ids, ...) are passed in by VLAPolicy with the actual "
        "values from <code>VocabBundle</code>; the fallbacks from <code>vla_config</code> "
        "(64/64/32/32/64) are only used if VLACVAE is built standalone.",
    ])

    code_block(story, "model/vla_cvae.py:96-120 - _encode_style (mask-aware)", """\
def _encode_style(self, qpos, actions, is_pad, state_mask=None, action_mask=None):
    B = qpos.size(0); device = qpos.device
    if state_mask is not None:
        qpos = qpos * state_mask.float()                  # zero padded state dims
    if action_mask is not None:
        actions = actions * action_mask[:, None, :].float()  # zero padded action dims

    cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)        # (B, 1, 512)
    qpos_tok = self.style_qpos_proj(qpos).unsqueeze(1)                # (B, 1, 512)
    act_toks = self.style_action_proj(actions)                        # (B, 100, 512)
    seq = torch.cat([cls, qpos_tok, act_toks], dim=1).permute(1, 0, 2).contiguous()
    # seq: (102, B, 512)
    pos = self.style_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (102, B, 512)

    always_valid = torch.zeros(B, 2, dtype=torch.bool, device=device)
    pad_mask = torch.cat([always_valid, is_pad], dim=1)               # (B, 102)
    out = self.style_encoder(seq, src_key_padding_mask=pad_mask, pos=pos)
    return self.latent_proj(out[0]).chunk(2, dim=-1)                  # 2x (B, 32)""")
    bullets(story, [
        "<b>Mask-aware variant</b> of ACT's <code>_encode_style</code>. Before the "
        "linear projections, padded state and action dims are explicitly zeroed so "
        "the model does not see arbitrary garbage from those slots — the result is "
        "the same as if those dims didn't exist.",
        "<b>state_mask shape</b> is <code>(B, 16)</code>; "
        "<code>state_mask.float()</code> &rarr; <code>(B, 16)</code> "
        "broadcasts against <code>qpos (B, 16)</code> elementwise.",
        "<b>action_mask shape</b> is <code>(B, 16)</code>; "
        "<code>action_mask[:, None, :]</code> &rarr; <code>(B, 1, 16)</code> "
        "broadcasts against <code>actions (B, 100, 16)</code>.",
        "Sequence layout and pad_mask construction are identical to ACT — only "
        "the inputs are higher-dimensional (16 instead of 8) before the projection.",
    ])

    code_block(story, "model/vla_cvae.py:122-137 - _encode_image (same as ACT)", """\
def _encode_image(self, image):
    B, N = image.shape[:2]
    flat = image.flatten(0, 1)                            # (B*N, 3, H, W)
    feat, pos = self.backbone(flat)
    if feat.dim() == 4:
        D, Hp, Wp = feat.shape[1:]
        feat = feat.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
        pos = pos.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
        feat = feat.permute(2, 0, 1).contiguous()         # (N*Hp*Wp, B, 512)
        pos = pos.permute(2, 0, 1).contiguous()
    else:
        S, _BN, D = feat.shape
        if N > 1:
            feat = feat.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
            pos = pos.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
    return feat, pos""")
    bullets(story, [
        "Identical structurally to ACT's <code>_encode_image</code>. The default VLA "
        "backbone is <code>dinov2_vits14+cellpose4</code>, so the dual-encoder branch "
        "(else) is the active path. With <code>CELLPOSE4_DIAMETER=180</code>, output "
        "is roughly <code>(409, B, 512)</code> for the default 240x320 input.",
    ])

    code_block(story, "model/vla_cvae.py:139-200 - forward", """\
def forward(self, image, qpos, instructions,
            robot_id, lab_id, embodiment_id, action_type_id, task_family_id,
            state_mask=None, action_mask=None,
            actions=None, is_pad=None):
    B = qpos.size(0); device = qpos.device
    if state_mask is not None:
        qpos = qpos * state_mask.float()                  # zero padded state dims

    if actions is not None:
        if is_pad is None:
            raise ValueError("is_pad is required when actions are provided")
        mu, logvar = self._encode_style(qpos, actions, is_pad, state_mask, action_mask)
        z = reparameterize(mu, logvar)                    # (B, 32)
    else:
        mu = torch.zeros(B, self.latent_dim, device=device)
        logvar = torch.zeros(B, self.latent_dim, device=device)
        z = torch.zeros(B, self.latent_dim, device=device)

    img_feat, img_pos = self._encode_image(image)         # (S_img, B, 512)
    lang_tokens, lang_pad = self.language_encoder(instructions)  # (32, B, 512), (B, 32)
    meta_tokens = self.embodiment(robot_id, lab_id, embodiment_id,
                                  action_type_id, task_family_id) # (5, B, 512)

    latent_tok = self.latent_to_src(z).unsqueeze(0)               # (1, B, 512)
    qpos_tok = self.qpos_to_src(qpos).unsqueeze(0)                # (1, B, 512)
    non_image = torch.cat([latent_tok, qpos_tok, meta_tokens, lang_tokens], dim=0)
    # non_image: (1 + 1 + 5 + 32, B, 512) = (39, B, 512)
    src = torch.cat([non_image, img_feat], dim=0)                 # (39 + S_img, B, 512)

    pos_non_image = self.extra_src_pos.weight[: non_image.size(0)]  # (39, 512)
    pos_non_image = pos_non_image.unsqueeze(1).expand(-1, B, -1)    # (39, B, 512)
    src_pos = torch.cat([pos_non_image, img_pos], dim=0)            # (39 + S_img, B, 512)

    fixed_valid = torch.zeros(B, 2 + self.embodiment.num_tokens,    # (B, 7)
                              dtype=torch.bool, device=device)
    img_valid = torch.zeros(B, img_feat.size(0), dtype=torch.bool, device=device)
    src_key_padding_mask = torch.cat([fixed_valid, lang_pad, img_valid], dim=1)
    # src_key_padding_mask: (B, 7 + 32 + S_img) = (B, 39 + S_img)

    hs = self.transformer(src, src_pos, self.query_embed.weight,
                          src_key_padding_mask=src_key_padding_mask)  # (100, B, 512)
    a_hat = self.action_head(hs.transpose(0, 1))                       # (B, 100, 16)
    if action_mask is not None:
        a_hat = a_hat * action_mask[:, None, :].float()                # zero invalid dims
    return a_hat, (mu, logvar)""")
    bullets(story, [
        "<b>Token concatenation order in the source</b>: latent (1) + qpos (1) + "
        "5 metadata + 32 language + S_img image. For the default DINOv2+Cellpose4 at "
        "240x320 with 1 camera and the default Cellpose 4 diameter scaling: "
        "<code>2 + 5 + 32 + ~409 = ~448</code> source tokens.",
        "<b>extra_src_pos.weight[: non_image.size(0)]</b> uses Python slicing on the "
        "embedding's <code>.weight</code> tensor — gets the first 39 rows, which is "
        "exactly <code>num_non_image_tokens</code>.",
        "<b>src_key_padding_mask construction</b>: zero (False = valid) for the 7 "
        "fixed tokens (latent + qpos + 5 meta), then the actual language pad_mask "
        "<code>(B, 32)</code>, then zero for all image tokens. Final shape "
        "<code>(B, 39 + S_img)</code>.",
        "<b>a_hat (B, 100, 16) * action_mask[:, None, :]</b> zeroes out predictions "
        "in invalid action dims before they reach the loss. Combined with the masked "
        "L1 in VLAPolicy._compute_loss, this means the model literally cannot waste "
        "capacity on padded slots.",
        "<b>Inference shortcut</b>: when <code>actions is None</code>, the style "
        "encoder is skipped and z = 0 (the prior mean). This matches ACT's behavior "
        "and is what the rollout uses.",
    ])

    code_block(story, "model/vla_cvae.py:203-204 - build_vla_cvae", """\
def build_vla_cvae(**kwargs) -> VLACVAE:
    return VLACVAE(**kwargs)""")

    # =====================================================================
    # model/vla_policy.py
    # =====================================================================
    h1(story, "model/vla_policy.py (NEW)")
    h2(story, "Purpose")
    body(story, "Wraps <code>VLACVAE</code> with: (1) per-robot normalization tables "
                "(one row per robot in vocab) registered as buffers, (2) the heterogeneous "
                "training loss that masks both padded timesteps AND padded action dims, "
                "and (3) a numpy-in/numpy-out <code>inference()</code> for the rollout "
                "that handles per-robot de-normalization and slicing back to the "
                "adapter's actual <code>action_dim</code>.")
    h2(story, "Shape / object contract")
    bullets(story, [
        "Buffer tables (one row per robot in vocab, all sized to MAX_*_DIM): "
        "<code>qpos_mean_table (V_robot, 16), qpos_std_table (V_robot, 16), "
        "action_mean_table (V_robot, 16), action_std_table (V_robot, 16)</code>.",
        "<code>image_mean (3, 1, 1)</code>, <code>image_std (3, 1, 1)</code>.",
        "<code>forward(...)</code> training: returns "
        "<code>{loss, l1, kl}</code> dict.",
        "<code>inference(image_np, qpos_np, instruction, ...)</code>: "
        "returns numpy <code>(100, action_dim)</code> in raw absolute units, "
        "sliced back to the adapter's actual action_dim.",
    ])

    code_block(story, "model/vla_policy.py:20-28 - vocab helpers", """\
def _coerce_vocabs(vocabs) -> VocabBundle:
    if isinstance(vocabs, VocabBundle):
        return vocabs
    return VocabBundle(**vocabs)

def _lookup(vocab: Dict[str, int], value: str) -> int:
    return int(vocab.get(str(value), vocab[C.UNKNOWN_TOKEN]))""")
    bullets(story, [
        "<code>_coerce_vocabs</code> handles both formats checkpoints can carry: "
        "an actual <code>VocabBundle</code> dataclass or its <code>as_dict()</code> "
        "form (which is what <code>save_vla_checkpoint</code> writes).",
        "<code>_lookup</code> is the same fallback-to-UNKNOWN logic as in "
        "<code>data/vla_dataset.py</code>; missing names map to ID 0.",
    ])

    code_block(story, "model/vla_policy.py:30-69 - VLAPolicy.__init__", """\
class VLAPolicy(nn.Module):
    def __init__(self, stats: dict, vocabs, kl_weight=C.KL_WEIGHT, **vla_kwargs):
        super().__init__()
        self.vocabs = _coerce_vocabs(vocabs)
        self.kl_weight = float(kl_weight)

        # Override embedding-table sizes with the actual vocab sizes.
        vla_kwargs.setdefault("num_robot_ids", len(self.vocabs.robot_ids))
        vla_kwargs.setdefault("num_lab_ids", len(self.vocabs.lab_ids))
        vla_kwargs.setdefault("num_embodiment_ids", len(self.vocabs.embodiment_ids))
        vla_kwargs.setdefault("num_action_type_ids", len(self.vocabs.action_type_ids))
        vla_kwargs.setdefault("num_task_family_ids", len(self.vocabs.task_family_ids))
        self.model = VLACVAE(**vla_kwargs)

        # Per-robot normalization tables: one row per vocab entry.
        V = len(self.vocabs.robot_ids)
        self.register_buffer("qpos_mean_table",   torch.zeros(V, C.MAX_STATE_DIM))   # (V, 16)
        self.register_buffer("qpos_std_table",    torch.ones(V,  C.MAX_STATE_DIM))   # (V, 16)
        self.register_buffer("action_mean_table", torch.zeros(V, C.MAX_ACTION_DIM))  # (V, 16)
        self.register_buffer("action_std_table",  torch.ones(V,  C.MAX_ACTION_DIM))  # (V, 16)
        for robot_name, rid in self.vocabs.robot_ids.items():
            if robot_name == C.UNKNOWN_TOKEN or robot_name not in stats["by_robot"]:
                continue
            robot_stats = stats["by_robot"][robot_name]
            self.qpos_mean_table[rid]   = torch.from_numpy(robot_stats["qpos_mean"])
            self.qpos_std_table[rid]    = torch.from_numpy(robot_stats["qpos_std"])
            self.action_mean_table[rid] = torch.from_numpy(robot_stats["action_mean"])
            self.action_std_table[rid]  = torch.from_numpy(robot_stats["action_std"])

        self.register_buffer("image_mean",
                             torch.from_numpy(stats["image_mean"]).view(3, 1, 1))
        self.register_buffer("image_std",
                             torch.from_numpy(stats["image_std"]).view(3, 1, 1))""")
    bullets(story, [
        "<b>Per-robot normalization tables</b>: each robot in vocab gets its own row "
        "of (mean, std) for both qpos and action, padded to 16 dims. Table shape is "
        "<code>(V_robot, 16)</code> where V_robot = number of robots in vocab "
        "(including the UNKNOWN slot at row 0).",
        "<b>UNKNOWN row stays at the init values</b> (mean=0, std=1) — a no-op "
        "normalization. Sane fallback for an unknown robot at inference.",
        "<b>setdefault(...)</b>: only overrides when the caller didn't pass the "
        "value. Lets the rollout (which may load weights from a checkpoint with "
        "different vocab sizes) override these explicitly.",
        "<b>Indexing into a buffer with [rid]</b>: assignment writes a row in place "
        "via the buffer's <code>__setitem__</code>. The buffer keeps tracking "
        "<code>.to(device)</code> the same way after this assignment.",
    ])

    code_block(story, "model/vla_policy.py:71-117 - forward", """\
def forward(self, image, qpos, instructions,
            robot_id, lab_id, embodiment_id, action_type_id, task_family_id,
            state_mask=None, action_mask=None,
            actions=None, is_pad=None):
    if actions is not None:
        if action_mask is None or is_pad is None:
            raise ValueError("action_mask and is_pad are required for training")
        a_hat, (mu, logvar) = self.model(
            image, qpos, instructions,
            robot_id, lab_id, embodiment_id, action_type_id, task_family_id,
            state_mask=state_mask, action_mask=action_mask,
            actions=actions, is_pad=is_pad,
        )
        return self._compute_loss(a_hat, actions, is_pad, action_mask, mu, logvar)

    a_hat, _ = self.model(
        image, qpos, instructions,
        robot_id, lab_id, embodiment_id, action_type_id, task_family_id,
        state_mask=state_mask, action_mask=action_mask,
    )
    return a_hat""")
    bullets(story, [
        "Same training-vs-eval split as ACTPolicy. Training callers must supply "
        "both <code>action_mask</code> and <code>is_pad</code> — without the action "
        "mask the loss couldn't ignore padded action dims.",
    ])

    code_block(story, "model/vla_policy.py:119-133 - _compute_loss (double-masked L1)", """\
def _compute_loss(self, a_hat, actions, is_pad, action_mask, mu, logvar):
    l1_unreduced = F.l1_loss(a_hat, actions, reduction="none")    # (B, 100, 16)
    valid = (~is_pad).unsqueeze(-1).float() * action_mask.unsqueeze(1).float()
    # valid:           (B, 100, 1)        *        (B, 1, 16)        = (B, 100, 16)
    l1 = (l1_unreduced * valid).sum() / valid.sum().clamp_min(1.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    total = l1 + self.kl_weight * kl
    return {"loss": total, "l1": l1.detach(), "kl": kl.detach()}""")
    bullets(story, [
        "<b>Two-axis masking</b>: <code>~is_pad (B, 100, 1)</code> covers padded "
        "<i>timesteps</i> (chunk extends past the trial's tail); "
        "<code>action_mask (B, 1, 16)</code> covers padded <i>action dims</i> (DOFs "
        "the robot doesn't have). Their elementwise product is <code>(B, 100, 16)</code> "
        "with 1.0 only where BOTH the timestep and the dim are real.",
        "<b>Denominator: valid.sum()</b> is the count of real action elements (not "
        "multiplied by action_dim like ACT, since the mask already counts elements "
        "directly).",
        "<b>KL formula</b> is identical to ACT — the latent is fixed at "
        "<code>latent_dim = 32</code> and is per-sample; <code>kl_weight = 10.0</code>.",
    ])

    code_block(story, "model/vla_policy.py:135-193 - inference", """\
@torch.no_grad()
def inference(self, image_np, qpos_np, instruction,
              robot_id=C.DEFAULT_ROBOT_ID, lab_id=C.DEFAULT_LAB_ID,
              embodiment=C.DEFAULT_EMBODIMENT, action_type=C.DEFAULT_ACTION_TYPE,
              task_family=C.DEFAULT_TASK_FAMILY,
              state_dim=None, action_dim=None) -> np.ndarray:
    self.eval()
    device = self.qpos_mean_table.device
    rid = _lookup(self.vocabs.robot_ids, robot_id)
    state_dim = int(state_dim if state_dim is not None else len(qpos_np))
    action_dim = int(action_dim if action_dim is not None else C.MAX_ACTION_DIM)

    img = self._preprocess_image(image_np).to(device).unsqueeze(0)        # (1, 1, 3, 240, 320)
    qpos = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)                    # (16,)
    qpos[:state_dim] = np.asarray(qpos_np, dtype=np.float32).reshape(-1)[:state_dim]
    state_mask = np.zeros(C.MAX_STATE_DIM, dtype=bool); state_mask[:state_dim] = True
    action_mask = np.zeros(C.MAX_ACTION_DIM, dtype=bool); action_mask[:action_dim] = True

    qpos_t = torch.from_numpy(qpos).to(device)
    qpos_t = ((qpos_t - self.qpos_mean_table[rid]) / self.qpos_std_table[rid]).unsqueeze(0)
    # qpos_t: (1, 16)
    state_mask_t = torch.from_numpy(state_mask).to(device).unsqueeze(0)   # (1, 16)
    action_mask_t = torch.from_numpy(action_mask).to(device).unsqueeze(0) # (1, 16)

    robot_id_t = torch.tensor([rid], dtype=torch.long, device=device)     # (1,)
    lab_id_t = torch.tensor([_lookup(...)], dtype=torch.long, device=device)
    embodiment_id_t = torch.tensor([_lookup(...)], dtype=torch.long, device=device)
    action_type_id_t = torch.tensor([_lookup(...)], dtype=torch.long, device=device)
    task_family_id_t = torch.tensor([_lookup(...)], dtype=torch.long, device=device)

    a_hat = self.forward(img, qpos_t, [instruction],
                         robot_id_t, lab_id_t, embodiment_id_t,
                         action_type_id_t, task_family_id_t,
                         state_mask=state_mask_t, action_mask=action_mask_t)
    # a_hat: (1, 100, 16) -- already zeroed at padded dims by VLACVAE.forward
    a = a_hat[0] * self.action_std_table[rid] + self.action_mean_table[rid]
    # a: (100, 16) -- de-normalized for this robot
    return a[:, :action_dim].cpu().numpy().astype(np.float32)
    # final: (100, action_dim) in raw absolute units""")
    bullets(story, [
        "<b>Per-robot de-normalization</b>: <code>action_std_table[rid]</code> picks "
        "row <code>rid</code> &rarr; <code>(16,)</code>; broadcast against "
        "<code>a_hat[0] (100, 16)</code> &rarr; <code>(100, 16)</code>. Same for "
        "<code>qpos</code> on the input side.",
        "<b>state_dim vs action_dim</b>: defaults are inferred from the supplied "
        "<code>qpos_np</code> length. The adapter (e.g. <code>SensapexDualAdapter</code>) "
        "passes its own canonical values (<code>state_dim=8, action_dim=8</code>).",
        "<b>Final slice <code>a[:, :action_dim]</code></b>: the model produces a "
        "16-D action chunk, but the adapter only consumes its first "
        "<code>action_dim</code> dims. Padded dims would be zero anyway because "
        "<code>action_mask</code> masked them out; the slice just makes the shape "
        "match the rest of the rollout pipeline.",
    ])

    code_block(story, "model/vla_policy.py:195-203 - _preprocess_image", """\
def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
    h, w = img_np.shape[:2]
    if (h, w) != (C.IMAGE_HEIGHT, C.IMAGE_WIDTH):
        pil = Image.fromarray(img_np).resize((C.IMAGE_WIDTH, C.IMAGE_HEIGHT),
                                              Image.BILINEAR)
        img_np = np.array(pil)
    x = torch.from_numpy(img_np).float() / 255.0
    x = x.permute(2, 0, 1)
    x = (x - self.image_mean) / self.image_std
    return x.unsqueeze(0)                                # (num_cam=1, 3, 240, 320)""")
    bullets(story, [
        "Identical to ACTPolicy's image preprocessing. Resize if needed, scale to "
        "[0,1], convert HWC&rarr;CHW, ImageNet-normalize, add the camera axis.",
    ])

    code_block(story, "model/vla_policy.py:206-219 - build_vla_policy", """\
def build_vla_policy(stats=None, vocabs=None,
                     stats_path=C.VLA_STATS_PATH,
                     kl_weight=C.KL_WEIGHT, **vla_kwargs) -> VLAPolicy:
    if stats is None or vocabs is None:
        with open(stats_path, "rb") as f:
            payload = pickle.load(f)
        stats = payload["stats"] if stats is None else stats
        vocabs = payload["vocabs"] if vocabs is None else vocabs
    return VLAPolicy(stats=stats, vocabs=vocabs, kl_weight=kl_weight, **vla_kwargs)""")
    bullets(story, [
        "Construction supports three modes: training (caller passes both stats and "
        "vocabs from the dataset), rollout (caller passes them from the loaded "
        "checkpoint), or fully implicit (load both from <code>VLA_STATS_PATH</code>).",
        "Note <code>**vla_kwargs</code> includes things like "
        "<code>backbone_name, freeze_backbone, language_backend, text_model_name</code>; "
        "the rollout pulls these from the checkpoint's saved config dict.",
    ])
