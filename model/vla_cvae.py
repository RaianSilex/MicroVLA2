"""Vision-language-action CVAE for heterogeneous micromanipulation.

This keeps the ACT action-chunking decoder, but conditions the transformer on:

    image tokens + instruction tokens + padded robot state + embodiment metadata

The output is always padded to MAX_ACTION_DIM. The policy wrapper masks invalid
action dimensions so single- and dual-manipulator demonstrations can train
together.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from config import vla_config as C
from model.backbone import build_backbone
from model.cvae import reparameterize
from model.embodiment import EmbodimentConditioner
from model.language_encoder import build_language_encoder
from model.transformer import build_encoder, build_transformer


class VLACVAE(nn.Module):
    def __init__(
        self,
        state_dim: int = C.MAX_STATE_DIM,
        action_dim: int = C.MAX_ACTION_DIM,
        hidden_dim: int = C.HIDDEN_DIM,
        latent_dim: int = C.LATENT_DIM,
        chunk_size: int = C.CHUNK_SIZE,
        num_cameras: int = C.NUM_CAMERAS,
        pretrained_backbone: bool = True,
        backbone_name: str = C.DEFAULT_BACKBONE,
        freeze_backbone: bool = True,
        language_backend: str = C.LANGUAGE_BACKEND,
        text_model_name: str = C.DEFAULT_TEXT_MODEL,
        max_language_tokens: int = C.MAX_LANGUAGE_TOKENS,
        num_robot_ids: int = C.NUM_ROBOT_IDS_FALLBACK,
        num_lab_ids: int = C.NUM_LAB_IDS_FALLBACK,
        num_embodiment_ids: int = C.NUM_EMBODIMENT_IDS_FALLBACK,
        num_action_type_ids: int = C.NUM_ACTION_TYPE_IDS_FALLBACK,
        num_task_family_ids: int = C.NUM_TASK_FAMILY_IDS_FALLBACK,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.chunk_size = int(chunk_size)
        self.num_cameras = int(num_cameras)
        self.max_language_tokens = int(max_language_tokens)

        self.backbone = build_backbone(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone,
            backbone_name=backbone_name,
            freeze=freeze_backbone,
        )
        self.language_encoder = build_language_encoder(
            backend=language_backend,
            model_name=text_model_name,
            hidden_dim=hidden_dim,
            max_tokens=max_language_tokens,
        )
        self.embodiment = EmbodimentConditioner(
            hidden_dim=hidden_dim,
            num_robot_ids=num_robot_ids,
            num_lab_ids=num_lab_ids,
            num_embodiment_ids=num_embodiment_ids,
            num_action_type_ids=num_action_type_ids,
            num_task_family_ids=num_task_family_ids,
        )

        self.transformer = build_transformer(d_model=hidden_dim)
        self.style_encoder = build_encoder(d_model=hidden_dim)

        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.style_qpos_proj = nn.Linear(state_dim, hidden_dim)
        self.style_action_proj = nn.Linear(action_dim, hidden_dim)
        self.style_pos_embed = nn.Embedding(1 + 1 + chunk_size, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)

        self.latent_to_src = nn.Linear(latent_dim, hidden_dim)
        self.qpos_to_src = nn.Linear(state_dim, hidden_dim)

        self.num_non_image_tokens = 2 + self.embodiment.num_tokens + self.max_language_tokens
        self.extra_src_pos = nn.Embedding(self.num_non_image_tokens, hidden_dim)

        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _encode_style(
        self,
        qpos: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = qpos.size(0)
        device = qpos.device
        if state_mask is not None:
            qpos = qpos * state_mask.float()
        if action_mask is not None:
            actions = actions * action_mask[:, None, :].float()

        cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)
        qpos_tok = self.style_qpos_proj(qpos).unsqueeze(1)
        act_toks = self.style_action_proj(actions)
        seq = torch.cat([cls, qpos_tok, act_toks], dim=1).permute(1, 0, 2).contiguous()
        pos = self.style_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        always_valid = torch.zeros(B, 2, dtype=torch.bool, device=device)
        pad_mask = torch.cat([always_valid, is_pad], dim=1)
        out = self.style_encoder(seq, src_key_padding_mask=pad_mask, pos=pos)
        return self.latent_proj(out[0]).chunk(2, dim=-1)

    def _encode_image(
        self,
        image: Optional[torch.Tensor],
        primary_feat: Optional[torch.Tensor] = None,
        aux_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if primary_feat is not None:
            # Cached raw encoder features (single-camera). Skip the frozen
            # encoders; the backbone still runs its trainable projection path.
            B, N = primary_feat.shape[0], self.num_cameras
            feat, pos = self.backbone(None, primary_feat=primary_feat, aux_feat=aux_feat)
        else:
            B, N = image.shape[:2]
            flat = image.flatten(0, 1)
            feat, pos = self.backbone(flat)
        if feat.dim() == 4:
            D, Hp, Wp = feat.shape[1:]
            feat = feat.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
            pos = pos.view(B, N, D, Hp, Wp).permute(0, 2, 1, 3, 4).flatten(2)
            feat = feat.permute(2, 0, 1).contiguous()
            pos = pos.permute(2, 0, 1).contiguous()
        else:
            S, _BN, D = feat.shape
            if N > 1:
                feat = feat.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
                pos = pos.view(S, B, N, D).permute(2, 0, 1, 3).reshape(N * S, B, D)
        return feat, pos

    def forward(
        self,
        image: torch.Tensor,
        qpos: torch.Tensor,
        instructions: Sequence[str] | str,
        robot_id: torch.Tensor,
        lab_id: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_type_id: torch.Tensor,
        task_family_id: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        img_primary_feat: Optional[torch.Tensor] = None,
        img_aux_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = qpos.size(0)
        device = qpos.device

        if state_mask is not None:
            qpos = qpos * state_mask.float()

        if actions is not None:
            if is_pad is None:
                raise ValueError("is_pad is required when actions are provided")
            mu, logvar = self._encode_style(qpos, actions, is_pad, state_mask, action_mask)
            z = reparameterize(mu, logvar)
        else:
            mu = torch.zeros(B, self.latent_dim, device=device)
            logvar = torch.zeros(B, self.latent_dim, device=device)
            z = torch.zeros(B, self.latent_dim, device=device)

        img_feat, img_pos = self._encode_image(image, img_primary_feat, img_aux_feat)
        lang_tokens, lang_pad = self.language_encoder(instructions)
        meta_tokens = self.embodiment(
            robot_id, lab_id, embodiment_id, action_type_id, task_family_id
        )

        latent_tok = self.latent_to_src(z).unsqueeze(0)
        qpos_tok = self.qpos_to_src(qpos).unsqueeze(0)
        non_image = torch.cat([latent_tok, qpos_tok, meta_tokens, lang_tokens], dim=0)
        src = torch.cat([non_image, img_feat], dim=0)

        pos_non_image = self.extra_src_pos.weight[: non_image.size(0)]
        pos_non_image = pos_non_image.unsqueeze(1).expand(-1, B, -1)
        src_pos = torch.cat([pos_non_image, img_pos], dim=0)

        fixed_valid = torch.zeros(
            B, 2 + self.embodiment.num_tokens, dtype=torch.bool, device=device
        )
        img_valid = torch.zeros(B, img_feat.size(0), dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat([fixed_valid, lang_pad, img_valid], dim=1)

        hs = self.transformer(
            src,
            src_pos,
            self.query_embed.weight,
            src_key_padding_mask=src_key_padding_mask,
        )
        a_hat = self.action_head(hs.transpose(0, 1))
        if action_mask is not None:
            a_hat = a_hat * action_mask[:, None, :].float()
        return a_hat, (mu, logvar)


def build_vla_cvae(**kwargs) -> VLACVAE:
    return VLACVAE(**kwargs)
