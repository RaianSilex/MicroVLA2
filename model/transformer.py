"""Transformer primitives for MicroACT.

DETR-style encoder / decoder with position embeddings injected at every
layer. These blocks are shared by:
    - the main ACT encoder-decoder (image + qpos + latent -> action chunk)
    - the CVAE style encoder in model/cvae.py (cls + qpos + actions -> latent)

Convention: sequence-first tensors (L, B, D), matching torch.nn.MultiheadAttention.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config as C


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _activation(name: str):
    if name == "relu": return F.relu
    if name == "gelu": return F.gelu
    if name == "glu":  return F.glu
    raise ValueError(f"unsupported activation: {name}")


def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
    return x if pos is None else x + pos


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = _with_pos(src, pos)
        src2 = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.LayerNorm] = None,
    ):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = _with_pos(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.multihead_attn(
            query=_with_pos(tgt, query_pos),
            key=_with_pos(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.LayerNorm] = None,
    ):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = tgt
        for layer in self.layers:
            out = layer(
                out,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        if self.norm is not None:
            out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Full encoder-decoder transformer (used for the main ACT policy)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = C.HIDDEN_DIM,
        nhead: int = C.NHEAD,
        num_encoder_layers: int = C.ENC_LAYERS,
        num_decoder_layers: int = C.DEC_LAYERS,
        dim_feedforward: int = C.DIM_FEEDFORWARD,
        dropout: float = C.DROPOUT,
        activation: str = "relu",
    ):
        super().__init__()
        enc_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.decoder = TransformerDecoder(
            dec_layer, num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,                            # (S, B, D)
        pos_embed: torch.Tensor,                      # (S, B, D)
        query_embed: torch.Tensor,                    # (Q, D) or (Q, B, D)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
    ) -> torch.Tensor:
        """Returns decoder output (Q, B, D)."""
        memory = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask, pos=pos_embed
        )

        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).expand(-1, src.size(1), -1)
        tgt = torch.zeros_like(query_embed)

        return self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )


def build_transformer(**kwargs) -> Transformer:
    return Transformer(**kwargs)


def build_encoder(
    d_model: int = C.HIDDEN_DIM,
    nhead: int = C.NHEAD,
    num_layers: int = C.ENC_LAYERS,
    dim_feedforward: int = C.DIM_FEEDFORWARD,
    dropout: float = C.DROPOUT,
    activation: str = "relu",
) -> TransformerEncoder:
    """Encoder-only stack, used by the CVAE style encoder."""
    layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation
    )
    return TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model))
