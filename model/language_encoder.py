"""Language encoders for MicroVLA.

The recommended path is a frozen Hugging Face encoder such as
DistilBERT/BERT. A small hash-token fallback exists for offline smoke tests,
but it is not a substitute for real language pretraining.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from config import vla_config as C


def _as_list(instructions: Sequence[str] | str) -> List[str]:
    if isinstance(instructions, str):
        return [instructions]
    return [str(x) for x in instructions]


class HuggingFaceTextEncoder(nn.Module):
    """Frozen Transformer text encoder with a trainable projection to hidden_dim."""

    def __init__(
        self,
        model_name: str = C.DEFAULT_TEXT_MODEL,
        hidden_dim: int = C.HIDDEN_DIM,
        max_tokens: int = C.MAX_LANGUAGE_TOKENS,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "HuggingFaceTextEncoder requires `transformers`. Install it or "
                "run with --language-backend simple for offline smoke tests."
            ) from exc

        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()

        model_dim = int(self.text_model.config.hidden_size)
        self.proj = nn.Linear(model_dim, hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()
        return self

    def forward(self, instructions: Sequence[str] | str) -> Tuple[torch.Tensor, torch.Tensor]:
        texts = _as_list(instructions)
        device = self.proj.weight.device
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.text_model(**enc).last_hidden_state                  # (B, L, H_txt)
        tokens = self.proj(out).permute(1, 0, 2).contiguous()               # (L, B, D)
        pad_mask = ~enc["attention_mask"].bool()                            # (B, L)
        return tokens, pad_mask


class SimpleHashTextEncoder(nn.Module):
    """Small deterministic tokenizer for dependency-free smoke tests."""

    _word_re = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

    def __init__(
        self,
        hidden_dim: int = C.HIDDEN_DIM,
        max_tokens: int = C.MAX_LANGUAGE_TOKENS,
        vocab_size: int = C.SIMPLE_TEXT_VOCAB_SIZE,
    ):
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

    def forward(self, instructions: Sequence[str] | str) -> Tuple[torch.Tensor, torch.Tensor]:
        texts = _as_list(instructions)
        device = self.embed.weight.device
        ids = torch.tensor([self._encode_one(t) for t in texts], dtype=torch.long, device=device)
        positions = torch.arange(self.max_tokens, dtype=torch.long, device=device)
        tokens = self.embed(ids) + self.pos_embed(positions).unsqueeze(0)   # (B, L, D)
        pad_mask = ids.eq(0)                                                # (B, L)
        return tokens.permute(1, 0, 2).contiguous(), pad_mask


def build_language_encoder(
    backend: str = C.LANGUAGE_BACKEND,
    model_name: str = C.DEFAULT_TEXT_MODEL,
    hidden_dim: int = C.HIDDEN_DIM,
    max_tokens: int = C.MAX_LANGUAGE_TOKENS,
) -> nn.Module:
    backend = str(backend).lower()
    if backend == "hf":
        return HuggingFaceTextEncoder(model_name=model_name, hidden_dim=hidden_dim, max_tokens=max_tokens)
    if backend == "simple":
        return SimpleHashTextEncoder(hidden_dim=hidden_dim, max_tokens=max_tokens)
    raise ValueError(f"Unsupported language backend: {backend!r}")
