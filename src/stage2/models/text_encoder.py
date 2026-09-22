"""Qwen3 text encoder for the T2I RAE recipe.

Mirrors RAEv2 ``src/stage2/models/embedders/text_encoder.py`` (the upstream
reference) and adapts it to GLD conventions:

  * Always loaded in bf16 to match the rest of the trainer's autocast.
  * Returns ``(B, T, D)`` hidden states with ``D = hidden_size`` of the
    underlying Qwen3 (1024 for Qwen3-0.6B). The trainer feeds this directly
    into the DiT cross-attn — no projection layer in between (the user
    explicitly chose to keep cross_attn_kdim = 1024 instead of pinning it to
    UMT5-XXL's 4096).
  * Tokenizer pads to ``max_length`` so all batches have the same shape.
    Returns the ``attention_mask`` so the trainer can pass it to
    ``ref_global``-aware DiT blocks if needed (current ``GAEFlow``
    does not consume an explicit mask; padded positions are still safe to
    keep because the encoder hidden state at pad tokens is small but not
    actively misleading — RAEv2 does the same).

Caching: this module does NOT cache embeddings to disk. The whole point of
moving away from the UMT5 cache is to avoid 463 GB of small files. With
Qwen3-0.6B (~600M params, ~1.2 GB bf16) on every rank the on-the-fly cost
is ~50 ms / forward at max_length=256 — small compared to a single DiT
training step.

Use:
    enc = Qwen3TextEncoder(model_name="Qwen/Qwen3-0.6B", max_length=256).cuda().eval()
    enc.requires_grad_(False)
    with torch.no_grad():
        out = enc(captions)  # captions: list[str] of length B
    ref_global = out["tokens"]              # (B, max_length, 1024) bf16
    attn_mask  = out["attention_mask"]      # (B, max_length) int
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn


def prefill_hf_cache(model_name: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
    """Force HF Hub to download / verify cache for ``model_name`` in the
    **current process only**, with full network access.

    Intended to be called on rank 0 before any other rank touches the
    cache. Without this, 8 ranks per node racing `from_pretrained` against
    the same `${HUGGINGFACE_HUB_CACHE}` produce fcntl lock contention on
    `_lock` / `.no_exist` / shard verification files, which sporadically
    blocks 30 s – several minutes — observed as rank-asymmetric forward
    stalls that trip the NCCL watchdog.

    After this call returns, the cache is fully materialized; subsequent
    ranks should set ``HF_HUB_OFFLINE=1`` and load purely from disk (no
    HTTP HEAD, no `_lock` files written, no contention).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)


class Qwen3TextEncoder(nn.Module):
    """Frozen Qwen3 text encoder returning per-token hidden states.

    Args:
        model_name: HuggingFace repo id (default ``Qwen/Qwen3-0.6B``).
        max_length: tokenizer pad/truncate length. RAEv2 uses 256; with that
            length, journeydb/long-caption captions that exceed 256 wpieces
            get truncated (rare but happens for long-caption split).
        torch_dtype: ``torch.bfloat16`` recommended — keeps
            memory at ~1.2 GB and runs flash-attn paths.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = int(max_length)
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # Some Qwen variants ship without an explicit pad token; fall
            # back to eos so padded positions don't raise during batching.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        )
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad_(False)

        self.feature_dim: int = int(self.text_model.config.hidden_size)

    @torch.no_grad()
    def forward(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        # Substitute empty / None captions with a single space so the
        # tokenizer doesn't drop them — empty strings collapse to a 0-length
        # sequence that breaks downstream batched cross-attn.
        cleaned = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]

        device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(
            cleaned,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_model(
            **tokens,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Last hidden state — RAEv2 takes hidden_states[-1] which is identical
        # to outputs.hidden_states[-1] for the AutoModelForCausalLM head.
        last_hidden = outputs.hidden_states[-1]  # (B, T, D)

        return {
            "tokens": last_hidden,
            "attention_mask": tokens["attention_mask"],
        }

    @torch.no_grad()
    def encode_null(self, batch_size: int, device: torch.device):
        """Convenience: returns the null-conditional embedding (single empty
        prompt broadcast to ``batch_size``). Useful for CFG sampling."""
        out = self.forward([""] * 1)
        return {
            "tokens": out["tokens"].expand(batch_size, -1, -1).to(device),
            "attention_mask": out["attention_mask"]
            .expand(batch_size, -1).to(device),
        }
