# Text strategies for Anima 3.8B (native Qwen3 0.6B + Qwen3.5 4B semantic stream).
#
# Text-encoder outputs are a flat list:
#   [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,      # native Anima conditioning
#    semantic_0, ..., semantic_{K-1}, semantic_mask]            # Qwen3.5 layers (K = 4)
#
# The Qwen3.5 states are cached trimmed to the prompt length; the dataset
# collator zero-pads them (and their mask) to the longest prompt in the batch.

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save

from library import anima_qwen35
from library.strategy_anima import (
    AnimaTextEncoderOutputsCachingStrategy,
    AnimaTextEncodingStrategy,
    AnimaTokenizeStrategy,
)
from library.strategy_base import TextEncodingStrategy, TokenizeStrategy
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

NATIVE_OUTPUT_COUNT = 4
NATIVE_KEYS = ("prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask")


def split_text_encoder_outputs(outputs: Sequence[Any], num_layers: int) -> tuple[list[Any], list[Any], Any]:
    """Split a flat 3.8B output list into (native[4], semantic_states[K], semantic_mask)."""
    expected = NATIVE_OUTPUT_COUNT + num_layers + 1
    if len(outputs) != expected:
        raise ValueError(f"Expected {expected} Anima 3.8B text-encoder outputs, got {len(outputs)}")
    native = list(outputs[:NATIVE_OUTPUT_COUNT])
    semantic = list(outputs[NATIVE_OUTPUT_COUNT : NATIVE_OUTPUT_COUNT + num_layers])
    return native, semantic, outputs[-1]


class Anima38BTokenizeStrategy(AnimaTokenizeStrategy):
    """Anima tokenization (Qwen3 + T5 ids) plus Qwen3.5 ids for the semantic connector."""

    def __init__(
        self,
        qwen3_path: str | None = None,
        t5_tokenizer_path: str | None = None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen35_tokenizer_path: str | None = None,
        qwen35_max_length: int = 1024,
    ) -> None:
        super().__init__(
            qwen3_path=qwen3_path,
            t5_tokenizer_path=t5_tokenizer_path,
            qwen3_max_length=qwen3_max_length,
            t5_max_length=t5_max_length,
        )
        self.qwen35_tokenizer = anima_qwen35.load_qwen35_tokenizer(qwen35_tokenizer_path)
        self.qwen35_max_length = int(qwen35_max_length)

    def tokenize(self, text: str | list[str]) -> list[torch.Tensor]:
        native = super().tokenize(text)
        qwen35_ids, qwen35_mask = anima_qwen35.tokenize_qwen35(self.qwen35_tokenizer, text, self.qwen35_max_length)
        return [*native, qwen35_ids, qwen35_mask]


class Anima38BTextEncodingStrategy(AnimaTextEncodingStrategy):
    """Encode native Anima conditioning plus selected Qwen3.5 hidden layers."""

    def __init__(self, layer_indices: Sequence[int], dropout_rate: float = 0.0) -> None:
        super().__init__(dropout_rate=dropout_rate)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self._uncond_outputs: list[torch.Tensor] | None = None  # each with batch dim 1, on CPU

    @property
    def num_outputs(self) -> int:
        return NATIVE_OUTPUT_COUNT + len(self.layer_indices) + 1

    def encode_semantic(
        self,
        qwen35_model: anima_qwen35.Qwen35HybridModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        device = qwen35_model.embed_tokens.weight.device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device, dtype=torch.long)
        _, intermediate = qwen35_model(
            input_ids,
            attention_mask=attention_mask,
            intermediate_output=list(self.layer_indices),
            return_final=False,
        )
        if not isinstance(intermediate, dict):
            raise RuntimeError("Qwen3.5 did not return the requested hidden layers")
        valid = attention_mask.bool().unsqueeze(-1)
        # ComfyUI hands the connector bfloat16 states; round the same way here.
        states = [intermediate[index].masked_fill(~valid, 0).to(torch.bfloat16) for index in self.layer_indices]
        return states, attention_mask

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        tokens: list[torch.Tensor],
        enable_dropout: bool = True,
    ) -> list[torch.Tensor]:
        """models: [qwen3_text_encoder, qwen35_model]; tokens: 4 native + [qwen35_ids, qwen35_mask]."""
        if len(models) < 2:
            raise ValueError("Anima 3.8B text encoding needs [qwen3, qwen35] models")
        native = super().encode_tokens(tokenize_strategy, [models[0]], list(tokens[:NATIVE_OUTPUT_COUNT]), enable_dropout=False)
        states, mask = self.encode_semantic(models[1], tokens[NATIVE_OUTPUT_COUNT], tokens[NATIVE_OUTPUT_COUNT + 1])
        outputs = [*native, *states, mask]
        if enable_dropout and self.dropout_rate > 0.0:
            if self._uncond_outputs is None:
                self.cache_uncond_embeddings(tokenize_strategy, models)
            outputs = self.drop_cached_text_encoder_outputs(*outputs)
        return outputs

    def cache_uncond_embeddings(self, tokenize_strategy: TokenizeStrategy, models: list[Any]) -> None:
        logger.info("Caching Anima 3.8B unconditional embeddings for caption dropout (empty caption)...")
        tokens = tokenize_strategy.tokenize("")
        with torch.no_grad():
            outputs = self.encode_tokens(tokenize_strategy, models, tokens, enable_dropout=False)
        self._uncond_outputs = [o.detach().cpu() for o in outputs]
        # Keep the native-only fields used by the parent class in sync.
        self._uncond_prompt_embeds, self._uncond_attn_mask, self._uncond_t5_input_ids, self._uncond_t5_attn_mask = (
            self._uncond_outputs[:NATIVE_OUTPUT_COUNT]
        )

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        *semantic_outputs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Replace randomly dropped captions with the empty-caption conditioning (both streams).

        ``semantic_outputs`` are the Qwen3.5 layer states followed by their mask.
        """
        outputs_list = [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, *semantic_outputs]
        if self.dropout_rate <= 0.0 or outputs_list[0] is None:
            return outputs_list
        batch_size = outputs_list[0].shape[0]
        drop = [i for i in range(batch_size) if random.random() < self.dropout_rate]
        if not drop:
            return outputs_list
        if self._uncond_outputs is None:
            raise RuntimeError("Caption dropout needs cached unconditional embeddings (cache_uncond_embeddings)")

        native, states, mask = split_text_encoder_outputs(outputs_list, len(self.layer_indices))
        u_native, u_states, u_mask = split_text_encoder_outputs(self._uncond_outputs, len(self.layer_indices))
        native = [t.clone() for t in native]
        u_len = int(u_mask.shape[1])
        needed = max(u_len, int(mask.shape[1]))
        states = [self._pad_tokens(s, needed).clone() for s in states]
        mask = self._pad_tokens(mask, needed).clone()
        for i in drop:
            for k, value in enumerate(native):
                value[i] = u_native[k][0].to(device=value.device, dtype=value.dtype)
            for k, value in enumerate(states):
                value[i] = 0
                value[i, :u_len] = u_states[k][0].to(device=value.device, dtype=value.dtype)
            mask[i] = 0
            mask[i, :u_len] = u_mask[0].to(device=mask.device, dtype=mask.dtype)
        return [*native, *states, mask]

    @staticmethod
    def _pad_tokens(value: torch.Tensor, length: int) -> torch.Tensor:
        if value.shape[1] >= length:
            return value
        pad = [0, 0] * (value.ndim - 2) + [0, length - value.shape[1]]
        return torch.nn.functional.pad(value, pad)


class Anima38BTextEncoderOutputsCachingStrategy(AnimaTextEncoderOutputsCachingStrategy):
    """Caches native Anima outputs plus trimmed Qwen3.5 layer states (bfloat16)."""

    ANIMA_TEXT_ENCODER_OUTPUTS_SAFETENSORS_SUFFIX = "_anima38b_te.safetensors"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        layer_indices: Sequence[int],
        qwen35_max_length: int,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.qwen35_max_length = int(qwen35_max_length)

    @property
    def semantic_keys(self) -> list[str]:
        return [f"qwen35_layer_{index}" for index in self.layer_indices]

    def _cache_signature(self) -> str:
        return f"layers={list(self.layer_indices)};qwen35_max_length={self.qwen35_max_length}"

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        path = os.path.splitext(npz_path)[0] + ".safetensors"
        if not os.path.exists(path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with safe_open(path, framework="pt") as handle:  # type: ignore[reportGeneralTypeIssues]
                keys = set(handle.keys())
                metadata = handle.metadata() or {}
        except Exception as error:  # noqa: BLE001 - corrupted cache is regenerated
            logger.warning(f"Error reading {path}, regenerating: {error}")
            return False
        required = {*NATIVE_KEYS, *self.semantic_keys, "qwen35_mask"}
        return required <= keys and metadata.get("anima38b_cache") == self._cache_signature()

    def load_outputs_npz(self, npz_path: str) -> list[np.ndarray]:
        path = os.path.splitext(npz_path)[0] + ".safetensors"
        with safe_open(path, framework="pt") as handle:  # type: ignore[reportGeneralTypeIssues]
            native = [
                handle.get_tensor("prompt_embeds").float().numpy(),
                handle.get_tensor("attn_mask").int().numpy(),
                handle.get_tensor("t5_input_ids").int().numpy(),
                handle.get_tensor("t5_attn_mask").int().numpy(),
            ]
            states = [handle.get_tensor(key).float().numpy() for key in self.semantic_keys]
            mask = handle.get_tensor("qwen35_mask").int().numpy()
        return [*native, *states, mask]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: list[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: list,
    ) -> None:
        if not isinstance(text_encoding_strategy, Anima38BTextEncodingStrategy):
            raise TypeError("Anima 3.8B caching needs Anima38BTextEncodingStrategy")
        captions = [info.caption for info in infos]
        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            outputs = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens, enable_dropout=False)
        native, states, mask = split_text_encoder_outputs(outputs, len(self.layer_indices))
        prompt_embeds = native[0].to(torch.bfloat16).cpu()
        attn_mask, t5_ids, t5_mask = (t.to(torch.int32).cpu() for t in native[1:])
        states = [s.to(torch.bfloat16).cpu() for s in states]
        mask = mask.to(torch.int32).cpu()

        for i, info in enumerate(infos):
            length = max(1, int(mask[i].sum().item()))
            item_states = [s[i, :length].contiguous() for s in states]
            item_mask = mask[i, :length].contiguous()
            if self.cache_to_disk:
                path = os.path.splitext(info.text_encoder_outputs_npz)[0] + ".safetensors"
                tensors = {
                    "prompt_embeds": prompt_embeds[i].contiguous(),
                    "attn_mask": attn_mask[i].contiguous(),
                    "t5_input_ids": t5_ids[i].contiguous(),
                    "t5_attn_mask": t5_mask[i].contiguous(),
                    "qwen35_mask": item_mask,
                }
                tensors.update(dict(zip(self.semantic_keys, item_states, strict=True)))
                safetensors_save(tensors, path, metadata={"anima38b_cache": self._cache_signature()})
            else:
                info.text_encoder_outputs = (
                    prompt_embeds[i].float().numpy(),
                    attn_mask[i].numpy(),
                    t5_ids[i].numpy(),
                    t5_mask[i].numpy(),
                    *(s.float().numpy() for s in item_states),
                    item_mask.numpy(),
                )
