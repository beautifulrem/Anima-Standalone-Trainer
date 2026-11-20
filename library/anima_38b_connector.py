# Anima 3.8B Qwen3.5 semantic connectors.
#
# Ported from GumGum10/comfyui-anima-3-8B (MIT License, Copyright (c) 2026
# GumGum10 contributors), which is the inference reference for Anima 3.8B.
# The math is kept identical so a LoRA trained here sees exactly the
# conditioning that ComfyUI/Forge produce at inference time.
#
# Two connector generations exist:
#
# * v1 ``anima_progressive_qwen35_cross_adapter_v1``: six Qwen3.5
#   cross-attention residuals inserted after the six frozen native adapter
#   blocks (``Anima-3.8B-expanded_adapter.safetensors``). Timestep independent.
# * v2 ``anima_qwen35_quality_anchored_semantic_connector_v2``: the frozen v1
#   ("quality anchor") plus a timestep-aware semantic resampler whose 64-token
#   bank is injected through six more cross-attention residuals. It must be run
#   once per denoising step because the bank depends on the timestep. The
#   released ``Anima-3.8B-v1.1.safetensors`` bundles it under
#   ``net.anima_v2_connector.*``.
#
# The connector does not own the native LLM adapter. It keeps an unregistered
# reference to ``dit.llm_adapter`` so the shared weights are moved, saved and
# (optionally) LoRA-patched exactly once, through the DiT.

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from library import anima_models
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


V1_ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"
V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"
SUPPORTED_ARCHITECTURES = (V1_ARCHITECTURE, V2_ARCHITECTURE)
BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
CONNECTOR_PREFIX = "anima_v2_connector."  # after the DiT's "net." prefix is stripped
BUNDLE_ADAPTER_METADATA_PREFIX = "anima_v2_adapter_"
LAYER_INDICES = (7, 15, 23, 31)
SEMANTIC_SOURCE_DIM = 2560


# ---------------------------------------------------------------------------
# Small building blocks matching ComfyUI's operations
# ---------------------------------------------------------------------------


def _attention(query_dim: int, context_dim: int, n_heads: int, head_dim: int, device, dtype) -> nn.Module:
    module = anima_models.LLMAdapterAttention(query_dim, context_dim, n_heads, head_dim)
    return module.to(device=device, dtype=dtype)


def _rms_norm(dim: int, device, dtype) -> nn.Module:
    return anima_models.LLMAdapterRMSNorm(dim, eps=1e-6).to(device=device, dtype=dtype)


def _attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(torch.bool)
    return mask.unsqueeze(1).unsqueeze(1) if mask.ndim == 2 else mask


def _embed_target(native_adapter: Any, target_input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return native_adapter.in_proj(native_adapter.embed(target_input_ids).to(dtype))


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Embed Anima's continuous [0, 1] flow timestep."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class MultiHeadAttention(nn.Module):
    """SDPA attention with independently sized query and context streams."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int, device, dtype) -> None:
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_dim = query_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, query_tokens, _ = query.shape
        context_tokens = context.shape[1]

        def heads(value: torch.Tensor, tokens: int) -> torch.Tensor:
            return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(query), query_tokens)
        k = heads(self.k_proj(context), context_tokens)
        v = heads(self.v_proj(context), context_tokens)
        mask = None
        if context_mask is not None:
            mask = context_mask.to(torch.bool).reshape(batch, 1, 1, context_tokens)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, self.query_dim)
        return self.o_proj(attended)


class TimestepModulatedNorm(nn.Module):
    """Parameter-free LayerNorm whose scale and shift come from the timestep."""

    def __init__(self, dim: int, device, dtype) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, device=device, dtype=dtype)

    def forward(self, value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SemanticResamplerBlock(nn.Module):
    """One timestep-aware query-resampler block (cross-attn, self-attn, SwiGLU)."""

    def __init__(self, dim: int, qwen_dim: int, num_heads: int, mlp_hidden_dim: int, device, dtype) -> None:
        super().__init__()
        self.cross_norm = TimestepModulatedNorm(dim, device, dtype)
        self.self_norm = TimestepModulatedNorm(dim, device, dtype)
        self.mlp_norm = TimestepModulatedNorm(dim, device, dtype)
        self.source_norm = nn.LayerNorm(qwen_dim, device=device, dtype=dtype)
        self.cross_attention = MultiHeadAttention(dim, qwen_dim, num_heads, device, dtype)
        self.self_attention = MultiHeadAttention(dim, dim, num_heads, device, dtype)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False, device=device, dtype=dtype)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype)

    def forward(
        self,
        queries: torch.Tensor,
        qwen_features: torch.Tensor,
        timestep_embedding: torch.Tensor,
        qwen_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        modulation = self.time_modulation(F.silu(timestep_embedding))
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = modulation.chunk(6, dim=-1)
        queries = queries + self.cross_attention(
            self.cross_norm(queries, cross_scale, cross_shift),
            self.source_norm(qwen_features),
            qwen_mask,
        )
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        normalized = self.mlp_norm(queries, mlp_scale, mlp_shift)
        gate, value = self.mlp_in(normalized).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class TimestepAwareSemanticResampler(nn.Module):
    """Preserve four Qwen layer streams and compress them into semantic queries."""

    def __init__(
        self,
        qwen_dim: int,
        output_dim: int,
        num_layers: int,
        num_queries: int,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        device,
        dtype,
        initialize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, model_dim, device=device, dtype=dtype))
        self.layer_embeddings = nn.Parameter(torch.empty(num_layers, 1, qwen_dim, device=device, dtype=dtype))
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, device=device, dtype=dtype),
        )
        self.blocks = nn.ModuleList(
            [SemanticResamplerBlock(model_dim, qwen_dim, num_heads, mlp_hidden_dim, device, dtype) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(model_dim, device=device, dtype=dtype)
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False, device=device, dtype=dtype)
        self.gradient_checkpointing = False
        if initialize_weights:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.layer_embeddings, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        timesteps: torch.Tensor,
        source_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} Qwen layers, got {len(hidden_states)}")
        batch = hidden_states[0].shape[0]
        layer_streams = [hidden + self.layer_embeddings[index].unsqueeze(0) for index, hidden in enumerate(hidden_states)]
        qwen_features = torch.cat(layer_streams, dim=1)
        qwen_mask = None
        if source_mask is not None:
            qwen_mask = torch.cat([source_mask] * self.num_layers, dim=1)

        time = sinusoidal_timestep_embedding(timesteps, self.model_dim).to(dtype=qwen_features.dtype)
        time = self.time_mlp(time)
        queries = self.query_tokens.expand(batch, -1, -1)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                queries = checkpoint(block, queries, qwen_features, time, qwen_mask, use_reentrant=False)
            else:
                queries = block(queries, qwen_features, time, qwen_mask)
        return self.output_projection(self.output_norm(queries))


# ---------------------------------------------------------------------------
# v1: progressive Qwen3.5 cross adapter (the v2 "quality anchor")
# ---------------------------------------------------------------------------


class ProgressiveQwen35CrossAdapter(nn.Module):
    """LLaMA-Pro style expansion of Anima's text connector.

    The six inherited adapter blocks are reused unchanged. After each one a new
    Qwen3.5 cross-attention residual is inserted.
    """

    architecture = V1_ARCHITECTURE
    requires_timesteps = False

    def __init__(
        self,
        native_adapter: Any,
        semantic_source_dim: int = SEMANTIC_SOURCE_DIM,
        layer_indices: Sequence[int] = LAYER_INDICES,
        initialize_from_native: bool = True,
        native_context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        # Unregistered on purpose: the DiT owns the native adapter.
        object.__setattr__(self, "_native_adapter", native_adapter)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.native_context_scale = float(native_context_scale)
        # Source checkpoint metadata, used to write the connector back when saving.
        self.base_metadata: dict[str, str] = {}
        self.adapter_metadata: dict[str, str] = {}
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads
        device = native_adapter.embed.weight.device
        dtype = native_adapter.embed.weight.dtype

        self.query_norms = nn.ModuleList([_rms_norm(model_dim, device, dtype) for _ in native_adapter.blocks])
        self.source_norms = nn.ModuleList([_rms_norm(semantic_source_dim, device, dtype) for _ in native_adapter.blocks])
        self.semantic_attentions = nn.ModuleList(
            [_attention(model_dim, semantic_source_dim, num_heads, head_dim, device, dtype) for _ in native_adapter.blocks]
        )
        self.layer_mix_logits = nn.Parameter(
            torch.zeros(len(native_adapter.blocks), len(self.layer_indices), device=device, dtype=dtype)
        )
        if initialize_from_native:
            self._initialize_from_native()

    @property
    def native_adapter(self) -> Any:
        return self._native_adapter

    @torch.no_grad()
    def _initialize_from_native(self) -> None:
        attention: Any
        query_norm: Any
        source_norm: Any
        for native, query_norm, source_norm, attention in zip(
            self.native_adapter.blocks, self.query_norms, self.source_norms, self.semantic_attentions, strict=True
        ):
            query_norm.weight.copy_(native.norm_cross_attn.weight)
            attention.q_proj.weight.copy_(native.cross_attn.q_proj.weight)
            attention.q_norm.weight.copy_(native.cross_attn.q_norm.weight)
            attention.k_norm.weight.copy_(native.cross_attn.k_norm.weight)
            nn.init.xavier_uniform_(attention.k_proj.weight)
            nn.init.xavier_uniform_(attention.v_proj.weight)
            attention.o_proj.weight.zero_()
            source_norm.weight.fill_(1.0)
        if self.layer_mix_logits.shape[1] > 1:
            self.layer_mix_logits.copy_(
                torch.linspace(
                    -0.5,
                    0.5,
                    self.layer_mix_logits.shape[1],
                    device=self.layer_mix_logits.device,
                    dtype=self.layer_mix_logits.dtype,
                ).repeat(self.layer_mix_logits.shape[0], 1)
            )

    @staticmethod
    def _mixed_source(hidden_states: Sequence[torch.Tensor], mix: torch.Tensor, block_index: int) -> torch.Tensor:
        mixed = hidden_states[0] * mix[block_index, 0]
        for layer_index in range(1, len(hidden_states)):
            mixed = mixed + hidden_states[layer_index] * mix[block_index, layer_index]
        return mixed

    def forward(
        self,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Sequence[torch.Tensor],
        target_attention_mask: torch.Tensor | None = None,
        native_source_mask: torch.Tensor | None = None,
        semantic_source_mask: torch.Tensor | None = None,
        include_inserted_blocks: bool = True,
        **_ignored,
    ) -> torch.Tensor:
        if len(semantic_hidden_states) != len(self.layer_indices):
            raise ValueError(f"Expected {len(self.layer_indices)} semantic layers, got {len(semantic_hidden_states)}")
        native_adapter = self.native_adapter
        target_mask = _attention_mask(target_attention_mask)
        native_mask = _attention_mask(native_source_mask)
        semantic_mask = _attention_mask(semantic_source_mask)

        x = _embed_target(native_adapter, target_input_ids, native_source.dtype)
        query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        native_positions = torch.arange(native_source.shape[1], device=x.device).unsqueeze(0)
        semantic_positions = torch.arange(semantic_hidden_states[0].shape[1], device=x.device).unsqueeze(0)
        query_rope = native_adapter.rotary_emb(x, query_positions)
        native_rope = native_adapter.rotary_emb(x, native_positions)
        semantic_rope = native_adapter.rotary_emb(x, semantic_positions)
        mix = self.layer_mix_logits.float().softmax(dim=-1).to(dtype=x.dtype)

        for index, native_block in enumerate(native_adapter.blocks):
            x = native_block(
                x,
                native_source,
                target_attention_mask=target_mask,
                source_attention_mask=native_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            if include_inserted_blocks:
                semantic_source = self.source_norms[index](self._mixed_source(semantic_hidden_states, mix, index))
                x = x + self.semantic_attentions[index](
                    self.query_norms[index](x),
                    mask=semantic_mask,
                    context=semantic_source,
                    position_embeddings=query_rope,
                    position_embeddings_context=semantic_rope,
                )

        context = native_adapter.norm(native_adapter.out_proj(x))
        if self.native_context_scale != 1.0:
            # scale*N + (E-N) == E + (scale-1)*N, as in the ComfyUI node.
            native_context = native_adapter(
                native_source,
                target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=native_source_mask,
            )
            context = context + (self.native_context_scale - 1.0) * native_context
        return context

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.state_dict().items()}

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.state_dict().keys())
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(f"Cross-adapter checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
        self.load_state_dict(dict(state), strict=True)


# ---------------------------------------------------------------------------
# v2: quality-anchored semantic connector
# ---------------------------------------------------------------------------


class QualityAnchoredSemanticConnectorV2(nn.Module):
    """Frozen v1 quality anchor plus a timestep-aware semantic resampler."""

    architecture = V2_ARCHITECTURE
    requires_timesteps = True

    def __init__(
        self,
        native_adapter: Any,
        semantic_source_dim: int = SEMANTIC_SOURCE_DIM,
        layer_indices: Sequence[int] = LAYER_INDICES,
        num_queries: int = 64,
        resampler_blocks: int = 6,
        resampler_dim: int = 2048,
        resampler_heads: int = 16,
        mlp_hidden_dim: int = 5632,
        initialize_from_native: bool = True,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_native_adapter", native_adapter)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.base_metadata: dict[str, str] = {}
        self.adapter_metadata: dict[str, str] = {}
        self.quality_anchor = ProgressiveQwen35CrossAdapter(
            native_adapter,
            semantic_source_dim=semantic_source_dim,
            layer_indices=self.layer_indices,
            initialize_from_native=initialize_from_native,
        )
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads
        device = native_adapter.embed.weight.device
        dtype = native_adapter.embed.weight.dtype

        self.semantic_resampler = TimestepAwareSemanticResampler(
            qwen_dim=semantic_source_dim,
            output_dim=model_dim,
            num_layers=len(self.layer_indices),
            num_queries=num_queries,
            num_blocks=resampler_blocks,
            model_dim=resampler_dim,
            num_heads=resampler_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            device=device,
            dtype=dtype,
            initialize_weights=initialize_from_native,
        )
        self.v2_query_norms = nn.ModuleList([_rms_norm(model_dim, device, dtype) for _ in native_adapter.blocks])
        self.v2_semantic_norms = nn.ModuleList([_rms_norm(model_dim, device, dtype) for _ in native_adapter.blocks])
        self.v2_attentions = nn.ModuleList(
            [_attention(model_dim, model_dim, num_heads, head_dim, device, dtype) for _ in native_adapter.blocks]
        )
        if initialize_from_native:
            self._initialize_v2_injections()

    @property
    def native_adapter(self) -> Any:
        return self._native_adapter

    @torch.no_grad()
    def _initialize_v2_injections(self) -> None:
        attention: Any
        query_norm: Any
        semantic_norm: Any
        for native, query_norm, semantic_norm, attention in zip(
            self.native_adapter.blocks, self.v2_query_norms, self.v2_semantic_norms, self.v2_attentions, strict=True
        ):
            query_norm.weight.copy_(native.norm_cross_attn.weight)
            semantic_norm.weight.fill_(1.0)
            attention.q_proj.weight.copy_(native.cross_attn.q_proj.weight)
            attention.q_norm.weight.copy_(native.cross_attn.q_norm.weight)
            attention.k_norm.weight.copy_(native.cross_attn.k_norm.weight)
            nn.init.xavier_uniform_(attention.k_proj.weight)
            nn.init.xavier_uniform_(attention.v_proj.weight)
            attention.o_proj.weight.zero_()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.semantic_resampler.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Sequence[torch.Tensor],
        target_attention_mask: torch.Tensor | None = None,
        native_source_mask: torch.Tensor | None = None,
        semantic_source_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        include_inserted_blocks: bool = True,
        include_v2: bool = True,
        **_ignored,
    ) -> torch.Tensor:
        if timesteps is None:
            raise ValueError("QualityAnchoredSemanticConnectorV2 requires diffusion timesteps")
        if len(semantic_hidden_states) != len(self.layer_indices):
            raise ValueError(f"Expected {len(self.layer_indices)} semantic layers, got {len(semantic_hidden_states)}")

        native_adapter = self.native_adapter
        anchor = self.quality_anchor
        target_mask = _attention_mask(target_attention_mask)
        native_mask = _attention_mask(native_source_mask)
        semantic_attention_mask = _attention_mask(semantic_source_mask)
        semantic_bank = None
        if include_inserted_blocks and include_v2:
            semantic_bank = self.semantic_resampler(semantic_hidden_states, timesteps, semantic_source_mask)

        x = _embed_target(native_adapter, target_input_ids, native_source.dtype)
        query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        native_positions = torch.arange(native_source.shape[1], device=x.device).unsqueeze(0)
        anchor_positions = torch.arange(semantic_hidden_states[0].shape[1], device=x.device).unsqueeze(0)
        bank_rope = None
        if semantic_bank is not None:
            bank_positions = torch.arange(semantic_bank.shape[1], device=x.device).unsqueeze(0)
            bank_rope = native_adapter.rotary_emb(x, bank_positions)
        query_rope = native_adapter.rotary_emb(x, query_positions)
        native_rope = native_adapter.rotary_emb(x, native_positions)
        anchor_rope = native_adapter.rotary_emb(x, anchor_positions)
        anchor_mix = anchor.layer_mix_logits.float().softmax(dim=-1).to(x.dtype)

        for index, native_block in enumerate(native_adapter.blocks):
            x = native_block(
                x,
                native_source,
                target_attention_mask=target_mask,
                source_attention_mask=native_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            if not include_inserted_blocks:
                continue
            anchor_source = anchor.source_norms[index](anchor._mixed_source(semantic_hidden_states, anchor_mix, index))
            x = x + anchor.semantic_attentions[index](
                anchor.query_norms[index](x),
                mask=semantic_attention_mask,
                context=anchor_source,
                position_embeddings=query_rope,
                position_embeddings_context=anchor_rope,
            )
            if semantic_bank is not None:
                x = x + self.v2_attentions[index](
                    self.v2_query_norms[index](x),
                    context=self.v2_semantic_norms[index](semantic_bank),
                    position_embeddings=query_rope,
                    position_embeddings_context=bank_rope,
                )

        return native_adapter.norm(native_adapter.out_proj(x))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.state_dict().items()}

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = set(self.state_dict().keys())
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(f"V2 checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
        self.load_state_dict(dict(state), strict=True)


SemanticConnector = ProgressiveQwen35CrossAdapter | QualityAnchoredSemanticConnectorV2


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def read_safetensors_metadata(path: str) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:  # type: ignore[reportGeneralTypeIssues]
        return dict(handle.metadata() or {})


def native_adapter_hash(native_state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 of the native LLM adapter, compatible with the ComfyUI bundler.

    ``native_state`` must use names relative to the adapter (``blocks.0...``).
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(native_state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bundle_connector_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    """Connector metadata embedded in a v1.1 bundle (``anima_v2_adapter_*`` keys)."""
    return {
        name[len(BUNDLE_ADAPTER_METADATA_PREFIX) :]: value
        for name, value in metadata.items()
        if name.startswith(BUNDLE_ADAPTER_METADATA_PREFIX)
    }


def is_bundle_metadata(metadata: Mapping[str, str]) -> bool:
    return metadata.get("architecture") == BUNDLE_ARCHITECTURE


def _metadata_number(metadata: Mapping[str, str], key: str, default: str, kind: type = int):
    raw = metadata.get(key, default)
    try:
        return kind(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid connector metadata {key}={raw!r}") from error


def layer_indices_from_metadata(metadata: Mapping[str, str]) -> tuple[int, ...]:
    raw = metadata.get("layer_indices")
    if not raw:
        return LAYER_INDICES
    try:
        return tuple(int(index) for index in json.loads(raw))
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"Invalid connector layer_indices metadata: {raw!r}") from error


def build_semantic_connector(
    native_adapter: Any,
    state: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str],
) -> SemanticConnector:
    """Instantiate a v1/v2 connector from adapter-relative tensors and metadata."""
    architecture = metadata.get("architecture")
    if architecture is None:
        # Bundles written by older tools may lack metadata; infer from keys.
        architecture = V2_ARCHITECTURE if any(k.startswith("semantic_resampler.") for k in state) else V1_ARCHITECTURE
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"Unsupported semantic connector architecture {architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}")
    if any(key.startswith("dit_blocks.") for key in state):
        raise RuntimeError(
            "This is a joint adapter+DiT training checkpoint, not a standalone connector. "
            "Materialize it into a diffusion-model checkpoint and an adapter-only checkpoint first."
        )
    if architecture == V1_ARCHITECTURE and any(key.startswith(("timestep_gates.", "anchor_deviation")) for key in state):
        raise RuntimeError("Timestep-gated or de-anchored v1 adapters are not supported (ComfyUI rejects them too).")

    layer_indices = layer_indices_from_metadata(metadata)
    if architecture == V2_ARCHITECTURE:
        connector: SemanticConnector = QualityAnchoredSemanticConnectorV2(
            native_adapter,
            semantic_source_dim=SEMANTIC_SOURCE_DIM,
            layer_indices=layer_indices,
            num_queries=_metadata_number(metadata, "semantic_query_tokens", "64"),
            resampler_blocks=_metadata_number(metadata, "semantic_resampler_blocks", "6"),
            resampler_dim=_metadata_number(metadata, "semantic_resampler_dim", "2048"),
            resampler_heads=_metadata_number(metadata, "semantic_resampler_heads", "16"),
            mlp_hidden_dim=_metadata_number(metadata, "semantic_resampler_mlp_hidden_dim", "5632"),
            initialize_from_native=False,
        )
    else:
        connector = ProgressiveQwen35CrossAdapter(
            native_adapter,
            semantic_source_dim=SEMANTIC_SOURCE_DIM,
            layer_indices=layer_indices,
            initialize_from_native=False,
            native_context_scale=_metadata_number(metadata, "native_context_scale", "1.0", float),
        )
    connector.load_trainable_state_dict(state)
    connector.requires_grad_(False)
    connector.eval()
    return connector


def check_native_adapter_hash(native_adapter: Any, metadata: Mapping[str, str], source: str) -> None:
    """Warn when a connector was trained against a different native adapter."""
    expected = metadata.get("frozen_native_sha256")
    if not expected:
        return
    try:
        actual = native_adapter_hash(native_adapter.state_dict())
    except Exception as error:  # noqa: BLE001 - diagnostics only
        logger.warning(f"Could not hash the native LLM adapter: {error}")
        return
    if actual != expected:
        logger.warning(
            f"{source}: the semantic connector was trained against a different native LLM adapter "
            f"(expected sha256 {expected[:12]}…, got {actual[:12]}…). "
            "This is expected after LLM-adapter fine-tuning or dtype conversion, otherwise check the model pair."
        )


def bundle_save_metadata(
    base_metadata: Mapping[str, str],
    adapter_metadata: Mapping[str, str],
    architecture: str,
    native_hash: str,
) -> dict[str, str]:
    """Metadata for a v1.1-style bundle (same layout as the ComfyUI ``bundle_v2.py`` tool).

    ``base_metadata`` is the source DiT's metadata (adapter keys are replaced),
    ``adapter_metadata`` the connector's own metadata without the prefix.
    """
    metadata = {
        name: value for name, value in base_metadata.items() if not name.startswith(BUNDLE_ADAPTER_METADATA_PREFIX)
    }
    metadata.setdefault("anima_v2_strength", "1.0")
    metadata.update(
        {
            "format": "pt",
            "architecture": BUNDLE_ARCHITECTURE,
            "anima_v2_bundle_format": BUNDLE_FORMAT,
            "anima_v2_connector_prefix": "net." + CONNECTOR_PREFIX,
            # The DiT's native adapter may have been fine-tuned: record the saved one.
            "anima_v2_native_sha256": native_hash,
        }
    )
    adapter = dict(adapter_metadata)
    adapter["architecture"] = architecture
    for name, value in adapter.items():
        metadata[BUNDLE_ADAPTER_METADATA_PREFIX + name] = value
    return metadata


def prepare_bundle_for_save(
    state_dict: dict[str, torch.Tensor],
    connector: SemanticConnector,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Turn a DiT state dict (``net.`` prefixed, already in the save dtype) into a 3.8B checkpoint.

    v2 connectors are saved inside the checkpoint (v1.1 bundle layout), so it loads
    in ComfyUI exactly like the released model. v1 connectors have no bundle format:
    they are dropped and must keep being loaded from their standalone file.
    """
    prefix = "net." + CONNECTOR_PREFIX
    if connector.architecture != V2_ARCHITECTURE:
        logger.warning(
            "The v1 semantic connector is not saved into the checkpoint; keep using its standalone file "
            "(--semantic_connector) with the saved model."
        )
        return {k: v for k, v in state_dict.items() if not k.startswith(prefix)}, {"format": "pt"}
    if not any(k.startswith(prefix) for k in state_dict):
        raise RuntimeError("The semantic connector tensors are missing from the state dict to save")
    adapter_prefix = "net.llm_adapter."
    native_hash = native_adapter_hash(
        {k[len(adapter_prefix) :]: v for k, v in state_dict.items() if k.startswith(adapter_prefix)}
    )
    metadata = bundle_save_metadata(
        connector.base_metadata, connector.adapter_metadata, connector.architecture, native_hash
    )
    return state_dict, metadata


def get_semantic_connector(dit: Any) -> SemanticConnector | None:
    """The connector attached to a DiT by ``load_anima_dit(load_semantic_connector=True)``, if any."""
    connector = getattr(dit, "anima_v2_connector", None)
    if isinstance(connector, (ProgressiveQwen35CrossAdapter, QualityAnchoredSemanticConnectorV2)):
        return connector
    return None


def semantic_layer_indices_for(dit_path: str, semantic_connector_path: str | None = None) -> tuple[int, ...]:
    """Qwen3.5 layers the connector reads, without loading the model."""
    if semantic_connector_path:
        return layer_indices_from_metadata(read_safetensors_metadata(semantic_connector_path))
    return layer_indices_from_metadata(bundle_connector_metadata(read_safetensors_metadata(dit_path)))


def is_anima_38b_checkpoint(path: str | None) -> bool:
    """True for Anima 3.8B DiT checkpoints (bundled connector or 52 blocks)."""
    if not path or not path.lower().endswith(".safetensors"):
        return False
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:  # type: ignore[reportGeneralTypeIssues]
            metadata = handle.metadata() or {}
            if is_bundle_metadata(metadata) or metadata.get("new_block_count") == "52":
                return True
            tensor_names = list(handle.keys())  # safe_open is not a mapping
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(f"Could not inspect {path}: {error}")
        return False
    return any(".blocks.51." in f".{name}" for name in tensor_names)
