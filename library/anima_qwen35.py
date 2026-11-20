# Qwen3.5 4B hybrid text encoder used by Anima 3.8B's semantic connector.
#
# Ported from GumGum10/comfyui-anima-3-8B (MIT License, Copyright (c) 2026
# GumGum10 contributors) and GumGum10/sd-scripts. The ComfyUI node is the
# inference reference for Anima 3.8B, so the math here intentionally mirrors it:
# the trainer must produce exactly the hidden states that ComfyUI/Forge feed the
# connector at inference time.

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


QWEN35_PAD_TOKEN_ID = 151643
QWEN35_DEFAULT_LAYER_INDICES = (7, 15, 23, 31)
QWEN35_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "qwen35_4b")
QWEN35_TOKENIZER_HF_REPO = "Qwen/Qwen3.5-4B"


class _CastLinear(nn.Linear):
    """Linear that keeps its storage dtype (e.g. FP8) and computes in the input dtype.

    This mirrors ComfyUI's manual-cast ops, which is how the reference node runs
    the FP8 ``qwen35_4b`` checkpoint (weights upcast, float32 activations).
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - matches nn.Linear
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias)


class _CastConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - matches nn.Conv1d
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return self._conv_forward(input, self.weight.to(input.dtype), bias)


class _CastEmbedding(nn.Embedding):
    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:  # noqa: A002
        out = F.embedding(input, self.weight)
        return out.to(out_dtype) if out_dtype is not None else out


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class _DirectRMSNorm(nn.Module):
    """Direct-scale RMSNorm used by Qwen3.5's gated delta block."""

    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        return y.to(dtype=x.dtype)


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm whose checkpoint weights are deltas around one."""

    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * (1.0 + self.weight.float())
        return y.to(dtype=x.dtype)


class Qwen35GatedDeltaNet(nn.Module):
    """Reference-equivalent Qwen3.5 GatedDeltaNet linear-attention block."""

    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32, d_gate=4096, conv_kernel=4, norm_dim=128, device=None, dtype=None):
        super().__init__()
        del d_inner, d_gate
        self.num_v_heads = n_groups
        self.num_k_heads = 16
        self.head_k_dim = norm_dim
        self.head_v_dim = norm_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = _CastLinear(hidden_size, self.conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = _CastLinear(hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_a = _CastLinear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_b = _CastLinear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.conv1d = _CastConv1d(
            self.conv_dim,
            self.conv_dim,
            conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.out_proj = _CastLinear(self.value_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = _DirectRMSNorm(self.head_v_dim, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))

    @staticmethod
    def _delta_rule_scan(query, key, value, g, beta):
        initial_dtype = query.dtype
        query = _l2norm(query, dim=-1).transpose(1, 2).contiguous().float()
        key = _l2norm(key, dim=-1).transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        batch, heads, seq_len, k_dim = key.shape
        v_dim = value.shape[-1]
        query = query * (1.0 / (k_dim**0.5))
        state = torch.zeros(batch, heads, k_dim, v_dim, device=query.device, dtype=torch.float32)
        outputs = []

        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

        return torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(initial_dtype)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)

        batch, seq_len, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp().to(hidden_states.device) * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.to(hidden_states.device).float()
        )
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        y = self._delta_rule_scan(query, key, value, g, beta)
        y = self.norm(y.reshape(-1, self.head_v_dim))
        y = y * F.silu(z.reshape(-1, self.head_v_dim).float()).to(y.dtype)
        return self.out_proj(y.reshape(batch, seq_len, self.value_dim))


def _apply_rotary_emb(x, cos, sin):
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_embed = (x_rot * cos) + (rotated * sin)
    return torch.cat((x_embed, x_pass), dim=-1)


def _precompute_freqs_cis(head_dim, max_seq_len, theta=10000000.0, device=None, dtype=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class Qwen35GatedSelfAttention(nn.Module):
    """Self-attention with a sigmoid output gate packed into q_proj (GQA 16/4, head_dim 256)."""

    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4, head_dim=256, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim

        self.q_proj = _CastLinear(hidden_size, 2 * self.inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = _CastLinear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = _CastLinear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = _CastLinear(self.inner_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape
        qg = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, L, self.inner_dim)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        k = k.repeat_interleave(self.gqa_ratio, dim=1)
        v = v.repeat_interleave(self.gqa_ratio, dim=1)

        attn_mask = attention_mask.to(dtype=q.dtype) if attention_mask is not None else None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attention_mask is None))
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


class Qwen35MLP(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216, device=None, dtype=None):
        super().__init__()
        self.gate_proj = _CastLinear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = _CastLinear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = _CastLinear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35HybridBlock(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216, use_ssm=True, has_mlp=True, device=None, dtype=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp
        self.input_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)
        if use_ssm:
            self.linear_attn = Qwen35GatedDeltaNet(hidden_size=hidden_size, device=device, dtype=dtype)
        else:
            self.self_attn = Qwen35GatedSelfAttention(hidden_size=hidden_size, device=device, dtype=dtype)
        if has_mlp:
            self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = Qwen35MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, linear_attention_mask=None):
        residual = x
        x_norm = self.input_layernorm(x)
        mixed = (
            self.linear_attn(x_norm, attention_mask=linear_attention_mask)
            if self.use_ssm
            else self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)
        )
        x = residual + mixed
        if self.has_mlp:
            x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen35HybridModel(nn.Module):
    """Qwen3.5 4B hybrid text backbone (24 GatedDeltaNet + 8 gated self-attention layers)."""

    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    ROTARY_DIM = 64
    ROPE_THETA = 10000000.0

    def __init__(self, dtype=None, device=None, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_layers = self.NUM_LAYERS
        # ComfyUI runs this encoder with float32 activations (weights upcast on
        # the fly), so float32 is the default to reproduce inference exactly.
        self.compute_dtype = compute_dtype
        self.embed_tokens = _CastEmbedding(self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Qwen35HybridBlock(
                    hidden_size=self.HIDDEN_SIZE,
                    intermediate_size=self.INTERMEDIATE_SIZE,
                    use_ssm=index not in self.SELF_ATTN_LAYERS,
                    has_mlp=index != self.NUM_LAYERS - 1,
                    device=device,
                    dtype=dtype,
                )
                for index in range(self.NUM_LAYERS)
            ]
        )
        # Projection head shipped in the Anima checkpoint. It is not used by the
        # semantic connector (which reads intermediate layers), but it is part of
        # the checkpoint so it is kept for strict loading.
        self.norm = nn.Sequential(
            _CastLinear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            Qwen35RMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            _CastLinear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def forward(self, *args, **kwargs):
        device_type = self.embed_tokens.weight.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self._forward(*args, **kwargs)

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        intermediate_output: int | Sequence[int] | None = None,
        return_final: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | dict[int, torch.Tensor] | None]:
        """Run the backbone.

        Returns ``(final_projection, intermediate)`` like the ComfyUI reference.
        ``intermediate`` is a dict ``{layer_index: hidden_state}`` when a list of
        indices is requested. When ``return_final`` is False the model stops after
        the deepest requested layer and returns ``None`` for the final projection.
        """
        hidden_states = self.embed_tokens(input_ids, out_dtype=self.compute_dtype)
        sequence_length = hidden_states.shape[1]
        rotary = _precompute_freqs_cis(
            self.ROTARY_DIM, sequence_length, theta=self.ROPE_THETA, device=hidden_states.device, dtype=hidden_states.dtype
        )

        linear_attention_mask = attention_mask
        attention_bias = None
        if attention_mask is not None:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            causal = torch.empty(
                sequence_length, sequence_length, dtype=hidden_states.dtype, device=hidden_states.device
            ).fill_(mask_fill).triu_(1)
            padding = 1.0 - attention_mask.to(hidden_states.dtype).reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1]).expand(
                attention_mask.shape[0], 1, sequence_length, attention_mask.shape[-1]
            )
            padding = padding.masked_fill(padding.to(torch.bool), mask_fill)
            attention_bias = causal + padding
        elif sequence_length > 1:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            attention_bias = torch.empty(
                sequence_length, sequence_length, dtype=hidden_states.dtype, device=hidden_states.device
            ).fill_(mask_fill).triu_(1)

        if isinstance(intermediate_output, int):
            wanted = {intermediate_output}
        elif intermediate_output is not None:
            wanted = {int(i) for i in intermediate_output}
        else:
            wanted = set()
        last_needed = max(wanted) if (wanted and not return_final) else self.NUM_LAYERS - 1

        intermediate = None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_bias,
                freqs_cis=rotary,
                linear_attention_mask=linear_attention_mask,
            )
            if isinstance(intermediate_output, int) and index == intermediate_output:
                intermediate = hidden_states.clone()
            elif intermediate_output is not None and not isinstance(intermediate_output, int) and index in wanted:
                if intermediate is None:
                    intermediate = {}
                intermediate[index] = hidden_states.clone()
            if index >= last_needed:
                break

        final = self.norm(hidden_states) if return_final else None
        return final, intermediate


def load_qwen35_tokenizer(path_or_repo: str | None = None):
    """Load the Qwen3.5 tokenizer.

    Resolution order: explicit path/repo, the bundled ``configs/qwen35_4b`` copy,
    then the Hugging Face repo ``Qwen/Qwen3.5-4B``.
    """
    from transformers import AutoTokenizer

    candidates: list[str] = []
    if path_or_repo:
        candidates.append(path_or_repo)
    if os.path.isfile(os.path.join(QWEN35_TOKENIZER_DIR, "tokenizer.json")):
        candidates.append(QWEN35_TOKENIZER_DIR)
    candidates.append(QWEN35_TOKENIZER_HF_REPO)

    last_error = None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate, trust_remote_code=False, local_files_only=os.path.isdir(candidate)
            )
        except Exception as error:  # noqa: BLE001 - try the next candidate
            last_error = error
            logger.warning(f"Could not load Qwen3.5 tokenizer from {candidate}: {error}")
            continue
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = QWEN35_PAD_TOKEN_ID
        tokenizer.padding_side = "right"
        logger.info(f"Loaded Qwen3.5 tokenizer from {candidate}")
        return tokenizer
    raise FileNotFoundError(f"Could not load the Qwen3.5 tokenizer from any of {candidates}") from last_error


def tokenize_qwen35(tokenizer, texts: str | list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize like the ComfyUI Anima 3.8B node.

    No special tokens are added; an empty prompt becomes a single pad token so the
    semantic stream is never fully masked (a fully masked SDPA row yields NaN).
    Sequences are right-padded to the longest prompt in the batch.
    """
    texts = [texts] if isinstance(texts, str) else list(texts)
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = [QWEN35_PAD_TOKEN_ID]
        sequences.append(ids[:max_length])
    longest = max(len(ids) for ids in sequences)
    input_ids = torch.full((len(sequences), longest), QWEN35_PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def load_qwen35_text_encoder(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    storage_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Qwen35HybridModel:
    """Load ``qwen35_4b.safetensors``.

    By default tensors keep their checkpoint dtype (the released file stores
    every linear weight as FP8 E4M3, about 4.5 GB) and are upcast to
    ``compute_dtype`` inside each layer, matching ComfyUI's manual-cast path.
    Pass ``storage_dtype`` to convert the weights at load time instead.
    """
    from safetensors import safe_open

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Qwen3.5 checkpoint not found: {checkpoint_path}")

    with torch.device("meta"):
        model = Qwen35HybridModel(compute_dtype=compute_dtype)

    expected = set(model.state_dict().keys())
    state = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:  # type: ignore[reportGeneralTypeIssues]
        keys = list(handle.keys())
        unexpected = sorted(set(keys) - expected)
        missing = sorted(expected - set(keys))
        if unexpected or missing:
            raise RuntimeError(f"Qwen3.5 checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        for key in keys:
            tensor = handle.get_tensor(key)
            if storage_dtype is not None and tensor.is_floating_point() and tensor.dtype != storage_dtype:
                tensor = tensor.to(storage_dtype)
            state[key] = tensor.to(device=device)

    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    storage = sorted({str(t.dtype).replace("torch.", "") for t in model.state_dict().values()})
    logger.info(
        f"Loaded Qwen3.5 4B text encoder from {checkpoint_path} (storage={storage}, compute={compute_dtype}, device={device})"
    )
    return model


def move_qwen35_text_encoder(model: Qwen35HybridModel, device: str | torch.device) -> Qwen35HybridModel:
    """Move the encoder, keeping FP8 storage where the backend supports it.

    Backends without FP8 tensors (e.g. Apple MPS) get bfloat16 storage, which
    represents every FP8 E4M3 value exactly.
    """
    device = torch.device(device)
    if device.type == "mps":
        fp8_dtypes = {getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)}
        if any(p.dtype in fp8_dtypes for p in model.parameters()):
            model.to(dtype=torch.bfloat16)
    return model.to(device)
