"""CPU tests for Anima 3.8B support (semantic connector, strategies, DiT wiring).

Run: python -m pytest tests/test_anima_38b.py -q
"""

import os
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import anima_38b_connector as C  # noqa: E402
from library import (  # noqa: E402
    anima_block_freeze,
    anima_models,
    anima_train_utils,
    anima_utils,
    strategy_anima_38b,
    strategy_base,
)

LAYERS = C.LAYER_INDICES


def _tiny_v2(adapter, seed=0):
    torch.manual_seed(seed)
    return C.QualityAnchoredSemanticConnectorV2(
        adapter, resampler_blocks=1, resampler_dim=64, resampler_heads=4, mlp_hidden_dim=96, num_queries=8
    )


def _randomize_v2_outputs(connector):
    with torch.no_grad():
        for name, parameter in connector.v2_attentions.named_parameters():
            if name.endswith("o_proj.weight"):
                torch.nn.init.normal_(parameter, std=0.02)


@pytest.fixture(scope="module")
def adapter():
    torch.manual_seed(0)
    return anima_models.LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, self_attn=True).eval()


def _inputs(batch=2, native_len=7, target_len=5, semantic_len=9):
    torch.manual_seed(1)
    native = torch.randn(batch, native_len, 1024)
    ids = torch.randint(0, 32128, (batch, target_len))
    states = [torch.randn(batch, semantic_len, C.SEMANTIC_SOURCE_DIM) for _ in LAYERS]
    mask = torch.ones(batch, semantic_len, dtype=torch.long)
    mask[-1, 6:] = 0
    return native, ids, states, mask


@torch.no_grad()
def test_fresh_connectors_equal_native_adapter(adapter):
    native, ids, states, mask = _inputs()
    expected = adapter(native, ids)
    v1 = C.ProgressiveQwen35CrossAdapter(adapter)
    v2 = _tiny_v2(adapter)
    assert torch.allclose(v1(native, ids, states, semantic_source_mask=mask), expected, atol=1e-5)
    out = v2(native, ids, states, semantic_source_mask=mask, timesteps=torch.tensor([0.2, 0.9]))
    assert torch.allclose(out, expected, atol=1e-5)


@torch.no_grad()
def test_connector_does_not_own_native_adapter(adapter):
    v2 = _tiny_v2(adapter)
    assert not any(name.startswith(("native_adapter", "quality_anchor.native_adapter")) for name in v2.state_dict())
    assert v2.native_adapter is adapter


@torch.no_grad()
def test_v2_is_timestep_dependent_and_padding_invariant(adapter):
    v2 = _tiny_v2(adapter)
    _randomize_v2_outputs(v2)
    native, ids, states, mask = _inputs()
    t_lo, t_hi = torch.full((2,), 0.1), torch.full((2,), 0.9)
    out_lo = v2(native, ids, states, semantic_source_mask=mask, timesteps=t_lo)
    out_hi = v2(native, ids, states, semantic_source_mask=mask, timesteps=t_hi)
    assert (out_lo - out_hi).abs().max() > 1e-4

    padded_states = [torch.cat([s, torch.zeros(2, 3, s.shape[-1])], dim=1) for s in states]
    padded_mask = torch.cat([mask, torch.zeros(2, 3, dtype=mask.dtype)], dim=1)
    out_padded = v2(native, ids, padded_states, semantic_source_mask=padded_mask, timesteps=t_lo)
    assert torch.allclose(out_lo, out_padded, atol=1e-4)


@torch.no_grad()
def test_build_from_state_dict_roundtrip(adapter):
    source = _tiny_v2(adapter, seed=3)
    metadata = {
        "architecture": C.V2_ARCHITECTURE,
        "layer_indices": "[7, 15, 23, 31]",
        "semantic_query_tokens": "8",
        "semantic_resampler_blocks": "1",
        "semantic_resampler_dim": "64",
        "semantic_resampler_heads": "4",
        "semantic_resampler_mlp_hidden_dim": "96",
    }
    rebuilt = C.build_semantic_connector(adapter, source.trainable_state_dict(), metadata)
    assert isinstance(rebuilt, C.QualityAnchoredSemanticConnectorV2)
    assert not any(p.requires_grad for p in rebuilt.parameters())
    native, ids, states, mask = _inputs()
    t = torch.tensor([0.4, 0.6])
    assert torch.equal(source(native, ids, states, semantic_source_mask=mask, timesteps=t), rebuilt(native, ids, states, semantic_source_mask=mask, timesteps=t))

    with pytest.raises(RuntimeError, match="mismatch"):
        state = source.trainable_state_dict()
        state.pop(next(iter(state)))
        C.build_semantic_connector(adapter, state, metadata)
    with pytest.raises(RuntimeError, match="joint"):
        C.build_semantic_connector(adapter, {"dit_blocks.0.x": torch.zeros(1)}, {"architecture": C.V1_ARCHITECTURE})


def test_native_adapter_hash_is_order_independent():
    a = {"b.weight": torch.ones(2, dtype=torch.bfloat16), "a.weight": torch.zeros(3)}
    b = dict(reversed(list(a.items())))
    assert C.native_adapter_hash(a) == C.native_adapter_hash(b)
    assert C.native_adapter_hash(a) != C.native_adapter_hash({**a, "a.weight": torch.ones(3)})


def _tiny_dit():
    torch.manual_seed(0)
    return anima_models.MiniTrainDIT(
        max_img_h=64, max_img_w=64, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
        model_channels=64, num_blocks=2, num_heads=4, crossattn_emb_channels=1024, pos_emb_cls="rope3d",
        use_adaln_lora=True, adaln_lora_dim=16, use_llm_adapter=True,
    ).eval()


@torch.no_grad()
def test_attach_bundled_connector_and_dit_forward(tmp_path):
    dit = _tiny_dit()
    connector = _tiny_v2(dit.llm_adapter, seed=5)
    _randomize_v2_outputs(connector)
    dit_state = {k: v.contiguous() for k, v in dit.state_dict().items()}
    native_hash = C.native_adapter_hash({k[len("llm_adapter."):]: v for k, v in dit_state.items() if k.startswith("llm_adapter.")})
    metadata = {
        "architecture": C.BUNDLE_ARCHITECTURE,
        "anima_v2_bundle_format": C.BUNDLE_FORMAT,
        "anima_v2_native_sha256": native_hash,
        "anima_v2_adapter_architecture": C.V2_ARCHITECTURE,
        "anima_v2_adapter_layer_indices": "[7, 15, 23, 31]",
        "anima_v2_adapter_semantic_query_tokens": "8",
        "anima_v2_adapter_semantic_resampler_blocks": "1",
        "anima_v2_adapter_semantic_resampler_dim": "64",
        "anima_v2_adapter_semantic_resampler_heads": "4",
        "anima_v2_adapter_semantic_resampler_mlp_hidden_dim": "96",
    }
    bundle = {f"net.{k}": v for k, v in dit_state.items()}
    bundle.update({f"net.anima_v2_connector.{k}": v.contiguous() for k, v in connector.state_dict().items()})
    path = tmp_path / "bundle.safetensors"
    save_file(bundle, str(path), metadata=metadata)

    # load_anima_dit only derives configs for production widths, so exercise the attach step directly.
    fresh = _tiny_dit()
    fresh.load_state_dict(dit_state, strict=False)
    stripped = {k[len("net.anima_v2_connector."):]: v for k, v in bundle.items() if k.startswith("net.anima_v2_connector.")}
    anima_utils._attach_semantic_connector(fresh, str(path), dit_state, stripped, None)
    assert isinstance(fresh.anima_v2_connector, C.QualityAnchoredSemanticConnectorV2)
    assert fresh.anima_v2_connector.native_adapter is fresh.llm_adapter
    assert sum(k.startswith("anima_v2_connector.") for k in fresh.state_dict()) == len(stripped)

    native, ids, states, mask = _inputs(batch=1, target_len=6)
    t5_mask = torch.ones(1, 6, dtype=torch.long)
    t5_mask[0, 4:] = 0
    t = torch.tensor([0.3])
    expected = connector(native, ids, states, target_attention_mask=t5_mask, semantic_source_mask=mask[:1], timesteps=t)
    expected[~t5_mask.bool()] = 0
    got = fresh.compute_crossattn_emb(native, t, None, ids, t5_mask, states, mask[:1])
    assert torch.allclose(got, expected, atol=1e-5)

    x = torch.randn(1, 16, 1, 8, 8)
    padding = torch.zeros(1, 1, 8, 8)
    out = fresh(x, t, native, padding_mask=padding, t5_input_ids=ids, t5_attn_mask=t5_mask,
                semantic_hidden_states=states, semantic_attention_mask=mask[:1])
    ref = fresh(x, t, got, padding_mask=padding)
    assert torch.allclose(out, ref, atol=1e-5)

    cond = anima_train_utils.build_sample_crossattn(fresh, [native, torch.ones(1, 7), ids, t5_mask, *states, mask[:1]], torch.device("cpu"))
    assert callable(cond)
    assert torch.allclose(cond(t), got, atol=1e-5)


def _strategy():
    return strategy_anima_38b.Anima38BTextEncodingStrategy(LAYERS, dropout_rate=1.0)


def _outputs(batch=2, sem_len=5):
    native = [torch.randn(batch, 4, 8), torch.ones(batch, 4), torch.ones(batch, 3), torch.ones(batch, 3)]
    states = [torch.randn(batch, sem_len, 6) for _ in LAYERS]
    mask = torch.ones(batch, sem_len, dtype=torch.long)
    return [*native, *states, mask]


def test_caption_dropout_replaces_both_streams():
    strategy = _strategy()
    uncond = _outputs(batch=1, sem_len=1)
    strategy._uncond_outputs = uncond
    dropped = strategy.drop_cached_text_encoder_outputs(*_outputs())
    native, states, mask = strategy_anima_38b.split_text_encoder_outputs(dropped, len(LAYERS))
    assert torch.equal(native[0][1], uncond[0][0])
    assert mask[:, 0].tolist() == [1, 1] and mask[:, 1:].sum() == 0
    assert torch.equal(states[2][0, 0], uncond[4 + 2][0, 0]) and states[2][:, 1:].abs().sum() == 0


def test_text_encoder_cache_roundtrip(tmp_path):
    layer_count = len(LAYERS)

    class FakeEncoding(strategy_anima_38b.Anima38BTextEncodingStrategy):
        def encode_tokens(self, tokenize_strategy, models, tokens, enable_dropout=True):
            out = _outputs(batch=2, sem_len=6)
            out[-1][1, 3:] = 0  # second caption is 3 tokens long
            return out

    caching = strategy_anima_38b.Anima38BTextEncoderOutputsCachingStrategy(True, 2, False, LAYERS, 1024)
    infos = [
        types.SimpleNamespace(caption="a", text_encoder_outputs_npz=str(tmp_path / f"img{i}_anima38b_te.npz"), text_encoder_outputs=None)
        for i in range(2)
    ]
    class FakeTokenize(strategy_base.TokenizeStrategy):
        def tokenize(self, text):
            return []

    caching.cache_batch_outputs(FakeTokenize(), [], FakeEncoding(LAYERS), infos)
    for i, length in enumerate((6, 3)):
        assert caching.is_disk_cached_outputs_expected(infos[i].text_encoder_outputs_npz)
        loaded = caching.load_outputs_npz(infos[i].text_encoder_outputs_npz)
        assert len(loaded) == 4 + layer_count + 1
        assert loaded[4].shape == (length, 6) and loaded[-1].shape == (length,)

    other = strategy_anima_38b.Anima38BTextEncoderOutputsCachingStrategy(True, 2, False, LAYERS, 512)
    assert not other.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz)


def _adapter_param(dit, name: str) -> torch.nn.Parameter:
    return dict(dit.llm_adapter.named_parameters())[name]


def _save_and_reload(tmp_path, dit, connector):
    path = tmp_path / "saved.safetensors"
    anima_utils.save_anima_model(str(path), dit.state_dict(), torch.float32, connector)
    from safetensors.torch import load_file

    return load_file(str(path)), C.read_safetensors_metadata(str(path))


@torch.no_grad()
def test_full_finetune_save_writes_v11_bundle(tmp_path):
    dit = _tiny_dit()
    connector = _tiny_v2(dit.llm_adapter, seed=7)
    connector.base_metadata = {"new_block_count": "52", "anima_v2_adapter_step": "old", "architecture": "old"}
    connector.adapter_metadata = {"layer_indices": "[7, 15, 23, 31]", "semantic_query_tokens": "8"}
    dit.add_module("anima_v2_connector", connector)
    _adapter_param(dit, "blocks.0.mlp.0.weight").add_(1.0)  # e.g. a fine-tuned native adapter

    saved, metadata = _save_and_reload(tmp_path, dit, connector)
    assert C.is_bundle_metadata(metadata) and metadata["anima_v2_bundle_format"] == C.BUNDLE_FORMAT
    assert metadata["anima_v2_connector_prefix"] == "net.anima_v2_connector."
    assert metadata["new_block_count"] == "52" and "anima_v2_adapter_step" not in metadata
    assert C.bundle_connector_metadata(metadata) == {**connector.adapter_metadata, "architecture": C.V2_ARCHITECTURE}
    native = {k[len("net.llm_adapter."):]: v for k, v in saved.items() if k.startswith("net.llm_adapter.")}
    assert metadata["anima_v2_native_sha256"] == C.native_adapter_hash(native)
    for name, value in connector.state_dict().items():
        assert torch.equal(saved["net.anima_v2_connector." + name], value)


@torch.no_grad()
def test_full_finetune_save_drops_standalone_v1_connector(tmp_path):
    dit = _tiny_dit()
    connector = C.ProgressiveQwen35CrossAdapter(dit.llm_adapter)
    dit.add_module("anima_v2_connector", connector)
    saved, metadata = _save_and_reload(tmp_path, dit, connector)
    assert not any(k.startswith("net.anima_v2_connector.") for k in saved)
    assert metadata == {"format": "pt"}


def test_connector_grad_follows_native_adapter():
    dit = _tiny_dit()
    connector = _tiny_v2(dit.llm_adapter, seed=9)
    _randomize_v2_outputs(connector)
    dit.add_module("anima_v2_connector", connector)
    dit.requires_grad_(False)
    native, ids, states, mask = _inputs(batch=1, target_len=6)
    t = torch.tensor([0.5])
    assert not dit.compute_crossattn_emb(native, t, None, ids, None, states, mask[:1]).requires_grad
    dit.llm_adapter.requires_grad_(True)  # full fine-tuning with a trainable native adapter
    out = dit.compute_crossattn_emb(native, t, None, ids, None, states, mask[:1])
    out.sum().backward()
    assert _adapter_param(dit, "blocks.0.mlp.0.weight").grad is not None
    assert all(p.grad is None for p in connector.parameters())


def test_inserted_only_freeze_supports_52_blocks():
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(52))
    dit.llm_adapter = torch.nn.Linear(2, 2)
    summary = anima_block_freeze.apply_inserted_only_training_freeze(dit)
    trainable = [i for i, b in enumerate(dit.blocks) if b.weight.requires_grad]
    assert trainable == list(anima_block_freeze.INSERTED_52_BLOCK_INDICES) == summary["inserted_block_indices"]
    assert not dit.llm_adapter.weight.requires_grad  # the trainer re-enables it unless --llm_adapter_lr 0
