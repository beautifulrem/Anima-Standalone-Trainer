# Anima 3.8B LoRA training

[Anima 3.8B](https://huggingface.co/lylogummy/Anima-3.8B) is a 52-block Anima DiT
(LLaMA-Pro expansion of the 40-block 2.9B) whose text conditioning adds a
**Qwen3.5 4B semantic stream** on top of the native Qwen3 0.6B path:

```
Qwen3 0.6B ─► native LLM adapter (6 blocks) ─┐
Qwen3.5 4B, hidden layers 7/15/23/31 ────────┴─► semantic connector ─► DiT cross-attention
```

The v2 connector shipped in `Anima-3.8B-v1.1.safetensors` contains a
timestep-aware resampler, so the conditioning changes with the diffusion
timestep and is recomputed every step.
`anima_38b_train_network.py` reproduces this path exactly as the
[ComfyUI node](https://github.com/GumGum10/comfyui-anima-3-8B) does at inference,
so LoRAs trained here see the same conditioning they get in ComfyUI.

## Files

| File (from `lylogummy/Anima-3.8B`) | Argument |
| --- | --- |
| `difussion_models/Anima-3.8B-v1.1.safetensors` (connector bundled) | `--dit_path` |
| `text_encoders/qwen35_4b.safetensors` | `--qwen35_path` |
| Qwen3 0.6B (`qwen_3_06b_base.safetensors`, same as Anima) | `--qwen3_path` |
| Qwen-Image VAE (same as Anima) | `--vae_path` |

You do not need `--semantic_connector` for v1.1, because the connector is
bundled. For the original `Anima-3.8B.safetensors`, pass
`--semantic_connector text_encoders/Anima-3.8B-expanded_adapter.safetensors`
(the v1 connector).

Do not pair the old DiT with the v1.1 connector. The DiT weights differ
between the two releases. The loader checks the native-adapter SHA-256 that
the bundle records, and warns on a mismatch.

The Qwen3.5 tokenizer ships in `configs/qwen35_4b`.

## Example

```bash
accelerate launch anima_38b_train_network.py \
  --dit_path Anima-3.8B-v1.1.safetensors \
  --qwen3_path qwen_3_06b_base.safetensors \
  --qwen35_path qwen35_4b.safetensors \
  --vae_path qwen_image_vae.safetensors \
  --dataset_config dataset.toml \
  --network_module networks.lora_anima --network_dim 32 --network_alpha 16 \
  --network_train_unet_only \
  --cache_latents_to_disk --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --mixed_precision bf16 --gradient_checkpointing --blocks_to_swap 20 \
  --learning_rate 1e-4 --optimizer_type AdamW8bit \
  --output_dir output --output_name my_lora --save_model_as safetensors
```

In the Web UI, set **Qwen3.5 4B Text Encoder Path** in Global Settings. The
server detects a 3.8B DiT by its bundle metadata or its 52 blocks, and then
launches `anima_38b_train_network.py` automatically.

## Notes

- **Only DiT networks are trained.** The DiT can be trained with LoRA, LoHa
  or LoKr, and optionally the native adapter blocks too
  (`--network_args train_llm_adapter=True`). Both text encoders and the
  connector stay frozen, and `--network_train_unet_only` is required.
- **Caching is strongly recommended.**
  - Use `--cache_text_encoder_outputs`. The Qwen3.5 states are stored
    trimmed to the prompt length, in bf16 safetensors files named
    `*_anima38b_te.safetensors`.
  - Each cached prompt token takes about 20 KB, so a 200-token caption needs
    about 4 MB.
  - If you change `--qwen35_max_token_length`, the cache is invalidated and
    rebuilt.
  - Without caching, Qwen3.5 (about 4.5 GB in FP8) stays on the GPU and runs
    every step.
- **Qwen3.5 precision.**
  - The encoder keeps its FP8 weights. By default it computes in float32,
    like ComfyUI.
  - `--qwen35_compute_dtype bfloat16` is faster but approximate.
  - Its outputs were checked layer by layer against Hugging Face
    `transformers` `Qwen3_5TextModel`; the maximum relative error is below
    1e-6.
- **Caption dropout** (`--caption_dropout_rate`) replaces both the native and
  the Qwen3.5 conditioning with the empty-prompt conditioning.
- **Sample images during training** recompute the connector at every step.
  Parallel CFG is disabled for these samples.
- **`--fp8_base`**: only the DiT blocks' Linear weights become fp8 (most of the
  parameters). The LLM adapter, the connector, embedders and norms stay in the
  training dtype, because element-wise ops cannot mix fp8 and float. This also
  fixes `--fp8_base` for regular Anima LoRA training, where casting the whole
  DiT used to fail in the LLM adapter.
- **Not supported** (rejected with a clear error):
  - full fine-tuning (`anima_train.py`, `anima_train_muon.py`);
  - TP/SP;
  - training the text encoders;
  - `--weighted_captions`.
- Every non-3.8B training script refuses an Anima 3.8B bundle. Otherwise it would
  train without the connector, and a full fine-tune would save the model with
  the connector removed.
- **Saved LoRAs** use the usual `lora_unet_blocks_N_*` names for blocks 0–51,
  and load in ComfyUI like any Anima LoRA.
