# Library Directory Audit Report

## Summary
The "Safe to Delete" files were audited to ensure their removal effectively "feature-gates" the trainer to exclude optional/legacy methods without breaking the core **Anima** or **Standard LoRA (train_network.py)** training processes.

**Recommendation:** Delete all files listed below to reduce clutter and potential confusion.

## Safe to Delete (Confirmed Unused)

### 1. Stable Diffusion 3 (SD3)
These files are specific to SD3 architecture and are not used by Anima or the generic SD1.5/2.1/SDXL trainer.
- `library/strategy_sd3.py`
- `library/sd3_models.py`
- `library/sd3_train_utils.py`
- `library/sd3_utils.py`

### 2. Hunyuan
These files are specific to Hunyuan Video/Image generation models.
- `library/strategy_hunyuan_image.py`
- `library/hunyuan_image_models.py`
- `library/hunyuan_image_modules.py`
- `library/hunyuan_image_text_encoder.py`
- `library/hunyuan_image_utils.py`
- `library/hunyuan_image_vae.py`

### 3. Lumina
These files are specific to Lumina-Next-T2I models.
- `library/strategy_lumina.py`
- `library/lumina_models.py`
- `library/lumina_train_util.py`
- `library/lumina_util.py`

### 4. Flux (Standard)
Anima uses its own implementation (`anima_models.py`, `strategy_anima.py`) derived from Flux but separate. The standard Flux library files are not imported by Anima scripts.
- `library/strategy_flux.py`
- `library/flux_models.py`
- `library/flux_train_utils.py`
- `library/flux_utils.py`

### 5. SDXL Specifics
Although `train_network.py` supports SDXL via `strategy_sd.py`, it does not utilize these specific standalone utility files. They are likely for `sdxl_train.py` or `sdxl_train_network.py` which are not present.
- `library/strategy_sdxl.py`
- `library/sdxl_lpw_stable_diffusion.py`
- `library/sdxl_model_util.py`
- `library/sdxl_original_control_net.py`
- `library/sdxl_original_unet.py`
- `library/sdxl_train_util.py`

### 6. Miscellaneous / Legacy
- `library/slicing_vae.py`: No references found.
- `library/hypernetwork.py`: Used only for Hypernetwork training (not supported/present in this repo).
- `library/chroma_models.py`: No references found.
- `library/original_unet.py`: Standard SD1.5 UNet definition (used by older scripts, not Anima).
- `library/lpw_stable_diffusion.py`: Long Prompt Weighting for SD1.5 (Anima uses Qwen3/T5).

## Critical Files (DO NOT DELETE)
- `library/strategy_sd.py`: Required by `train_network.py` (parent class of `AnimaNetworkTrainer`).
- `library/train_util.py`: Core utility library.
- `library/anima_*.py`: Core Anima files.
- `library/fp8_optimization_utils.py`: Used by `lora_utils.py`.
