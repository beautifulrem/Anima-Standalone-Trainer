# Anima model loading/saving utilities

import os
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device  # kept for potential future use

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import anima_models


# Keys that should stay in high precision (float32/bfloat16, not quantized)
KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']


def load_safetensors(path: str, device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Load a safetensors file and optionally cast to dtype."""
    sd = load_file(path, device=device)
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def load_anima_dit(
    dit_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cpu",
    transformer_dtype: Optional[torch.dtype] = None,
    llm_adapter_path: Optional[str] = None,
    disable_mmap: bool = False,
    load_semantic_connector: bool = False,
    semantic_connector_path: "str | None" = None,
    ignore_semantic_connector: bool = False,
) -> anima_models.MiniTrainDIT:
    """Load the MiniTrainDIT model from safetensors.

    Args:
        dit_path: Path to DiT safetensors file
        dtype: Base dtype for model parameters
        device: Device to load to
        transformer_dtype: Optional separate dtype for transformer blocks (lower precision)
        llm_adapter_path: Optional separate path for LLM adapter weights
        disable_mmap: If True, disable memory-mapped loading (reduces peak memory)
        load_semantic_connector: Attach the Anima 3.8B Qwen3.5 semantic connector as
            ``dit.anima_v2_connector`` (from the bundled checkpoint or ``semantic_connector_path``)
        semantic_connector_path: Optional standalone v1/v2 semantic connector file
        ignore_semantic_connector: Drop a bundled connector instead of raising (inference without Qwen3.5)
    """
    if transformer_dtype is None:
        transformer_dtype = dtype

    logger.info(f"Loading Anima DiT from {dit_path}")
    if disable_mmap:
        from library.safetensors_utils import load_safetensors as load_safetensors_no_mmap
        state_dict = load_safetensors_no_mmap(dit_path, device="cpu", disable_mmap=True)
    else:
        state_dict = load_file(dit_path, device="cpu")

    # Normalize checkpoint prefixes
    prefixes = (
        "net.",
        "model.diffusion_model.",
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        new_state_dict[k] = v

    state_dict = new_state_dict

    # Anima 3.8B v1.1 bundles its Qwen3.5 semantic connector under net.anima_v2_connector.*
    from library import anima_38b_connector

    connector_state = {
        k[len(anima_38b_connector.CONNECTOR_PREFIX):]: state_dict.pop(k)
        for k in list(state_dict.keys())
        if k.startswith(anima_38b_connector.CONNECTOR_PREFIX)
    }
    if connector_state and not load_semantic_connector:
        message = (
            f"{os.path.basename(dit_path)} is an Anima 3.8B checkpoint with a bundled Qwen3.5 semantic connector "
            f"({len(connector_state)} tensors)."
        )
        if not ignore_semantic_connector:
            # Training without the connector would condition the DiT differently from inference,
            # and a full fine-tune would save a checkpoint with the connector silently removed.
            raise RuntimeError(
                message + " This script does not support it; use anima_38b_train_network.py (LoRA) for Anima 3.8B."
            )
        logger.warning(message + " Ignoring it: only the native Qwen3 adapter is used.")
        connector_state = {}

    # Derive config from state_dict
    dit_config = anima_models.get_dit_config(state_dict)

    # Detect LLM adapter
    if llm_adapter_path is not None:
        use_llm_adapter = True
        dit_config['use_llm_adapter'] = True
        llm_adapter_state_dict = load_safetensors(llm_adapter_path, device="cpu")
    elif 'llm_adapter.out_proj.weight' in state_dict:
        use_llm_adapter = True
        dit_config['use_llm_adapter'] = True
        llm_adapter_state_dict = None  # Loaded as part of DiT
    else:
        use_llm_adapter = False
        llm_adapter_state_dict = None

    logger.info(f"DiT config: model_channels={dit_config['model_channels']}, num_blocks={dit_config['num_blocks']}, "
                f"num_heads={dit_config['num_heads']}, use_llm_adapter={use_llm_adapter}")

    # Build model normally on CPU — buffers get proper values from __init__
    dit = anima_models.MiniTrainDIT(**dit_config)

    # Merge LLM adapter weights into state_dict if loaded separately
    if use_llm_adapter and llm_adapter_state_dict is not None:
        for k, v in llm_adapter_state_dict.items():
            state_dict[f"llm_adapter.{k}"] = v

    # Load checkpoint: strict=False keeps buffers not in checkpoint (e.g. pos_embedder.seq)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [k for k in missing if not any(
            buf_name in k for buf_name in ('seq', 'dim_spatial_range', 'dim_temporal_range', 'inv_freq')
        )]
        if unexpected_missing:
            logger.warning(f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}")
    if unexpected:
        logger.info(f"Unexpected keys in checkpoint (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if load_semantic_connector:
        _attach_semantic_connector(dit, dit_path, state_dict, connector_state, semantic_connector_path)

    # Apply per-parameter dtype (high precision for 1D/critical, transformer_dtype for rest)
    for name, p in dit.named_parameters():
        dtype_to_use = dtype if (
            any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1
        ) else transformer_dtype
        p.data = p.data.to(dtype=dtype_to_use)

    dit.to(device)
    logger.info(f"Loaded Anima DiT successfully. Parameters: {sum(p.numel() for p in dit.parameters()):,}")
    return dit


def _attach_semantic_connector(
    dit: anima_models.MiniTrainDIT,
    dit_path: str,
    dit_state_dict: "dict[str, torch.Tensor]",
    bundled_state: "dict[str, torch.Tensor]",
    semantic_connector_path: "str | None",
) -> None:
    """Build the Anima 3.8B semantic connector and attach it as ``dit.anima_v2_connector``."""
    from library import anima_38b_connector as connector_lib

    if not getattr(dit, "use_llm_adapter", False) or not hasattr(dit, "llm_adapter"):
        raise RuntimeError("The Anima 3.8B semantic connector needs a DiT with an LLM adapter.")

    dit_metadata = connector_lib.read_safetensors_metadata(dit_path) if dit_path.lower().endswith(".safetensors") else {}
    if semantic_connector_path:
        if bundled_state:
            logger.info(f"Using {semantic_connector_path} instead of the connector bundled in {os.path.basename(dit_path)}")
        state = load_file(semantic_connector_path, device="cpu")
        metadata = connector_lib.read_safetensors_metadata(semantic_connector_path)
        source = os.path.basename(semantic_connector_path)
        expected_hash = metadata.get("frozen_native_sha256")
    elif bundled_state:
        state = bundled_state
        metadata = connector_lib.bundle_connector_metadata(dit_metadata)
        source = f"{os.path.basename(dit_path)} (bundled)"
        expected_hash = dit_metadata.get("anima_v2_native_sha256") or metadata.get("frozen_native_sha256")
    else:
        raise RuntimeError(
            f"{os.path.basename(dit_path)} has no bundled Qwen3.5 semantic connector. Use the bundled "
            "Anima-3.8B-v1.1.safetensors, or pass --semantic_connector with a standalone v1/v2 connector "
            "(e.g. Anima-3.8B-expanded_adapter.safetensors for the original Anima-3.8B.safetensors)."
        )

    if expected_hash:
        native_state = {k[len("llm_adapter."):]: v for k, v in dit_state_dict.items() if k.startswith("llm_adapter.")}
        actual_hash = connector_lib.native_adapter_hash(native_state)
        if actual_hash != expected_hash:
            logger.warning(
                f"{source}: the connector was trained against a different native LLM adapter "
                f"(expected sha256 {expected_hash[:12]}, got {actual_hash[:12]}). Check that the DiT and connector match."
            )
        else:
            logger.info(f"{source}: native LLM adapter hash matches ({actual_hash[:12]})")

    connector = connector_lib.build_semantic_connector(dit.llm_adapter, state, metadata)
    connector.base_metadata = dit_metadata
    connector.adapter_metadata = dict(metadata)
    dit.add_module("anima_v2_connector", connector)
    n_params = sum(p.numel() for p in connector.parameters())
    logger.info(
        f"Attached Anima 3.8B semantic connector {connector.architecture} from {source}: "
        f"{n_params:,} parameters, Qwen3.5 layers {list(connector.layer_indices)}"
    )


def load_anima_vae(vae_path: str, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    """Load WanVAE from a safetensors/pth file.

    Returns (vae_model, mean_tensor, std_tensor, scale).
    """
    from library.anima_models import ANIMA_VAE_MEAN, ANIMA_VAE_STD

    logger.info(f"Loading Anima VAE from {vae_path}")

    # VAE config (fixed for WanVAE)
    vae_config = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )

    from library.anima_vae import WanVAE_

    # Build model
    with torch.device('meta'):
        vae = WanVAE_(**vae_config)

    # Load state dict
    if vae_path.endswith('.safetensors'):
        vae_sd = load_file(vae_path, device='cpu')
    else:
        vae_sd = torch.load(vae_path, map_location='cpu', weights_only=True)

    vae.load_state_dict(vae_sd, assign=True)
    vae = vae.eval().requires_grad_(False).to(device, dtype=dtype)

    # Create normalization tensors
    mean = torch.tensor(ANIMA_VAE_MEAN, dtype=dtype, device=device)
    std = torch.tensor(ANIMA_VAE_STD, dtype=dtype, device=device)
    scale = [mean, 1.0 / std]

    logger.info(f"Loaded Anima VAE successfully.")
    return vae, mean, std, scale


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'qwen3_06b')
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(qwen3_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'qwen3_06b')
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith('.safetensors'):
            state_dict = load_file(qwen3_path, device='cpu')
        else:
            state_dict = torch.load(qwen3_path, map_location='cpu', weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_sd[k[len('model.'):]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 't5_old')
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, 'spiece.model'),
            tokenizer_file=os.path.join(config_dir, 'tokenizer.json'),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(
    save_path: str,
    dit_state_dict: "dict[str, torch.Tensor]",
    dtype: "torch.dtype | None" = None,
    semantic_connector=None,
):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        dtype: Optional dtype to cast to before saving
        semantic_connector: The DiT's Anima 3.8B connector (``dit.anima_v2_connector``), if any.
            A v2 connector is written back in the v1.1 bundle layout and metadata.
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            v = v.to(dtype)
        prefixed_sd['net.' + k] = v.contiguous()

    metadata = {'format': 'pt'}
    if semantic_connector is not None:
        from library import anima_38b_connector

        prefixed_sd, metadata = anima_38b_connector.prepare_bundle_for_save(prefixed_sd, semantic_connector)
    save_file(prefixed_sd, save_path, metadata=metadata)
    logger.info(f"Saved Anima model to {save_path}")


def vae_encode(tensor: torch.Tensor, vae, scale):
    """Encode tensor through WanVAE with normalization.

    Args:
        tensor: Input tensor (B, C, T, H, W) in [-1, 1] range
        vae: WanVAE_ model
        scale: [mean, 1/std] list

    Returns:
        Normalized latents
    """
    return vae.encode(tensor, scale)


def vae_decode(latents: torch.Tensor, vae, scale):
    """Decode latents through WanVAE with denormalization.

    Args:
        latents: Normalized latents
        vae: WanVAE_ model
        scale: [mean, 1/std] list

    Returns:
        Decoded tensor in [-1, 1] range
    """
    return vae.decode(latents, scale)
