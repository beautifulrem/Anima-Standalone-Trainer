# Anima 3.8B helpers shared by the training scripts.
#
# The LoRA trainer (anima_38b_train_network.py) is a NetworkTrainer subclass.
# The full fine-tuning loops (anima_train.py, anima_train_muon.py and their
# TP/SP wrapper) are monolithic, so they use Anima38BFullTraining to switch
# strategies, text encoders and DiT arguments when the checkpoint is 3.8B.

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

import torch

from library import anima_38b_connector, anima_qwen35, strategy_anima_38b
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

QWEN35_COMPUTE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def add_anima_38b_arguments(parser: argparse.ArgumentParser, qwen35_required: bool = False) -> None:
    """Arguments for Anima 3.8B (Qwen3.5 semantic connector)."""
    parser.add_argument(
        "--qwen35_path",
        type=str,
        required=qwen35_required,
        default=None,
        help="Anima 3.8B only: path to the Qwen3.5 4B text encoder (text_encoders/qwen35_4b.safetensors)",
    )
    parser.add_argument(
        "--qwen35_tokenizer_path",
        type=str,
        default=None,
        help="Qwen3.5 tokenizer directory or HF repo (default: bundled configs/qwen35_4b, then Qwen/Qwen3.5-4B)",
    )
    parser.add_argument(
        "--qwen35_max_token_length",
        type=int,
        default=1024,
        help="Maximum Qwen3.5 prompt tokens (default 1024, same as the ComfyUI node)",
    )
    parser.add_argument(
        "--qwen35_compute_dtype",
        type=str,
        default="float32",
        choices=list(QWEN35_COMPUTE_DTYPES),
        help="Activation dtype for Qwen3.5 (default float32, matching ComfyUI; bfloat16 is faster but approximate)",
    )
    parser.add_argument(
        "--semantic_connector",
        type=str,
        default=None,
        help="Standalone v1/v2 semantic connector. Not needed for Anima-3.8B-v1.1.safetensors, which bundles it. "
        "For the original Anima-3.8B.safetensors use text_encoders/Anima-3.8B-expanded_adapter.safetensors",
    )


def is_anima_38b_run(args: argparse.Namespace) -> bool:
    """True when the DiT is an Anima 3.8B checkpoint or a standalone connector is given."""
    if getattr(args, "semantic_connector", None):
        return True
    return anima_38b_connector.is_anima_38b_checkpoint(getattr(args, "dit_path", None))


def validate_anima_38b_args(args: argparse.Namespace) -> None:
    if not getattr(args, "qwen35_path", None):
        raise ValueError("Anima 3.8B: --qwen35_path (text_encoders/qwen35_4b.safetensors) is required")
    if not 1 <= int(args.qwen35_max_token_length) <= 4096:
        raise ValueError("--qwen35_max_token_length must be between 1 and 4096")
    if getattr(args, "weighted_captions", False):
        # Anima has no prompt-weight support (strategy_anima), and ComfyUI ignores weights for Qwen3.5.
        raise ValueError("Anima 3.8B: --weighted_captions is not supported (Anima does not implement prompt weights)")


def load_qwen35_text_encoder(args: argparse.Namespace) -> anima_qwen35.Qwen35HybridModel:
    logger.info("Loading Qwen3.5 4B text encoder...")
    return anima_qwen35.load_qwen35_text_encoder(
        args.qwen35_path,
        device="cpu",
        compute_dtype=QWEN35_COMPUTE_DTYPES[args.qwen35_compute_dtype],
    )


def move_text_encoders(text_encoders: Sequence[Any], device: torch.device | str, dtype: torch.dtype | None = None) -> None:
    """Move [qwen3, qwen35]; Qwen3.5 keeps its FP8 storage where the device supports it."""
    for text_encoder in text_encoders:
        if isinstance(text_encoder, anima_qwen35.Qwen35HybridModel):
            anima_qwen35.move_qwen35_text_encoder(text_encoder, device)
        elif dtype is None:
            text_encoder.to(device)
        else:
            text_encoder.to(device, dtype=dtype)


def semantic_dit_kwargs(
    semantic_outputs: Sequence[Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """DiT keyword arguments from the Qwen3.5 part of the outputs: [state_0..K-1, mask]."""
    *states, mask = semantic_outputs
    return {
        "semantic_hidden_states": [s.to(device, dtype=dtype) for s in states],
        "semantic_attention_mask": mask.to(device),
    }


def freeze_semantic_connector(dit: torch.nn.Module) -> None:
    """The connector is a frozen bridge; only the DiT (and optionally its native adapter) train."""
    connector = getattr(dit, "anima_v2_connector", None)
    if connector is not None:
        connector.requires_grad_(False)


class Anima38BFullTraining:
    """Switches a full fine-tuning loop to Anima 3.8B conditioning."""

    def __init__(self, args: argparse.Namespace) -> None:
        validate_anima_38b_args(args)
        self.args = args
        self.layer_indices = anima_38b_connector.semantic_layer_indices_for(args.dit_path, args.semantic_connector)
        logger.info(f"Anima 3.8B: Qwen3.5 semantic connector on layers {list(self.layer_indices)}")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Anima38BFullTraining | None:
        return cls(args) if is_anima_38b_run(args) else None

    def tokenize_strategy(self, qwen3_tokenizer, t5_tokenizer) -> strategy_anima_38b.Anima38BTokenizeStrategy:
        args = self.args
        return strategy_anima_38b.Anima38BTokenizeStrategy(
            qwen3_tokenizer=qwen3_tokenizer,
            t5_tokenizer=t5_tokenizer,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
            qwen35_tokenizer_path=args.qwen35_tokenizer_path,
            qwen35_max_length=args.qwen35_max_token_length,
        )

    def text_encoding_strategy(self, dropout_rate: float) -> strategy_anima_38b.Anima38BTextEncodingStrategy:
        return strategy_anima_38b.Anima38BTextEncodingStrategy(self.layer_indices, dropout_rate=dropout_rate)

    def caching_strategy(
        self, cache_to_disk: bool, batch_size: int, skip_check: bool
    ) -> strategy_anima_38b.Anima38BTextEncoderOutputsCachingStrategy:
        return strategy_anima_38b.Anima38BTextEncoderOutputsCachingStrategy(
            cache_to_disk, batch_size, skip_check, self.layer_indices, self.args.qwen35_max_token_length, is_partial=False
        )

    def load_qwen35(self) -> anima_qwen35.Qwen35HybridModel:
        return load_qwen35_text_encoder(self.args)

    def dit_load_kwargs(self) -> dict[str, Any]:
        return {"load_semantic_connector": True, "semantic_connector_path": self.args.semantic_connector}

    def prepare_dit(self, dit: torch.nn.Module) -> None:
        freeze_semantic_connector(dit)
        if getattr(dit, "num_blocks", 52) != 52:
            logger.warning(f"Anima 3.8B normally has 52 DiT blocks; this checkpoint has {dit.num_blocks}")

    def dit_kwargs(self, semantic_outputs: Sequence[Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        if len(semantic_outputs) != len(self.layer_indices) + 1:
            raise ValueError(
                f"Expected {len(self.layer_indices)} Qwen3.5 states and a mask, got {len(semantic_outputs)} tensors"
            )
        return semantic_dit_kwargs(semantic_outputs, device, dtype)
