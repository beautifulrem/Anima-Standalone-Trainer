# Anima 3.8B LoRA training script.
#
# Anima 3.8B (https://huggingface.co/lylogummy/Anima-3.8B) is a 52-block Anima
# DiT whose text conditioning combines the native Qwen3 0.6B path with a Qwen3.5
# 4B "semantic connector". This trainer reproduces the ComfyUI inference path
# (GumGum10/comfyui-anima-3-8B) during training:
#
#   Qwen3 0.6B  ──► native LLM adapter ─┐
#   Qwen3.5 4B layers 7/15/23/31 ───────┴─► semantic connector (v2: per timestep) ──► DiT
#
# The connector and both text encoders stay frozen; only the network (LoRA,
# LoHa, ...) on the DiT is trained, optionally including the native adapter
# blocks the connector reuses (network_args train_llm_adapter=True).

import argparse
import json
import logging

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from anima_train_network import AnimaNetworkTrainer  # noqa: E402
from anima_train_network import setup_parser as setup_anima_parser  # noqa: E402
from library import anima_qwen35, strategy_anima_38b, train_util  # noqa: E402
from library.utils import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class Anima38BNetworkTrainer(AnimaNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.qwen35_text_encoder = None
        self.semantic_layer_indices: tuple[int, ...] | None = None
        self.semantic_connector_architecture: str | None = None

    # ------------------------------------------------------------------ args
    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        if not args.network_train_unet_only:
            raise ValueError(
                "Anima 3.8B: training the text encoders is not supported; add --network_train_unet_only "
                "(use network_args train_llm_adapter=True to also train the native LLM adapter)."
            )
        if args.weighted_captions:
            raise ValueError("Anima 3.8B: --weighted_captions is not supported")
        if not 1 <= args.qwen35_max_token_length <= 4096:
            raise ValueError("--qwen35_max_token_length must be between 1 and 4096")
        if not args.cache_text_encoder_outputs:
            logger.warning(
                "Anima 3.8B without --cache_text_encoder_outputs keeps the Qwen3.5 4B encoder on the GPU "
                "(~4.5 GB FP8) and runs it every step. Caching is strongly recommended."
            )

    def get_dit_load_kwargs(self, args) -> dict:
        return {"load_semantic_connector": True, "semantic_connector_path": args.semantic_connector}

    # ------------------------------------------------------------ models
    def load_target_model(self, args, weight_dtype, accelerator):
        model_type, text_encoders, vae, dit = super().load_target_model(args, weight_dtype, accelerator)

        connector = getattr(dit, "anima_v2_connector", None)
        if connector is None:  # load_anima_dit raises earlier; keep a clear error just in case
            raise RuntimeError("The Anima 3.8B semantic connector was not attached to the DiT")
        self.semantic_layer_indices = tuple(connector.layer_indices)
        self.semantic_connector_architecture = connector.architecture
        if dit.num_blocks != 52:
            logger.warning(f"Anima 3.8B normally has 52 DiT blocks; this checkpoint has {dit.num_blocks}")

        logger.info("Loading Qwen3.5 4B text encoder...")
        self.qwen35_text_encoder = anima_qwen35.load_qwen35_text_encoder(
            args.qwen35_path,
            device="cpu",
            compute_dtype=_DTYPES[args.qwen35_compute_dtype],
        )
        return model_type, [text_encoders[0], self.qwen35_text_encoder], vae, dit

    def get_tokenize_strategy(self, args):
        self.tokenize_strategy = strategy_anima_38b.Anima38BTokenizeStrategy(
            qwen3_path=args.qwen3_path,
            t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
            qwen35_tokenizer_path=args.qwen35_tokenizer_path,
            qwen35_max_length=args.qwen35_max_token_length,
        )
        self.qwen3_tokenizer = self.tokenize_strategy.qwen3_tokenizer
        self.t5_tokenizer = self.tokenize_strategy.t5_tokenizer
        return self.tokenize_strategy

    def _require_layer_indices(self) -> tuple[int, ...]:
        if self.semantic_layer_indices is None:
            raise RuntimeError("Semantic connector layer indices are unknown before the DiT is loaded")
        return self.semantic_layer_indices

    def get_text_encoding_strategy(self, args):
        self.text_encoding_strategy = strategy_anima_38b.Anima38BTextEncodingStrategy(
            self._require_layer_indices(), dropout_rate=getattr(args, "caption_dropout_rate", 0.0)
        )
        return self.text_encoding_strategy

    def get_text_encoder_outputs_caching_strategy(self, args):
        if not args.cache_text_encoder_outputs:
            return None
        return strategy_anima_38b.Anima38BTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            self._require_layer_indices(),
            args.qwen35_max_token_length,
            is_partial=False,
        )

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False] * len(text_encoders)

    def cast_text_encoder(self, args):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Keep Qwen3.5 in its FP8 storage dtype; it upcasts per layer (like ComfyUI).
        return False

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        pass

    def _move_text_encoders(self, text_encoders, device, weight_dtype):
        text_encoders[0].to(device, dtype=weight_dtype)
        anima_qwen35.move_qwen35_text_encoder(text_encoders[1], device)  # FP8 storage is kept where supported

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if not args.cache_text_encoder_outputs:
            self._move_text_encoders(text_encoders, accelerator.device, weight_dtype)
            return

        org_vae_device = next(vae.parameters()).device
        org_unet_device = unet.device
        if not args.lowram:
            logger.info("move vae and unet to cpu to save memory")
            vae.to("cpu")
            unet.to("cpu")
            clean_memory_on_device(accelerator.device)

        logger.info("move text encoders to gpu")
        self._move_text_encoders(text_encoders, accelerator.device, weight_dtype)
        tokenize_strategy = self.tokenize_strategy
        text_encoding_strategy = self.text_encoding_strategy
        if not isinstance(tokenize_strategy, strategy_anima_38b.Anima38BTokenizeStrategy) or not isinstance(
            text_encoding_strategy, strategy_anima_38b.Anima38BTextEncodingStrategy
        ):
            raise RuntimeError("Anima 3.8B strategies are not initialized")
        with accelerator.autocast():
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            if args.sample_prompts is not None and self.sample_prompts_te_outputs is None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
                outputs = {}
                with torch.no_grad():
                    for prompt_dict in train_util.load_prompts(args.sample_prompts):
                        for prompt in (prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")):
                            if prompt not in outputs:
                                tokens = tokenize_strategy.tokenize(prompt)
                                encoded = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens, enable_dropout=False
                                )
                                outputs[prompt] = [t.cpu() for t in encoded]
                self.sample_prompts_te_outputs = outputs

            if getattr(args, "caption_dropout_rate", 0.0) > 0.0:
                text_encoding_strategy.cache_uncond_embeddings(tokenize_strategy, text_encoders)

        accelerator.wait_for_everyone()
        logger.info("move text encoders back to cpu")
        for text_encoder in text_encoders:
            text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)
        if not args.lowram:
            logger.info("move vae and unet back to original device")
            vae.to(org_vae_device)
            unet.to(org_unet_device)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # The frozen connector only needs autograd when LoRA patches the native
        # adapter blocks it reuses; otherwise skip storing its activations.
        unet.semantic_connector_grad = bool(getattr(network, "train_llm_adapter", False))

    # ------------------------------------------------------------ training
    def get_extra_dit_kwargs(self, text_encoder_conds, accelerator, weight_dtype) -> dict:
        _, states, mask = strategy_anima_38b.split_text_encoder_outputs(
            list(text_encoder_conds), len(self._require_layer_indices())
        )
        return {
            "semantic_hidden_states": [s.to(accelerator.device, dtype=weight_dtype) for s in states],
            "semantic_attention_mask": mask.to(accelerator.device),
        }

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        from library import anima_train_utils

        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        if te is not None and any(t is None for t in te):
            te = None
        anima_train_utils.sample_images(
            accelerator, args, epoch, global_step, unet, vae, self.vae_scale,
            te, self.tokenize_strategy, self.text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_anima_variant"] = "anima-3.8b"
        metadata["ss_semantic_connector_architecture"] = self.semantic_connector_architecture or ""
        metadata["ss_semantic_connector_layers"] = json.dumps(list(self.semantic_layer_indices or ()))
        metadata["ss_qwen35_max_token_length"] = str(args.qwen35_max_token_length)


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_anima_parser()
    parser.add_argument(
        "--qwen35_path",
        type=str,
        required=True,
        help="Path to the Anima 3.8B Qwen3.5 4B text encoder (text_encoders/qwen35_4b.safetensors)",
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
        choices=list(_DTYPES),
        help="Activation dtype for Qwen3.5 (default float32, matching ComfyUI; bfloat16 is faster but approximate)",
    )
    parser.add_argument(
        "--semantic_connector",
        type=str,
        default=None,
        help="Standalone v1/v2 semantic connector. Not needed for Anima-3.8B-v1.1.safetensors, which bundles it. "
        "For the original Anima-3.8B.safetensors use text_encoders/Anima-3.8B-expanded_adapter.safetensors",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = Anima38BNetworkTrainer()
    trainer.train(args)
