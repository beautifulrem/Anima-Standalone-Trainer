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
# The connector and Qwen3.5 stay frozen. The network (LoRA, LoHa, ...) trains
# the DiT, optionally the native adapter blocks the connector reuses
# (network_args train_llm_adapter=True) and, like regular Anima, the Qwen3 0.6B
# text encoder (without --network_train_unet_only; needs uncached TE outputs).

import argparse
import json
import logging

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from anima_train_network import AnimaNetworkTrainer  # noqa: E402
from anima_train_network import setup_parser as setup_anima_parser  # noqa: E402
from library import anima_38b_train_utils, strategy_anima_38b, train_util  # noqa: E402
from library.utils import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


class Anima38BNetworkTrainer(AnimaNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.qwen35_text_encoder = None
        self.semantic_layer_indices: tuple[int, ...] | None = None
        self.semantic_connector_architecture: str | None = None

    # ------------------------------------------------------------------ args
    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        anima_38b_train_utils.validate_anima_38b_args(args)
        if not args.network_train_unet_only:
            logger.info("Anima 3.8B: training a network on the Qwen3 0.6B text encoder too (Qwen3.5 stays frozen)")
        elif not args.cache_text_encoder_outputs:
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
        self.qwen35_text_encoder = anima_38b_train_utils.load_qwen35_text_encoder(args)
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
        # Only Qwen3 0.6B can be trained; Qwen3.5 is always frozen.
        return [self.is_train_text_encoder(args)] + [False] * (len(text_encoders) - 1)

    def cast_text_encoder(self, args):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Keep Qwen3.5 in its FP8 storage dtype; it upcasts per layer (like ComfyUI).
        return False

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if index == 0:
            super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0:
            super().prepare_text_encoder_fp8(index, text_encoder, te_weight_dtype, weight_dtype)

    def _move_text_encoders(self, text_encoders, device, weight_dtype):
        anima_38b_train_utils.move_text_encoders(text_encoders, device, weight_dtype)

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
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        # The frozen connector needs autograd when gradients must pass through it:
        # to the native adapter blocks it reuses (train_llm_adapter) or back to the
        # Qwen3 text encoder (text-encoder LoRA). Otherwise skip storing its
        # activations. Networks without the flag (e.g. LyCORIS) may patch the adapter.
        trains_adapter = bool(getattr(network, "train_llm_adapter", True))
        unet.semantic_connector_grad = trains_adapter or self.is_train_text_encoder(args)

    # ------------------------------------------------------------ training
    def get_extra_dit_kwargs(self, text_encoder_conds, accelerator, weight_dtype) -> dict:
        _, states, mask = strategy_anima_38b.split_text_encoder_outputs(
            list(text_encoder_conds), len(self._require_layer_indices())
        )
        return anima_38b_train_utils.semantic_dit_kwargs([*states, mask], accelerator.device, weight_dtype)

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
    anima_38b_train_utils.add_anima_38b_arguments(parser, qwen35_required=True)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = Anima38BNetworkTrainer()
    trainer.train(args)
