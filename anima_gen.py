
import argparse
import os
import sys
import torch
import threading
from accelerate import Accelerator
from library import anima_utils, strategy_anima, anima_train_utils, train_util
from library.utils import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

# Global models cache for server mode
CACHED_MODELS = None
APP_ACCELERATOR = None
CURRENT_LORA = {"path": None, "mul": 1.0}

def manage_lora(accelerator, models, target_path, target_mul):
    global CURRENT_LORA
    
    # Check if change is needed
    if CURRENT_LORA["path"] == target_path and abs(CURRENT_LORA["mul"] - target_mul) < 1e-6:
        return # No change
        
    import networks.lora_anima
    
    # Unmerge current if exists
    if CURRENT_LORA["path"]:
        logger.info(f"Unmerging previous LoRA: {CURRENT_LORA['path']}")
        try:
            # Create network with negative multiplier to subtract
            net, sd = networks.lora_anima.create_network_from_weights(
                multiplier=-CURRENT_LORA["mul"], 
                file=CURRENT_LORA["path"],
                ae=models["vae"],
                text_encoders=[models["qwen3"]],
                unet=models["dit"],
                for_inference=True
            )
            net.merge_to([models["qwen3"]], models["dit"], sd, models["dtype"], accelerator.device)
            del net, sd, 
            torch.cuda.empty_cache()
            CURRENT_LORA["path"] = None
        except Exception as e:
            logger.error(f"Failed to unmerge LoRA: {e}")
            
    # Merge new if specified
    if target_path:
        logger.info(f"Merging new LoRA: {target_path} (x{target_mul})")
        try:
            net, sd = networks.lora_anima.create_network_from_weights(
                multiplier=target_mul,
                file=target_path,
                ae=models["vae"],
                text_encoders=[models["qwen3"]],
                unet=models["dit"],
                for_inference=True
            )
            net.merge_to([models["qwen3"]], models["dit"], sd, models["dtype"], accelerator.device)
            del net, sd
            torch.cuda.empty_cache()
            CURRENT_LORA["path"] = target_path
            CURRENT_LORA["mul"] = target_mul
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")

def load_models(args, accelerator):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Qwen3
    logger.info(f"Loading Qwen3 from {args.qwen3_path}")
    qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3_path, dtype=weight_dtype, device="cpu"
    )
    qwen3_text_encoder.eval()
    
    # Load DiT
    logger.info(f"Loading DiT from {args.dit_path}")
    transformer_dtype = None
    if hasattr(args, 'transformer_dtype') and args.transformer_dtype is not None:
         dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
         transformer_dtype = dtype_map.get(args.transformer_dtype, None)

    dit = anima_utils.load_anima_dit(
        args.dit_path,
        dtype=weight_dtype,
        device="cpu",
        transformer_dtype=transformer_dtype,
    )
    dit.eval()
    
    # Load VAE
    logger.info(f"Loading VAE from {args.vae_path}")
    vae, _, _, vae_scale = anima_utils.load_anima_vae(
        args.vae_path, dtype=torch.bfloat16, device="cpu"
    )
    vae.eval()

    # Load LoRA if specified (Merge into base models)
    if args.network_weights:
        # Initial LoRA load 
        logger.info(f"Loading LoRA weights from {args.network_weights}")
        import networks.lora_anima
        network, weights_sd = networks.lora_anima.create_network_from_weights(
            multiplier=args.network_mul,
            file=args.network_weights,
            ae=vae,
            text_encoders=[qwen3_text_encoder],
            unet=dit,
            for_inference=True
        )
        
        network.merge_to([qwen3_text_encoder], dit, weights_sd, weight_dtype, "cpu")
        logger.info(f"LoRA merged with multiplier {args.network_mul}")
        
        global CURRENT_LORA
        CURRENT_LORA["path"] = args.network_weights
        CURRENT_LORA["mul"] = args.network_mul

    # Move to GPU
    dit.to(accelerator.device)
    qwen3_text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    vae_scale = [t.to(accelerator.device) for t in vae_scale]

    return {
        "dit": dit,
        "qwen3": qwen3_text_encoder,
        "vae": vae,
        "vae_scale": vae_scale,
        "tokenize_strategy": strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3_path,
            qwen3_max_length=args.qwen3_max_token_length,
        ),
        "text_encoding_strategy": strategy_anima.AnimaTextEncodingStrategy(),
        "dtype": weight_dtype
    }

def perform_generation(args, models, accelerator):
    logger.info("Starting generation...")
    
    # Update args for specific request if needed (e.g. prompt file might have changed content)
    # The models are already loaded, so we just pass them
    
    anima_train_utils.sample_images(
        accelerator, args, 0, 0, 
        models["dit"], 
        models["vae"], 
        models["vae_scale"],
        models["qwen3"], 
        models["tokenize_strategy"], 
        models["text_encoding_strategy"]
    )
    logger.info("Generation finished.")

def run_server(args, accelerator):
    global CACHED_MODELS
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("Flask is required for server mode. Please install it.")
        sys.exit(1)

    app = Flask(__name__)
    
    # Load models once on startup
    CACHED_MODELS = load_models(args, accelerator)
    logger.info("Models loaded. Server ready.")

    @app.route('/generate', methods=['POST'])
    def handle_generate():
        try:
            data = request.json
            if 'sample_prompts' in data:
                args.sample_prompts = data['sample_prompts']
            
            if 'flow_shift' in data:
                args.flow_shift = float(data['flow_shift'])
            
            # Handle Dynamic LoRA Switching
            req_weights = data.get('network_weights')
            req_mul = float(data.get('network_mul', 1.0))
            
            manage_lora(accelerator, CACHED_MODELS, req_weights, req_mul)

            perform_generation(args, CACHED_MODELS, accelerator)
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/ping', methods=['GET'])
    def ping():
        return jsonify({"status": "ready"})

    @app.route('/stop', methods=['POST'])
    def stop():
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()
        else:
            # Fallback for other servers or older werkzeug
            os._exit(0) 
        return jsonify({"success": True})

    app.run(host='0.0.0.0', port=args.server_port, debug=False, use_reloader=False)

def main():
    parser = argparse.ArgumentParser()
    # Add standard arguments
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    
    # Add minimal others needed
    parser.add_argument("--sample_prompts", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=None)
    
    # Fake args usually set by read_config_from_file or other places if missing
    parser.add_argument("--caption_dropout_rate", type=float, default=0.0)
    parser.add_argument("--sample_at_first", action="store_true", default=True)
    parser.add_argument("--sample_every_n_steps", type=int, default=None)
    parser.add_argument("--sample_every_n_epochs", type=int, default=None)
    parser.add_argument("--network_weights", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--network_mul", type=float, default=1.0, help="LoRA multiplier")
    
    parser.add_argument("--server_port", type=int, default=None, help="Run in server mode on this port")

    args = parser.parse_args()
    
    # Force sample_at_first to True for this script
    args.sample_at_first = True

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    if args.server_port:
        logger.info(f"Starting Anima Generation Server on port {args.server_port}")
        run_server(args, accelerator)
    else:
        # Standard one-shot mode
        models = load_models(args, accelerator)
        perform_generation(args, models, accelerator)

if __name__ == "__main__":
    main()
