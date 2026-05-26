# Shared network base for LyCORIS-family modules (LoHa, LoKr, etc).
# Ported from sd-scripts/networks/network_base.py with Anima knobs:
#   - te_prefixes pinned to ["lora_te1"] for Anima (matches lora_anima.py:701)
#   - type_dims / emb_dims / train_block_indices kwargs on AdditionalNetwork
#   - on_step_start no-op for train_network.py:1525 contract
#   - _is_tp_active(unet) helper for LoHa/LoKr TP/SP guard

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from library.sdxl_original_unet import InferSdxlUNet2DConditionModel
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArchConfig:
    unet_target_modules: List[str]
    te_target_modules: List[str]
    unet_prefix: str
    te_prefixes: List[str]
    default_excludes: List[str] = field(default_factory=list)
    adapter_target_modules: List[str] = field(default_factory=list)
    unet_conv_target_modules: List[str] = field(default_factory=list)


def detect_arch_config(unet, text_encoders) -> ArchConfig:
    """Detect architecture from model structure and return ArchConfig."""
    from library.sdxl_original_unet import SdxlUNet2DConditionModel

    # SDXL first
    if unet is not None and (
        issubclass(unet.__class__, SdxlUNet2DConditionModel) or issubclass(unet.__class__, InferSdxlUNet2DConditionModel)
    ):
        return ArchConfig(
            unet_target_modules=["Transformer2DModel"],
            te_target_modules=["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"],
            unet_prefix="lora_unet",
            te_prefixes=["lora_te1", "lora_te2"],
            default_excludes=[],
            unet_conv_target_modules=["ResnetBlock2D", "Downsample2D", "Upsample2D"],
        )

    # Anima: look for Block class in named_modules
    module_class_names = set()
    if unet is not None:
        for module in unet.modules():
            module_class_names.add(type(module).__name__)

    if "Block" in module_class_names:
        return ArchConfig(
            unet_target_modules=["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"],
            te_target_modules=["Qwen3Attention", "Qwen3MLP", "Qwen3SdpaAttention", "Qwen3FlashAttention2"],
            unet_prefix="lora_unet",
            te_prefixes=["lora_te1"],  # pinned to match lora_anima.LORA_PREFIX_TEXT_ENCODER
            default_excludes=[r".*(_modulation|_norm|_embedder|final_layer).*"],
            adapter_target_modules=["LLMAdapterTransformerBlock"],
        )

    raise ValueError(f"Cannot auto-detect architecture for LyCORIS. Module classes found: {sorted(module_class_names)}")


def _parse_anima_kwargs(kwargs: Dict, unet) -> Tuple[Optional[List[Optional[int]]], Optional[List[int]], Optional[List[bool]]]:
    """Parse Anima-specific kwargs from --network_args. Mirrors lora_anima.py:535-595."""
    def _opt_int(v):
        return int(v) if v is not None else None

    type_dims = [
        _opt_int(kwargs.get("self_attn_dim", None)),
        _opt_int(kwargs.get("cross_attn_dim", None)),
        _opt_int(kwargs.get("mlp_dim", None)),
        _opt_int(kwargs.get("mod_dim", None)),
        _opt_int(kwargs.get("llm_adapter_dim", None)),
    ]
    if all(d is None for d in type_dims):
        type_dims = None

    emb_dims = kwargs.get("emb_dims", None)
    if emb_dims is not None:
        emb_dims = emb_dims.strip()
        if emb_dims.startswith("[") and emb_dims.endswith("]"):
            emb_dims = emb_dims[1:-1]
        emb_dims = [int(d) for d in emb_dims.split(",")]
        assert len(emb_dims) == 3, f"invalid emb_dims: {emb_dims}, must be 3 dimensions (x_embedder, t_embedder, final_layer)"

    train_block_indices = kwargs.get("train_block_indices", None)
    if train_block_indices is not None:
        num_blocks = len(unet.blocks) if hasattr(unet, "blocks") else 999
        train_block_indices = _parse_block_selection(train_block_indices, num_blocks)

    return type_dims, emb_dims, train_block_indices


def _parse_block_selection(selection: str, total_blocks: int) -> List[bool]:
    """Parse "all" / "none" / "" / "0,2,5-7" into a bool list of length total_blocks."""
    if selection == "all":
        return [True] * total_blocks
    if selection == "none" or selection == "":
        return [False] * total_blocks
    selected = [False] * total_blocks
    for r in selection.split(","):
        if "-" in r:
            start, end = map(str.strip, r.split("-"))
            start, end = int(start), int(end)
            assert 0 <= start < total_blocks and 0 <= end < total_blocks and start <= end
            for i in range(start, end + 1):
                selected[i] = True
        else:
            index = int(r)
            assert 0 <= index < total_blocks
            selected[index] = True
    return selected


def _str_to_bool(v) -> bool:
    """Tri-state bool from --network_args strings."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def _is_tp_active(unet) -> bool:
    """Return True if the unet has any TP-sharded Linear children.

    Used by LoHa/LoKr to refuse construction under wd_parallel TP/SP.
    Scans by class name so this works even when wd_parallel isn't importable.
    """
    if unet is None:
        return False
    tp_class_names = ("ColumnParallelLinear", "RowParallelLinear")
    for module in unet.modules():
        if type(module).__name__ in tp_class_names:
            return True
    return False


def _parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, Union[int, float]]:
    """Parse a string of key-value pairs separated by commas."""
    pairs = {}
    for pair in kv_pair_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning(f"Invalid format: {pair}, expected 'key=value'")
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            pairs[key] = int(value) if is_int else float(value)
        except ValueError:
            logger.warning(f"Invalid value for {key}: {value}")
    return pairs


class AdditionalNetwork(torch.nn.Module):
    """Generic Additional network supporting LoHa, LoKr, and similar module types.

    Constructed with a module_class parameter to inject the specific module type.
    Generalized from lora_anima.py:LoRANetwork, supports the same Anima knobs
    (type_dims / emb_dims / train_block_indices / train_llm_adapter) so LoHa
    and LoKr configs port cleanly between LoRA and LoHa/LoKr.
    """

    def __init__(
        self,
        text_encoders: list,
        unet,
        arch_config: ArchConfig,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[torch.nn.Module] = None,
        module_kwargs: Optional[Dict] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        train_llm_adapter: bool = False,
        type_dims: Optional[List[Optional[int]]] = None,
        emb_dims: Optional[List[int]] = None,
        train_block_indices: Optional[List[bool]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert module_class is not None, "module_class must be specified"

        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.train_llm_adapter = train_llm_adapter
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs
        self.arch_config = arch_config
        self.type_dims = type_dims
        self.emb_dims = emb_dims
        self.train_block_indices = train_block_indices

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if module_kwargs is None:
            module_kwargs = {}

        if modules_dim is not None:
            logger.info(f"create {module_class.__name__} network from weights")
        else:
            logger.info(f"create {module_class.__name__} network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(exclude_patterns)
        include_re_patterns = str_to_re_patterns(include_patterns)

        def create_modules(
            prefix: str,
            root_module: torch.nn.Module,
            target_replace_modules: Optional[List[str]],
            default_dim: Optional[int] = None,
            is_unet: bool = False,
            filter: Optional[str] = None,
            include_conv2d_if_filter: bool = False,
        ) -> Tuple[List[torch.nn.Module], List[str]]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_modules is None or module.__class__.__name__ in target_replace_modules:
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # `filter` is a positive selector for the emb_dims second-pass;
                            # if set and not matched, skip immediately. A matched filter
                            # is also an explicit user request → bypass default_excludes.
                            force_incl_conv2d = False
                            explicit_include = False
                            if filter is not None:
                                if filter not in lora_name:
                                    continue
                                force_incl_conv2d = include_conv2d_if_filter
                                explicit_include = True

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                                    explicit_include = True  # loading a saved module
                            else:
                                if self.reg_dims is not None:
                                    for reg, d in self.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = self.alpha
                                            logger.info(f"Module {original_name} matched with regex '{reg}' -> dim: {dim}")
                                            explicit_include = True  # reg_dims is explicit
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1 or force_incl_conv2d:
                                        dim = default_dim if default_dim is not None else self.lora_dim
                                        alpha_val = self.alpha
                                    elif is_conv2d and self.conv_lora_dim is not None:
                                        dim = self.conv_lora_dim
                                        alpha_val = self.conv_alpha

                                # Anima per-type dim dispatch (mirrors lora_anima.py:797-825).
                                # An explicit non-None type_dims entry overrides default_excludes.
                                if is_unet and self.type_dims is not None and dim is not None:
                                    identifier_order = [
                                        (4, ("llm_adapter",)),
                                        (3, ("adaln_modulation",)),
                                        (0, ("self_attn",)),
                                        (1, ("cross_attn",)),
                                        (2, ("mlp",)),
                                    ]
                                    for idx, ids in identifier_order:
                                        d = self.type_dims[idx]
                                        if d is not None and all(id_str in lora_name for id_str in ids):
                                            dim = d
                                            explicit_include = True
                                            break

                                # Anima block-index gating (mirrors lora_anima.py:814-825)
                                if is_unet and dim and self.train_block_indices is not None and "blocks_" in lora_name:
                                    parts = lora_name.split("_")
                                    for pi, part in enumerate(parts):
                                        if part == "blocks" and pi + 1 < len(parts):
                                            try:
                                                block_index = int(parts[pi + 1])
                                                if not self.train_block_indices[block_index]:
                                                    dim = 0
                                            except (ValueError, IndexError):
                                                pass
                                            break

                            # Apply exclude/include patterns AFTER dim determination so
                            # that explicit user knobs (filter, type_dims, reg_dims,
                            # modules_dim) can override default_excludes. User-supplied
                            # include_patterns also override exclude_patterns.
                            if not explicit_include:
                                excluded = any(p.fullmatch(original_name) for p in exclude_re_patterns)
                                included = any(p.fullmatch(original_name) for p in include_re_patterns)
                                if excluded and not included:
                                    if verbose:
                                        logger.info(f"exclude: {original_name}")
                                    continue

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha_val,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                **module_kwargs,
                            )
                            lora.original_name = original_name
                            loras.append(lora)

                    if target_replace_modules is None:
                        break
            return loras, skipped

        # Text encoders
        self.text_encoder_loras: List[torch.nn.Module] = []
        skipped_te = []
        if text_encoders is not None:
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                te_prefix = arch_config.te_prefixes[i] if i < len(arch_config.te_prefixes) else arch_config.te_prefixes[0]
                logger.info(f"create {module_class.__name__} for Text Encoder {i+1} (prefix={te_prefix}):")
                te_loras, te_skipped = create_modules(te_prefix, text_encoder, arch_config.te_target_modules)
                logger.info(f"create {module_class.__name__} for Text Encoder {i+1}: {len(te_loras)} modules.")
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # UNet/DiT
        target_modules = list(arch_config.unet_target_modules)
        if modules_dim is not None or conv_lora_dim is not None:
            target_modules.extend(arch_config.unet_conv_target_modules)
        if train_llm_adapter and arch_config.adapter_target_modules:
            target_modules.extend(arch_config.adapter_target_modules)

        self.unet_loras: List[torch.nn.Module]
        self.unet_loras, skipped_un = create_modules(
            arch_config.unet_prefix, unet, target_modules, is_unet=True,
        )

        # Anima emb_dims pass: x_embedder / t_embedder / final_layer
        if self.emb_dims:
            for filter_name, in_dim in zip(["x_embedder", "t_embedder", "final_layer"], self.emb_dims):
                if not in_dim:
                    continue
                emb_loras, _ = create_modules(
                    arch_config.unet_prefix, unet, None,
                    default_dim=in_dim,
                    is_unet=True,
                    filter=filter_name,
                    include_conv2d_if_filter=(filter_name == "x_embedder"),
                )
                self.unet_loras.extend(emb_loras)

        logger.info(f"create {module_class.__name__} for UNet/DiT: {len(self.unet_loras)} modules.")

        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable modules for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable modules for UNet/DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        te_prefixes = self.arch_config.te_prefixes
        unet_prefix = self.arch_config.unet_prefix

        for key in weights_sd.keys():
            if any(key.startswith(p) for p in te_prefixes):
                apply_text_encoder = True
            elif key.startswith(unet_prefix):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable modules for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable modules for UNet/DiT")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            reg_groups = {}
            reg_lrs_list = list(self.reg_lrs.items()) if self.reg_lrs is not None else []

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}")
                        break

                for name, param in lora.named_parameters():
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {"lora": {}, "plus": {}, "lr": reg_lr}
                        if loraplus_ratio is not None and self._is_plus_param(name):
                            reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                        else:
                            reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                        continue

                    if loraplus_ratio is not None and self._is_plus_param(name):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = reg_lr * loraplus_ratio if loraplus_ratio is not None else reg_lr
                    else:
                        param_data["lr"] = reg_lr
                    if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(desc + (" plus" if key == "plus" else ""))

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr
                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            for te_idx, te_prefix in enumerate(self.arch_config.te_prefixes):
                te_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(te_prefix)]
                if len(te_loras) > 0:
                    te_lr = text_encoder_lr[te_idx] if te_idx < len(text_encoder_lr) else text_encoder_lr[0]
                    logger.info(f"Text Encoder {te_idx+1} ({te_prefix}): {len(te_loras)} modules, LR {te_lr}")
                    params, descriptions = assemble_params(te_loras, te_lr, loraplus_ratio)
                    all_params.extend(params)
                    lr_descriptions.extend([f"textencoder {te_idx+1}" + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def _is_plus_param(self, name: str) -> bool:
        return "lora_up" in name or "hada_w2_a" in name or "lokr_w1" in name

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def on_step_start(self, text_encoder, unet):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        loras = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False
