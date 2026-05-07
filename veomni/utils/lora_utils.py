# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open

from ..distributed.parallel_state import get_parallel_state


def freeze_parameters(model: nn.Module):
    # Freeze parameters
    model.requires_grad_(False)
    model.eval()
    model.train()


def add_lora_to_model(
    model: nn.Module,
    lora_rank=4,
    lora_alpha=4,
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
    init_lora_weights="kaiming",
    pretrained_lora_path=None,
    state_dict_converter=None,
    lora_target_modules_support=None,
):
    model.lora_alpha = lora_alpha
    if init_lora_weights == "kaiming":
        init_lora_weights = True

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )

    for lora_target_module in lora_config.target_modules:
        if lora_target_module not in lora_target_modules_support:
            raise ValueError(f"lora_target_module {lora_target_module} not in lora_target_modules_support")

    model = inject_adapter_in_model(lora_config, model)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype=torch.float32)

    # Lora pretrained lora weights
    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
        )


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def patch_fsdp2_lora_weight_loading(model: torch.nn.Module):
    """Patch FSDP2 weight-loading helpers for PEFT-wrapped models.

    Patches ``_convert_weight_key`` to remap base-checkpoint keys into the
    PEFT namespace (``base_model.model.`` prefix + ``base_layer.`` splice),
    and ``_init_parameter`` to apply PEFT-standard LoRA initialisation
    (kaiming for lora_A, zeros for lora_B) for params missing from the base
    checkpoint.
    """
    parameter_name = next(model.named_parameters())[0]
    if not parameter_name.startswith("base_model."):
        return

    overrides: Dict[str, str] = {}
    for fqn, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        inner = fqn[len("base_model.model.") :] if fqn.startswith("base_model.model.") else fqn
        inner_dot = inner + ("." if inner else "")
        wrap_dot = fqn + ("." if fqn else "") + "base_layer."
        for pname, _ in module.base_layer.named_parameters():
            overrides[inner_dot + pname] = wrap_dot + pname
        for bname, _ in module.base_layer.named_buffers():
            overrides[inner_dot + bname] = wrap_dot + bname

    from veomni.models import module_utils

    orig_convert = module_utils._convert_weight_key
    orig_init = module_utils._init_parameter

    def patched_convert_weight_key(key, m, _orig=orig_convert, _model=model, _overrides=overrides):
        key = _orig(key, m)
        if m is _model:
            key = _overrides.get(key, "base_model.model." + key)
        return key

    _lora_layers_initialized: set = set()

    def patched_init_parameter(module, name, _orig=orig_init, _initialized=_lora_layers_initialized, _model=model):
        if module is not _model or not any(piece.startswith("lora_") for piece in name.split(".")):
            _orig(module, name)
            return
        pieces = name.split(".")
        lora_layer = module
        lora_layer_fqn_parts = []
        for piece in pieces:
            if piece.startswith("lora_"):
                break
            lora_layer_fqn_parts.append(piece)
            lora_layer = getattr(lora_layer, piece)
        lora_layer_fqn = ".".join(lora_layer_fqn_parts)
        if lora_layer_fqn not in _initialized and hasattr(lora_layer, "reset_lora_parameters"):
            _initialized.add(lora_layer_fqn)
            for adapter in getattr(lora_layer, "lora_A", {}).keys():
                lora_layer.reset_lora_parameters(adapter, init_lora_weights=True)

    module_utils._convert_weight_key = patched_convert_weight_key
    module_utils._init_parameter = patched_init_parameter


def patch_fsdp1_lora_weight_loading(model: torch.nn.Module):
    def patch_parallel_load_safetensors(weights_path, func, model: torch.nn.Module, **kwargs):
        import math

        import torch.distributed as dist

        shard_states = func(weights_path, **kwargs)
        parameter_name = next(model.named_parameters())[0]
        if parameter_name.startswith("base_model."):  # using lora peft will add prefix "base_model"
            shard_states = {"base_model.model." + k: v for k, v in shard_states.items()}
        for fqn, module in model.named_modules():
            fqn_dot = fqn + ("." if fqn else "")
            if hasattr(module, "base_layer"):  # using lora peft will insert "base_layer"
                for pname, _ in module.base_layer.named_parameters():
                    old_name = fqn_dot + pname
                    if old_name in shard_states:
                        wrap_name = fqn_dot + "base_layer." + pname
                        shard_states[wrap_name] = shard_states.pop(old_name)

            # Pre-fill lora_A (kaiming) and lora_B (zeros) for LoRA layers whose
            # weights are not already present in shard_states (i.e. scratch training).
            # This mirrors PEFT's standard initialisation and avoids the fallback
            # random init inside parallel_init_fsdp_fn.
            #
            # IMPORTANT: parallel_load_safetensors uses a sharded-ownership scheme
            # where only the owning rank stores the actual tensor and all others store
            # the owner's rank index. We assign rank 0 as owner for all LoRA init
            # tensors so that create_and_sync_state uses a consistent broadcast src.
            for adapter_name, lora_a_linear in getattr(module, "lora_A", {}).items():
                lora_b_linear = module.lora_B[adapter_name]
                a_key = fqn_dot + f"lora_A.{adapter_name}.weight"
                b_key = fqn_dot + f"lora_B.{adapter_name}.weight"
                if a_key not in shard_states:
                    if dist.get_rank() == 0:
                        a_w = lora_a_linear.weight
                        t = torch.empty(a_w.shape, dtype=a_w.dtype if not a_w.is_meta else torch.float32)
                        nn.init.kaiming_uniform_(t, a=math.sqrt(5))
                        shard_states[a_key] = t
                    else:
                        shard_states[a_key] = 0
                if b_key not in shard_states:
                    if dist.get_rank() == 0:
                        b_w = lora_b_linear.weight
                        shard_states[b_key] = torch.zeros(
                            b_w.shape, dtype=b_w.dtype if not b_w.is_meta else torch.float32
                        )
                    else:
                        shard_states[b_key] = 0
        return shard_states

    from functools import partial

    from veomni.distributed import torch_parallelize

    torch_parallelize.parallel_load_safetensors = partial(
        patch_parallel_load_safetensors,
        func=torch_parallelize.parallel_load_safetensors,
        model=model,
    )


def patch_fsdp_lora_weight_loading(model: nn.Module):
    """Patch weight-loading helpers so that PEFT-wrapped models load correctly.

    Routes to the FSDP2-specific patch (``patch_load_model_weights``) or the
    FSDP1/legacy patch (``patch_parallel_load_safetensors``) based on the
    current data-parallel mode.
    """
    if get_parallel_state().dp_mode == "fsdp2":
        patch_fsdp2_lora_weight_loading(model)
    elif get_parallel_state().dp_mode == "fsdp1":
        patch_fsdp1_lora_weight_loading(model)
