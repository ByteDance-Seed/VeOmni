# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native dense MXFP8 preparation and exact master-weight state I/O."""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from fnmatch import fnmatchcase
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from veomni.arguments import MXFP8Config


_PROJECTION = re.compile(r"model\.layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj)")


def resolve_mxfp8_config(model, override: MXFP8Config | None) -> MXFP8Config | None:
    """An explicit recipe wins; an unset training config preserves HF metadata."""
    if override is not None:
        if not isinstance(override, MXFP8Config):
            raise TypeError("mixed_precision.mxfp8 must be an MXFP8Config or None.")
        return override
    saved = getattr(getattr(model, "config", None), "veomni_mxfp8", None)
    if saved is None:
        return None
    if not isinstance(saved, dict):
        raise ValueError("config.veomni_mxfp8 must be an MXFP8 recipe dictionary.")
    return MXFP8Config(**saved)


def _select_linears(model, config):
    modules = dict(model.named_modules())
    references = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    selected = {}
    for pattern in config.target_modules:
        matches = [name for name in modules if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"MXFP8 target pattern {pattern!r} matched no modules.")
        for name in matches:
            module = modules[name]
            if type(module) is not nn.Linear:
                raise ValueError(f"MXFP8 target {name!r} must be a plain nn.Linear.")
            if references[id(module.weight)] != 1:
                raise ValueError(f"MXFP8 target {name!r} has a shared weight.")
            if type(module.weight) is not nn.Parameter:
                raise ValueError(f"MXFP8 target {name!r} must be converted before sharding or quantization.")
            if module.in_features % 32 or module.out_features % 32:
                raise ValueError(f"MXFP8 target {name!r} dimensions must be divisible by 32.")
            if any(
                getattr(module, attr)
                for attr in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                    "_state_dict_hooks",
                    "_state_dict_pre_hooks",
                    "_load_state_dict_pre_hooks",
                    "_load_state_dict_post_hooks",
                )
            ):
                raise ValueError(f"MXFP8 conversion cannot discard hooks on {name!r}.")
            selected[name] = module
    return selected


def validate_mxfp8_training(
    model,
    mixed_precision,
    parallel_state,
    *,
    compile_enabled=False,
    offload=False,
    activation_offload=False,
    is_peft_model=False,
    low_precision_reduce_scatter_comm=False,
    expert_weight_fp8_all_gather=False,
):
    """Validate before any model mutation, quantization or FSDP wrapping."""
    config = resolve_mxfp8_config(model, mixed_precision.mxfp8)
    if config is None or not config.enable:
        return config
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        raise ValueError("MXFP8 currently supports standalone dense Qwen3 only.")
    if parallel_state.dp_mode != "fsdp2" or any(
        getattr(parallel_state, name, False)
        for name in (
            "dp_replicate_enabled",
            "sp_enabled",
            "cp_enabled",
            "tp_enabled",
            "pp_enabled",
            "any_extra_parallel_enabled",
            "async_enabled",
        )
    ):
        raise ValueError("MXFP8 requires FSDP2 without HSDP, SP/CP, TP/PP, ExtraParallel or asynchronous attention.")
    if (
        not mixed_precision.enable
        or mixed_precision.param_dtype != "bfloat16"
        or mixed_precision.reduce_dtype != "float32"
    ):
        raise ValueError("MXFP8 requires enabled BF16 parameter / FP32 reduction mixed precision.")
    if compile_enabled or offload or activation_offload or is_peft_model or hasattr(model, "peft_config"):
        raise ValueError("MXFP8 does not support compile, offload or LoRA.")
    if low_precision_reduce_scatter_comm or expert_weight_fp8_all_gather:
        raise ValueError("MXFP8 cannot be combined with other low-precision communication options.")
    for module in model.modules():
        module_config = getattr(module, "config", None)
        ops = getattr(module_config, "ops_implementation", None) or {}
        qat = ops.get("qat_implementation", "none") if isinstance(ops, Mapping) else ops.qat_implementation
        if qat not in (None, "none") or getattr(module, "qat_implementation", "none") not in (None, "none"):
            raise ValueError("MXFP8 cannot be combined with QAT.")
    selected = _select_linears(model, config)
    for name, module in selected.items():
        if not _PROJECTION.fullmatch(name):
            raise ValueError(f"MXFP8 target {name!r} is not a supported Qwen3 decoder projection.")
        if module.bias is not None:
            raise ValueError(f"MXFP8 target {name!r} must be bias-free.")
        if config.enable_fsdp_all_gather and module.out_features % (32 * parallel_state.dp_shard_size):
            raise ValueError(f"MXFP8 target {name!r} requires uniform row shards aligned to 32 rows.")
    _validate_native_runtime()
    return config


def _validate_native_runtime():
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_torch_device

    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        raise RuntimeError("Native MXFP8 training requires NVIDIA CUDA SM100+; emulation is test-only.")
    device = get_torch_device()
    index = device.current_device()
    if device.get_device_capability(index) < (10, 0):
        raise RuntimeError(f"Native MXFP8 requires SM100+ on the current rank's cuda:{index}.")
    try:
        installed = version("torchao")
    except PackageNotFoundError as exc:
        raise ImportError("Native MXFP8 requires torchao==0.18.0 from the GPU extra.") from exc
    if installed.split("+")[0] != "0.18.0":
        raise RuntimeError(f"MXFP8 targets torchao==0.18.0, got {installed}.")
    _probe_native_gemm(index)


@lru_cache(maxsize=None)
def _probe_native_gemm(device_index):
    from veomni.utils.device import get_device_type, get_torch_device

    from .tensors import _quantize

    # Probe the actual native operation: a capability number alone is not
    # evidence that the installed CUDA/cuBLAS stack supports this architecture.
    try:
        with torch.no_grad(), torch.device(get_device_type(), device_index):
            operand = _quantize(torch.ones(32, 32, dtype=torch.bfloat16), emulate=False)
            torch.mm(operand, operand.t())
            get_torch_device().synchronize(device_index)
    except (RuntimeError, NotImplementedError) as exc:
        raise RuntimeError(f"Native MXFP8 GEMM is unavailable on cuda:{device_index}; no emulation fallback.") from exc


def unwrap_mxfp8_tensor(tensor):
    """Expose master storage, retaining DTensor placement; never export a compute view."""
    if isinstance(tensor, DTensor):
        local = tensor.to_local()
        master = unwrap_mxfp8_tensor(local)
        if master is local:
            return tensor
        return DTensor.from_local(
            master,
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
            run_check=False,
            shape=tensor.shape,
            stride=tensor.stride(),
        )
    module = sys.modules.get(__name__ + ".tensors")
    if module is not None:
        if isinstance(tensor, module.MXFP8Weight):
            return tensor._master
        if isinstance(tensor, module.MXFP8GatheredWeight):
            raise ValueError("Reshard before exporting MXFP8 weights: gathered compute payloads are not masters.")
    return tensor


def _state_dict_post_hook(module, state_dict, prefix, local_metadata):
    for name in module._parameters:
        key = prefix + name
        if isinstance(state_dict.get(key), torch.Tensor):
            state_dict[key] = unwrap_mxfp8_tensor(state_dict[key])


def _convert_linears(model, config, *, emulate=False):
    """Internal conversion seam; CPU oracles explicitly opt into emulated GEMMs."""
    from .tensors import MXFP8Linear

    selected = _select_linears(model, config)
    if any(module.weight.dtype != torch.float32 for module in selected.values()):
        raise ValueError("MXFP8 conversion requires FP32 master weights.")
    for name, module in selected.items():
        replacement = MXFP8Linear.from_float(module, all_gather=config.enable_fsdp_all_gather, emulate=emulate)
        replacement.register_state_dict_post_hook(_state_dict_post_hook)
        if name:
            model.set_submodule(name, replacement)
        else:
            model = replacement
    return model


def prepare_mxfp8(model, config):
    """Apply a validated recipe after FP32 upcast and before FSDP2 materialization."""
    if config is not None:
        if config.enable:
            model = _convert_linears(model, config)
        # HF exports store ordinary floating weights plus enough metadata to
        # rebuild the recipe. An explicit disable must survive another export.
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.veomni_mxfp8 = asdict(config)
    return model
