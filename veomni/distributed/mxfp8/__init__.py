"""FSDP transport for projections already selected by the model's linear row."""

import sys
from functools import lru_cache

import torch
from torch import nn


def bound_linears(model):
    tensors = sys.modules.get(__name__ + ".tensors")
    if tensors is None:
        return []
    return [module for module in model.modules() if isinstance(module, tensors.MXFP8Linear)]


def prepare_fsdp_transport(model, precision, state, *, gather, compile_enabled, offload, incompatible_comm):
    projections = bound_linears(model)
    if not projections:
        if gather:
            raise ValueError("mxfp8_all_gather requires at least one MXFP8 linear row.")
        return
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("MXFP8 is supported on standalone dense Qwen3 only.")
    if (
        not state.fsdp_enabled
        or state.dp_mode != "fsdp2"
        or any(
            getattr(state, flag, False)
            for flag in (
                "dp_replicate_enabled",
                "sp_enabled",
                "cp_enabled",
                "tp_enabled",
                "pp_enabled",
                "any_extra_parallel_enabled",
                "async_enabled",
            )
        )
    ):
        raise ValueError("MXFP8 requires ordinary multi-rank FSDP2.")
    if not precision.enable or precision.param_dtype != "bfloat16" or precision.reduce_dtype != "float32":
        raise ValueError("MXFP8 requires BF16 parameters and FP32 reductions.")
    if compile_enabled or offload or incompatible_comm or precision.float8.enable:
        raise ValueError("MXFP8 does not support compile, offload, LoRA or other low-precision communication.")
    for module in projections:
        if module.weight.dtype != torch.float32:
            raise ValueError("MXFP8 requires FP32 master weights before FSDP2 wraps them.")
        if gather and module.out_features % (32 * state.dp_shard_size):
            raise ValueError("MXFP8 AllGather requires output rows divisible by 32 times the shard count.")
    _validate_native_runtime()
    if gather:
        from .tensors import MXFP8Weight

        for module in projections:
            module.weight = nn.Parameter(
                MXFP8Weight(module.weight.detach(), emulate=module._emulate), requires_grad=module.weight.requires_grad
            )


def _validate_native_runtime():
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_torch_device

    if not IS_CUDA_AVAILABLE or torch.version.hip is not None:
        raise RuntimeError("Native MXFP8 training requires NVIDIA CUDA SM100+; emulation is test-only.")
    device = get_torch_device()
    index = device.current_device()
    if device.get_device_capability(index) < (10, 0):
        raise RuntimeError(f"Native MXFP8 requires SM100+ on the current rank's cuda:{index}.")
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("torchao")
    except PackageNotFoundError as exc:
        raise ImportError("Native MXFP8 requires torchao==0.18.0 from the float8 extra.") from exc
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
