# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""SM-specific CUDA FA4 kernel selection and one-time CUTLASS setup."""

from functools import cache

import torch

from ......utils.device import get_gpu_compute_capability


KERNEL_UNSUPPORTED = "unsupported"
KERNEL_CUTLASS = "cutlass"
KERNEL_CUTE_JIT = "cute_jit"
_CUTLASS_STACK_SIZE = 8192
CUDA_DEVICE_TYPE = "cuda"


def get_kernel_mode(device: torch.device) -> str:
    """Resolve the CUDA FA4 implementation selected for the query device."""
    if device.type != CUDA_DEVICE_TYPE:
        return KERNEL_UNSUPPORTED

    compute_capability = get_gpu_compute_capability(device)
    if 90 <= compute_capability < 100:
        return KERNEL_CUTLASS
    if compute_capability >= 100:
        return KERNEL_CUTE_JIT
    return KERNEL_UNSUPPORTED


@cache
def prepare_kernel(device: torch.device) -> tuple[str, dict[str, object] | None]:
    """Prepare the SM-specific CUDA FA4 implementation once per device."""
    kernel_mode = get_kernel_mode(device)
    if kernel_mode == KERNEL_UNSUPPORTED:
        compute_capability = get_gpu_compute_capability(device) if device.type == CUDA_DEVICE_TYPE else 0
        hardware = f"SM{compute_capability}" if compute_capability else device.type
        raise RuntimeError(
            f"VeOmni `magi_attention` does not support {hardware}; "
            "use an NVIDIA SM90 GPU with the CUTLASS overlay or an SM100+ GPU with CUTE DSL/JIT."
        )

    if kernel_mode == KERNEL_CUTLASS:
        return kernel_mode, _prepare_cutlass(device)

    # The CUTE DSL path needs no VeOmni setup. MagiAttention compiles and
    # caches the SM100+ kernel when the FA4 function invokes fa4_fwd.
    return kernel_mode, None


def validate_cutlass_inputs(
    query: torch.Tensor,
    value: torch.Tensor,
    softcap: float,
    build_flags: dict[str, object],
) -> int:
    """Reject inputs excluded from the installed SM90 CUTLASS matrix.

    Returns the compiled head-dim bucket used for FA4 metadata tiles.
    """
    if query.dtype == torch.float16:
        if build_flags.get("FLASHATTENTION_DISABLE_FP16", False):
            raise TypeError("The installed MagiAttention SM90 CUTLASS backend does not include FP16 kernels.")
    elif query.dtype != torch.bfloat16:
        raise TypeError(
            "MagiAttention's SM90 CUTLASS backend requires BF16"
            f"{' or FP16' if not build_flags.get('FLASHATTENTION_DISABLE_FP16', False) else ''} inputs, "
            f"got {query.dtype}."
        )

    if value.shape[-1] != query.shape[-1]:
        raise ValueError(
            "The installed MagiAttention SM90 CUTLASS backend requires query and value to have the same "
            f"head dimension, got {query.shape[-1]} and {value.shape[-1]}."
        )

    head_dim = query.shape[-1]
    compiled_head_dims = [
        dim for dim in (64, 96, 128, 192, 256) if not build_flags.get(f"FLASHATTENTION_DISABLE_HDIM{dim}", False)
    ]
    compiled_head_dim = next((dim for dim in compiled_head_dims if head_dim <= dim), None)
    if compiled_head_dim is None:
        raise ValueError(
            f"The installed MagiAttention SM90 CUTLASS backend does not include a kernel for head_dim={head_dim}."
        )

    if softcap != 0.0 and build_flags.get("FLASHATTENTION_DISABLE_SOFTCAP", False):
        raise ValueError("The installed MagiAttention SM90 CUTLASS backend does not include softcap kernels.")

    return compiled_head_dim


def _prepare_cutlass(device: torch.device) -> dict[str, object]:
    try:
        from flash_attn_cute.ffa_fa3 import flash_attn_interface
        from flash_attn_cute.ffa_fa3.flash_attn_config import CONFIG

        required_symbols = ("_flash_attn_forward", "_flash_attn_backward")
        if not all(callable(getattr(flash_attn_interface, name, None)) for name in required_symbols):
            raise ImportError("CUTLASS FFA backend is missing required forward or backward entry points.")
    except (ImportError, OSError, RuntimeError) as error:
        raise ImportError(
            "VeOmni `magi_attention` on SM90 requires MagiAttention's precompiled CUTLASS FFA backend. "
            "Run `uv sync --extra gpu --dev`, then `bash scripts/kernel/install_magi_sm90.sh`."
        ) from error

    build_flags = CONFIG.get("build_flags", {})
    required_flags = {
        "FLASHATTENTION_DISABLE_BACKWARD": False,
        "FLASHATTENTION_DISABLE_SM90": False,
        "FLASHATTENTION_DISABLE_ARBITRARY": False,
    }
    incompatible_flags = [name for name, expected in required_flags.items() if build_flags.get(name) is not expected]
    if incompatible_flags:
        raise RuntimeError(
            "The installed MagiAttention SM90 CUTLASS backend has an incompatible build configuration: "
            f"{', '.join(incompatible_flags)}."
        )

    _install_tile_size_compatibility()
    _ensure_cutlass_stack_size(device)
    return build_flags


def _install_tile_size_compatibility() -> None:
    # MagiAttention 1.1.1 imports this helper from ``utils``, while the
    # corresponding flash-attn-cute revision publishes it from ``tile_size``.
    # Expose the expected name before importing MagiAttention so FA4AttnArg
    # builds Q2K/K2Q metadata with the kernel's actual tile sizes.
    from flash_attn_cute import tile_size as flash_attn_tile_size
    from flash_attn_cute import utils as flash_attn_utils

    if not hasattr(flash_attn_utils, "get_tile_sizes_by_backend"):
        flash_attn_utils.get_tile_sizes_by_backend = flash_attn_tile_size.get_tile_sizes_by_backend


def _ensure_cutlass_stack_size(device: torch.device) -> None:
    """Raise the CUDA thread stack limit required by CUTLASS FFA backward."""
    try:
        from cuda.bindings import runtime
    except ImportError as error:
        raise ImportError(
            "VeOmni `magi_attention` on SM90 requires `cuda-bindings`; install VeOmni with the `gpu` extra."
        ) from error

    with torch.cuda.device(device):
        get_error, current_stack_size = runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)
        if get_error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to query the CUDA device stack size: {get_error}.")
        if current_stack_size >= _CUTLASS_STACK_SIZE:
            return

        (set_error,) = runtime.cudaDeviceSetLimit(
            runtime.cudaLimit.cudaLimitStackSize,
            _CUTLASS_STACK_SIZE,
        )
        if set_error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                "Failed to configure the CUDA device stack required by "
                f"MagiAttention CUTLASS arbitrary-mask backward: {set_error}."
            )
        verify_error, configured_stack_size = runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)
        if verify_error != runtime.cudaError_t.cudaSuccess or configured_stack_size < _CUTLASS_STACK_SIZE:
            raise RuntimeError(
                "Failed to verify the CUDA device stack required by MagiAttention CUTLASS arbitrary-mask backward: "
                f"status={verify_error}, configured={configured_stack_size}, required={_CUTLASS_STACK_SIZE}."
            )
