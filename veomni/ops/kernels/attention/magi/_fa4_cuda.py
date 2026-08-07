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
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA FA4 preparation, autograd, and SM-specific backend dispatch."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from threading import Lock

import torch

from .....utils.device import get_gpu_compute_capability


_MAGI_KERNEL_UNSUPPORTED = "unsupported"
_MAGI_KERNEL_CUTLASS = "cutlass"
_MAGI_KERNEL_CUTE_JIT = "cute_jit"
_MAGI_CUTLASS_STACK_SIZE = 8192
_CUDA_DEVICE_TYPE = "cuda"


@dataclass(frozen=True)
class _FA4CacheEntry:
    key: tuple[object, ...]
    attn_arg: object
    metadata_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


_FA4_CACHE_LOCK = Lock()
_fa4_cache_entry: _FA4CacheEntry | None = None


def _get_magi_kernel_mode(device: torch.device) -> str:
    """Resolve the CUDA FA4 implementation selected for the query device."""
    if device.type != _CUDA_DEVICE_TYPE:
        return _MAGI_KERNEL_UNSUPPORTED

    compute_capability = get_gpu_compute_capability(device)
    if 90 <= compute_capability < 100:
        return _MAGI_KERNEL_CUTLASS
    if compute_capability >= 100:
        return _MAGI_KERNEL_CUTE_JIT
    return _MAGI_KERNEL_UNSUPPORTED


@cache
def _prepare_default_magi_kernel(device: torch.device) -> tuple[str, dict[str, object] | None]:
    """Prepare the SM-specific CUDA FA4 implementation once per device."""
    kernel_mode = _get_magi_kernel_mode(device)
    if kernel_mode == _MAGI_KERNEL_UNSUPPORTED:
        compute_capability = get_gpu_compute_capability(device) if device.type == _CUDA_DEVICE_TYPE else 0
        hardware = f"SM{compute_capability}" if compute_capability else device.type
        raise RuntimeError(
            f"VeOmni `magi_attention` does not support {hardware}; "
            "use an NVIDIA SM90 GPU with the CUTLASS overlay or an SM100+ GPU with CUTE DSL/JIT."
        )

    if kernel_mode == _MAGI_KERNEL_CUTLASS:
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
        incompatible_flags = [
            name for name, expected in required_flags.items() if build_flags.get(name) is not expected
        ]
        if incompatible_flags:
            raise RuntimeError(
                "The installed MagiAttention SM90 CUTLASS backend has an incompatible build configuration: "
                f"{', '.join(incompatible_flags)}."
            )

        _install_magi_tile_size_compatibility()
        _ensure_magi_cutlass_stack_size(device)
        return kernel_mode, build_flags

    # The CUTE DSL path needs no VeOmni setup. MagiAttention compiles and
    # caches the SM100+ kernel when the FA4 function invokes fa4_fwd.
    return kernel_mode, None


def _validate_magi_cutlass_inputs(
    query: torch.Tensor,
    value: torch.Tensor,
    softcap: float,
    build_flags: dict[str, object],
) -> int:
    """Reject inputs excluded from the installed SM90 CUTLASS matrix."""
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


def _install_magi_tile_size_compatibility() -> None:
    # MagiAttention 1.1.1 imports this helper from ``utils``, while the
    # corresponding flash-attn-cute revision publishes it from ``tile_size``.
    # Expose the expected name before importing MagiAttention so FA4AttnArg
    # builds Q2K/K2Q metadata with the kernel's actual tile sizes.
    from flash_attn_cute import tile_size as flash_attn_tile_size
    from flash_attn_cute import utils as flash_attn_utils

    if not hasattr(flash_attn_utils, "get_tile_sizes_by_backend"):
        flash_attn_utils.get_tile_sizes_by_backend = flash_attn_tile_size.get_tile_sizes_by_backend


def _ensure_magi_cutlass_stack_size(device: torch.device) -> None:
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
        if current_stack_size >= _MAGI_CUTLASS_STACK_SIZE:
            return

        (set_error,) = runtime.cudaDeviceSetLimit(
            runtime.cudaLimit.cudaLimitStackSize,
            _MAGI_CUTLASS_STACK_SIZE,
        )
        if set_error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(
                "Failed to configure the CUDA device stack required by "
                f"MagiAttention CUTLASS arbitrary-mask backward: {set_error}."
            )
        verify_error, configured_stack_size = runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)
        if verify_error != runtime.cudaError_t.cudaSuccess or configured_stack_size < _MAGI_CUTLASS_STACK_SIZE:
            raise RuntimeError(
                "Failed to verify the CUDA device stack required by MagiAttention CUTLASS arbitrary-mask backward: "
                f"status={verify_error}, configured={configured_stack_size}, required={_MAGI_CUTLASS_STACK_SIZE}."
            )


def _get_or_prepare_fa4_attn_arg(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int | None = None,
) -> object:
    """Reuse prepared FA4 mask metadata across layers with matching inputs."""
    global _fa4_cache_entry

    metadata_head_dim = query.shape[-1] if metadata_head_dim is None else metadata_head_dim
    cache_key = _make_fa4_cache_key(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
    with _FA4_CACHE_LOCK:
        if cache_key is not None and _fa4_cache_entry is not None and _fa4_cache_entry.key == cache_key:
            return _fa4_cache_entry.attn_arg

        attn_arg = _prepare_fa4_attn_arg(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
        _fa4_cache_entry = (
            _FA4CacheEntry(
                key=cache_key,
                attn_arg=attn_arg,
                metadata_tensors=(q_ranges, k_ranges, attn_type_map),
            )
            if cache_key is not None
            else None
        )
        return attn_arg


def _prepare_fa4_attn_arg(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int | None = None,
) -> object:
    """Build upstream FA4 metadata once for a new mask and attention shape."""
    metadata_head_dim = query.shape[-1] if metadata_head_dim is None else metadata_head_dim
    device_context = torch.cuda.device(query.device) if query.device.type == _CUDA_DEVICE_TYPE else nullcontext()
    with device_context:
        from magi_attention.common.ranges import AttnRanges
        from magi_attention.meta.collection.calc_meta import FA4AttnArg

        q_ranges_list: list[list[int]] = q_ranges.cpu().tolist()
        k_ranges_list: list[list[int]] = k_ranges.cpu().tolist()
        attn_type_map_list: list[int] = (
            [0] * len(q_ranges_list) if attn_type_map is None else attn_type_map.cpu().tolist()
        )
        return FA4AttnArg(
            q_ranges=AttnRanges.from_ranges(q_ranges_list),
            k_ranges=AttnRanges.from_ranges(k_ranges_list),
            attn_type_map=attn_type_map_list,
            seqlen_q=query.shape[0],
            seqlen_k=key.shape[0],
            headdim=metadata_head_dim,
        )


def _make_fa4_cache_key(
    query: torch.Tensor,
    key: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    metadata_head_dim: int,
) -> tuple[object, ...] | None:
    """Identify unchanged FA4 metadata inputs without reading tensor values."""
    tensor_keys: list[tuple[int, int] | None] = []
    for tensor in (q_ranges, k_ranges, attn_type_map):
        if tensor is None:
            tensor_keys.append(None)
            continue
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors do not expose a version counter, so in-place
            # mutation cannot be detected safely.
            return None
        tensor_keys.append((id(tensor), version))

    return (
        query.device,
        query.dtype,
        tuple(query.shape),
        tuple(key.shape),
        metadata_head_dim,
        *tensor_keys,
    )


class _MagiFA4Function(torch.autograd.Function):
    """Run FA4 autograd with an explicit prepared FA4AttnArg."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        softmax_scale: float | None,
        softcap: float,
        attn_arg: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        softmax_scale = query.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        device_context = torch.cuda.device(query.device) if query.device.type == _CUDA_DEVICE_TYPE else nullcontext()
        with device_context:
            from magi_attention.functional.fa4 import fa4_fwd

            output, lse = fa4_fwd(
                q=query,
                k=key,
                v=value,
                sink=None,
                attn_arg=attn_arg,
                softmax_scale=softmax_scale,
                softcap=softcap,
            )
        ctx.save_for_backward(query, key, value, output, lse, q_ranges, k_ranges, attn_type_map)
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.attn_arg = attn_arg
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args: object) -> tuple[torch.Tensor | None, ...]:
        query, key, value, output, lse, _, _, _ = ctx.saved_tensors
        device_context = torch.cuda.device(query.device) if query.device.type == _CUDA_DEVICE_TYPE else nullcontext()
        with device_context:
            from magi_attention.functional.fa4 import fa4_bwd

            grad_query, grad_key, grad_value, _ = fa4_bwd(
                do=grad_output,
                q=query,
                k=key,
                v=value,
                sink=None,
                o=output,
                lse=lse,
                attn_arg=ctx.attn_arg,
                softmax_scale=ctx.softmax_scale,
                softcap=ctx.softcap,
                deterministic=False,
            )
        return grad_query, grad_key, grad_value, None, None, None, None, None, None


def _fa4_cuda_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None,
    *,
    softmax_scale: float | None,
    softcap: float,
):
    """Run the SM-specific CUDA FA4 backend with prepared mask metadata."""
    kernel_mode, build_flags = _prepare_default_magi_kernel(query.device)
    metadata_head_dim = None
    if kernel_mode == _MAGI_KERNEL_CUTLASS:
        if build_flags is None:
            raise RuntimeError("MagiAttention's SM90 CUTLASS backend did not provide build configuration.")
        compiled_head_dim = _validate_magi_cutlass_inputs(query, value, softcap, build_flags)
        # CUTLASS arbitrary-mask tiles follow the compiled bucket rather than
        # the smaller runtime head dimension served by that bucket.
        metadata_head_dim = compiled_head_dim

    device_context = torch.cuda.device(query.device) if query.device.type == _CUDA_DEVICE_TYPE else nullcontext()
    with device_context:
        try:
            from magi_attention.api import AttnForwardMeta
        except ImportError as error:
            raise ImportError(
                "VeOmni `magi_attention` requires the optional `magi-attention` package. "
                "Install VeOmni with the `gpu` extra."
            ) from error

    attn_arg = _get_or_prepare_fa4_attn_arg(
        query,
        key,
        q_ranges,
        k_ranges,
        attn_type_map,
        metadata_head_dim,
    )
    output, lse = _MagiFA4Function.apply(
        query,
        key,
        value,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale,
        softcap,
        attn_arg,
    )
    return output, AttnForwardMeta(lse=lse, max_logits=None)
