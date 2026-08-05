# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""MagiAttention Flex Flash Attention backend and CP1/SP-aware adapter."""

from dataclasses import dataclass
from functools import cache
from typing import Callable, Optional

import torch

from ....distributed.parallel_state import get_parallel_state
from ....utils.device import get_gpu_compute_capability
from .ulysses import prepare_ulysses_qkv, restore_ulysses_output


_MAGI_KERNEL_UNSUPPORTED = "unsupported"
_MAGI_KERNEL_CUTLASS = "cutlass"
_MAGI_KERNEL_CUTE_JIT = "cute_jit"
_MAGI_CUTLASS_STACK_SIZE = 8192


@dataclass(frozen=True)
class MagiAttentionMask:
    """Range-based attention mask consumed by MagiAttention's FFA kernel.

    ``q_ranges`` and ``k_ranges`` contain paired half-open token ranges with
    shape ``[num_ranges, 2]`` and dtype ``torch.int32``. ``attn_type_map`` is
    optional; when present, its values are ``0=full``, ``1=causal``,
    ``2=inverse causal``, and ``3=bidirectional causal``.
    """

    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _validate_range_tensor("q_ranges", self.q_ranges)
        _validate_range_tensor("k_ranges", self.k_ranges)

        if self.q_ranges.shape[0] != self.k_ranges.shape[0]:
            raise ValueError(
                "MagiAttentionMask q_ranges and k_ranges must contain the same number of ranges, "
                f"got {self.q_ranges.shape[0]} and {self.k_ranges.shape[0]}."
            )
        if self.q_ranges.device != self.k_ranges.device:
            raise ValueError(
                "MagiAttentionMask q_ranges and k_ranges must be on the same device, "
                f"got {self.q_ranges.device} and {self.k_ranges.device}."
            )

        _validate_range_values("q_ranges", self.q_ranges)
        _validate_range_values("k_ranges", self.k_ranges)

        if self.attn_type_map is not None:
            if self.attn_type_map.dtype != torch.int32:
                raise TypeError(
                    f"MagiAttentionMask attn_type_map must have dtype torch.int32, got {self.attn_type_map.dtype}."
                )
            if self.attn_type_map.ndim != 1 or self.attn_type_map.shape[0] != self.q_ranges.shape[0]:
                raise ValueError(
                    "MagiAttentionMask attn_type_map must have shape [num_ranges], "
                    f"got {tuple(self.attn_type_map.shape)} for {self.q_ranges.shape[0]} ranges."
                )
            if self.attn_type_map.device != self.q_ranges.device:
                raise ValueError(
                    "MagiAttentionMask attn_type_map must be on the same device as q_ranges and k_ranges, "
                    f"got {self.attn_type_map.device} and {self.q_ranges.device}."
                )
            if not self.attn_type_map.is_contiguous():
                raise ValueError("MagiAttentionMask attn_type_map must be contiguous.")
            _validate_attn_type_values(self.attn_type_map)


def _validate_range_tensor(name: str, ranges: torch.Tensor) -> None:
    if not isinstance(ranges, torch.Tensor):
        raise TypeError(f"MagiAttentionMask {name} must be a torch.Tensor, got {type(ranges).__name__}.")
    if ranges.dtype != torch.int32:
        raise TypeError(f"MagiAttentionMask {name} must have dtype torch.int32, got {ranges.dtype}.")
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"MagiAttentionMask {name} must have shape [num_ranges, 2], got {tuple(ranges.shape)}.")
    if ranges.shape[0] == 0:
        raise ValueError(f"MagiAttentionMask {name} must contain at least one range.")


def _require_all(condition: torch.Tensor, message: str) -> None:
    """Require every element to be true without synchronizing CUDA to Python."""
    all_true = condition.all()
    if all_true.device.type == "cpu":
        if not bool(all_true):
            raise ValueError(message)
        return

    torch._assert_async(all_true, message)


def _validate_range_values(name: str, ranges: torch.Tensor) -> None:
    starts, ends = ranges.unbind(dim=1)
    valid_ranges = (starts >= 0) & (starts < ends)
    _require_all(
        valid_ranges,
        f"MagiAttentionMask {name} must contain non-empty half-open ranges with non-negative starts.",
    )


def _validate_attn_type_values(attn_type_map: torch.Tensor) -> None:
    valid_types = (attn_type_map >= 0) & (attn_type_map <= 3)
    _require_all(valid_types, "MagiAttentionMask attn_type_map values must be in [0, 3].")


def _get_magi_kernel_mode(device: torch.device) -> str:
    """Resolve the FA4 implementation selected for the query device."""
    if device.type != "cuda":
        return _MAGI_KERNEL_UNSUPPORTED

    compute_capability = get_gpu_compute_capability(device)
    if 90 <= compute_capability < 100:
        return _MAGI_KERNEL_CUTLASS
    if compute_capability >= 100:
        return _MAGI_KERNEL_CUTE_JIT
    return _MAGI_KERNEL_UNSUPPORTED


@cache
def _prepare_default_magi_kernel(device: torch.device) -> tuple[str, dict[str, object] | None]:
    """Prepare the hardware-specific FA4 implementation once per device."""
    kernel_mode = _get_magi_kernel_mode(device)
    if kernel_mode == _MAGI_KERNEL_UNSUPPORTED:
        compute_capability = get_gpu_compute_capability(device) if device.type == "cuda" else 0
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
    # caches the SM100+ kernel when ffa_fa4_func invokes flash_attn_cute.
    return kernel_mode, None


def _validate_magi_cutlass_inputs(
    query: torch.Tensor,
    value: torch.Tensor,
    softcap: float,
    build_flags: dict[str, object],
) -> None:
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
    if not any(head_dim <= compiled_dim for compiled_dim in compiled_head_dims):
        raise ValueError(
            f"The installed MagiAttention SM90 CUTLASS backend does not include a kernel for head_dim={head_dim}."
        )

    if softcap != 0.0 and build_flags.get("FLASHATTENTION_DISABLE_SOFTCAP", False):
        raise ValueError("The installed MagiAttention SM90 CUTLASS backend does not include softcap kernels.")


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


def _default_magi_attention_forward(
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
    """Load and call MagiAttention's architecture-portable FA4 facade."""
    kernel_mode, build_flags = _prepare_default_magi_kernel(query.device)
    if kernel_mode == _MAGI_KERNEL_CUTLASS:
        if build_flags is None:
            raise RuntimeError("MagiAttention's SM90 CUTLASS backend did not provide build configuration.")
        _validate_magi_cutlass_inputs(query, value, softcap, build_flags)

    try:
        from magi_attention.api import AttnForwardMeta
        from magi_attention.functional import ffa_fa4_func
    except ImportError as error:
        raise ImportError(
            "VeOmni `magi_attention` requires the optional `magi-attention` package. "
            "Install VeOmni with the `gpu` extra."
        ) from error

    # MagiAttention dispatches this facade by query.device:
    # SM90 -> precompiled ffa_fa3 CUTLASS; SM100+ -> CUTE DSL/JIT.
    output, lse = ffa_fa4_func(
        query,
        key,
        value,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
        softcap=softcap,
    )
    return output, AttnForwardMeta(lse=lse, max_logits=None)


# Module-level patch slot for MagiAttention's FA4 facade. Keeping the import in
# the default callable preserves CPU-safe VeOmni imports.
_magi_attention_forward: Callable = _default_magi_attention_forward


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.ndim != 4:
            raise ValueError(
                f"MagiAttention requires {name} in [batch, heads, sequence, head_dim] layout, "
                f"got shape {tuple(tensor.shape)}."
            )
        if any(dim == 0 for dim in tensor.shape):
            raise ValueError(f"MagiAttention does not support {name} tensors with zero dimensions.")

    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError(
            "MagiAttention currently requires batch size 1 because FFA consumes packed three-dimensional tensors, "
            f"got query/key/value batch sizes {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
        )
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(
            f"MagiAttention GQA requires query heads ({query.shape[1]}) to be divisible by "
            f"key/value heads ({key.shape[1]})."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "MagiAttention requires query, key, and value on the same device, "
            f"got {query.device}, {key.device}, and {value.device}."
        )


def magi_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: MagiAttentionMask | None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run MagiAttention FFA for CP1 with optional VeOmni Ulysses exchange."""
    del module, kwargs

    if not isinstance(attention_mask, MagiAttentionMask):
        raise TypeError(f"MagiAttention requires a MagiAttentionMask, got {type(attention_mask).__name__}.")
    if dropout != 0.0:
        raise ValueError(f"MagiAttention FFA does not support attention dropout, got dropout={dropout}.")
    if sliding_window is not None:
        raise ValueError(
            "MagiAttention does not accept standalone sliding_window metadata; encode visibility in MagiAttentionMask."
        )

    _validate_qkv(query, key, value)
    if attention_mask.q_ranges.device != query.device:
        raise ValueError(
            "MagiAttentionMask tensors must be on the same device as query/key/value, "
            f"got mask device {attention_mask.q_ranges.device} and query device {query.device}."
        )

    parallel_state = get_parallel_state()
    if parallel_state.cp_size != 1:
        raise ValueError(f"MagiAttention FFA currently supports cp_size == 1, got cp_size={parallel_state.cp_size}.")

    ulysses_enabled = parallel_state.ulysses_enabled and not skip_ulysses
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if ulysses_enabled:
        query, key, value, _ = prepare_ulysses_qkv(
            query,
            key,
            value,
            group=parallel_state.ulysses_group,
            ulysses_size=parallel_state.ulysses_size,
        )

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    output, meta = _magi_attention_forward(
        query,
        key,
        value,
        attention_mask.q_ranges,
        attention_mask.k_ranges,
        attention_mask.attn_type_map,
        softmax_scale=scaling,
        softcap=0.0 if softcap is None else softcap,
    )

    output = output.unsqueeze(0)
    lse = meta.lse
    if lse is not None:
        lse = lse.unsqueeze(0).unsqueeze(-1)

    if ulysses_enabled:
        output = restore_ulysses_output(output, group=parallel_state.ulysses_group)
        if lse is not None:
            lse = restore_ulysses_output(lse, group=parallel_state.ulysses_group)

    if lse is not None:
        lse = lse.squeeze(-1).transpose(1, 2).contiguous()

    return output, lse
