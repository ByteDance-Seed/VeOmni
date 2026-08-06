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

"""Tensor-native MagiAttention mask contract and Transformers builder."""

from dataclasses import dataclass
from typing import Callable

import torch
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    bidirectional_mask_function,
    causal_mask_function,
)

from .....distributed.parallel_state import get_parallel_state


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


def create_magi_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    *,
    device: torch.device | str | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    attn_type_map: torch.Tensor | None = None,
    skip_ulysses: bool = False,
    **kwargs,
) -> MagiAttentionMask:
    """Build a range mask for standard, packed, or model-defined MagiAttention.

    Transformers' mask registry supplies lengths and a canonical mask function
    for unpacked attention. Models with packed or mixed visibility should call
    this builder directly with cumulative sequence lengths or explicit ranges.
    """
    del kwargs

    if batch_size != 1:
        raise ValueError(f"MagiAttention mask creation requires physical batch size 1, got {batch_size}.")
    if q_offset != 0 or kv_offset != 0:
        raise ValueError(
            "MagiAttention mask creation does not support KV-cache offsets; "
            f"got q_offset={q_offset} and kv_offset={kv_offset}."
        )

    if device is None:
        for tensor in (q_ranges, k_ranges, cu_seq_lens_q, cu_seq_lens_k, attention_mask):
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
    if device is None:
        raise ValueError("MagiAttention mask creation requires a device or tensor metadata.")
    device = torch.device(device)

    has_explicit_ranges = q_ranges is not None or k_ranges is not None
    has_cu_seqlens = cu_seq_lens_q is not None or cu_seq_lens_k is not None
    if has_explicit_ranges and has_cu_seqlens:
        raise ValueError("Pass either explicit MagiAttention ranges or cumulative sequence lengths, not both.")

    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise TypeError(
                "MagiAttention mask creation accepts only a 2D all-valid attention mask, "
                f"got {type(attention_mask).__name__} with shape "
                f"{getattr(attention_mask, 'shape', None)}."
            )
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"MagiAttention attention_mask batch size must be {batch_size}, got {attention_mask.shape[0]}."
            )
        if not has_explicit_ranges and not has_cu_seqlens:
            raise ValueError(
                "The registered MagiAttention mask builder cannot recover packed boundaries from a 2D attention mask. "
                "Pass cu_seq_lens_q/k or explicit q_ranges/k_ranges from the model."
            )
        _require_all(
            attention_mask.to(device=device, dtype=torch.bool),
            "MagiAttention requires an all-valid 2D attention mask; encode padding as independent ranges.",
        )

    parallel_state = get_parallel_state()
    sequence_scale = parallel_state.ulysses_size if parallel_state.ulysses_enabled and not skip_ulysses else 1
    full_q_length = q_length * sequence_scale
    full_kv_length = kv_length * sequence_scale
    if attention_mask is not None and attention_mask.shape[-1] != full_kv_length:
        raise ValueError(
            "MagiAttention attention_mask must describe the full post-Ulysses key sequence, "
            f"got mask length {attention_mask.shape[-1]} and expected {full_kv_length}."
        )

    if has_explicit_ranges:
        if q_ranges is None or k_ranges is None:
            raise ValueError("q_ranges and k_ranges must be provided together.")
        q_ranges = q_ranges.to(device=device).contiguous()
        k_ranges = k_ranges.to(device=device).contiguous()
        if attn_type_map is not None:
            attn_type_map = attn_type_map.to(device=device).contiguous()
        result = MagiAttentionMask(q_ranges, k_ranges, attn_type_map)
        _require_all(
            result.q_ranges[:, 1] <= full_q_length,
            f"MagiAttention q_ranges must end within the full query length ({full_q_length}).",
        )
        _require_all(
            result.k_ranges[:, 1] <= full_kv_length,
            f"MagiAttention k_ranges must end within the full key length ({full_kv_length}).",
        )
        return result

    if attn_type_map is not None:
        raise ValueError("attn_type_map is only accepted with explicit q_ranges and k_ranges.")

    if mask_function is causal_mask_function:
        causal = True
    elif mask_function is bidirectional_mask_function:
        causal = False
    else:
        raise ValueError(
            "The registered MagiAttention mask builder supports only canonical causal or bidirectional masks. "
            "Packed and model-specific visibility must call create_magi_mask with cumulative sequence lengths "
            "or explicit ranges."
        )

    if has_cu_seqlens:
        if cu_seq_lens_q is None:
            raise ValueError("cu_seq_lens_q is required when cumulative sequence lengths are provided.")
        if cu_seq_lens_k is None:
            cu_seq_lens_k = cu_seq_lens_q
        q_ranges = _ranges_from_cu_seqlens(
            "cu_seq_lens_q",
            cu_seq_lens_q,
            device=device,
            sequence_length=full_q_length,
        )
        k_ranges = _ranges_from_cu_seqlens(
            "cu_seq_lens_k",
            cu_seq_lens_k,
            device=device,
            sequence_length=full_kv_length,
        )
    else:
        q_ranges = torch.tensor([[0, full_q_length]], device=device, dtype=torch.int32)
        k_ranges = torch.tensor([[0, full_kv_length]], device=device, dtype=torch.int32)

    if causal:
        attn_type_map = torch.ones(q_ranges.shape[0], device=device, dtype=torch.int32)
    else:
        attn_type_map = None

    result = MagiAttentionMask(q_ranges, k_ranges, attn_type_map)
    if causal:
        _require_all(
            result.q_ranges.diff(dim=1) == result.k_ranges.diff(dim=1),
            "Packed causal MagiAttention requires matching query and key segment lengths.",
        )
    return result


def register_veomni_magi_attention_mask_builder() -> None:
    """Register the standard MagiAttention range-mask builder with Transformers."""
    ALL_MASK_ATTENTION_FUNCTIONS.register("veomni_magi_attention_with_sp", create_magi_mask)


def _ranges_from_cu_seqlens(
    name: str,
    cu_seq_lens: torch.Tensor,
    *,
    device: torch.device,
    sequence_length: int,
) -> torch.Tensor:
    if not isinstance(cu_seq_lens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(cu_seq_lens).__name__}.")
    if cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2:
        raise ValueError(f"{name} must have shape [num_sequences + 1], got {tuple(cu_seq_lens.shape)}.")

    cu_seq_lens = cu_seq_lens.to(device=device)
    _require_all(cu_seq_lens[:1] == 0, f"{name} must start at 0.")
    _require_all(
        cu_seq_lens[-1:] == sequence_length,
        f"{name} must end at the full sequence length ({sequence_length}).",
    )
    return torch.stack((cu_seq_lens[:-1], cu_seq_lens[1:]), dim=1).contiguous()


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


__all__ = [
    "MagiAttentionMask",
    "create_magi_mask",
    "register_veomni_magi_attention_mask_builder",
]
