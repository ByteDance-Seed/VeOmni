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

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn

from ..utils import logging
from .parallel_state import get_parallel_state
from .utils import check_any_fqn_match


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class PackedSequenceRange:
    segment_start: int
    segment_end: int
    token_start: int
    token_end: int
    max_length: int
    linear_attn_segment_start: Optional[int] = None
    linear_attn_segment_end: Optional[int] = None


_chunk_mbs_ranges: ContextVar[Optional[list[PackedSequenceRange]]] = ContextVar("chunk_mbs_ranges", default=None)


@contextmanager
def chunk_mbs_context(ranges: Optional[list[PackedSequenceRange]]) -> Iterator[None]:
    token = _chunk_mbs_ranges.set(ranges)
    try:
        yield
    finally:
        _chunk_mbs_ranges.reset(token)


def build_chunk_mbs_ranges(batch: dict[str, Any], config: Any) -> Optional[list[PackedSequenceRange]]:
    if not getattr(config, "enable", False):
        return None

    chunk_mbs = getattr(config, "chunk_mbs", 1)
    if chunk_mbs < 1:
        raise ValueError(f"chunk_mbs_config.chunk_mbs must be >= 1, got {chunk_mbs}.")

    cu_seq_lens_q = batch.get("cu_seq_lens_q")
    strict = getattr(config, "strict", True)
    if cu_seq_lens_q is None:
        if strict:
            raise ValueError("ChunkMBS requires packed-sequence FlashAttention kwargs: missing cu_seq_lens_q.")
        return None

    if cu_seq_lens_q.ndim != 1:
        raise ValueError(f"cu_seq_lens_q must be a 1D tensor, got shape {tuple(cu_seq_lens_q.shape)}.")

    if cu_seq_lens_q.device.type != "cpu":
        if strict:
            raise RuntimeError("ChunkMBS ranges must be built before cu_seq_lens_q is moved off CPU.")
        return None

    cu_values = [int(v) for v in cu_seq_lens_q.tolist()]
    if not cu_values or cu_values[0] != 0:
        raise ValueError("ChunkMBS requires cu_seq_lens_q to start from 0.")
    segment_lengths = [end - start for start, end in zip(cu_values, cu_values[1:])]
    if any(length <= 0 for length in segment_lengths):
        raise ValueError("ChunkMBS requires strictly increasing cu_seq_lens_q.")

    linear_attn_values = _linear_attn_cu_values(batch.get("linear_attn_cu_seq_lens_q"), cu_values[-1])
    num_segments = len(segment_lengths)
    if num_segments <= chunk_mbs:
        return None

    ranges: list[PackedSequenceRange] = []
    for segment_start in range(0, num_segments, chunk_mbs):
        segment_end = min(segment_start + chunk_mbs, num_segments)
        token_start = cu_values[segment_start]
        token_end = cu_values[segment_end]
        max_length = max(segment_lengths[segment_start:segment_end])
        linear_attn_segment_start = None
        linear_attn_segment_end = None
        if linear_attn_values is not None:
            linear_attn_segment_start = bisect_right(linear_attn_values, token_start)
            linear_attn_segment_end = bisect_left(linear_attn_values, token_end)
        ranges.append(
            PackedSequenceRange(
                segment_start=segment_start,
                segment_end=segment_end,
                token_start=token_start,
                token_end=token_end,
                max_length=max_length,
                linear_attn_segment_start=linear_attn_segment_start,
                linear_attn_segment_end=linear_attn_segment_end,
            )
        )
    return ranges


def _linear_attn_cu_values(cu_seq_lens: Optional[torch.Tensor], expected_total: int) -> Optional[list[int]]:
    if cu_seq_lens is None:
        return None
    if cu_seq_lens.ndim != 1:
        raise ValueError(f"linear_attn_cu_seq_lens_q must be a 1D tensor, got shape {tuple(cu_seq_lens.shape)}.")
    if cu_seq_lens.device.type != "cpu":
        raise RuntimeError("ChunkMBS ranges must be built before linear_attn_cu_seq_lens_q is moved off CPU.")
    cu_values = [int(v) for v in cu_seq_lens.tolist()]
    if not cu_values or cu_values[0] != 0:
        raise ValueError("ChunkMBS requires linear_attn_cu_seq_lens_q to start from 0.")
    if any(end <= start for start, end in zip(cu_values, cu_values[1:])):
        raise ValueError("ChunkMBS requires strictly increasing linear_attn_cu_seq_lens_q.")
    if cu_values[-1] != expected_total:
        raise ValueError("ChunkMBS requires linear_attn_cu_seq_lens_q to end at cu_seq_lens_q[-1].")
    return cu_values


def apply_chunk_mbs(model: nn.Module, config: Any) -> nn.Module:
    if not getattr(config, "enable", False):
        return model

    parallel_state = get_parallel_state()
    if parallel_state.sp_enabled:
        raise RuntimeError("ChunkMBS currently supports packed sequences without sequence parallelism.")
    if parallel_state.any_extra_parallel_enabled:
        raise RuntimeError("ChunkMBS currently does not support ExtraParallel/MoE.")

    patterns = list(getattr(config, "apply_modules", []) or [])
    if not patterns:
        raise ValueError("chunk_mbs_config.apply_modules must be non-empty when ChunkMBS is enabled.")

    strict = getattr(config, "strict", True)
    target_modules = _find_target_modules(model, patterns)
    if not target_modules:
        message = f"ChunkMBS did not match any module from apply_modules={patterns!r}."
        if strict:
            raise ValueError(message)
        logger.warning_rank0(message)
        return model

    for fqn, module in target_modules:
        _wrap_module_forward(module, config)
        logger.info_rank0(f"Enable ChunkMBS for module: {fqn}")
    _wrap_gradient_checkpointing_methods(model, [module for _, module in target_modules])
    return model


def _find_target_modules(model: nn.Module, patterns: Iterable[str]) -> list[tuple[str, nn.Module]]:
    normalized_patterns = [_normalize_pattern(pattern) for pattern in patterns]
    target_modules: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for fqn, module in model.named_modules():
        if id(module) in seen:
            continue
        if check_any_fqn_match(normalized_patterns, fqn):
            target_modules.append((fqn, module))
            seen.add(id(module))
    return target_modules


def _normalize_pattern(pattern: str) -> str:
    return pattern.replace("{*}", "*")


def _wrap_module_forward(module: nn.Module, config: Any) -> None:
    if getattr(module, "_chunk_mbs_wrapped", False):
        return

    orig_forward = module.forward
    _wrap_gradient_checkpointing_func(module)

    @wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        ranges = _chunk_mbs_ranges.get()
        if not ranges:
            return orig_forward(*args, **kwargs)
        return _chunked_forward(orig_forward, ranges, config, args, kwargs)

    module.forward = wrapped_forward
    module._chunk_mbs_wrapped = True


def _wrap_gradient_checkpointing_func(module: nn.Module) -> None:
    if not getattr(module, "gradient_checkpointing", False):
        return
    checkpoint_func = getattr(module, "_gradient_checkpointing_func", None)
    if checkpoint_func is None:
        raise RuntimeError("ChunkMBS found gradient checkpointing enabled without a checkpoint function.")
    if getattr(checkpoint_func, "_chunk_mbs_wrapped", False):
        return

    @wraps(checkpoint_func)
    def wrapped_checkpoint_func(function, *args, **kwargs):
        ranges = _chunk_mbs_ranges.get()
        if not ranges:
            return checkpoint_func(function, *args, **kwargs)

        @wraps(function)
        def forward_with_chunk_mbs_context(*forward_args, **forward_kwargs):
            with chunk_mbs_context(ranges):
                return function(*forward_args, **forward_kwargs)

        return checkpoint_func(forward_with_chunk_mbs_context, *args, **kwargs)

    wrapped_checkpoint_func._chunk_mbs_wrapped = True
    module._gradient_checkpointing_func = wrapped_checkpoint_func


def _wrap_gradient_checkpointing_methods(model: nn.Module, target_modules: list[nn.Module]) -> None:
    if getattr(model, "_chunk_mbs_gradient_checkpointing_wrapped", False):
        return

    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if gradient_checkpointing_enable is not None:

        @wraps(gradient_checkpointing_enable)
        def wrapped_enable(*args, **kwargs):
            result = gradient_checkpointing_enable(*args, **kwargs)
            for module in target_modules:
                _wrap_gradient_checkpointing_func(module)
            return result

        model.gradient_checkpointing_enable = wrapped_enable

    model._chunk_mbs_gradient_checkpointing_wrapped = True


def _chunked_forward(
    orig_forward: Any,
    ranges: list[PackedSequenceRange],
    config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if kwargs.get("use_cache") or any(
        kwargs.get(key) is not None for key in ("past_key_value", "past_key_values", "layer_past")
    ):
        raise RuntimeError("ChunkMBS is only supported for training without KV cache.")

    sequence_dim = getattr(config, "sequence_dim", 1)
    hidden_states = _get_hidden_states(args, kwargs)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("ChunkMBS expects hidden_states to be a torch.Tensor.")

    sequence_dim = _canonical_dim(hidden_states, sequence_dim)
    full_seq_len = hidden_states.shape[sequence_dim]
    if ranges[-1].token_end != full_seq_len:
        raise ValueError(
            f"ChunkMBS range end ({ranges[-1].token_end}) does not match hidden_states sequence length "
            f"({full_seq_len})."
        )
    outputs = []
    for seq_range in ranges:
        chunk_args, chunk_kwargs = _replace_hidden_states(
            args, kwargs, _slice_tensor(hidden_states, seq_range, sequence_dim)
        )
        chunk_kwargs = _slice_kwargs(chunk_kwargs, seq_range, full_seq_len, sequence_dim)
        outputs.append(orig_forward(*chunk_args, **chunk_kwargs))

    return _concat_outputs(outputs, sequence_dim)


def _get_hidden_states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("hidden_states")


def _replace_hidden_states(
    args: tuple[Any, ...], kwargs: dict[str, Any], hidden_states: torch.Tensor
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args:
        return (hidden_states, *args[1:]), kwargs
    chunk_kwargs = dict(kwargs)
    chunk_kwargs["hidden_states"] = hidden_states
    return args, chunk_kwargs


def _slice_kwargs(
    kwargs: dict[str, Any],
    seq_range: PackedSequenceRange,
    full_seq_len: int,
    sequence_dim: int,
) -> dict[str, Any]:
    chunk_kwargs = dict(kwargs)

    for key in ("position_ids", "position_embeddings", "cache_position"):
        if key in chunk_kwargs:
            chunk_kwargs[key] = _slice_by_matching_length(chunk_kwargs[key], seq_range, full_seq_len, sequence_dim)

    if "attention_mask" in chunk_kwargs:
        chunk_kwargs["attention_mask"] = _slice_attention_mask(chunk_kwargs["attention_mask"], seq_range, full_seq_len)

    for key in ("cu_seq_lens_q", "cu_seq_lens_k"):
        if key in chunk_kwargs and chunk_kwargs[key] is not None:
            chunk_kwargs[key] = _slice_cu_seq_lens(chunk_kwargs[key], seq_range)

    for key in ("max_length_q", "max_length_k"):
        if key in chunk_kwargs:
            chunk_kwargs[key] = seq_range.max_length

    if "linear_attn_cu_seq_lens_q" in chunk_kwargs and chunk_kwargs["linear_attn_cu_seq_lens_q"] is not None:
        chunk_kwargs["linear_attn_cu_seq_lens_q"] = _slice_linear_attn_cu_seq_lens(
            chunk_kwargs["linear_attn_cu_seq_lens_q"], seq_range
        )
    return chunk_kwargs


def _slice_tensor(tensor: torch.Tensor, seq_range: PackedSequenceRange, dim: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(seq_range.token_start, seq_range.token_end)
    return tensor[tuple(slices)]


def _slice_by_matching_length(data: Any, seq_range: PackedSequenceRange, full_seq_len: int, preferred_dim: int) -> Any:
    if isinstance(data, torch.Tensor):
        dim = _matching_dim(data, full_seq_len, preferred_dim)
        if dim is None:
            return data
        return _slice_tensor(data, seq_range, dim)
    if isinstance(data, tuple):
        return tuple(_slice_by_matching_length(v, seq_range, full_seq_len, preferred_dim) for v in data)
    if isinstance(data, list):
        return [_slice_by_matching_length(v, seq_range, full_seq_len, preferred_dim) for v in data]
    if isinstance(data, dict):
        return {k: _slice_by_matching_length(v, seq_range, full_seq_len, preferred_dim) for k, v in data.items()}
    return data


def _slice_attention_mask(data: Any, seq_range: PackedSequenceRange, full_seq_len: int) -> Any:
    if isinstance(data, torch.Tensor):
        sliced = data
        for dim, size in reversed(list(enumerate(data.shape))):
            if size == full_seq_len:
                sliced = _slice_tensor(sliced, seq_range, dim)
        return sliced
    if isinstance(data, dict):
        return {k: _slice_attention_mask(v, seq_range, full_seq_len) for k, v in data.items()}
    return data


def _slice_cu_seq_lens(cu_seq_lens: torch.Tensor, seq_range: PackedSequenceRange) -> torch.Tensor:
    local = cu_seq_lens.narrow(0, seq_range.segment_start, seq_range.segment_end - seq_range.segment_start + 1)
    return local - cu_seq_lens[seq_range.segment_start]


def _slice_linear_attn_cu_seq_lens(cu_seq_lens: torch.Tensor, seq_range: PackedSequenceRange) -> torch.Tensor:
    if cu_seq_lens.ndim != 1:
        raise ValueError(f"linear_attn_cu_seq_lens_q must be a 1D tensor, got shape {tuple(cu_seq_lens.shape)}.")
    if seq_range.linear_attn_segment_start is None or seq_range.linear_attn_segment_end is None:
        raise RuntimeError("ChunkMBS ranges were built without linear_attn_cu_seq_lens_q.")

    start = cu_seq_lens.new_tensor([seq_range.token_start])
    end = cu_seq_lens.new_tensor([seq_range.token_end])
    inner = cu_seq_lens.narrow(
        0,
        seq_range.linear_attn_segment_start,
        seq_range.linear_attn_segment_end - seq_range.linear_attn_segment_start,
    )
    return torch.cat((start, inner, end), dim=0) - start


def _matching_dim(tensor: torch.Tensor, full_seq_len: int, preferred_dim: int) -> Optional[int]:
    if tensor.ndim == 0:
        return None
    preferred_dim = _canonical_dim(tensor, preferred_dim)
    if tensor.shape[preferred_dim] == full_seq_len:
        return preferred_dim
    for dim, size in enumerate(tensor.shape):
        if size == full_seq_len:
            return dim
    return None


def _canonical_dim(tensor: torch.Tensor, dim: int) -> int:
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"Invalid sequence_dim={dim} for tensor shape {tuple(tensor.shape)}.")
    return dim


def _concat_outputs(outputs: list[Any], sequence_dim: int) -> torch.Tensor:
    if not outputs:
        raise ValueError("ChunkMBS produced no outputs.")
    if not all(isinstance(output, torch.Tensor) for output in outputs):
        raise TypeError("ChunkMBS currently supports modules returning a single tensor.")
    return torch.cat(outputs, dim=sequence_dim)
