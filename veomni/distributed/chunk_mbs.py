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
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
from transformers.modeling_layers import GradientCheckpointingLayer

from ..utils import logging
from .parallel_state import get_parallel_state


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
_chunk_mbs_checkpoint_func: ContextVar[Optional[Any]] = ContextVar("chunk_mbs_checkpoint_func", default=None)


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
    if cu_seq_lens_q is None:
        raise ValueError("ChunkMBS requires packed-sequence FlashAttention kwargs: missing cu_seq_lens_q.")

    _validate_cu_seq_lens(cu_seq_lens_q, "cu_seq_lens_q")
    cu_seq_lens_k = batch.get("cu_seq_lens_k")
    if cu_seq_lens_k is not None:
        _validate_cu_seq_lens(cu_seq_lens_k, "cu_seq_lens_k")
        if not torch.equal(cu_seq_lens_q, cu_seq_lens_k):
            raise ValueError("ChunkMBS currently requires identical cu_seq_lens_q and cu_seq_lens_k.")
    max_length_q = batch.get("max_length_q")
    max_length_k = batch.get("max_length_k")
    if max_length_q is not None and max_length_k is not None and max_length_q != max_length_k:
        raise ValueError("ChunkMBS currently requires identical max_length_q and max_length_k.")

    cu_values = [int(v) for v in cu_seq_lens_q.tolist()]
    if not cu_values or cu_values[0] != 0:
        raise ValueError("ChunkMBS requires cu_seq_lens_q to start from 0.")
    segment_lengths = [end - start for start, end in zip(cu_values, cu_values[1:])]
    if any(length <= 0 for length in segment_lengths):
        raise ValueError("ChunkMBS requires strictly increasing cu_seq_lens_q.")

    linear_attn_values = _linear_attn_cu_values(batch.get("linear_attn_cu_seq_lens_q"), cu_values[-1])
    linear_attn_boundaries = set(linear_attn_values) if linear_attn_values is not None else None
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
            if token_start not in linear_attn_boundaries or token_end not in linear_attn_boundaries:
                raise ValueError("ChunkMBS chunk boundaries must align with linear_attn_cu_seq_lens_q boundaries.")
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
    _validate_cu_seq_lens(cu_seq_lens, "linear_attn_cu_seq_lens_q")
    cu_values = [int(v) for v in cu_seq_lens.tolist()]
    if not cu_values or cu_values[0] != 0:
        raise ValueError("ChunkMBS requires linear_attn_cu_seq_lens_q to start from 0.")
    if any(end <= start for start, end in zip(cu_values, cu_values[1:])):
        raise ValueError("ChunkMBS requires strictly increasing linear_attn_cu_seq_lens_q.")
    if cu_values[-1] != expected_total:
        raise ValueError("ChunkMBS requires linear_attn_cu_seq_lens_q to end at cu_seq_lens_q[-1].")
    return cu_values


def _validate_cu_seq_lens(cu_seq_lens: Any, name: str) -> None:
    if not isinstance(cu_seq_lens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(cu_seq_lens).__name__}.")
    if cu_seq_lens.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor, got shape {tuple(cu_seq_lens.shape)}.")
    if cu_seq_lens.dtype != torch.int32:
        raise TypeError(f"{name} must use torch.int32, got {cu_seq_lens.dtype}.")
    if cu_seq_lens.device.type != "cpu":
        raise RuntimeError(f"ChunkMBS ranges must be built before {name} is moved off CPU.")


def apply_chunk_mbs(model: nn.Module, config: Any) -> nn.Module:
    if not getattr(config, "enable", False):
        return model

    parallel_state = get_parallel_state()
    if parallel_state.sp_enabled:
        raise RuntimeError("ChunkMBS currently supports packed sequences without sequence parallelism.")
    if getattr(parallel_state, "tp_enabled", False) or getattr(parallel_state, "pp_enabled", False):
        raise RuntimeError("ChunkMBS currently does not support tensor or pipeline parallelism.")
    if parallel_state.any_extra_parallel_enabled:
        raise RuntimeError("ChunkMBS currently does not support ExtraParallel/MoE.")

    target_classes = _decoder_layer_class_names(model)
    if len(target_classes) != 1:
        raise ValueError(
            "ChunkMBS requires exactly one decoder layer class in model._no_split_modules, "
            f"got {sorted(target_classes)!r}."
        )
    target_modules = _find_target_modules(model, target_classes)
    if not target_modules:
        raise ValueError("ChunkMBS did not match any decoder layer listed in model._no_split_modules.")
    if any(_contains_moe_submodule(module) for _, module in target_modules):
        raise RuntimeError("ChunkMBS currently does not support MoE decoder layers.")

    target_stacks = {fqn.rpartition(".")[0] for fqn, _ in target_modules}
    if len(target_stacks) != 1:
        raise ValueError(f"ChunkMBS requires exactly one decoder stack, got {sorted(target_stacks)!r}.")

    incompatible_modules = [
        fqn for fqn, module in target_modules if not isinstance(module, GradientCheckpointingLayer)
    ]
    if incompatible_modules:
        raise TypeError(
            "ChunkMBS requires decoder layers to inherit transformers.modeling_layers.GradientCheckpointingLayer, "
            f"got incompatible modules {incompatible_modules!r}."
        )

    for fqn, module in target_modules:
        _wrap_module_forward(module)
        logger.info_rank0(f"Enable ChunkMBS for module: {fqn}")
    _wrap_gradient_checkpointing_methods(model, [module for _, module in target_modules])
    return model


def _decoder_layer_class_names(model: nn.Module) -> set[str]:
    no_split_modules = getattr(model, "_no_split_modules", None) or []
    return {name for name in no_split_modules if isinstance(name, str) and name.endswith("DecoderLayer")}


def _contains_moe_submodule(module: nn.Module) -> bool:
    return any(
        "moe" in submodule.__class__.__name__.lower() or "expert" in submodule.__class__.__name__.lower()
        for submodule in module.modules()
    )


def _find_target_modules(model: nn.Module, target_classes: set[str]) -> list[tuple[str, nn.Module]]:
    target_modules: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for fqn, module in model.named_modules():
        if id(module) in seen:
            continue
        if module.__class__.__name__ in target_classes:
            target_modules.append((fqn, module))
            seen.add(id(module))
    return target_modules


def _wrap_module_forward(module: nn.Module) -> None:
    if getattr(module, "_chunk_mbs_wrapped", False):
        return

    orig_forward = module.forward
    _wrap_gradient_checkpointing_func(module)

    @wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        ranges = _chunk_mbs_ranges.get()
        if not ranges:
            return orig_forward(*args, **kwargs)
        return _chunked_forward(orig_forward, ranges, args, kwargs)

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

        token = _chunk_mbs_checkpoint_func.set(checkpoint_func)
        try:
            return function(*args, **kwargs)
        finally:
            _chunk_mbs_checkpoint_func.reset(token)

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
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if kwargs.get("use_cache") or any(
        kwargs.get(key) is not None for key in ("past_key_value", "past_key_values", "layer_past")
    ):
        raise RuntimeError("ChunkMBS is only supported for training without KV cache.")

    hidden_states = _get_hidden_states(args, kwargs)
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("ChunkMBS expects hidden_states to be a torch.Tensor.")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(
            "ChunkMBS expects packed hidden_states with shape [1, sequence, hidden], "
            f"got {tuple(hidden_states.shape)}."
        )

    sequence_dim = _canonical_dim(hidden_states, 1)
    full_seq_len = hidden_states.shape[sequence_dim]
    if ranges[-1].token_end != full_seq_len:
        raise ValueError(
            f"ChunkMBS range end ({ranges[-1].token_end}) does not match hidden_states sequence length "
            f"({full_seq_len})."
        )
    outputs = []
    checkpoint_func = _chunk_mbs_checkpoint_func.get()
    for seq_range in ranges:
        chunk_args, chunk_kwargs = _replace_hidden_states(
            args, kwargs, _slice_tensor(hidden_states, seq_range, sequence_dim)
        )
        chunk_kwargs = _slice_kwargs(chunk_kwargs, seq_range, full_seq_len)
        if checkpoint_func is None:
            outputs.append(orig_forward(*chunk_args, **chunk_kwargs))
        else:
            outputs.append(checkpoint_func(orig_forward, *chunk_args, **chunk_kwargs))

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
) -> dict[str, Any]:
    chunk_kwargs = dict(kwargs)

    if "position_ids" in chunk_kwargs:
        chunk_kwargs["position_ids"] = _slice_by_sequence_dim(
            chunk_kwargs["position_ids"], seq_range, full_seq_len, -1, "position_ids"
        )
    if "position_embeddings" in chunk_kwargs:
        chunk_kwargs["position_embeddings"] = _slice_by_sequence_dim(
            chunk_kwargs["position_embeddings"], seq_range, full_seq_len, -2, "position_embeddings"
        )
    if "cache_position" in chunk_kwargs:
        chunk_kwargs["cache_position"] = _slice_by_sequence_dim(
            chunk_kwargs["cache_position"], seq_range, full_seq_len, -1, "cache_position"
        )

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


def _slice_by_sequence_dim(
    data: Any,
    seq_range: PackedSequenceRange,
    full_seq_len: int,
    dim: int,
    name: str,
) -> Any:
    if isinstance(data, torch.Tensor):
        dim = _canonical_dim(data, dim)
        if data.shape[dim] != full_seq_len:
            raise ValueError(
                f"{name} sequence dimension must have length {full_seq_len}, got shape {tuple(data.shape)}."
            )
        return _slice_tensor(data, seq_range, dim)
    if isinstance(data, tuple):
        return tuple(_slice_by_sequence_dim(v, seq_range, full_seq_len, dim, name) for v in data)
    if isinstance(data, list):
        return [_slice_by_sequence_dim(v, seq_range, full_seq_len, dim, name) for v in data]
    if isinstance(data, dict):
        return {k: _slice_by_sequence_dim(v, seq_range, full_seq_len, dim, name) for k, v in data.items()}
    return data


def _slice_attention_mask(data: Any, seq_range: PackedSequenceRange, full_seq_len: int) -> Any:
    if isinstance(data, torch.Tensor):
        if data.ndim == 0:
            return data
        sliced = data
        if data.ndim == 2 and data.shape == (full_seq_len, full_seq_len):
            candidate_dims = (-2, -1)
        else:
            candidate_dims = (-1,) if data.ndim <= 2 else (-2, -1)
        for dim in candidate_dims:
            canonical_dim = _canonical_dim(data, dim)
            if data.shape[canonical_dim] == full_seq_len:
                sliced = _slice_tensor(sliced, seq_range, canonical_dim)
        return sliced
    if isinstance(data, tuple):
        return tuple(_slice_attention_mask(v, seq_range, full_seq_len) for v in data)
    if isinstance(data, list):
        return [_slice_attention_mask(v, seq_range, full_seq_len) for v in data]
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
