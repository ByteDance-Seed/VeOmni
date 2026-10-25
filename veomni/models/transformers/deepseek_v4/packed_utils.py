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

"""Packed-sequence helpers for DeepSeek V4 compressed attention."""

from typing import Callable

import torch

from veomni.distributed.context_parallel import empty_compressed_rows, rebase_window_indices
from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates
from veomni.utils.device import get_torch_device
from veomni.utils.seqlen_pos_transform_utils import packed_sequence_slices_from_cu_seqlens


_PACKED_MASK_ERROR = (
    "DeepSeek V4 packed attention received an attention_mask with masked-out "
    "positions alongside cu_seq_lens_q that span the full sequence. Express "
    "padding through cu_seq_lens_q, which the sparse path reads, instead of a "
    "dense mask, which it drops."
)


def resolve_packed_sequence_slices(
    packed_sequence_slices: tuple[tuple[int, int], ...] | None,
    cu_seq_lens_q: torch.Tensor | None,
    sequence_length: int,
) -> tuple[tuple[int, int], ...] | None:
    """Prefer host slices; copy GPU ``cu_seq_lens_q`` only when slices are missing.

    Training collators pass ``packed_sequence_slices`` as Python tuples. The
    public model entry may still receive only a GPU cumulative-length tensor;
    that fallback is allowed to synchronize.
    """
    if packed_sequence_slices is not None:
        _require_slices_span_sequence(packed_sequence_slices, sequence_length)
        return packed_sequence_slices
    if not isinstance(cu_seq_lens_q, torch.Tensor):
        return None
    host = cu_seq_lens_q if cu_seq_lens_q.device.type == "cpu" else cu_seq_lens_q.detach().cpu()
    slices = packed_sequence_slices_from_cu_seqlens(host)
    _require_slices_span_sequence(slices, sequence_length)
    return slices


def ensure_unmasked_packed_attention(
    attention_mask: torch.Tensor | None,
    *,
    attention_mask_is_all_ones: bool | None,
) -> None:
    """Refuse a dense mask with zeros before the sparse path drops it.

    The collator records ``attention_mask_is_all_ones`` on CPU. A GPU
    ``attention_mask.all()`` stays off the training hot path: only a CPU mask
    or CUDA sync-debug mode still reduces the tensor.
    """
    if attention_mask_is_all_ones is True:
        return
    if attention_mask_is_all_ones is False:
        raise ValueError(_PACKED_MASK_ERROR)
    if not isinstance(attention_mask, torch.Tensor):
        return
    if attention_mask.device.type != "cpu" and not _device_sync_debug_enabled():
        return
    if not bool(attention_mask.all()):
        raise ValueError(_PACKED_MASK_ERROR)


def _require_slices_span_sequence(slices: tuple[tuple[int, int], ...], sequence_length: int) -> None:
    boundaries = [slices[0][0], *(end for _, end in slices)] if slices else []
    if not slices or boundaries[0] != 0 or boundaries[-1] != sequence_length:
        raise ValueError(
            "DeepSeek V4 packed cu_seq_lens_q must span the full sequence; "
            f"got {boundaries} for length {sequence_length}"
        )


def _device_sync_debug_enabled() -> bool:
    device_module = get_torch_device()
    getter = getattr(device_module, "get_sync_debug_mode", None)
    return bool(getter is not None and getter())


def build_packed_compression_metadata(
    reference: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_slices: tuple[tuple[int, int], ...],
    compress_rates: tuple[int, ...],
) -> dict[int, dict[str, torch.Tensor]]:
    """Build reusable packed window indices and causal ranges once per forward.

    ``reference`` supplies device and dtype only. Its shape is unused, so a
    scalar placeholder is enough; do not allocate a full hidden-state tensor.
    """
    metadata = {}
    for compress_rate in dict.fromkeys(compress_rates):
        window_starts_list = [
            window_start
            for start, end in sequence_slices
            for window_start in range(start, end - compress_rate + 1, compress_rate)
        ]
        window_starts = torch.tensor(window_starts_list, device=reference.device, dtype=torch.long)
        offsets = torch.arange(compress_rate, device=reference.device, dtype=torch.long)
        window_indices = window_starts[:, None] + offsets[None, :]

        range_starts = torch.empty(position_ids.shape[1], device=reference.device, dtype=torch.int32)
        range_ends = torch.empty_like(range_starts)
        compressed_offset = 0
        for start, end in sequence_slices:
            compressed_count = (end - start) // compress_rate
            range_starts[start:end] = compressed_offset
            visible_counts = torch.minimum(
                (position_ids[0, start:end] + 1) // compress_rate,
                position_ids.new_tensor(compressed_count),
            )
            range_ends[start:end] = compressed_offset + visible_counts.to(torch.int32)
            compressed_offset += compressed_count

        metadata[compress_rate] = {
            "window_starts": window_starts,
            "window_indices": window_indices,
            "range_starts": range_starts,
            "range_ends": range_ends,
        }
    return metadata


def shard_packed_compression_metadata(
    packed_metadata: dict[str, torch.Tensor],
    *,
    window_begin: int,
    window_end: int,
    local_seq_len: int,
    cp_rank: int,
    halo: int,
) -> dict[str, torch.Tensor]:
    """Restrict global packed compression metadata to one context-parallel shard.

    A window belongs to the rank that owns its first token, so the per-window
    arrays are sliced to this rank's contiguous ``[window_begin, window_end)``
    run. Per-query arrays are sliced by query row instead, and their *values*
    stay untouched: the compressed array is replicated, so a compressed slot
    means the same thing on every rank.

    ``window_indices`` addresses ``[left halo | local shard | right halo]`` and
    so shifts by ``halo`` as well as by the shard offset, while ``window_starts``
    is only ever used to index ``position_ids``, which carries no halo.
    """
    window_starts = packed_metadata["window_starts"]
    queries = slice(cp_rank * local_seq_len, (cp_rank + 1) * local_seq_len)
    local_metadata = dict(packed_metadata)
    local_metadata["window_starts"] = window_starts[window_begin:window_end] - cp_rank * local_seq_len
    local_metadata["window_indices"] = rebase_window_indices(
        packed_metadata["window_indices"][window_begin:window_end], local_seq_len, cp_rank, halo
    )
    local_metadata["range_starts"] = packed_metadata["range_starts"][queries]
    local_metadata["range_ends"] = packed_metadata["range_ends"][queries]
    return local_metadata


def compress_packed_windows(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    head_dim: int,
    compress_rate: int,
    kv_norm: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    rope_layer_type: str,
    position_ids: torch.Tensor,
    packed_metadata: dict[str, torch.Tensor],
    *,
    overlap: bool,
    apply_rope: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Compress complete windows without crossing packed sequence boundaries.

    Dynamic batching represents a packed microbatch as ``[1, total_tokens]``
    and resets ``position_ids`` to zero at each sequence boundary. Selecting
    window ends from local positions keeps incomplete tails out of the next
    sequence and lets every operation stay on device.

    ``apply_rope`` is injected so callers in the generated modeling pass their
    module-global ``apply_rotary_pos_emb``, whose local ``VeomniOp`` selection may
    use the fused Triton backend. It is required rather than defaulted to the eager
    reference: a defaulted call site would silently keep eager while every other
    one is fused, which no test would catch.

    ``kv`` and ``gate`` may be pre-extended with halos, as context parallelism
    does, provided ``window_indices`` has been rebased onto the extended buffer
    by :func:`shard_packed_compression_metadata` and the left halo is at least
    ``compress_rate`` wide. That width is what keeps
    ``previous_indices = current_indices - compress_rate`` non-negative at the
    first owned window; a narrower halo would silently read window 0's overlap
    half from the wrong tokens instead, because ``clamp_min(0)`` cannot tell the
    difference. ``position_ids`` and ``window_starts`` stay unhaloed and
    shard-local, since positions are per-sample and never cross a rank.
    """
    if kv.shape[0] != 1:
        raise ValueError(f"Packed DeepSeek V4 compression expects batch size 1, got {kv.shape[0]}")

    window_starts = packed_metadata["window_starts"]
    if window_starts.numel() == 0:
        return empty_compressed_rows(kv, gate, head_dim)

    current_indices = packed_metadata["window_indices"]
    current_kv = kv[0, current_indices]
    current_gate = gate[0, current_indices] + position_bias.to(gate.dtype)

    if overlap:
        previous_indices = (current_indices - compress_rate).clamp_min(0)
        previous_kv = kv[0, previous_indices, :head_dim]
        previous_gate = gate[0, previous_indices, :head_dim] + position_bias[:, :head_dim].to(gate.dtype)
        has_previous_window = position_ids[0, window_starts] >= compress_rate
        previous_gate = previous_gate.masked_fill(~has_previous_window[:, None, None], float("-inf"))
        window_kv = torch.cat([previous_kv, current_kv[..., head_dim:]], dim=1)
        window_gate = torch.cat([previous_gate, current_gate[..., head_dim:]], dim=1)
    else:
        window_kv = current_kv
        window_gate = current_gate

    # `sum` follows autocast's fp32_set_opt_dtype policy, so an implicit `dtype`
    # silently returns fp32 under autocast and leaks out through `kv_norm` into
    # the bf16-only TileLang kernels. Accumulate in fp32 explicitly, cast back.
    compressed = kv_norm(
        (window_kv * window_gate.softmax(dim=1, dtype=torch.float32).to(window_kv.dtype))
        .sum(dim=1, dtype=torch.float32)
        .to(window_kv.dtype)
    ).unsqueeze(0)
    window_positions = position_ids[0, window_starts]
    cos, sin = rotary_emb(
        compressed,
        position_ids=window_positions.unsqueeze(0),
        layer_type=rope_layer_type,
    )
    return apply_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)


def packed_compressed_causal_ranges(
    packed_metadata: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the local compressed-KV interval visible to every packed query."""
    return packed_metadata["range_starts"], packed_metadata["range_ends"]


def isolate_packed_causal_mask_(
    causal_mask: torch.Tensor | None,
    sequence_slices: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Mask previous packed samples in an explicit causal/sliding-window mask."""
    if causal_mask is None or not isinstance(causal_mask, torch.Tensor):
        raise ValueError("DeepSeek V4 packed training requires an explicit tensor attention mask")
    blocked_value = False if causal_mask.dtype == torch.bool else torch.finfo(causal_mask.dtype).min
    for start, end in sequence_slices[1:]:
        causal_mask[..., start:end, :start] = blocked_value
    return causal_mask


__all__ = [
    "CompressedCandidates",
    "build_packed_compression_metadata",
    "compress_packed_windows",
    "ensure_unmasked_packed_attention",
    "isolate_packed_causal_mask_",
    "packed_compressed_causal_ranges",
    "resolve_packed_sequence_slices",
    "shard_packed_compression_metadata",
]
