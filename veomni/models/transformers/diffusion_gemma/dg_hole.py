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
"""What ``canvas_batching: flat_seq`` needs before the decoder runs.

The other two modes give each canvas its own row, so visibility is a key-side
mask and nothing about the query matters. Packing the canvases onto one
sequence makes visibility depend on which canvas a query sits in, and the whole
mode rests on that dependence being an *interval*: canvas ``k`` reads a
contiguous run of the cache and nothing else.

That is true of what the pipeline builds -- ``prompt + k C``, narrowed on a
sliding layer to the last ``W-1`` visible positions, with left padding only
ever in front. It is checked rather than assumed, because a hole in the middle
of the run is exactly what an interval cannot express, and a kernel handed one
would read across it silently.
"""

from __future__ import annotations

from typing import Callable

import torch

from ....distributed.parallel_state import get_parallel_state


_KERNEL: Callable | None = None


def _load_hole_kernel():
    """Deferred so importing the model does not require Triton."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from ....ops.kernels.dllm_hole_attention import dllm_hole_attention

    _KERNEL = dllm_hole_attention
    return _KERNEL


def hole_decoder_attention(
    query_states: torch.Tensor,
    encoder_key_states: torch.Tensor,
    encoder_value_states: torch.Tensor,
    canvas_key_states: torch.Tensor,
    canvas_value_states: torch.Tensor,
    clean_lo: torch.Tensor,
    clean_hi: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Run the hole kernel over a packed row of canvases. Returns ``(B, H, T, D)``."""
    return _load_hole_kernel()(
        query_states,
        encoder_key_states,
        encoder_value_states,
        canvas_key_states,
        canvas_value_states,
        clean_lo,
        clean_hi,
        sm_scale,
    )


def canvas_visible_intervals(encoder_visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-canvas ``[lo, hi)`` over the encoder cache, from a per-canvas mask.

    ``encoder_visible`` is ``(batch, canvases, cache_length)`` and boolean. The
    returned pair is int32 ``(batch, canvases)``, half-open, and empty rows come
    back as ``lo == hi == 0``.

    Refuses a visible set that is not one contiguous run. Left padding puts its
    zeros before the run and the future canvases put theirs after it, so any
    zero *between* two ones means the caller built something this mode cannot
    represent -- and silently reading across the gap would leak the answer the
    canvas is denoising.
    """
    if encoder_visible.dim() != 3:
        raise ValueError(
            f"`encoder_visible` must be (batch, canvases, cache_length), got {tuple(encoder_visible.shape)}"
        )
    visible = encoder_visible.bool()
    cache_length = visible.shape[-1]
    positions = torch.arange(cache_length, device=visible.device)

    any_visible = visible.any(dim=-1)
    first = torch.where(visible, positions, torch.full_like(positions, cache_length)).amin(dim=-1)
    last = torch.where(visible, positions, torch.full_like(positions, -1)).amax(dim=-1)
    lo = torch.where(any_visible, first, torch.zeros_like(first))
    hi = torch.where(any_visible, last + 1, torch.zeros_like(last))

    if not torch.equal(visible.sum(dim=-1), hi - lo):
        raise ValueError(
            "A canvas's visible encoder positions are not one contiguous run. `flat_seq` "
            "carries visibility as an interval, which cannot express a hole in the middle "
            "of the run; use `canvas_batching: serial` or `flat_batch` for such a mask."
        )
    return lo.to(torch.int32), hi.to(torch.int32)


def sliding_visible_intervals(
    lo: torch.Tensor, hi: torch.Tensor, sliding_window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow an interval to a sliding layer's last ``sliding_window - 1`` keys.

    Mirrors ``_sliding_encoder_keys``, which keeps the visible positions at or
    after ``last - (W - 1) + 1``. The window is measured against the canvas's
    *own* visible prefix, which is the reason it has to be recomputed per canvas
    rather than shared with the full-attention layers.
    """
    if sliding_window is None or sliding_window <= 0:
        return lo, hi
    narrowed = torch.maximum(lo, hi - (sliding_window - 1))
    return torch.minimum(narrowed, hi).to(torch.int32), hi


def dense_hole_mask(
    lo: torch.Tensor,
    hi: torch.Tensor,
    cache_length: int,
    canvas_length: int,
) -> torch.Tensor:
    """The same visibility as a ``(batch, 1, K*C, cache + K*C)`` boolean mask.

    This is what the sdpa path attends through, and what the kernel is checked
    against. It is built from the intervals rather than beside them, so the two
    paths cannot drift: if the interval is wrong, both are wrong the same way
    and the equivalence against ``serial`` is what catches it.
    """
    batch, canvases = lo.shape
    device = lo.device
    tokens = canvases * canvas_length
    canvas_of_query = torch.arange(tokens, device=device) // canvas_length

    keys = torch.arange(cache_length, device=device)
    query_lo = lo[:, canvas_of_query].unsqueeze(-1)
    query_hi = hi[:, canvas_of_query].unsqueeze(-1)
    clean = (keys >= query_lo) & (keys < query_hi)

    canvas_of_key = torch.arange(tokens, device=device) // canvas_length
    noise = canvas_of_query.unsqueeze(-1) == canvas_of_key.unsqueeze(0)
    noise = noise.unsqueeze(0).expand(batch, tokens, tokens)

    return torch.cat([clean, noise], dim=-1).unsqueeze(1)


def canvas_owner_segments(lo: torch.Tensor, hi: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Which packed sample each canvas reads from, as an index into the segments.

    Packing samples into one row leaves the canvases needing to know whose
    visible prefix they continue -- their positions come from that prefix's end
    relative to the owning sample, and nothing else in the batch says which
    sample that is. It does not have to be declared: a canvas's visible run
    already lies inside exactly one sample's slice of the cache, so the run *is*
    the attribution, and the case where it is not is precisely the leak this
    mode has to refuse.

    ``lo`` / ``hi`` are ``(1, canvases)`` half-open runs over the packed cache,
    as ``canvas_visible_intervals`` returns them, and ``cu_seqlens`` is the
    encoder's segment prefix sum. Returns ``(1, canvases)`` int64.

    Refuses a run straddling two samples and a run that is empty. The first is a
    canvas reading another sample's context; the second has no sample to be
    attributed to, and on this path it also cannot be the padding it would be
    elsewhere -- packing exists so that there is none.
    """
    if lo.shape != hi.shape:
        raise ValueError(f"`lo` and `hi` must agree in shape, got {tuple(lo.shape)} and {tuple(hi.shape)}")
    if lo.dim() != 2 or lo.shape[0] != 1:
        raise ValueError(
            f"a packed row carries its canvases in one row, so `lo` is (1, canvases), got {tuple(lo.shape)}"
        )

    bounds = cu_seqlens.to(device=lo.device, dtype=torch.int64)
    starts, ends = bounds[:-1], bounds[1:]
    low, high = lo.to(torch.int64), hi.to(torch.int64)

    if not bool((high > low).all()):
        empty = (high <= low)[0].nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"canvases {empty} of a packed row see nothing of the encoder cache, so there is no sample to "
            "take their positions from. An empty run means a padded canvas, and a packed row is built to "
            "hold the real ones only -- pad the canvas count and the padding is what gets trained on."
        )

    owner = torch.searchsorted(bounds, low.reshape(-1), right=True) - 1
    owner = owner.clamp_(0, starts.numel() - 1)
    if not bool((high.reshape(-1) <= ends[owner]).all()):
        straddling = (high.reshape(-1) > ends[owner]).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"canvases {straddling} of a packed row read past the end of the sample they start in. A "
            "canvas may only see its own sample's context: reading across the boundary would hand it "
            "another sample's prompt, and on the last canvases another sample's clean answer."
        )
    return owner.reshape(lo.shape)


def packed_canvas_positions(
    lo: torch.Tensor,
    hi: torch.Tensor,
    cu_seqlens: torch.Tensor,
    canvas_length: int,
) -> torch.Tensor:
    """Decoder positions for a packed row, as ``(1, canvases * canvas_length)``.

    A canvas continues immediately after the clean prefix it can attend to. For
    canvas ``k`` of one sample this is ``prompt_length + k * canvas_length``,
    matching generation, even though the encoder cache physically also contains
    the clean current and future canvases. Packing adds an absolute sample offset
    to the visible interval, so subtract the owning sample's start before building
    positions. Otherwise both hidden future tokens and preceding packed samples
    shift RoPE while remaining invisible to attention.
    """
    owner = canvas_owner_segments(lo, hi, cu_seqlens)
    bounds = cu_seqlens.to(device=lo.device, dtype=torch.int64)
    visible_prefix_length = hi.to(torch.int64) - bounds[:-1][owner]
    block = torch.arange(canvas_length, device=lo.device, dtype=torch.int64)
    return (visible_prefix_length.unsqueeze(-1) + block).reshape(1, -1)


def should_use_hole_kernel(config, device: torch.device, dropout: float) -> bool:
    """Whether ``flat_seq`` may run the Triton kernel instead of dense sdpa.

    Fail-closed on dropout, which has no place in the kernel's softmax, rather
    than quietly attending through a dense mask that costs the K canvases the
    kernel exists to skip.

    Under sequence parallelism this is the *only* decoder path there is, and the
    sdpa fallback is the one that cannot run: its dense hole mask spans the
    whole encoder row, which is the axis SP sharded, so it would be built
    against a cache missing most of its keys with every shape still lining up.
    ``ulysses_decoder_split`` gathers the cache back to full length inside the
    attention, which restores what the kernel needs -- a canvas's visible run as
    one readable interval -- so a CPU device, not SP, is what refuses here.
    """
    if not getattr(config, "attn_dg_prefix", False):
        return False
    if dropout:
        raise ValueError(
            "`attn_dg_prefix` with `canvas_batching: flat_seq` cannot apply attention dropout; "
            "set `attention_dropout: 0.0` or turn the kernel off."
        )
    if device.type != "cuda" and get_parallel_state().sp_enabled:
        raise ValueError(
            "sequence parallelism needs the hole kernel, which is Triton and so needs CUDA. The dense "
            "sdpa fallback builds its mask over the whole encoder row, and under SP this rank holds only "
            "a slice of that row's cache."
        )
    return device.type == "cuda"
