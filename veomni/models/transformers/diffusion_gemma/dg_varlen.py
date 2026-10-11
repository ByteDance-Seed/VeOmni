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
"""What the encoder needs before it can read a row holding several samples.

``flat_seq`` packs the canvases of one sample onto the decoder's sequence axis
and leaves the encoder alone: it still runs ``(B, P)``, one row per sample.
Packing *samples* moves the problem into the encoder, where the mask has to
become block diagonal -- sample ``s`` may read its own prefix and nothing to the
left of it, even though that text is now in the same row.

A mask would express it and would also pay for it. sdpa and FFPA form the whole
``(sum P)^2`` score matrix and mask afterwards, so a packed row costs what the
longest-times-widest rectangle costs rather than what the segments cost; see
constraint 32 for the 5.9-12.1x that measures out to. The two backends here take
the segment boundaries as data instead:

* the 25 sliding layers are ``head_dim`` 256 with a window, which
  ``flash_attn_varlen_func`` serves directly, and
* the 5 global layers are 512 wide, past flash's 256 ceiling, which is what
  ``wide_varlen_attention`` was written for.

Both read ``cu_seqlens``, so the boundaries travel as an ``O(S)`` vector and no
quadratic mask is built at all.
"""

from __future__ import annotations

from typing import Callable

import torch

from ....distributed.parallel_state import get_parallel_state


_FLASH_VARLEN: Callable | None = None
_WIDE_VARLEN: Callable | None = None

#: Above this the encoder has no fused kernel other than `wide_varlen_attention`:
#: flash refuses a head dimension past 256 outright.
FLASH_HEAD_DIM_LIMIT = 256


def _load_flash_varlen():
    """Deferred so importing the model does not require flash-attn."""
    global _FLASH_VARLEN
    if _FLASH_VARLEN is None:
        from flash_attn import flash_attn_varlen_func

        _FLASH_VARLEN = flash_attn_varlen_func
    return _FLASH_VARLEN


def _load_wide_varlen():
    """Deferred so importing the model does not require Triton."""
    global _WIDE_VARLEN
    if _WIDE_VARLEN is None:
        from ....ops.kernels.wide_varlen_attention import wide_varlen_attention

        _WIDE_VARLEN = wide_varlen_attention
    return _WIDE_VARLEN


def encoder_segments(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(cu_seqlens, max_seqlen)`` for a packed row, read off its positions.

    A packed row restarts its positions at every sample, which is already how
    this repository marks a segment boundary for varlen attention, so the
    boundaries need no second channel to travel down and cannot disagree with
    the ones RoPE saw. ``position_ids`` is ``(1, total_tokens)``.

    Reading them costs a host synchronisation, which is why the encoder does it
    once per forward and hands the result to all thirty layers rather than
    letting each derive its own.
    """
    if position_ids.dim() != 2 or position_ids.shape[0] != 1:
        raise ValueError(
            f"a packed encoder row has `position_ids` of shape (1, total_tokens), got {tuple(position_ids.shape)}. "
            "Packing puts every sample in one row, so a second row would need its own boundaries."
        )
    positions = position_ids[0]
    total_tokens = positions.numel()
    if total_tokens == 0:
        raise ValueError("a packed encoder row cannot be empty")

    starts = (positions == 0).nonzero(as_tuple=True)[0]
    if starts.numel() == 0 or int(starts[0]) != 0:
        raise ValueError(
            "a packed encoder row must start a sample at index 0, so `position_ids[0, 0]` must be 0. "
            "Left padding has no place in a packed row -- pack the real tokens and let `cu_seqlens` "
            "say where each sample ends."
        )

    boundaries = torch.cat([starts, torch.tensor([total_tokens], device=starts.device)])
    lengths = boundaries.diff()

    # Positions must run 0..L-1 inside each segment. Checked because a row that
    # merely restarts somewhere is not enough: RoPE has already been applied
    # from these positions, so a gap or a repeat would leave the segment
    # boundaries and the rotary phases describing different sequences, and the
    # attention would be self-consistently wrong.
    expected = torch.arange(total_tokens, device=positions.device) - boundaries[:-1].repeat_interleave(lengths)
    if not torch.equal(positions, expected):
        raise ValueError(
            "a packed encoder row must carry positions that run 0..L-1 within each sample; got a "
            "sequence that restarts at 0 but does not increase by one in between. RoPE is applied "
            "from these same positions, so this would rotate the keys as one sequence while "
            "attending to them as another."
        )

    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=positions.device)
    cu_seqlens[1:] = lengths.cumsum(0).to(torch.int32)
    return cu_seqlens, int(lengths.max())


def block_diagonal_causal_mask(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """The same visibility as a ``(1, 1, T, T)`` boolean mask.

    What the sdpa path attends through when no kernel is available, and what the
    kernels are checked against. Quadratic in the packed length by construction
    -- that is the cost the varlen backends exist to avoid, not an oversight --
    so it belongs in a test or on a CPU, not in a training step.

    ``sliding_window`` counts the keys a query may see *including itself*, which
    is the convention ``create_sliding_window_causal_mask`` produces: a window of
    8 lets query 7 see keys 0..7 and query 8 see keys 1..8.
    """
    device = cu_seqlens.device
    positions = torch.arange(total_tokens, device=device)
    lengths = cu_seqlens.diff().to(torch.long)
    segment = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    if segment.numel() != total_tokens:
        raise ValueError(
            f"`cu_seqlens` covers {segment.numel()} tokens but the row holds {total_tokens}; "
            "the prefix sum must end at the packed length"
        )

    keep = (segment[:, None] == segment[None, :]) & (positions[:, None] >= positions[None, :])
    if sliding_window:
        keep &= positions[:, None] - positions[None, :] < sliding_window
    return keep[None, None]


def varlen_encoder_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim: int,
    sliding_window: int | None,
    sm_scale: float,
) -> torch.Tensor:
    """Causal attention within each sample of a packed encoder row.

    Takes ``(1, heads, total_tokens, head_dim)`` as the layer produces it and
    returns ``(1, total_tokens, heads, head_dim)``, the layout the caller
    reshapes from -- the same contract ``ffpa_attention_forward`` honours.

    Under sequence parallelism ``total_tokens`` is this rank's shard on the way
    in and out, and the whole row in between: a pair of all-to-alls trades the
    sequence shard for a head shard, because attention is the one place that
    needs every key. ``cu_seqlens`` and ``max_seqlen`` therefore describe the
    *full* row -- the boundaries were read before the shard was taken, and a
    shard cannot reconstruct them. That also leaves the sliding window measured
    over the real sequence rather than over a shard of it.
    """
    # `sp_enabled` and not `ulysses_enabled`, to match every other gate in this
    # model. The two differ only at `cp_size > 1` with `ulysses_size == 1`, which
    # the preflight refuses -- and if they were allowed to disagree here, the row
    # would be sharded by `shard_packed_row` while attention never gathered it
    # back, which is a wrong answer rather than a failure.
    if get_parallel_state().sp_enabled:
        return _ulysses_varlen_encoder_attention(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            max_seqlen,
            head_dim=head_dim,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
        )
    return _local_varlen_encoder_attention(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
    )


def _ulysses_varlen_encoder_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim: int,
    sliding_window: int | None,
    sm_scale: float,
) -> torch.Tensor:
    """The same attention with the row gathered and the heads split.

    Layouts are the fiddly part. The all-to-all helpers are written for
    ``(..., seq, heads, ...)``, this function is handed ``(1, heads, seq, dim)``,
    and the local kernels return ``(1, seq, heads, dim)`` -- so the head axis
    moves twice and the two calls are not symmetric in their arguments.
    """
    from .dg_sequence_parallel import _ulysses, ulysses_head_split

    from ....distributed.sequence_parallel import gather_heads_scatter_seq  # isort: skip

    group, sp_size, _ = _ulysses()
    local_tokens = query_states.shape[2]
    query_states, key_states, value_states = ulysses_head_split(
        query_states, key_states, value_states, seq_dim=2, head_dim=1
    )
    # Host-side product, not `int(cu_seqlens[-1])`: the encoder already checked once
    # that the boundaries cover the unsharded row, and `local * sp_size` is that
    # length. Re-reading the device tensor here would be one sync per layer per
    # pass (sixty on a 30-layer stack under checkpointing) for an invariant that
    # cannot change between layers.
    if query_states.shape[2] != local_tokens * sp_size:
        raise ValueError(
            f"the gathered row holds {query_states.shape[2]} tokens against "
            f"{local_tokens} on this rank times `sp_size` {sp_size}. Under sequence "
            "parallelism the all-to-all must restore the row the boundaries describe."
        )

    attn_output = _local_varlen_encoder_attention(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
    )
    return gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2, group=group)


def _local_varlen_encoder_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim: int,
    sliding_window: int | None,
    sm_scale: float,
) -> torch.Tensor:
    """One rank's worth, with every key it needs already present.

    The head width picks the backend, not a config flag, because the two are not
    interchangeable: flash is faster where it applies and refuses a head past
    256, and the wide kernel has no window. A sliding layer that is somehow 512
    wide would therefore silently lose its window here, so it is refused.
    """
    if sliding_window and head_dim > FLASH_HEAD_DIM_LIMIT:
        raise ValueError(
            f"a sliding layer {head_dim} wide has no varlen backend: flash refuses a head dimension "
            f"past {FLASH_HEAD_DIM_LIMIT} and `wide_varlen_attention` implements no window, so it "
            "would attend over the whole prefix instead of the last `sliding_window` keys."
        )

    if head_dim > FLASH_HEAD_DIM_LIMIT:
        attn_output = _load_wide_varlen()(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            max_seqlen=max_seqlen,
            sm_scale=sm_scale,
        )
        return attn_output.transpose(1, 2)

    if query_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"flash has no {query_states.dtype} path, and a layer {head_dim} wide has no other varlen "
            "backend -- `wide_varlen_attention` starts above "
            f"{FLASH_HEAD_DIM_LIMIT}. Run the model in bfloat16, or widen this layer."
        )

    # flash wants `(total_tokens, heads, head_dim)`, one row's worth without the
    # batch axis, and takes GQA from the head counts.
    query = query_states[0].transpose(0, 1)
    key = key_states[0].transpose(0, 1)
    value = value_states[0].transpose(0, 1)
    attn_output = _load_flash_varlen()(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=sm_scale,
        causal=True,
        # flash counts keys to the left *besides* the query's own, where the
        # mask this replaces counts the query's own among them.
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
    )
    return attn_output.unsqueeze(0)


def should_use_varlen_encoder(config, device: torch.device, dropout: float) -> bool:
    """Whether a packed encoder row may run the varlen kernels.

    Fail-closed, and on a sharper edge than ``should_use_hole_kernel``'s. There,
    refusing costs the dense mask the kernel was avoiding and the answer is the
    same. Here the fallback is not the ordinary causal mask -- that one would let
    every sample read the samples packed in front of it -- so a refusal has to
    reach a *block-diagonal* mask or nothing at all. Everything below therefore
    raises, except the one case that has a correct slow path to fall to.
    """
    if dropout:
        raise ValueError(
            "a packed encoder row cannot apply attention dropout: neither varlen backend "
            "implements it. Set `attention_dropout: 0.0`."
        )
    if get_parallel_state().sp_enabled and device.type != "cuda":
        raise ValueError(
            f"a packed encoder row on {device.type} falls back to a block-diagonal mask, which is "
            "materialized at the full row length on every rank and so is exactly what sequence "
            "parallelism was turned on to avoid. The all-to-all path needs the varlen kernels, "
            "which are CUDA-only."
        )
    if getattr(config, "use_bidirectional_attention", None) == "all":
        raise ValueError(
            "a packed encoder row needs causal attention within each sample, but "
            "`use_bidirectional_attention: all` makes the encoder bidirectional. Neither varlen "
            "backend implements a bidirectional block."
        )
    if device.type == "cuda":
        return True

    # The one correct fallback: a block-diagonal mask, quadratic in the packed
    # length and 5.9-12.1x slower than these kernels (constraint 32). Reachable
    # because a CPU has nothing faster to offer -- but only through sdpa.
    # `block_diagonal_causal_mask` is boolean, and `eager_attention_forward`
    # *adds* its mask to the scores, so eager would add 1.0 to the positions the
    # mask means to keep and 0.0 to the ones it means to drop and attend
    # everywhere. `create_diffusion_decoder_attention_mask`'s `to_4d` already
    # carries that scar for the decoder. flex wants a `BlockMask` and would take
    # a dense boolean the same way.
    implementation = getattr(config, "_attn_implementation", "sdpa")
    if implementation != "sdpa":
        raise ValueError(
            f"a packed encoder row off CUDA falls back to a block-diagonal mask, which only sdpa "
            f"reads as this code builds it, but `attn_implementation` is {implementation!r}. eager adds "
            "its mask to the scores, so a boolean one would leave every sample visible to every "
            "other; flex expects a BlockMask. Set `attn_implementation: sdpa`."
        )
    return False
