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
"""Where DiffusionGemma meets Ulysses sequence parallelism.

VeOmni carries the two all-to-alls inside ``flash_attention_forward`` and hands
them to a model through the ``attn_implementation`` rewrite, which covers
``flash_attention_2/3/4`` and nothing else. DiffusionGemma cannot take that
route: its decoder attends bidirectionally over ``[encoder cache | canvas]``,
which FA2 reads as a padding mask over the query length, and its 5 global layers
are 512 wide against flash's 256 ceiling. So the model has to reach the
primitives itself, and this module is where it does.

The axis being sharded is the *encoder row*. That is the axis that grows: the
canvases are ``K`` blocks of ``canvas_length`` and bounded by ``max_canvases``,
while the encoder row is ``data.max_seq_len`` and is the reason to want SP at
all.

One consequence decides which attention backend SP is built on. Ulysses gathers
the sequence back to full length inside attention and shards the *heads*
instead, so whatever attention materializes still spans the whole row. Under
sdpa that is `create_causal_mask`'s dense ``(B, 1, L, L)``, which SP does not
shard at all and which grows as ``L^2``. The varlen kernels in ``dg_varlen``
take their boundaries as an ``O(L)`` ``cu_seqlens`` vector and build no mask, so
they are the only path that scales in ``L`` -- which makes
``canvas_batching: flat_seq_pack`` a precondition for SP here rather than an
alternative to it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel.loss import SumOverShards
from ....utils.constants import IGNORE_INDEX


def _ulysses(state=None) -> tuple[Optional[ProcessGroup], int, int]:
    """``(group, size, rank)`` for the sequence-parallel dimension.

    Read off the ``ulysses`` mesh dimension and never off ``sp``, and that is not
    a stylistic choice. ``sp`` is a *flattened* mesh dimension, which
    ``DeviceMesh`` registers in ``_mesh_resources`` -- a ``threading.local``. The
    autograd engine runs the backward on its own thread, so a flattened
    dimension that resolves in a layer's forward raises ``Mesh dimension 'sp'
    does not exist`` when gradient checkpointing replays that layer. Not every
    caller here sits inside a checkpointed layer -- ``sum_over_shards`` and
    ``sharded_ar_slice`` run in ``encoder_ar_term``, outside all of them -- but
    the attention ones do, and one accessor that is right everywhere is cheaper
    than a rule about which callers are exposed.

    ``ulysses`` is a real dimension of the mesh object and answers on any
    thread. It is also the whole of SP for this model, since the preflight
    refuses ``cp_size > 1``.

    ``sp_size`` is a product of plain integers and safe anywhere; the fallback
    for rank is what lets a single-process test drive these functions with no
    group at all.
    """
    state = state if state is not None else get_parallel_state()
    group = state.ulysses_group
    if group is None:
        return None, state.sp_size, state.sp_rank
    return group, dist.get_world_size(group), dist.get_rank(group)


def sequence_parallel_preflight(config) -> None:
    """Refuse the ways of asking for sequence parallelism that this model cannot honour.

    What SP needs from the recipe is ``canvas_batching: flat_seq_pack`` and
    ``attn_dg_prefix: true``, and the reason for the first is not a missing
    feature: SP shards the encoder row, and the padded modes have no single row
    to shard -- one row per sample, its length set by the longest prompt in the
    batch. Nor could they use one, since those rows attend through an sdpa mask
    that is quadratic in exactly the length SP was turned on to grow, and that
    mask is replicated on every rank. The second flag is what routes the decoder
    to the hole kernel; without it the refusal would fire only after the dense
    mask had already been allocated.

    Refused loudly because the alternative is not a crash. Nothing in the padded
    path slices anything -- ``DiffusionGemmaCollator`` never passes through
    ``MainCollator``, so no ``SequenceParallelCollator`` runs -- and no
    all-to-all is issued, so every rank would recompute the same micro-batch and
    agree on it. Correct, ``sp_size`` times the cost, reported as a healthy run.

    **This assumes ``init_parallel_state`` has already run**, which is worth
    stating because ``get_parallel_state`` is fail-*open*: uninitialised, it logs
    a warning and hands back a single-process default, and this refusal would
    wave the run through. ``BaseTrainer._setup`` initialises the topology before
    ``_build_model``, so the shipped path is safe; an entrypoint that built a
    model first would silently lose the check.
    """
    state = get_parallel_state()
    if not state.sp_enabled:
        return

    if state.cp_size > 1:
        raise ValueError(
            f"DiffusionGemma supports Ulysses sequence parallelism only, but `cp_size` is "
            f"{state.cp_size}. Ring/context parallelism splits attention across ranks by keys, "
            "which none of this model's kernels implement. Set `cp_size: 1` and use `ulysses_size`."
        )

    # EP + Ulysses is allowed. Expert modules are fully_shard'd on `ep_fsdp`, whose
    # members do not include the SP peers, but `torch_parallelize` then overrides the
    # FSDP divide factor to `world_size` (`ParallelState.extra_parallel_gradient_divide_factor`).
    # That is the same path Qwen3-MoE / DeepSeek-V3 use: SP ranks still replicate the
    # EP-local expert shard, and dividing by world_size folds the SP dimension in without
    # re-apportioning (dp_shard, ulysses, ep).

    if getattr(config, "canvas_batching", None) != "flat_seq_pack":
        raise ValueError(
            f"sequence parallelism needs `canvas_batching: flat_seq_pack`, got "
            f"{getattr(config, 'canvas_batching', None)!r} with `ulysses_size` {state.ulysses_size}. SP "
            "shards the encoder row, and the padded modes have one row per sample padded to the "
            "longest prompt -- no single row to shard, and their sdpa mask is quadratic in the "
            "length SP was turned on to grow. Nothing would fail: each rank would recompute the "
            "whole micro-batch and agree on it, at sp_size times the cost."
        )

    # `attn_dg_prefix` is what routes `flat_seq_pack` to the hole kernel, which is
    # the only decoder path that can read a sharded cache. Without it the decoder
    # falls through to concatenating the cache onto the canvas keys, refused later
    # -- but only after `_flat_seq_sdpa_mask` has already built a dense mask over
    # the full encoder row, which is the L² allocation SP was turned on to avoid.
    # Checked here so a misconfigured run fails at model build rather than on an
    # OOM inside the first decoder layer.
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if not getattr(text_config, "attn_dg_prefix", False):
        raise ValueError(
            f"sequence parallelism needs `attn_dg_prefix: true`, and it is off with `ulysses_size` "
            f"{state.ulysses_size}. That flag is what routes `flat_seq_pack` to the hole kernel, "
            "the only decoder path that can read a sharded cache. Without it the decoder would "
            "fall through to concatenating the cache onto the canvas keys -- refused later, but "
            "only after building a dense mask over the full encoder row, which is the L² allocation "
            "SP was turned on to avoid. Set `attn_dg_prefix: true`."
        )


def sp_pad_tokens(attention_mask: torch.Tensor) -> int:
    """How many tokens at the end of a packed row are sequence-parallel padding.

    A packed row is built out of real tokens, so its padding mask is all ones and
    ``DiffusionGemmaEncoderModel`` refuses anything else -- a padded row packed as
    if it were dense would read across the padding of the sample in front of it.
    SP alignment is the one exception, and it is narrow enough to recognise on
    sight: a trailing run of at most ``sp_size - 1`` zeros. Recognised rather than
    trusted, because "the collator said so" is not available here and a genuinely
    padded row must still be refused.
    """
    state = get_parallel_state()
    if not state.sp_enabled:
        return 0
    zeros = (attention_mask == 0).reshape(-1)
    count = int(zeros.sum())
    if count == 0:
        return 0
    if count >= state.sp_size or not bool(zeros[-count:].all()):
        raise ValueError(
            f"a packed encoder row may carry sequence-parallel padding -- at most {state.sp_size - 1} "
            f"zeros, all at the end -- but its `attention_mask` has {count} zeros, "
            f"{'not all trailing' if count < state.sp_size else 'too many to be alignment'}. Packing "
            "exists so the padding is not there: pack the real tokens and let the restarts in "
            "`position_ids` carry the boundaries."
        )
    return count


def shard_packed_row(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep this rank's contiguous slice of a packed encoder row.

    ``DiffusionGemmaCollator`` has already padded the row to a multiple of
    ``sp_size``, with the tail as a segment of its own, so this only slices. It
    is deliberately *after* the boundaries have been read: ``cu_seqlens``
    describes the whole row and cannot be recovered from a shard, and the
    attention will gather back to full length anyway.

    ``position_ids`` is sliced with the embeddings, so RoPE rotates each token to
    the phase it has within its own sample. The pair the layers then hold --
    local positions, global boundaries -- is the one Ulysses needs, and the
    caller must not swap either half for the other.
    """
    _, sp_size, sp_rank = _ulysses()
    total = inputs_embeds.shape[1]
    if total % sp_size:
        raise ValueError(
            f"a packed encoder row of {total} tokens does not divide across {sp_size} sequence-parallel "
            "ranks. `DiffusionGemmaCollator` pads the row to a multiple of `sp_size`; a row that "
            "reaches here unpadded came from somewhere else, or from a collator built with the wrong "
            "`sp_size`."
        )
    if position_ids.shape[1] != total:
        raise ValueError(
            f"`position_ids` covers {position_ids.shape[1]} tokens and the row holds {total}. Both are "
            "expected at full length here -- the boundaries are read from these positions before the "
            "row is sharded, so a pre-sliced `position_ids` would describe segments that do not exist."
        )

    local = total // sp_size
    start = sp_rank * local
    return (
        inputs_embeds[:, start : start + local],
        position_ids[:, start : start + local],
    )


def sharded_ar_slice(
    encoder_hidden: torch.Tensor,
    ar_labels: torch.Tensor,
    ar_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This rank's share of the encoder AR term, plus the one state it must borrow.

    The targets arrive at full length and stay correct there: they are built from
    ``input_ids``, the padding mask and ``cu_seqlens``, none of which is sharded,
    so each token's ``1/N_s`` weight is its own sample's and not a shard's.

    The borrowing is the whole reason this is not a slice. The term reads token
    ``t``'s label off the hidden state at ``t-1``, and for the first token of
    every shard that state lives on the rank before. One vector is fetched from
    each neighbour so the shift closes locally, which keeps the sharded sum equal
    to the unsharded one instead of quietly dropping ``sp_size - 1`` tokens --
    a difference of a few tokens in thousands, i.e. exactly the size that hides.
    """
    from ....distributed.sequence_parallel import gather_outputs

    group, sp_size, sp_rank = _ulysses()
    local = encoder_hidden.shape[1]
    total = ar_labels.shape[1]
    if total != local * sp_size:
        raise ValueError(
            f"the AR targets cover {total} tokens and the encoder returned {local} on each of "
            f"{sp_size} ranks. Both come from the same batch: the targets are built at full length "
            "and the hidden states are sharded, so a mismatch means one of them was already sliced."
        )

    # `gather_outputs` and not a send/recv pair: this is `sp_size` vectors of
    # `hidden_size`, so the collective costs nothing worth avoiding, and its
    # backward is an all-reduce that returns each rank's own slice -- which is
    # how the borrowed state's gradient finds its way back to the rank that owns
    # the parameters that produced it.
    boundary = gather_outputs(encoder_hidden[:, -1:, :], gather_dim=1, group=group)

    # Rank 0 prepends too, and takes the last rank's state to do it. Not because
    # position 0 has a predecessor -- it has none -- but because the gather's
    # backward is itself an all-reduce, and a rank that leaves `boundary` out of
    # its graph does not run that backward while the others wait in it. So the
    # borrowed vector is kept and made inert instead, which is why the weight
    # below has to be zero rather than merely expected to be.
    # Checked on every rank although only rank 0 depends on it, because
    # `float()` is a device-to-host sync and one that happens on a single rank
    # each step makes that rank's stream the odd one out for no reason. The
    # weights are the whole row's here, so every rank reads the same element.
    if float(ar_weights[:, 0].abs().max()) != 0.0:
        raise ValueError(
            "the AR term gives token 0 of the row a non-zero weight, so it is asking for a prediction "
            "from a state before the row began. `encoder_ar_targets` clears the first token of every "
            "segment; if that changed, the wrapped-around state rank 0 prepends here stops being inert."
        )

    start = sp_rank * local
    previous = (sp_rank - 1) % sp_size
    hidden = torch.cat([boundary[:, previous : previous + 1, :], encoder_hidden], dim=1)
    if sp_rank != 0:
        return hidden, ar_labels[:, start - 1 : start + local], ar_weights[:, start - 1 : start + local]

    ignored = torch.full_like(ar_labels[:, :1], IGNORE_INDEX)
    return (
        hidden,
        torch.cat([ignored, ar_labels[:, :local]], dim=1),
        torch.cat([torch.zeros_like(ar_weights[:, :1]), ar_weights[:, :local]], dim=1),
    )


def sum_over_shards(loss: torch.Tensor) -> torch.Tensor:
    """The AR term summed over the group, so the logged number does not depend on ``sp_size``.

    See `SumOverShards` for why the backward scales by ``sp_size``.
    """
    state = get_parallel_state()
    if not state.sp_enabled:
        return loss
    group, sp_size, _ = _ulysses(state)
    return SumOverShards.apply(loss, group, sp_size)


def refuse_decoder_path_under_sequence_parallelism(what: str, why: str) -> None:
    """Close a decoder path that sequence parallelism has not been built for.

    The supported one is the hole kernel under ``canvas_batching:
    flat_seq_pack``, which is what the preflight already requires. The others
    reach the same sharded cache without a way to make it whole:

    * ``dg_prefix`` gathers each canvas's visible prefix out of the cache in one
      piece, from a mask it re-derives itself, and both of those describe the
      full row rather than this rank's slice of it.
    * concatenating the cache onto the canvas keys and handing the pair to sdpa
      builds a dense mask against a cache missing ``(sp_size - 1)/sp_size`` of
      its keys, whose shapes still line up.
    * the ``serial`` and ``flat_batch`` canvas loops do not carry the intervals
      the split below needs, and the preflight refuses those modes anyway.

    Refused with the path named rather than left to the kernels: the shapes are
    consistent in every one of these cases, so what a rank would otherwise get
    is an answer computed against a fraction of the context.
    """
    state = get_parallel_state()
    if not state.sp_enabled:
        return
    raise ValueError(
        f"{what} does not run under sequence parallelism (`ulysses_size` is {state.sp_size}): {why} Each "
        f"rank holds one {state.sp_size}th of the encoder KV cache along the sequence. The supported "
        "decoder path is the hole kernel under `canvas_batching: flat_seq_pack`, which all-to-alls the "
        "cache back to full length inside the attention -- set `attn_dg_prefix: true` and leave the canvas "
        "batching alone."
    )


def full_cache_length(local_length: int) -> int:
    """The encoder row's length, given what this rank's slice of the cache holds.

    The decoder measures every canvas's visible run against the cache in *cache
    coordinates*, and those come from the replicated decoder mask, which spans
    the whole row. So the length the interval arithmetic needs is the row's, not
    the shard's -- and the attention makes that true by gathering the cache back
    to full length before it reads those intervals.
    """
    state = get_parallel_state()
    if not state.sp_enabled:
        return local_length
    return local_length * state.sp_size


def _kv_repeats(q_heads: int, kv_heads: int, sp_size: int) -> int:
    """How many times the KV heads have to be repeated for one per rank.

    ``1`` when there are already enough. The repetition costs the memory GQA was
    saving, on the KV only, and is the only option -- a rank with zero KV heads
    has nothing to attend to. It stays correct because ``repeat_interleave``
    duplicates each head in place: the ranks that receive copies of KV head
    ``i`` are exactly the ranks that receive the query heads whose group is
    ``i``, since both are cut into contiguous blocks in the same order.

    The counts arrive as tensor shapes, so the messages below name both config
    fields a layer's KV heads can come from rather than guessing: sliding layers
    read ``num_key_value_heads`` and global layers ``num_global_key_value_heads``,
    and naming only the first would point a reader at the wrong key on one layer
    type out of two.
    """
    kv_field = "num_key_value_heads / num_global_key_value_heads"
    if q_heads % sp_size:
        raise ValueError(
            f"num_attention_heads ({q_heads}) must be divisible by sp_size ({sp_size}): Ulysses gives "
            "each rank a whole number of query heads."
        )
    if sp_size <= kv_heads:
        if kv_heads % sp_size:
            raise ValueError(
                f"the KV head count ({kv_heads}, from {kv_field}) must be divisible by sp_size "
                f"({sp_size}) when there are enough of them to go round, or the ranks would not agree "
                "on which query heads share a KV head."
            )
        return 1
    if sp_size % kv_heads:
        raise ValueError(
            f"sp_size ({sp_size}) exceeds the KV head count ({kv_heads}, from {kv_field}) and is not "
            "a multiple of it, so the KV heads cannot be repeated to one per rank. Use an sp_size that "
            "divides or is divided by the KV head count."
        )
    return sp_size // kv_heads


def repeat_kv_heads(tensor: torch.Tensor, dim: int, repeats: int) -> torch.Tensor:
    """Duplicate each KV head in place, ``repeats`` times.

    One function rather than the six call sites it replaces, because *which*
    repetition this is is the correctness argument: ``repeat_interleave`` yields
    ``[h0, h0, h1, h1]`` and a tile would yield ``[h0, h1, h0, h1]``. Both give
    every rank one KV head of the right shape and only the first gives each rank
    the head its contiguous block of query heads belongs to. Kept public so the
    test that asserts that pairing can call this instead of restating it.
    """
    return torch.repeat_interleave(tensor, dim=dim, repeats=repeats)


def ulysses_head_split(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    seq_dim: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trade the sequence shard for a head shard: ``L/sp x H`` becomes ``L x H/sp``.

    The mirror of ``flash_attention_forward``'s Ulysses patch, including its way
    out of ``sp_size > num_key_value_heads``.
    """
    from ....distributed.sequence_parallel import gather_seq_scatter_heads

    group, sp_size, _ = _ulysses()

    repeats = _kv_repeats(query_states.shape[head_dim], key_states.shape[head_dim], sp_size)
    if repeats > 1:
        key_states = repeat_kv_heads(key_states, head_dim, repeats)
        value_states = repeat_kv_heads(value_states, head_dim, repeats)

    return tuple(
        gather_seq_scatter_heads(tensor, seq_dim=seq_dim, head_dim=head_dim, group=group)
        for tensor in (query_states, key_states, value_states)
    )


def _local_heads(tensor: torch.Tensor, head_dim: int, sp_size: int, sp_rank: int) -> torch.Tensor:
    """This rank's block of heads out of a tensor every rank holds in full.

    Cut the way ``all_to_all_tensor`` cuts its scatter dimension --
    ``tensor_split`` into contiguous blocks, rank ``r`` taking block ``r`` -- so
    a replicated tensor sliced here lines up head for head with a sharded one
    that arrived through the all-to-all.
    """
    heads = tensor.shape[head_dim]
    if heads % sp_size:
        raise ValueError(f"{heads} heads do not divide across {sp_size} sequence-parallel ranks")
    return tensor.tensor_split(sp_size, dim=head_dim)[sp_rank]


def ulysses_decoder_split(
    query_states: torch.Tensor,
    canvas_key_states: torch.Tensor,
    canvas_value_states: torch.Tensor,
    cache_key_states: torch.Tensor,
    cache_value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make one decoder layer's attention local, on ``(B, heads, tokens, dim)``.

    Two halves arriving in two different states, which is what makes this not a
    copy of ``ulysses_head_split``:

    * the **cache** is sharded along the sequence, ``sp_size`` slices of one
      encoder row, and every head is present. It is all-to-alled: full length,
      ``heads/sp`` of them. That restores the premise the hole kernel is built
      on -- a canvas's visible run is one interval it can read in one piece --
      at the traffic measured in ``scripts/bench_encoder_sequence_parallel.py``
      as 0.3-0.4% of a step against 0.8-1.5% for gathering the cache instead,
      and with the same bytes resident as the shard it replaces.
    * the **canvas** half is replicated: the collator never sharded it, since
      the canvases are ``K`` blocks of ``canvas_length`` bounded by
      ``max_canvases`` and so are not the axis that grows. Its heads are sliced
      locally, no collective involved.

    The output has to be put back together by :func:`gather_attention_heads`,
    not by ``gather_heads_scatter_seq``: the canvas axis is replicated on the
    way in and stays replicated on the way out. Sharding it as well is a
    separate change -- it would shard the diffusion loss with it.
    """
    from ....distributed.sequence_parallel import gather_seq_scatter_heads

    group, sp_size, sp_rank = _ulysses()

    repeats = _kv_repeats(query_states.shape[1], cache_key_states.shape[1], sp_size)
    if repeats > 1:
        cache_key_states = repeat_kv_heads(cache_key_states, 1, repeats)
        cache_value_states = repeat_kv_heads(cache_value_states, 1, repeats)
        canvas_key_states = repeat_kv_heads(canvas_key_states, 1, repeats)
        canvas_value_states = repeat_kv_heads(canvas_value_states, 1, repeats)

    return (
        _local_heads(query_states, 1, sp_size, sp_rank),
        _local_heads(canvas_key_states, 1, sp_size, sp_rank),
        _local_heads(canvas_value_states, 1, sp_size, sp_rank),
        gather_seq_scatter_heads(cache_key_states, seq_dim=2, head_dim=1, group=group),
        gather_seq_scatter_heads(cache_value_states, seq_dim=2, head_dim=1, group=group),
    )


def gather_attention_heads(attn_output: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """Put the head-sharded decoder output back together on every rank.

    An all-gather and not the all-to-all that ends the encoder's attention,
    because the tokens were replicated to begin with: what each rank is missing
    is heads, and every rank needs all of them to run ``o_proj``.

    The gradient is why this is the right call rather than merely a convenient
    one. Every rank computes the same loss from the same gathered output, so the
    incoming gradient is the same on all of them; ``gather_outputs``'s backward
    all-reduces it and keeps this rank's heads, which multiplies the local
    gradient by ``sp_size`` -- exactly what FSDP's *average* over the SP group
    then divides out to leave the sum over heads that the unsharded run
    computes. The same factor ``sum_over_shards`` applies to the AR term, and
    for the same reason.
    """
    from ....distributed.sequence_parallel import gather_outputs

    state = get_parallel_state()
    if not state.sp_enabled:
        return attn_output
    group, _, _ = _ulysses(state)
    return gather_outputs(attn_output, gather_dim=head_dim, group=group)
