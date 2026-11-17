"""Sequence-parallel shard handoff between SeedOmni modules.

Every SeedOmni module normally ends its ``post_forward`` by all-gathering its
``1 / sp_size`` slice back into a full-sequence tensor on ``conversation_list``,
and the next module immediately re-packs that full sequence and slices it again.
On a long sequence that round trip costs a full-size tensor, its backward
buffer, and one all-gather per boundary.

``ParallelState.skip_sp_gather_between_modules`` replaces it with a direct shard
handoff: the producer writes **only this rank's window** onto ``item.value``,
and the consumer concatenates those windows, pads back to the window width, and
runs. Nothing here is model-specific — these are tensor primitives over the
``conversation_list`` carrier, shared by the text encoder, the vision towers and
the AR backbones alike.

Two rules keep the handoff exact:

1. **An item never holds padding.** :func:`split_sp_sharded_tensor` emits the
   intersection of a segment with this rank's window and drops the SP pad tail,
   so ``item.value`` is always real rows; the consumer restores the padded width
   with :func:`pad_sp_shard`. Padding is a property of the *window*, not of the
   data. Carrying it in the carrier made the two lengths that must agree
   (``meta[_OMNI_TOKEN_LEN]`` and ``value.size(0)``) disagree on exactly one
   item, which is what a zero-masking pass over the tail used to paper over.
2. **Lengths come from ``meta``, never from ``value``.** Once a value holds a
   shard its row count is the window's, so every length is read through
   :func:`~.llm_packing.item_token_len`, which prefers ``_OMNI_TOKEN_LEN`` (the
   CPU preprocessors stamp it at *full* span length, and it is never sliced).

A batch whose items are all one modality needs no gather at all: the shards
already concatenate into the backbone's row in carrier order. A batch that
interleaves modalities has to gather per producer first, because the packed
order is not the order any single producer wrote — see
:func:`gather_sp_sharded_groups`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor, sp_pad

from ...utils.conversation import ConversationItem, iter_desired_items
from .llm_packing import conversation_metadata, item_token_len, item_value_2d


def sp_local_len(valid_len: int, sp_size: int) -> int:
    """Rows one rank owns when a ``valid_len`` sequence is split over ``sp_size`` ranks."""
    valid_len = int(valid_len)
    if sp_size <= 1:
        return valid_len
    return (valid_len + (-valid_len % sp_size)) // sp_size


def sp_valid_in_window(valid_len: int, sp_rank: int, sp_size: int) -> int:
    """Real (non-pad) rows this rank's window holds."""
    local_len = sp_local_len(valid_len, sp_size)
    start = sp_rank * local_len
    return max(0, min(start + local_len, int(valid_len)) - start)


def split_sp_sharded_tensor(
    local_tensor: torch.Tensor,
    lengths: Sequence[int],
    sp_rank: int,
    sp_size: int,
) -> list[torch.Tensor]:
    """Split this rank's ``[local_len, ...]`` window into per-segment slices.

    ``lengths`` are the segments' **unsharded** token counts, in carrier order.
    Each slice is that segment's intersection with this rank's window
    ``[sp_rank * local_len, (sp_rank + 1) * local_len)``. The SP pad tail is not
    emitted and a segment wholly outside the window gets a zero-row slice, so no
    item ever receives padding — the consumer re-creates the padded width with
    :func:`pad_sp_shard`.
    """
    if not lengths:
        return []
    valid_len = sum(int(length) for length in lengths)
    local_len = sp_local_len(valid_len, sp_size)
    if local_tensor.size(0) != local_len:
        raise ValueError(
            f"split_sp_sharded_tensor: expected local_tensor dim 0 == {local_len} "
            f"(valid_len={valid_len}, sp_size={sp_size}), got {local_tensor.size(0)}."
        )
    shard_start = sp_rank * local_len
    valid_rows = sp_valid_in_window(valid_len, sp_rank, sp_size)
    out: list[torch.Tensor] = []
    seg_start = 0
    for length in lengths:
        seg_end = seg_start + int(length)
        lo = min(max(seg_start - shard_start, 0), valid_rows)
        hi = min(max(seg_end - shard_start, 0), valid_rows)
        out.append(local_tensor[lo:hi])
        seg_start = seg_end
    return out


def pad_sp_shard(
    local: torch.Tensor,
    valid_len: int,
    sp_size: int,
    *,
    dim: int = 0,
) -> torch.Tensor:
    """Pad this rank's real rows back up to its window width with zeros.

    :func:`split_sp_sharded_tensor` keeps padding out of the carrier, so the
    consumer restores it here: the kernels downstream are launched from a
    ``cu_seqlens`` computed over the *padded* row and their inputs have to match.
    The pad rows carry no gradient and no label ever targets them, so they cannot
    reach the loss.

    More rows than the window means a value that was never sliced (a producer
    that gathered, or a full span left on the carrier); that is raised here
    rather than padded, because padding cannot shorten a tensor.
    """
    window = sp_local_len(valid_len, sp_size)
    if local.size(dim) > window:
        raise ValueError(
            f"pad_sp_shard: {local.size(dim)} rows exceed the window width {window} "
            f"(valid_len={valid_len}, sp_size={sp_size}); a value on the carrier was never sliced."
        )
    pad = window - local.size(dim)
    if pad <= 0:
        return local
    shape = list(local.shape)
    shape[dim] = pad
    return torch.cat([local, local.new_zeros(shape)], dim=dim)


def single_modality_shard_batch(conversations: list[list[ConversationItem]] | None) -> bool:
    """Whether this batch's carrier holds one modality, already sharded, in backbone order.

    Then no gather is needed: the per-item shards concatenate straight into this
    rank's row. Decided from ``item.type`` and ``meta`` alone, so every rank of
    the ``sp_group`` reaches the same answer — this predicate chooses between a
    path with a collective and one without, and a rank-local answer would leave
    half the group inside a gather. A value that was never sliced is caught by
    :func:`pad_sp_shard`, not here.
    """
    ps = get_parallel_state()
    if ps.sp_size <= 1 or not ps.skip_sp_gather_between_modules or not conversations:
        return False
    items = [item for sample in conversations for item in sample if not item.is_dummy]
    if not items or len({item.type for item in items}) != 1:
        return False
    return all(isinstance(value := item_value_2d(item), torch.Tensor) and value.dim() >= 2 for item in items)


def gather_sp_sharded_tensor(local: torch.Tensor, valid_len: int, group: Any) -> torch.Tensor:
    """All-gather one producer's rows back to the full ``[valid_len, ...]`` sequence.

    Each rank holds only its own window, so the slices differ in length — a
    segment lying entirely outside one rank's window is zero rows there — and a
    collective needs equal inputs. Every rank therefore pads to the same window
    width first, and the gathered windows are trimmed to their own real rows and
    concatenated in rank order. Slicing the gathered tensor to ``valid_len``
    would instead interleave each rank's padding into the middle of the sequence.

    ``valid_len`` must be the *producer's* full length, so ``sp_local_len``
    yields the width that producer sliced with.
    """
    ps = get_parallel_state()
    padded = pad_sp_shard(local, valid_len, ps.sp_size, dim=0)
    gathered = gather_outputs(padded, gather_dim=0, group=group)
    windows = list(torch.split(gathered, padded.size(0), dim=0))
    real = [
        window.narrow(0, 0, sp_valid_in_window(valid_len, rank, ps.sp_size)) for rank, window in enumerate(windows)
    ]
    return torch.cat(real, dim=0)


def gather_sp_sharded_items(
    items: Sequence[ConversationItem],
    sp_group: Any,
    *,
    meta_list_keys: Sequence[str] = (),
) -> None:
    """All-gather one producer's sharded ``item.value``s back onto their items.

    ``meta_list_keys`` names per-item ``meta`` entries holding a *list* of
    same-length row tensors (Qwen3-VL's per-layer DeepStack features); each is
    gathered alongside ``value`` so its layers stay aligned with their rows.
    """
    if not items:
        return
    lengths = [item_token_len(item) for item in items]
    valid_len = sum(lengths)
    if valid_len == 0:
        return
    local_cat = torch.cat([item_value_2d(item) for item in items], dim=0)
    for item, chunk in zip(
        items, torch.split(gather_sp_sharded_tensor(local_cat, valid_len, sp_group), lengths, dim=0), strict=True
    ):
        item.value = chunk

    for key in meta_list_keys:
        first = items[0].meta.get(key)
        if not first:
            continue
        layers = [
            gather_sp_sharded_tensor(torch.cat([item.meta[key][i] for item in items], dim=0), valid_len, sp_group)
            for i in range(len(first))
        ]
        splits = [torch.split(layer, lengths, dim=0) for layer in layers]
        for index, item in enumerate(items):
            item.meta[key] = [layer_splits[index] for layer_splits in splits]


def gather_sp_sharded_groups(
    groups: Sequence[Sequence[ConversationItem]],
    sp_group: Any,
    *,
    meta_list_keys: Sequence[str] = (),
) -> None:
    """All-gather each producer's shards back onto its own items.

    A group is one module's output items **in the order that module wrote them**,
    not in carrier order: the vision tower, for instance, emits all images and
    then all videos, so gathering its items in carrier order would split one
    sharded sequence into pieces whose windows overlap. Each group must therefore
    be exactly the subsequence whose concatenation the producer sliced — see
    :func:`modality_groups`.
    """
    for items in groups:
        if items:
            keys = tuple(key for key in meta_list_keys if key in items[0].meta)
            gather_sp_sharded_items(items, sp_group, meta_list_keys=keys)


def modality_groups(conversations: list[list[ConversationItem]] | None) -> list[list[ConversationItem]]:
    """The shipped modules' producer grouping: text in carrier order, then media in ViT order.

    Each group must be one producer's **whole** output, in the order that
    producer concatenated it, because that is the sequence it sliced:

    * The text encoder embeds every text item in carrier order
      (``pack_input_ids`` walks the carrier), so its group is carrier order.
    * The vision tower hands images and videos to one ViT call and always runs
      images first, then videos (``process_qwen3vl_visual_items`` calls
      ``_add(image_items)`` before ``_add(video_items)``). This mirrors that
      selection exactly, including the step where an all-dummy batch feeds its
      image placeholders through the ViT as the FSDP anchor.

    Grouping media in carrier order instead would break as soon as a video turn
    precedes an image turn: each rank would reassemble the two producers'
    windows from the wrong offsets.
    """
    if not conversations:
        return []
    groups: list[list[ConversationItem]] = []
    text = [item for sample in conversations for item in sample if item.type == "text"]
    if text:
        groups.append(text)
    image_items = list(iter_desired_items(conversations, types=["image"], roles=["user"]))
    video_items = list(iter_desired_items(conversations, types=["video"], roles=["user"]))
    real_images = [item for item in image_items if not item.is_dummy]
    # No real media at all: the tower runs its dummy image placeholders instead.
    media = (real_images + video_items) if (real_images or video_items) else image_items
    if media:
        groups.append(media)
    return groups


def sp_varlen_row(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    valid_len: int,
    *,
    pos_dim: int,
    group: Any,
    sample_seqlens: Sequence[int],
) -> tuple[torch.Tensor | None, torch.Tensor, tuple[Any, Any, int, int]]:
    """Pad a full-row position/mask pair to the window width and derive its varlen kwargs.

    Returns ``(attention_mask_padded, position_ids_window, (cu_q, cu_k, max_q, max_k))``.
    Every branch of the handoff ends here: the mask stays full-length (attention
    reads it as a row mask, and families that take no mask get ``None`` back), the
    position ids are narrowed to this rank's window, and ``cu_seqlens`` is built
    over the *padded* row because the attention all-to-all reconstructs the full
    sequence before the kernel runs.

    The boundaries come from ``sample_seqlens`` rather than from
    ``prepare_fa_kwargs_from_position_ids``: that helper finds them at
    ``position_ids == 0`` on a flattened row, which for a multi-row (M-RoPE)
    ``[3, 1, L]`` tensor finds three zeros per sample start and produces garbage
    offsets. The lengths are known here anyway, so the boundaries and the single
    pad segment are stated directly for every family.
    """
    mask = None if attention_mask is None else sp_pad(attention_mask, dim=1, pad_value=1)
    pids = sp_pad(position_ids, dim=pos_dim, pad_value=0)
    padded_len = pids.size(pos_dim)
    tail_padding = padded_len - int(valid_len)
    bounds = [0]
    for length in sample_seqlens:
        bounds.append(bounds[-1] + int(length))
    if tail_padding > 0:
        bounds.append(padded_len)
    cu_q = cu_k = torch.tensor(bounds, dtype=torch.int32, device=pids.device)
    # ``cu_q`` and ``cu_k`` are the same tensor, so one maximum has to describe the
    # longest segment of both. The pad tail is a segment like any other: when it is
    # longer than every real sample, it — not ``max(sample_seqlens)`` — is what the
    # kernel has to size for. (``flash_attn_varlen_func`` launches from
    # ``max_seqlen_q``, so under-reporting it is not merely inconsistent metadata.)
    max_q = max_k = max(max(sample_seqlens, default=0), tail_padding)
    return mask, slice_input_tensor(pids, dim=pos_dim, padding=False, group=group), (cu_q, cu_k, max_q, max_k)


def pack_sp_sharded_conversations_for_forward(
    conversations: list[list[ConversationItem]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, list[int]] | None:
    """Pack a single-modality batch from ``item.value`` shards, or ``None`` when one cannot.

    Returns ``(inputs_embeds, attention_mask, position_ids, inputs_embeds_shape,
    sample_seqlens)``. ``inputs_embeds`` is this rank's window, already padded to
    the window width; the metadata tensors are over the **full** row, like every
    other packer's, so the caller pads and slices them once whether or not the
    batch was sharded.

    Returning ``None`` rather than exposing a separate predicate keeps the
    decision and the packing in one place: the caller either gets a ready-to-run
    batch or falls back to the gather path, without asking the same question
    twice.
    """
    if not single_modality_shard_batch(conversations):
        return None
    ps = get_parallel_state()
    items = [item for sample in conversations for item in sample if not item.is_dummy]
    # Lengths come from ``meta``: the values are windows, so ``naflatten`` on them
    # would report the window's row count where every consumer expects the full
    # span, and ``split_sp_sharded_tensor`` would then be handed a length that
    # already sums to this rank's rows.
    full_lengths = [item_token_len(item) for item in items]
    valid_len = sum(full_lengths)
    values = [item_value_2d(item) for item in items]
    position_ids, attention_mask, sample_seqlens = conversation_metadata(conversations, device)
    embeds = pad_sp_shard(torch.cat(values, dim=0).unsqueeze(0), valid_len, ps.sp_size, dim=1)
    inputs_embeds_shape = torch.tensor([[length] for length in full_lengths], dtype=torch.long, device="cpu")
    return embeds, attention_mask, position_ids, inputs_embeds_shape, sample_seqlens


__all__ = [
    "gather_sp_sharded_groups",
    "modality_groups",
    "pack_sp_sharded_conversations_for_forward",
    "pad_sp_shard",
    "single_modality_shard_batch",
    "sp_local_len",
    "sp_valid_in_window",
    "sp_varlen_row",
    "split_sp_sharded_tensor",
]
