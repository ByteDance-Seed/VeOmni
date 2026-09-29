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

"""Vocab-parallel (``emb``) embedding lookup and tied projection.

Both ops take this rank's plain ``[vocab // emb_size, hidden]`` rows. The weight
gradient they return covers every token of the ``emb`` group, but only this
data-parallel replica's tokens: callers run them inside the FSDP2 unshard hooks
of the owning module, which reduce-scatter and scale that gradient like any
other parameter's.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _group_size_and_rank(group: Optional["dist.ProcessGroup"]) -> tuple[int, int]:
    if group is None:
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


def _check_ids_in_range(ids: torch.Tensor, vocab_size: int, group: Optional["dist.ProcessGroup"]) -> None:
    """Fail on every rank together, before any dispatch, if some rank holds an out-of-range id.

    Ids are bucketed by floor division, so a negative id would land in bucket
    ``-1`` that nothing collects and leave its output row uninitialized. Raising
    on the offending rank alone would leave the others blocked in the next
    collective until it times out.
    """
    lo, hi = (int(ids.min()), int(ids.max())) if ids.numel() > 0 else (0, 0)
    bad = torch.tensor([int(lo < 0 or hi >= vocab_size)], device=ids.device)
    if group is not None:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group)
    if bad.item():
        raise RuntimeError(
            f"AllToAllEmbedding: token ids must lie in [0, {vocab_size}); "
            f"this rank's ids span [{lo}, {hi}] (some rank of the emb group is out of range)."
        )


class AllToAllEmbedding(torch.autograd.Function):
    """Vocab-parallel embedding via all-to-all token dispatch over the ``emb`` group.

    Each ``emb`` rank owns a contiguous vocabulary shard ``embedding_table`` of
    shape ``[vocab // emb_size, hidden]``. Global token ids are dispatched
    (all-to-all) to their owning rank, looked up locally on that rank's shard,
    then the embeddings are shipped back (all-to-all) and reassembled in input
    order. Backward routes the per-token grads back the same way and index-adds
    them (in fp32) into the local shard.

    A single shard (no group, or a group of one) owns the whole table, so each
    exchange below becomes the identity and is aliased rather than performed.
    Note it is aliased and not merely skipped: every collective here writes into
    a freshly ``empty`` buffer, so dropping the call alone would return
    uninitialized memory forward and a silently all-zero table gradient back.
    """

    @staticmethod
    def forward(ctx, group: Optional["dist.ProcessGroup"], input_tensor: torch.Tensor, embedding_table: torch.Tensor):
        emb_size, emb_rank = _group_size_and_rank(group)
        sharded = emb_size > 1
        vocab_size_per_rank, embedding_dim = embedding_table.shape

        raw_shape = input_tensor.shape
        input_flat = input_tensor.reshape(-1)
        num_input_ids = input_flat.shape[0]
        _check_ids_in_range(input_flat, vocab_size_per_rank * emb_size, group if sharded else None)

        # --- Dispatching logic: which rank owns each id ---
        id_rank = input_flat // vocab_size_per_rank
        full_rank_index = torch.argsort(id_rank, stable=True)
        send_counts = torch.bincount(id_rank, minlength=emb_size)
        send_rank_count_list = send_counts.tolist()

        # --- 1st collective: exchange per-pair counts ---
        if sharded:
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=group)
            receive_rank_count_list = recv_counts.tolist()
        else:
            receive_rank_count_list = send_rank_count_list

        # --- 2nd collective: exchange token ids ---
        send_ids = input_flat[full_rank_index].contiguous()
        if sharded:
            recv_ids = torch.empty(sum(receive_rank_count_list), dtype=input_flat.dtype, device=input_flat.device)
            dist.all_to_all_single(
                recv_ids,
                send_ids,
                output_split_sizes=receive_rank_count_list,
                input_split_sizes=send_rank_count_list,
                group=group,
            )
        else:
            recv_ids = send_ids

        # --- Local lookup on this rank's shard ---
        local_indices = recv_ids - emb_rank * vocab_size_per_rank
        embs = F.embedding(local_indices, embedding_table)

        # --- 3rd collective: ship looked-up embeddings back ---
        if sharded:
            embs_recv = torch.empty(num_input_ids, embedding_dim, dtype=embs.dtype, device=embs.device)
            dist.all_to_all_single(
                embs_recv,
                embs.contiguous(),
                output_split_sizes=send_rank_count_list,
                input_split_sizes=receive_rank_count_list,
                group=group,
            )
        else:
            embs_recv = embs

        # --- Reassemble to original input order ---
        output = torch.empty(num_input_ids, embedding_dim, dtype=embs.dtype, device=embs.device)
        if num_input_ids > 0:
            output[full_rank_index] = embs_recv
        output = output.view(*raw_shape, embedding_dim)

        ctx.save_for_backward(local_indices, full_rank_index)
        ctx.group = group
        ctx.sharded = sharded
        ctx.embedding_table_shape = embedding_table.shape
        ctx.embedding_table_dtype = embedding_table.dtype
        ctx.receive_rank_count_list = receive_rank_count_list
        ctx.send_rank_count_list = send_rank_count_list
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_indices, full_rank_index = ctx.saved_tensors
        embedding_dim = ctx.embedding_table_shape[1]

        grad_send_buf = grad_output.reshape(-1, embedding_dim)[full_rank_index].contiguous()
        if ctx.sharded:
            grad_recv_buf = torch.empty(
                sum(ctx.receive_rank_count_list),
                embedding_dim,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            dist.all_to_all_single(
                grad_recv_buf,
                grad_send_buf,
                output_split_sizes=ctx.receive_rank_count_list,
                input_split_sizes=ctx.send_rank_count_list,
                group=ctx.group,
            )
        else:
            grad_recv_buf = grad_send_buf

        # Accumulate in fp32: frequent tokens sum many rows into one.
        grad_embedding_table = torch.zeros(ctx.embedding_table_shape, device=grad_output.device, dtype=torch.float32)
        if grad_recv_buf.numel() > 0:
            grad_embedding_table.index_add_(0, local_indices, grad_recv_buf.float())

        # Gradients for (group, input_tensor, embedding_table)
        return None, None, grad_embedding_table.to(ctx.embedding_table_dtype)


def _all_gather_vocab(weight: torch.Tensor, group: "dist.ProcessGroup", emb_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(weight) for _ in range(emb_size)]
    dist.all_gather(gathered, weight.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


class VocabParallelLinear(torch.autograd.Function):
    """Tied-embedding output projection when the vocab is sharded over the ``emb`` group.

    Symmetric to :class:`AllToAllEmbedding`: each ``emb`` rank owns a contiguous
    vocabulary shard ``weight`` of shape ``[vocab // emb_size, hidden]``. To
    produce full-vocab logits for this rank's tokens, the shards are all-gathered
    over the ``emb`` group (concatenated in rank order to match the ``Shard(0)``
    vocab layout) into the full ``[vocab, hidden]`` weight and projected locally.

    Only the local shard is saved for backward, which gathers it again rather
    than keeping the full table alive between forward and backward. The
    full-vocab weight grad is then reduce-scattered over the ``emb`` group so
    each rank keeps its own chunk's grad, summed over the group's tokens.
    """

    @staticmethod
    def forward(ctx, group: Optional["dist.ProcessGroup"], hidden: torch.Tensor, weight: torch.Tensor):
        emb_size, _ = _group_size_and_rank(group)
        weight_full = _all_gather_vocab(weight, group, emb_size) if emb_size > 1 else weight
        logits = F.linear(hidden, weight_full)

        ctx.save_for_backward(hidden, weight)
        ctx.group = group
        ctx.emb_size = emb_size
        return logits

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        hidden, weight = ctx.saved_tensors
        group, emb_size = ctx.group, ctx.emb_size
        _, needs_hidden_grad, needs_weight_grad = ctx.needs_input_grad

        grad_hidden = grad_weight = None
        if needs_hidden_grad:
            weight_full = _all_gather_vocab(weight, group, emb_size) if emb_size > 1 else weight
            grad_hidden = grad_logits @ weight_full
        if needs_weight_grad:
            gl = grad_logits.reshape(-1, grad_logits.shape[-1])
            h = hidden.reshape(-1, hidden.shape[-1])
            grad_weight_full = gl.transpose(0, 1) @ h  # [vocab, hidden]
            if emb_size > 1:
                grad_weight = torch.empty_like(weight, dtype=grad_weight_full.dtype)
                dist.reduce_scatter_tensor(grad_weight, grad_weight_full.contiguous(), group=group)
            else:
                grad_weight = grad_weight_full

        # Gradients for (group, hidden, weight)
        return None, grad_hidden, grad_weight


__all__ = ["AllToAllEmbedding", "VocabParallelLinear"]
