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

"""All-to-all lookup of a vocab-sharded (``emb``) embedding table.

The op takes this rank's plain ``[vocab // emb_size, hidden]`` rows. The weight
gradient it returns covers every token of the ``emb`` group, but only this
data-parallel replica's tokens: callers run it inside the FSDP2 unshard hooks
of the owning module, which reduce-scatter and scale that gradient like any
other parameter's.

Every collective, backward included, spans the whole ``emb`` group: each rank
must call the op the same number of times in the same order (with empty inputs
if it has no tokens) and backpropagate through all of them, or the others block.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _group_size_and_rank(group: Optional["dist.ProcessGroup"]) -> tuple[int, int]:
    if group is None:
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


class AllToAllEmbedding(torch.autograd.Function):
    """Vocab-parallel embedding via all-to-all token dispatch over the ``emb`` group.

    Each ``emb`` rank owns a contiguous vocabulary shard ``embedding_table`` of
    shape ``[vocab // emb_size, hidden]``. Global token ids are dispatched
    (all-to-all) to their owning rank, looked up locally on that rank's shard,
    then the embeddings are shipped back (all-to-all) and reassembled in input
    order. Backward routes the per-token grads back the same way and index-adds
    them into the local shard, in fp32 over only the rows this step touched.

    Every rank learns in the first exchange whether any rank holds an id outside
    ``[0, vocab)``, and all of them raise before dispatching ids. Bucketing by
    floor division would otherwise send a negative id to bucket ``-1`` that no
    rank collects, and raising on one rank alone would leave the rest blocked.

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
        vocab_size = vocab_size_per_rank * emb_size

        # Bucket each id by the rank that owns its row.
        in_range = (input_flat >= 0) & (input_flat < vocab_size)
        id_rank = torch.where(in_range, input_flat // vocab_size_per_rank, 0)
        full_rank_index = torch.argsort(id_rank, stable=True)
        send_counts = torch.bincount(id_rank, minlength=emb_size)
        # Per destination rank: (id count, "some id of mine is out of range").
        send_meta = torch.stack([send_counts, (~in_range).any().long().expand(emb_size)], dim=1)

        # 1st collective: per-pair counts plus each rank's out-of-range flag.
        if sharded:
            recv_meta = torch.empty_like(send_meta)
            dist.all_to_all_single(recv_meta, send_meta, group=group)
        else:
            recv_meta = send_meta
        send_meta_list, recv_meta_list = torch.stack([send_meta, recv_meta]).tolist()
        if any(bad for _, bad in recv_meta_list):
            bad_ids = input_flat[~in_range]
            span = f"[{int(bad_ids.min())}, {int(bad_ids.max())}]" if bad_ids.numel() > 0 else "all in range"
            raise RuntimeError(
                f"AllToAllEmbedding: token ids must lie in [0, {vocab_size}); "
                f"some rank of the emb group is out of range (this rank's offending ids: {span})."
            )
        send_rank_count_list = [count for count, _ in send_meta_list]
        receive_rank_count_list = [count for count, _ in recv_meta_list]

        # 2nd collective: send each id to its owner.
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

        # Look up the received ids in this rank's shard.
        local_indices = recv_ids - emb_rank * vocab_size_per_rank
        embs = F.embedding(local_indices, embedding_table)

        # 3rd collective: send the embeddings back.
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

        # Restore the input order.
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

        grad_embedding_table = torch.zeros(
            ctx.embedding_table_shape, device=grad_output.device, dtype=ctx.embedding_table_dtype
        )
        if grad_recv_buf.numel() > 0:
            # Accumulate in fp32 (frequent tokens sum many rows into one), but only
            # over touched rows: a full-table fp32 buffer would triple the peak.
            # ``unique`` costs one host sync; ``index_copy_`` needs long indices.
            rows, inverse = torch.unique(local_indices.long(), return_inverse=True)
            row_grads = torch.zeros(rows.numel(), embedding_dim, device=grad_output.device, dtype=torch.float32)
            row_grads.index_add_(0, inverse, grad_recv_buf.float())
            grad_embedding_table.index_copy_(0, rows, row_grads.to(ctx.embedding_table_dtype))

        # Gradients for (group, input_tensor, embedding_table)
        return None, None, grad_embedding_table


__all__ = ["AllToAllEmbedding"]
