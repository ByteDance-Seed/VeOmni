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

"""Vocab-parallel (``emb``) embedding ops."""

import torch
import torch.distributed as dist


class AllToAllEmbedding(torch.autograd.Function):
    """Vocab-parallel embedding via all-to-all token dispatch over the ``emb`` group.

    Each ``emb`` rank owns a contiguous vocabulary shard ``embedding_table`` of
    shape ``[vocab // emb_size, hidden]``. Global token ids are dispatched
    (all-to-all) to their owning rank, looked up locally on that rank's shard,
    then the embeddings are shipped back (all-to-all) and reassembled in input
    order. Backward routes the per-token grads back the same way and index-adds
    them into the local shard — so the shard's gradient is complete after the
    backward all-to-all (no separate all-reduce / FSDP reduce-scatter needed).
    """

    @staticmethod
    def forward(ctx, group: "dist.ProcessGroup", input_tensor: torch.Tensor, embedding_table: torch.Tensor):
        emb_size = dist.get_world_size(group) if group else 1
        emb_rank = dist.get_rank(group) if group else 0

        vocab_size_per_rank = embedding_table.shape[0]
        start_id = emb_rank * vocab_size_per_rank

        _raw_shape = input_tensor.shape
        embedding_dim = embedding_table.shape[1]
        input_flat = input_tensor.reshape(-1)
        num_input_ids = input_flat.shape[0]

        # --- Dispatching logic: which rank owns each id ---
        id_rank = torch.clamp_max(input_flat // vocab_size_per_rank, emb_size - 1)
        index = torch.arange(num_input_ids, device=input_flat.device)
        rank_index = [index[id_rank == i] for i in range(emb_size)]
        send_rank_count_list = [int(len(ri)) for ri in rank_index]
        full_rank_index = torch.cat(rank_index, dim=0)

        # --- 1st collective: exchange per-pair counts ---
        send_rank_count = torch.tensor(send_rank_count_list, device=input_flat.device)
        all_send_rank_count = [torch.zeros_like(send_rank_count) for _ in range(emb_size)]
        if group:
            dist.all_gather(all_send_rank_count, send_rank_count, group=group)
        all_send_rank_count = torch.stack(all_send_rank_count, dim=0)
        receive_rank_count_list = [int(x) for x in all_send_rank_count[:, emb_rank].tolist()]

        # --- 2nd collective: exchange token ids ---
        send_ids = input_flat[full_rank_index].contiguous()
        recv_ids = torch.empty(sum(receive_rank_count_list), dtype=input_flat.dtype, device=input_flat.device)
        if group:
            dist.all_to_all_single(
                recv_ids,
                send_ids,
                output_split_sizes=receive_rank_count_list,
                input_split_sizes=send_rank_count_list,
                group=group,
            )

        # --- Local lookup on this rank's shard ---
        local_indices = recv_ids - start_id
        # Synchronous bounds check: surface a clear Python error (with the actual
        # numbers) instead of an async CUDA "gather index out of bounds" assert.
        if local_indices.numel() > 0:
            lo = int(local_indices.min())
            hi = int(local_indices.max())
            if lo < 0 or hi >= vocab_size_per_rank:
                raise RuntimeError(
                    f"AllToAllEmbedding: local index out of range [0,{vocab_size_per_rank}) on "
                    f"emb_rank={emb_rank}/{emb_size}: min={lo}, max={hi}, start_id={start_id}, "
                    f"recv_ids=[{int(recv_ids.min())},{int(recv_ids.max())}], "
                    f"input_ids=[{int(input_flat.min())},{int(input_flat.max())}]."
                )
        embs = torch.nn.functional.embedding(local_indices, embedding_table)

        # --- 3rd collective: ship looked-up embeddings back ---
        embs_recv = torch.empty(num_input_ids, embedding_dim, dtype=embs.dtype, device=embs.device)
        if group:
            dist.all_to_all_single(
                embs_recv,
                embs.contiguous(),
                output_split_sizes=send_rank_count_list,
                input_split_sizes=receive_rank_count_list,
                group=group,
            )

        # --- Reassemble to original input order ---
        output = torch.empty(num_input_ids, embedding_dim, dtype=embs.dtype, device=embs.device)
        if num_input_ids > 0:
            output[full_rank_index] = embs_recv
        output = output.view(*_raw_shape, embedding_dim)

        ctx.save_for_backward(local_indices, full_rank_index)
        ctx.group = group
        ctx.embedding_table_shape = embedding_table.shape
        ctx.receive_rank_count_list = receive_rank_count_list
        ctx.send_rank_count_list = send_rank_count_list
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_indices, full_rank_index = ctx.saved_tensors
        group = ctx.group
        embedding_table_shape = ctx.embedding_table_shape
        receive_rank_count_list = ctx.receive_rank_count_list
        send_rank_count_list = ctx.send_rank_count_list
        embedding_dim = embedding_table_shape[1]

        grad_output_flat = grad_output.reshape(-1, embedding_dim)
        grad_send_buf = grad_output_flat[full_rank_index].contiguous()
        grad_recv_buf = torch.empty(
            sum(receive_rank_count_list),
            embedding_dim,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        if group:
            dist.all_to_all_single(
                grad_recv_buf,
                grad_send_buf,
                output_split_sizes=receive_rank_count_list,
                input_split_sizes=send_rank_count_list,
                group=group,
            )

        grad_embedding_table = torch.zeros(embedding_table_shape, device=grad_output.device, dtype=grad_output.dtype)
        if grad_recv_buf.numel() > 0:
            grad_embedding_table.index_add_(0, local_indices, grad_recv_buf)

        # Gradients for (group, input_tensor, embedding_table)
        return None, None, grad_embedding_table


class VocabParallelLinear(torch.autograd.Function):
    """Tied-embedding output projection when the vocab is sharded over the ``emb`` group.

    Symmetric to :class:`AllToAllEmbedding`: each ``emb`` rank owns a contiguous
    vocabulary shard ``weight`` of shape ``[vocab // emb_size, hidden]`` (the
    hidden dim is already reconstructed by the caller's ``full_tensor()``). To
    produce full-vocab logits for this rank's tokens, the shards are all-gathered
    over the ``emb`` group (concatenated in rank order to match the ``Shard(0)``
    vocab layout) into the full ``[vocab, hidden]`` weight and projected locally.

    Backward computes the full-vocab weight grad from this rank's tokens, then
    reduce-scatters it over the ``emb`` group so each rank keeps only its own
    chunk's (cross-rank summed) grad — mirroring the embedding's grad routing, so
    the local shard's gradient is complete without a separate all-reduce.
    """

    @staticmethod
    def forward(ctx, group: "dist.ProcessGroup", hidden: torch.Tensor, weight: torch.Tensor):
        emb_size = dist.get_world_size(group) if group else 1
        if group and emb_size > 1:
            gathered = [torch.empty_like(weight) for _ in range(emb_size)]
            dist.all_gather(gathered, weight.contiguous(), group=group)
            weight_full = torch.cat(gathered, dim=0)
        else:
            weight_full = weight

        logits = torch.nn.functional.linear(hidden, weight_full)

        ctx.save_for_backward(hidden, weight_full)
        ctx.group = group
        ctx.emb_size = emb_size
        ctx.vocab_local = weight.shape[0]
        return logits

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        hidden, weight_full = ctx.saved_tensors
        group = ctx.group
        emb_size = ctx.emb_size

        gl = grad_logits.reshape(-1, grad_logits.shape[-1])
        h = hidden.reshape(-1, hidden.shape[-1])

        grad_hidden = (gl @ weight_full).view_as(hidden)
        grad_weight_full = gl.transpose(0, 1) @ h  # [vocab, hidden]

        if group and emb_size > 1:
            grad_weight_local = torch.empty(
                ctx.vocab_local,
                grad_weight_full.shape[1],
                dtype=grad_weight_full.dtype,
                device=grad_weight_full.device,
            )
            dist.reduce_scatter_tensor(grad_weight_local, grad_weight_full.contiguous(), group=group)
        else:
            grad_weight_local = grad_weight_full

        # Gradients for (group, hidden, weight)
        return None, grad_hidden, grad_weight_local


__all__ = ["AllToAllEmbedding", "VocabParallelLinear"]
