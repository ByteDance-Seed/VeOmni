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

"""Tied output projection of a vocab-sharded (``emb``) embedding table.

Like :mod:`.all_to_all`, the op takes this rank's plain ``[vocab // emb_size,
hidden]`` rows and runs inside the FSDP2 unshard hooks of the owning module,
which reduce-scatter and scale the weight gradient it returns. Every
collective, backward included, spans the whole ``emb`` group.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .all_to_all import _group_size_and_rank


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

    Forward and backward each materialize the full ``[vocab, hidden]`` weight,
    and backward its full gradient too: sized for a text vocabulary, not for a
    table too large to gather on one device.
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


__all__ = ["VocabParallelLinear"]
