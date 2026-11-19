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


from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .comm import get_unified_sequence_parallel_group


class ReduceLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        loss: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        group=None,
    ) -> torch.Tensor:
        # ``group`` defaults to the comm-global unified SP group (single-model
        # path). SeedOmni V2 passes each module's OWN SP group so heterogeneous
        # per-module SP reduces over the right ranks (the comm-global unified
        # group is just the last-built module's and would be wrong here).
        if group is None:
            group = get_unified_sequence_parallel_group()
        loss = torch.where(num_valid_tokens > 0, loss, torch.zeros_like(loss))

        local_num_tokens = num_valid_tokens.detach().clone()
        loss *= num_valid_tokens
        dist.all_reduce(loss, group=group)
        dist.all_reduce(num_valid_tokens, group=group)
        ctx.save_for_backward(local_num_tokens, num_valid_tokens)
        ctx.sp_world_size = dist.get_world_size(group) if group else 1

        # FIX: When ALL ranks in the SP group have zero valid tokens,
        # global num_valid_tokens = 0 after all_reduce, causing 0/0 = NaN.
        # This NaN propagates through element_mul_kernel in Liger backward,
        # corrupting the entire model via FSDP all-reduce.
        # Return zero loss instead to safely skip this micro-batch.
        return loss / num_valid_tokens.clamp_min(1)

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        local_num_tokens, global_num_tokens = ctx.saved_tensors

        # FIX: Mirror the forward guard — zero grad when global tokens = 0,
        # preventing NaN grad_output from corrupting downstream parameters.
        grad_output = ctx.sp_world_size * local_num_tokens * grad_output / global_num_tokens.clamp(min=1)
        return grad_output, None, None


def reduce_sequence_parallel_loss(loss: torch.Tensor, num_valid_tokens: torch.Tensor, group=None) -> torch.Tensor:
    return ReduceLoss.apply(loss, num_valid_tokens, group)


class SumOverShards(torch.autograd.Function):
    """Report the whole group's sum; leave each rank differentiating only its own part.

    For a quantity that is a sum over tokens -- a loss sum, or a per-expert
    statistic -- the shards already add up to the unsharded value, so no
    reweighting is needed, only the addition.

    The gradient is the part that is not free. FSDP shards over ``dp_shard_sp``
    and *averages* over it, so ``sp_size`` ranks each holding one shard's
    derivative would deliver ``1/sp_size`` of the total. Scaling here cancels
    that, which is the same factor ``reduce_sequence_parallel_loss`` applies to a
    token-mean loss.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, sp_size: int) -> torch.Tensor:
        ctx.sp_size = sp_size
        total = tensor.detach().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
        return total

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.sp_size, None, None


def sum_over_ulysses_shards(tensor: torch.Tensor) -> torch.Tensor:
    """``SumOverShards`` over the Ulysses group; the identity when Ulysses is off.

    Reads the ``ulysses`` mesh dimension, not ``sp``: ``sp`` is a flattened
    dimension that only resolves on the thread that flattened it, so it fails in
    a backward replayed by gradient checkpointing.
    """
    from ..parallel_state import get_parallel_state

    state = get_parallel_state()
    if not state.ulysses_enabled:
        return tensor
    return SumOverShards.apply(tensor, state.ulysses_group, state.ulysses_size)
