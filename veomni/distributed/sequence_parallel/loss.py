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

from .comm import (
    get_unified_sequence_parallel_group,
    get_unified_sequence_parallel_world_size,
)


class ReduceLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.Function, loss: torch.Tensor, num_valid_tokens: torch.Tensor) -> torch.Tensor:
        loss = torch.where(num_valid_tokens > 0, loss, torch.zeros_like(loss))

        local_num_tokens = num_valid_tokens.detach().clone()
        loss *= num_valid_tokens
        group = get_unified_sequence_parallel_group()
        dist.all_reduce(loss, group=group)
        dist.all_reduce(num_valid_tokens, group=group)
        ctx.save_for_backward(local_num_tokens, num_valid_tokens)

        # FIX: When ALL ranks in the SP group have zero valid tokens,
        # global num_valid_tokens = 0 after all_reduce, causing 0/0 = NaN.
        # This NaN propagates through element_mul_kernel in Liger backward,
        # corrupting the entire model via FSDP all-reduce.
        # Return zero loss instead to safely skip this micro-batch.
        return loss / num_valid_tokens.clamp_min(1)

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        local_num_tokens, global_num_tokens = ctx.saved_tensors

        # FIX: Mirror the forward guard — zero grad when global tokens = 0,
        # preventing NaN grad_output from corrupting downstream parameters.
        grad_output = (
            get_unified_sequence_parallel_world_size()
            * local_num_tokens
            * grad_output
            / global_num_tokens.clamp(min=1)
        )
        return grad_output, None


def reduce_sequence_parallel_loss(loss: torch.Tensor, num_valid_tokens: torch.Tensor) -> torch.Tensor:
    return ReduceLoss.apply(loss, num_valid_tokens)
