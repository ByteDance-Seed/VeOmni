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

"""NPU situ fused MoE forward (Kimi-K3 SituGLU experts).

A sibling of ``npu_group_gemm.py``'s swiglu fused MoE where the activation
between the two group gemms is Kimi-K3's SituGLU
(``beta * tanh(gate/beta) * sigmoid(gate) * up``) instead of SwiGLU. Both the
single-device and the expert-parallel paths are provided; the token-permute /
group-gemm helpers are reused from
``npu_group_gemm`` so the numerics and scheduling stay identical to the
standard fused MoE.
"""

from typing import Optional

import torch
import torch.distributed as dist

from .....distributed.moe import cumsum_to_counts, dispatch_to_ep_class
from .....distributed.parallel_state import get_parallel_state
from .....utils.device import IS_NPU_AVAILABLE
from ...situ_and_mul.kimi_k3.triton import situ_and_mul
from ..standard.npu import npu_group_gemm


# NPU-only op: guard torch_npu so the module stays importable on non-NPU hosts
# (e.g. GPU CI collecting tests); the forward is only selected on NPU anyway.
if IS_NPU_AVAILABLE:
    import torch_npu


def _situ(x: torch.Tensor, beta: float, linear_beta: Optional[float]) -> torch.Tensor:
    return situ_and_mul(x, beta=beta, linear_beta=linear_beta)


def _npu_situ_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """NPU single-device situ fused MoE forward pass (non-EP)."""
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, fc1_weight, tokens_per_expert)
    intermediate_activations = _situ(intermediate_hidden_states, beta, linear_beta)
    output = npu_group_gemm(intermediate_activations, fc2_weight.transpose(1, 2), tokens_per_expert)
    hidden_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return hidden_states


def _npu_situ_expert_math(permute_tokens, cumsum, fc1_weight, fc2_weight, beta=1.0, linear_beta=None):
    """Two Ascend grouped GEMMs with SituGLU between them.

    Takes the shape ``dispatch_to_ep_class`` provides, so the all-to-all plumbing is
    shared with every other fused row; only the per-rank math is this variant's
    (mirrors ``standard/npu._npu_expert_math`` with SituGLU in place of SwiGLU).
    ``npu_group_gemm`` pins ``group_list_type=1`` -- per-group sizes -- so the
    cumulative ``cumsum`` is converted back to counts first.
    """
    counts = cumsum_to_counts(cumsum)
    intermediate = npu_group_gemm(permute_tokens, fc1_weight, counts)
    activated = _situ(intermediate, beta, linear_beta)
    return npu_group_gemm(activated, fc2_weight, counts)


def npu_situ_ep_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    ep_group: Optional[dist.ProcessGroup] = None,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """NPU expert-parallel situ fused MoE forward pass.

    The all-to-all plumbing is shared with every other fused row through
    ``dispatch_to_ep_class`` (which also applies the routing weights on the
    combined tokens); only the per-rank math is Ascend's SituGLU group GEMM.
    """
    if fc1_1_2_weight is not None:
        fc1_weight = fc1_1_2_weight
    else:
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)
    fc1_weight = fc1_weight.transpose(1, 2)
    return dispatch_to_ep_class(
        _npu_situ_expert_math,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_weight,
        fc2_weight.transpose(1, 2),
        beta,
        linear_beta,
    )


def npu_situ_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    beta: float = 1.0,
    linear_beta: float | None = None,
):
    """Dispatch to the EP or single-device situ fused MoE path.

    ``num_experts`` is the *global* expert count; with EP enabled the weight
    tensors hold this rank's ``num_experts / ep_size`` local experts.
    """
    if get_parallel_state().ep_enabled:
        final_hidden_states = npu_situ_ep_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            ep_group=get_parallel_state().ep_group,
            beta=beta,
            linear_beta=linear_beta,
        )
    else:
        final_hidden_states = _npu_situ_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
            beta=beta,
            linear_beta=linear_beta,
        )
    return final_hidden_states
