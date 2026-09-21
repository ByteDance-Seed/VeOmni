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

"""NPU fused MoE-LoRA kernels (owned by the LoRA stack).

The NPU counterpart of :mod:`veomni.lora.ops.moe_group_gemm` (the Triton
implementation). It lives here — next to the Triton LoRA kernel — so both
backends' LoRA-aware fused MoE forwards sit under ``veomni.lora.ops`` and are
bound uniformly by :func:`veomni.lora.ops.bind_lora_moe_kernels`.

Unlike the Triton kernel (which re-implements the whole EP dispatch/combine +
grouped-gemm with hand-written ``autograd.Function`` backwards), the NPU path
*composes* the base NPU MoE primitives that already live in
``veomni.ops.kernels.moe.npu_group_gemm``:

  * ``dispatch_preprocess`` / ``alltoall_dispatch`` / ``alltoall_combine`` —
    the exact all-to-all EP dispatch/combine the base fused MoE uses, and
  * ``npu_group_gemm`` — the grouped-matmul ``autograd.Function`` (its own
    backward),

then injects three seed-style LoRA deltas (gate, up, down). Backward is derived
by autograd (no hand-written kernel), because every op in the composition is
differentiable. MoE-LoRA and expert parallelism are orthogonal: the EP handling
is identical to the base kernel, the only extra work is the LoRA delta on the
*dispatched* tokens using the *local* (already EP-sharded) LoRA weights.

Two LoRA layouts (mirroring ``veomni.lora.moe_layers``):
  * shared (Mode 2, ``LoraSharedExperts``)     — one 2-D LoRA pair reused by
    every expert ⇒ the delta is a plain matmul over *all* dispatched tokens
    (expert-independent), so no per-expert grouping is needed.
  * independent (Mode 1, ``LoraIndependentExperts``) — a 3-D per-expert LoRA
    pair ⇒ the delta is a grouped-gemm over the same per-expert token groups
    the base experts use, driven by the same ``group_list`` (token counts).

The eager reference these must match line-for-line lives in
``veomni.lora.moe_layers.{LoraSharedExperts,LoraIndependentExperts}._eager_forward``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from ...distributed.parallel_state import get_parallel_state
from ...ops.kernels.moe.npu_group_gemm import (
    alltoall_combine,
    alltoall_dispatch,
    dispatch_preprocess,
    npu_group_gemm,
)


def _lora_gate_up_delta(
    x: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    group_list: torch.Tensor,
    independent: bool,
) -> torch.Tensor:
    """Seed-style gate+up LoRA delta on dispatched tokens ``x`` (``[N, H]``).

    Returns ``[N, 2I]`` (gate half then up half) so it can be added straight
    onto the base ``gate_up`` group-gemm output before ``npu_swiglu``. The two
    halves keep separate rank-r adapters (see ``moe_layers`` for why).

    ``group_list`` is the per-(local-)expert token count driving the grouped
    gemm — identical to the count the base experts use, so the LoRA groups line
    up with the base groups token-for-token.
    """
    if independent:
        # 3-D per-expert LoRA: ``npu_group_gemm(x, W, counts)`` computes the
        # per-expert ``x @ W`` (base kernel passes ``W.transpose(1, 2)`` to
        # realise ``F.linear``, so we do the same for A/B).
        gate = npu_group_gemm(
            npu_group_gemm(x, lora_a_gate.transpose(1, 2), group_list),
            lora_b_gate.transpose(1, 2),
            group_list,
        )
        up = npu_group_gemm(
            npu_group_gemm(x, lora_a_up.transpose(1, 2), group_list),
            lora_b_up.transpose(1, 2),
            group_list,
        )
    else:
        # 2-D shared LoRA: identical for every expert ⇒ plain matmul over all
        # dispatched tokens, no grouping.
        gate = F.linear(F.linear(x, lora_a_gate), lora_b_gate)
        up = F.linear(F.linear(x, lora_a_up), lora_b_up)
    return torch.cat([gate * lora_scale_gate, up * lora_scale_up], dim=-1)


def _lora_down_delta(
    mid: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_down: float,
    group_list: torch.Tensor,
    independent: bool,
) -> torch.Tensor:
    """Down-projection LoRA delta on the post-SwiGLU intermediate ``mid`` (``[N, I]`` -> ``[N, H]``)."""
    if independent:
        down = npu_group_gemm(
            npu_group_gemm(mid, lora_a_down.transpose(1, 2), group_list),
            lora_b_down.transpose(1, 2),
            group_list,
        )
    else:
        down = F.linear(F.linear(mid, lora_a_down), lora_b_down)
    return down * lora_scale_down


def _npu_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
    independent: bool,
) -> torch.Tensor:
    """NPU single-device (non-EP) fused MoE forward + seed-style two-LoRA."""
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    # ``selected_experts`` is int64 (topk). ``npu_moe_token_permute`` wants
    # int32 indices; ``npu_grouped_matmul``'s ``groupList`` wants int64 counts
    # (int32 raises EZ1001). ``torch.histc`` only accepts float input and
    # returns float, so bin on float32 then cast counts to int64.
    selected_experts = selected_experts.to(torch.int32)
    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts)
    tokens_per_expert = torch.histc(selected_experts.to(torch.float32), bins=num_experts, min=0, max=num_experts).to(
        torch.int64
    )

    gate_up = npu_group_gemm(permuted_hidden_states, fc1_1_2_weight.transpose(1, 2), tokens_per_expert)
    gate_up = gate_up + _lora_gate_up_delta(
        permuted_hidden_states,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_scale_gate,
        lora_scale_up,
        tokens_per_expert,
        independent,
    )
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), tokens_per_expert)
    output = output + _lora_down_delta(
        intermediate, lora_a_down, lora_b_down, lora_scale_down, tokens_per_expert, independent
    )
    return torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)


def _npu_ep_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
    independent: bool,
    ep_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """NPU expert-parallel fused MoE forward + seed-style two-LoRA.

    EP dispatch/combine is identical to
    :func:`veomni.ops.kernels.moe.npu_group_gemm.npu_ep_fused_moe_forward`; the
    LoRA deltas are added on the dispatched local tokens with the local (already
    EP-sharded for independent, replicated for shared) LoRA weights, driven by
    ``num_global_sum_tokens_per_local_expert`` — the same per-local-expert token
    count the base group-gemm uses.
    """
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    # NPU expert routing/indexing (dispatch_preprocess + alltoall_dispatch)
    # expects int32; ``selected_experts`` is int64 (topk), so cast up front to
    # match the non-EP path and avoid dtype mismatches in the dispatch.
    selected_experts = selected_experts.to(torch.int32)
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    hidden_states, unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )

    # ``npu_grouped_matmul`` requires int64 groupList (same as non-EP path).
    group_list = num_global_sum_tokens_per_local_expert.to(torch.int64)
    gate_up = npu_group_gemm(hidden_states, fc1_1_2_weight.transpose(1, 2), group_list)
    gate_up = gate_up + _lora_gate_up_delta(
        hidden_states,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_scale_gate,
        lora_scale_up,
        group_list,
        independent,
    )
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = npu_group_gemm(intermediate, fc2_weight.transpose(1, 2), group_list)
    output = output + _lora_down_delta(
        intermediate, lora_a_down, lora_b_down, lora_scale_down, group_list, independent
    )

    output = alltoall_combine(
        output,
        routing_weights,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
    return output


def _npu_lora_moe_dispatch(independent: bool, **kwargs) -> torch.Tensor:
    """Route to the EP or non-EP NPU LoRA MoE forward based on parallel state."""
    if get_parallel_state().ep_enabled:
        return _npu_ep_fused_lora_moe_forward(
            independent=independent, ep_group=get_parallel_state().ep_group, **kwargs
        )
    return _npu_fused_lora_moe_forward(independent=independent, **kwargs)


def npu_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> torch.Tensor:
    """NPU Mode-2 (shared) fused MoE-LoRA forward.

    Signature matches ``veomni.lora.ops.fused_lora_moe_forward`` /
    ``group_gemm_fused_lora_moe_forward`` so it is a drop-in ``triton``
    replacement bound by :func:`veomni.lora.ops.bind_lora_moe_kernels`.
    """
    return _npu_lora_moe_dispatch(
        independent=False,
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )


def npu_fused_independent_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate: torch.Tensor,
    lora_b_gate: torch.Tensor,
    lora_a_up: torch.Tensor,
    lora_b_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> torch.Tensor:
    """NPU Mode-1 (independent per-expert) fused MoE-LoRA forward.

    Same shape contract as :func:`npu_fused_lora_moe_forward` except every LoRA
    tensor carries a leading expert dim (``[E, r, ...]`` / ``[E, ..., r]``).
    Drop-in replacement for ``group_gemm_fused_independent_lora_moe_forward``.
    """
    return _npu_lora_moe_dispatch(
        independent=True,
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_2_weight=fc1_1_2_weight,
        fc2_weight=fc2_weight,
        lora_a_gate=lora_a_gate,
        lora_b_gate=lora_b_gate,
        lora_a_up=lora_a_up,
        lora_b_up=lora_b_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
        lora_scale_gate=lora_scale_gate,
        lora_scale_up=lora_scale_up,
        lora_scale_down=lora_scale_down,
    )
