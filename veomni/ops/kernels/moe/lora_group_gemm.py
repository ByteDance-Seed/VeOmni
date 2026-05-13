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
"""Triton fused MoE forward with shared MoE-LoRA (Mode 2), non-EP.

Numerics: identical to the eager wrapper at
``veomni.utils.moe_lora.LoraSharedExperts._eager_forward`` modulo
fused/grouped-gemm rounding. The single ``autograd.Function`` mirrors
the layout of the non-LoRA ``MergedFc1TritonFusedMoeExpertFunction``:

* ``MergedFc1TritonFusedLoRAMoeExpertFunction`` — fused experts layout
  (single ``[E, 2I, H]`` fc1 weight, single fused gate+up LoRA pair).
  LoRA delta is added to ``fc1_output`` before chunk/silu, and to
  ``fc2_output``.

LoRA delta math:

* fc1: ``Δfc1 = (S @ A.T) @ B.T * scale``, where ``S`` is the scattered
  hidden state (one row per (token, top-k slot)). The LoRA pair is shared
  across experts but evaluated per scattered row — equivalent to evaluating
  it once on ``hidden_states`` and indexing by ``scatter_index``.
* fc2: ``Δfc2 = (W @ A_dn.T) @ B_dn.T * scale_dn``, where
  ``W = fc1_weighted_output`` (i.e. ``mid * routing_weight``). Both base
  ``down`` and the LoRA delta are linear in ``mid``, so applying the routing
  weight before fc2 is equivalent to weighting the per-expert output
  afterwards (matches eager).

Backward is hand-derived (the underlying ``group_gemm_same_nk`` /
``group_gemm_same_mn`` are leaf triton calls with no autograd integration).
The LoRA parameters get closed-form gradients; the base activations
accumulate the LoRA contribution into ``grad_scatter_output`` /
``grad_fc1_weighted_output`` so the chain through the existing base
backward stays unchanged.

Phase 4 scope:
    * Mode 2 LoRA only: a single 2-D ``A``/``B`` pair per target parameter,
      shared across experts. Per-expert (Mode 1) LoRA needs leading
      ``num_experts`` dims and is left to a future iteration.
    * Non-EP only. EP support comes in Phase 5 via the
      ``EPMergedFc1GroupGemm``-equivalent path.
"""

from __future__ import annotations

import torch

from ....distributed.parallel_state import get_parallel_state
from ._kernels.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from ._kernels.kernel.moe import expert_histogram, moe_gather, moe_scatter


class MergedFc1TritonFusedLoRAMoeExpertFunction(torch.autograd.Function):
    """Fused MoE forward + shared MoE-LoRA (Mode 2), fused experts layout, non-EP.

    Inputs (forward):
        num_experts: ``E``, the global expert count for this layer.
        gate_weights: ``[B*S, topk]`` routing weights per (token, slot).
        expert_index: ``[B*S, topk]`` selected expert ids per (token, slot).
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E, H, I]`` down base weight.
        lora_a_gu: ``[r, H]``  shared LoRA A on fused gate+up.
        lora_b_gu: ``[2I, r]`` shared LoRA B on fused gate+up.
        lora_a_dn: ``[r, I]``  shared LoRA A on down.
        lora_b_dn: ``[H, r]``  shared LoRA B on down.
        lora_scale_gu: scaling for the gate+up LoRA delta (``alpha / r``
            or ``alpha / sqrt(r)`` for rsLoRA).
        lora_scale_dn: scaling for the down LoRA delta.

    Output:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gu,
        lora_b_gu,
        lora_a_dn,
        lora_b_dn,
        lora_scale_gu,
        lora_scale_dn,
    ):
        splits = expert_histogram(expert_index, num_experts)
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
        scatter_output = moe_scatter(hidden_states, scatter_index)  # [T, H]   T = B*S*topk
        cumsum_t = torch.cumsum(splits, dim=0)

        # Base fc1 (group-gemm): [T, 2I]
        fc1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # LoRA fc1 delta on gate+up (shared across experts).
        # F.linear(x, W) computes x @ W.T, so:
        #   tmp_gu = scatter_output @ lora_a_gu.T   shape [T, r]
        #   delta  = tmp_gu @ lora_b_gu.T * scale   shape [T, 2I]
        tmp_gu = torch.nn.functional.linear(scatter_output, lora_a_gu)  # [T, r]
        lora_delta_gu = torch.nn.functional.linear(tmp_gu, lora_b_gu) * lora_scale_gu  # [T, 2I]
        fc1_output = fc1_output + lora_delta_gu

        # Standard fused MoE post-fc1.
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)  # views, no copy
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output  # mid in eager terms — [T, I]

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight  # [T, I]

        # Base fc2 (group-gemm): [T, H]
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # LoRA fc2 delta on down (shared across experts).
        tmp_dn = torch.nn.functional.linear(fc1_weighted_output, lora_a_dn)  # [T, r]
        lora_delta_dn = torch.nn.functional.linear(tmp_dn, lora_b_dn) * lora_scale_dn  # [T, H]
        fc2_output = fc2_output + lora_delta_dn

        expert_output = moe_gather(fc2_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.lora_scale_gu = lora_scale_gu
        ctx.lora_scale_dn = lora_scale_dn
        ctx.save_for_backward(
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            lora_a_gu,
            lora_b_gu,
            lora_a_dn,
            lora_b_dn,
            tmp_gu,
            tmp_dn,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            lora_a_gu,
            lora_b_gu,
            lora_a_dn,
            lora_b_dn,
            tmp_gu,
            tmp_dn,
        ) = ctx.saved_tensors
        scale_gu = ctx.lora_scale_gu
        scale_dn = ctx.lora_scale_dn

        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MoE step 10: undo gather → grad on per-(token,slot) fc2 output.
        grad_fc2_output = moe_scatter(grad_output, scatter_index)  # [T, H]

        # ---- LoRA fc2 backward (closed form). ---------------------------
        # Forward: lora_delta_dn = tmp_dn @ lora_b_dn.T * scale_dn,
        #          tmp_dn        = fc1_weighted_output @ lora_a_dn.T.
        # grad_lora_delta_dn = grad_fc2_output (it was added into fc2_output).
        grad_tmp_dn = torch.nn.functional.linear(grad_fc2_output, lora_b_dn.t()) * scale_dn  # [T, r]
        grad_lora_b_dn = grad_fc2_output.t().to(tmp_dn.dtype) @ tmp_dn * scale_dn  # [H, r]
        grad_lora_a_dn = grad_tmp_dn.t().to(fc1_weighted_output.dtype) @ fc1_weighted_output  # [r, I]
        grad_fc1_weighted_output_lora = torch.nn.functional.linear(grad_tmp_dn, lora_a_dn.t())  # [T, I]

        # MoE step 9 (base) — dgrad of fc2 wrt fc1_weighted_output.
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )  # [T, I]
        grad_fc1_weighted_output = grad_fc1_weighted_output + grad_fc1_weighted_output_lora

        # MoE step 9 (base) — wgrad of fc2.
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MoE step 8: split routing-weight scale through fc1_weighted_output = fc1_activation * sgw.
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # MoE step 7: chain through silu(gate) * up.
        # Recompute silu output to save memory (matches existing function).
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)  # [T, 2I]

        # ---- LoRA fc1 backward (closed form). ---------------------------
        # Forward: lora_delta_gu = tmp_gu @ lora_b_gu.T * scale_gu,
        #          tmp_gu        = scatter_output @ lora_a_gu.T.
        # grad_lora_delta_gu = grad_fc1_output (added into fc1_output before chunk).
        grad_tmp_gu = torch.nn.functional.linear(grad_fc1_output, lora_b_gu.t()) * scale_gu  # [T, r]
        grad_lora_b_gu = grad_fc1_output.t().to(tmp_gu.dtype) @ tmp_gu * scale_gu  # [2I, r]
        grad_lora_a_gu = grad_tmp_gu.t().to(scatter_output.dtype) @ scatter_output  # [r, H]
        grad_scatter_output_lora = torch.nn.functional.linear(grad_tmp_gu, lora_a_gu.t())  # [T, H]

        # MoE step 4 (base) — single dgrad for merged fc1.
        grad_scatter_output = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )
        grad_scatter_output = grad_scatter_output + grad_scatter_output_lora

        # MoE step 4 (base) — single wgrad for merged fc1.
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=scatter_output,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MoE step 3.
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
            grad_lora_a_gu,  # lora_a_gu
            grad_lora_b_gu,  # lora_b_gu
            grad_lora_a_dn,  # lora_a_dn
            grad_lora_b_dn,  # lora_b_dn
            None,  # lora_scale_gu
            None,  # lora_scale_dn
        )


def group_gemm_fused_lora_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    lora_a_gate_up: torch.Tensor,
    lora_b_gate_up: torch.Tensor,
    lora_a_down: torch.Tensor,
    lora_b_down: torch.Tensor,
    lora_scale_gate_up: float,
    lora_scale_down: float,
) -> torch.Tensor:
    """Triton grouped-gemm fused MoE forward with shared MoE-LoRA (Mode 2).

    Args:
        num_experts: number of experts ``E`` in this MoE layer.
        routing_weights: ``[B*S, topk]`` per-(token, slot) routing weights.
        selected_experts: ``[B*S, topk]`` per-(token, slot) selected expert ids.
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E, H, I]`` down base weight.
        lora_a_gate_up: ``[r, H]``  shared LoRA A on fused gate+up.
        lora_b_gate_up: ``[2I, r]`` shared LoRA B on fused gate+up.
        lora_a_down: ``[r, I]``  shared LoRA A on down.
        lora_b_down: ``[H, r]``  shared LoRA B on down.
        lora_scale_gate_up: scaling for the gate+up LoRA delta.
        lora_scale_down: scaling for the down LoRA delta.

    Returns:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.

    Constraints in this phase:
        * Mode 2 LoRA only — single 2-D ``A``/``B`` pair per target parameter,
          shared across experts.
        * Non-EP only. ``get_parallel_state().ep_enabled`` is checked here and
          raises ``NotImplementedError``; EP comes in Phase 5.
    """
    if get_parallel_state().ep_enabled:
        raise NotImplementedError(
            "group_gemm_fused_lora_moe_forward: expert parallelism (EP) support is not "
            "implemented yet (Phase 5). Disable EP or fall back to eager for now."
        )
    return MergedFc1TritonFusedLoRAMoeExpertFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate_up,
        lora_b_gate_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate_up,
        lora_scale_down,
    )
