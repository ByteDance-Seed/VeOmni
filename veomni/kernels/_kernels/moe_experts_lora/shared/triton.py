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
# See the License for the specific language governing limitations
# under the License.

"""shared MoE-LoRA Triton implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state
from ...moe_experts.shared.dispatch import expert_histogram, moe_gather, moe_scatter
from ...moe_experts.shared.group_gemm import group_gemm_same_mn, group_gemm_same_nk


class MergedFc1TritonFusedLoRAMoeExpertFunction(torch.autograd.Function):
    """Fused MoE forward + shared seed-style two-LoRA on fused gate_up (Mode 2), non-EP.

    Inputs (forward):
        num_experts: ``E``, the global expert count for this layer.
        gate_weights: ``[B*S, topk]`` routing weights per (token, slot).
        expert_index: ``[B*S, topk]`` selected expert ids per (token, slot).
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E, H, I]`` down base weight.
        lora_a_gate / lora_b_gate: shared LoRA pair on the gate half
            (``[r, H]`` / ``[I, r]``).
        lora_a_up / lora_b_up: shared LoRA pair on the up half
            (``[r, H]`` / ``[I, r]``).
        lora_a_down / lora_b_down: shared LoRA pair on down
            (``[r, I]`` / ``[H, r]``).
        lora_scale_gate / lora_scale_up / lora_scale_down: per-spec scaling
            (typically all equal to ``alpha / r`` or ``alpha / sqrt(r)``;
            kept separate so per-spec scales are future-proof).

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
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
    ):
        splits = expert_histogram(expert_index, num_experts)
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
        scatter_output = moe_scatter(hidden_states, scatter_index)  # [T, H]   T = B*S*topk
        cumsum_t = torch.cumsum(splits, dim=0)
        max_t = scatter_output.shape[0]

        # Base fc1 (group-gemm): [T, 2I]
        fc1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=max_t,
            transpose_a=False,
            transpose_b=True,
        )

        # Two independent shared LoRA deltas on gate / up halves.
        # F.linear(x, W) computes x @ W.T, so each chain is:
        #   tmp_side = scatter_output @ lora_a_side.T          [T, r]
        #   delta_side = tmp_side    @ lora_b_side.T * scale    [T, I]
        tmp_gate = torch.nn.functional.linear(scatter_output, lora_a_gate)  # [T, r]
        delta_gate = torch.nn.functional.linear(tmp_gate, lora_b_gate) * lora_scale_gate  # [T, I]
        tmp_up = torch.nn.functional.linear(scatter_output, lora_a_up)  # [T, r]
        delta_up = torch.nn.functional.linear(tmp_up, lora_b_up) * lora_scale_up  # [T, I]
        # Cat per-half deltas into [T, 2I] and add into fc1_output *before*
        # chunk + silu — LoRA must enter the pre-activation linear.
        fc1_output = fc1_output + torch.cat([delta_gate, delta_up], dim=-1)

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
            max_M=max_t,
            transpose_a=False,
            transpose_b=True,
        )

        # LoRA fc2 delta on down (shared across experts).
        tmp_down = torch.nn.functional.linear(fc1_weighted_output, lora_a_down)  # [T, r]
        lora_delta_down = torch.nn.functional.linear(tmp_down, lora_b_down) * lora_scale_down  # [T, H]
        fc2_output = fc2_output + lora_delta_down

        expert_output = moe_gather(fc2_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.lora_scale_gate = lora_scale_gate
        ctx.lora_scale_up = lora_scale_up
        ctx.lora_scale_down = lora_scale_down
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
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
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
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        ) = ctx.saved_tensors
        scale_gate = ctx.lora_scale_gate
        scale_up = ctx.lora_scale_up
        scale_down = ctx.lora_scale_down

        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)
        max_t = grad_output.shape[0]

        # MoE step 10: undo gather → grad on per-(token,slot) fc2 output.
        grad_fc2_output = moe_scatter(grad_output, scatter_index)  # [T, H]

        # ---- LoRA fc2 backward (closed form). ---------------------------
        # Forward: lora_delta_down = tmp_down @ lora_b_down.T * scale_down,
        #          tmp_down        = fc1_weighted_output @ lora_a_down.T.
        # grad_lora_delta_down = grad_fc2_output (it was added into fc2_output).
        grad_tmp_down = torch.nn.functional.linear(grad_fc2_output, lora_b_down.t()) * scale_down  # [T, r]
        grad_lora_b_down = grad_fc2_output.t().to(tmp_down.dtype) @ tmp_down * scale_down  # [H, r]
        grad_lora_a_down = grad_tmp_down.t().to(fc1_weighted_output.dtype) @ fc1_weighted_output  # [r, I]
        grad_fc1_weighted_output_lora = torch.nn.functional.linear(grad_tmp_down, lora_a_down.t())  # [T, I]

        # MoE step 9 (base) — dgrad of fc2 wrt fc1_weighted_output.
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=max_t,
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
                max_K=max_t,
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

        # ---- LoRA fc1 backward (per-half closed form). ------------------
        # ``grad_fc1_output`` was the gradient flowing into the merged
        # ``fc1_output`` *after* the LoRA add ⇒ each half-slice is the
        # gradient flowing into the matching ``delta_side`` chain. chunk
        # gives non-contiguous views into the [T, 2I] block; matmul handles
        # those, but we materialise contiguous copies for the wgrad
        # transpose-then-matmul to avoid pathological strides.
        grad_delta_gate, grad_delta_up = grad_fc1_output.chunk(2, dim=-1)
        grad_delta_gate = grad_delta_gate.contiguous()
        grad_delta_up = grad_delta_up.contiguous()

        # Gate half.
        grad_tmp_gate = torch.nn.functional.linear(grad_delta_gate, lora_b_gate.t()) * scale_gate  # [T, r]
        grad_lora_b_gate = grad_delta_gate.t().to(tmp_gate.dtype) @ tmp_gate * scale_gate  # [I, r]
        grad_lora_a_gate = grad_tmp_gate.t().to(scatter_output.dtype) @ scatter_output  # [r, H]
        grad_scatter_output_gate = torch.nn.functional.linear(grad_tmp_gate, lora_a_gate.t())  # [T, H]

        # Up half.
        grad_tmp_up = torch.nn.functional.linear(grad_delta_up, lora_b_up.t()) * scale_up  # [T, r]
        grad_lora_b_up = grad_delta_up.t().to(tmp_up.dtype) @ tmp_up * scale_up  # [I, r]
        grad_lora_a_up = grad_tmp_up.t().to(scatter_output.dtype) @ scatter_output  # [r, H]
        grad_scatter_output_up = torch.nn.functional.linear(grad_tmp_up, lora_a_up.t())  # [T, H]

        # MoE step 4 (base) — single dgrad for merged fc1, accumulated with
        # the per-half LoRA contributions so the input gradient is right.
        grad_scatter_output = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum_t,
            max_M=max_t,
            transpose_b=False,
        )
        grad_scatter_output = grad_scatter_output + grad_scatter_output_gate + grad_scatter_output_up

        # MoE step 4 (base) — single wgrad for merged fc1.
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=scatter_output,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum_t,
                max_K=max_t,
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
            grad_lora_a_gate,  # lora_a_gate
            grad_lora_b_gate,  # lora_b_gate
            grad_lora_a_up,  # lora_a_up
            grad_lora_b_up,  # lora_b_up
            grad_lora_a_down,  # lora_a_down
            grad_lora_b_down,  # lora_b_down
            None,  # lora_scale_gate
            None,  # lora_scale_up
            None,  # lora_scale_down
        )


# ──────────────────────────────────────────────────────────────────────────────
# EP variants — operate on permuted local-expert tokens, no routing-weight
# chain inside (applied later via ``tokens_post_all2all`` → ``unpermute``).
# Mirror :class:`veomni.distributed.moe.EPMergedFc1GroupGemm`.
# ──────────────────────────────────────────────────────────────────────────────


class EPMergedFc1SharedLoRAGroupGemm(torch.autograd.Function):
    """EP fused MoE forward + shared seed-style two-LoRA on fused gate_up (Mode 2).

    Operates on ``permute_tokens`` already redistributed by
    ``token_pre_all2all``: each row corresponds to one (token, expert-slot)
    pair routed to a *local* expert on this rank. ``cumsum`` carries the
    per-local-expert token counts. Routing-weight scaling is applied
    afterwards by ``tokens_post_all2all`` / ``unpermute``, so neither fc2
    output nor the down-LoRA delta multiplies by gate weights here — base
    and LoRA deltas remain linear in ``mid``, so the two conventions agree.

    Inputs (forward):
        permute_tokens: ``[T_local, H]`` permuted hidden states (one row
            per (token, slot) routed to a local expert).
        cumsum: ``[E_local]`` cumulative count of tokens per local expert.
        fc1_1_2_weight: ``[E_local, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E_local, H, I]`` down base weight.
        lora_a_gate / lora_b_gate: shared LoRA pair on the gate half
            (``[r, H]`` / ``[I, r]``, rank-invariant).
        lora_a_up / lora_b_up: shared LoRA pair on the up half
            (``[r, H]`` / ``[I, r]``).
        lora_a_down / lora_b_down: shared LoRA pair on down
            (``[r, I]`` / ``[H, r]``).
        lora_scale_gate / lora_scale_up / lora_scale_down: per-spec scaling.

    Output:
        ``[T_local, H]`` — same shape as ``permute_tokens``.
    """

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
    ):
        max_t = permute_tokens.shape[0]

        # Base fc1: [T_local, 2I]
        fc1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_a=False,
            transpose_b=True,
        )

        # Two independent shared LoRA deltas on gate / up halves —
        # rank-invariant ⇒ plain F.linear chains. Cat per-half deltas into
        # [T_local, 2I] and add into fc1_output before chunk + silu.
        tmp_gate = torch.nn.functional.linear(permute_tokens, lora_a_gate)  # [T_local, r]
        delta_gate = torch.nn.functional.linear(tmp_gate, lora_b_gate) * lora_scale_gate  # [T_local, I]
        tmp_up = torch.nn.functional.linear(permute_tokens, lora_a_up)  # [T_local, r]
        delta_up = torch.nn.functional.linear(tmp_up, lora_b_up) * lora_scale_up  # [T_local, I]
        fc1_output = fc1_output + torch.cat([delta_gate, delta_up], dim=-1)

        # silu(gate) * up — no routing-weight scaling here (applied after all2all combine).
        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_act = fc1_1_activation * fc1_2_output  # [T_local, I]

        # Base fc2: [T_local, H]
        fc2_output = group_gemm_same_nk(
            a=fc1_act,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_a=False,
            transpose_b=True,
        )

        # Shared LoRA delta on down.
        tmp_down = torch.nn.functional.linear(fc1_act, lora_a_down)  # [T_local, r]
        lora_delta_down = torch.nn.functional.linear(tmp_down, lora_b_down) * lora_scale_down  # [T_local, H]
        fc2_output = fc2_output + lora_delta_down

        ctx.lora_scale_gate = lora_scale_gate
        ctx.lora_scale_up = lora_scale_up
        ctx.lora_scale_down = lora_scale_down
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            fc1_act,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [T_local, H] — already routing-weight-aware via the upstream
        # tokens_post_all2all chain, so no per-row scattered-gate-weight handling here.
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            fc1_act,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            tmp_gate,
            tmp_up,
            tmp_down,
        ) = ctx.saved_tensors
        scale_gate = ctx.lora_scale_gate
        scale_up = ctx.lora_scale_up
        scale_down = ctx.lora_scale_down

        max_t = grad_output.shape[0]

        # ---- LoRA fc2 backward (closed form). ---------------------------
        grad_tmp_down = torch.nn.functional.linear(grad_output, lora_b_down.t()) * scale_down  # [T_local, r]
        grad_lora_b_down = grad_output.t().to(tmp_down.dtype) @ tmp_down * scale_down  # [H, r]
        grad_lora_a_down = grad_tmp_down.t().to(fc1_act.dtype) @ fc1_act  # [r, I]
        grad_fc1_act_lora = torch.nn.functional.linear(grad_tmp_down, lora_a_down.t())  # [T_local, I]

        # Base fc2 dgrad → grad_fc1_act, then accumulate LoRA contribution.
        grad_fc1_act = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_b=False,
        )  # [T_local, I]
        grad_fc1_act = grad_fc1_act + grad_fc1_act_lora

        # Base fc2 wgrad.
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_act,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=max_t,
                transpose_a=True,
                transpose_b=False,
            )

        # silu chain — recompute fc1_1_activation to save memory (matches EPMergedFc1GroupGemm).
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        grad_fc1_1_activation = grad_fc1_act * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_act
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)  # [T_local, 2I]

        # ---- LoRA fc1 backward (per-half closed form). ------------------
        grad_delta_gate, grad_delta_up = grad_fc1_output.chunk(2, dim=-1)
        grad_delta_gate = grad_delta_gate.contiguous()
        grad_delta_up = grad_delta_up.contiguous()

        # Gate half.
        grad_tmp_gate = torch.nn.functional.linear(grad_delta_gate, lora_b_gate.t()) * scale_gate  # [T_local, r]
        grad_lora_b_gate = grad_delta_gate.t().to(tmp_gate.dtype) @ tmp_gate * scale_gate  # [I, r]
        grad_lora_a_gate = grad_tmp_gate.t().to(permute_tokens.dtype) @ permute_tokens  # [r, H]
        grad_permute_gate = torch.nn.functional.linear(grad_tmp_gate, lora_a_gate.t())  # [T_local, H]

        # Up half.
        grad_tmp_up = torch.nn.functional.linear(grad_delta_up, lora_b_up.t()) * scale_up  # [T_local, r]
        grad_lora_b_up = grad_delta_up.t().to(tmp_up.dtype) @ tmp_up * scale_up  # [I, r]
        grad_lora_a_up = grad_tmp_up.t().to(permute_tokens.dtype) @ permute_tokens  # [r, H]
        grad_permute_up = torch.nn.functional.linear(grad_tmp_up, lora_a_up.t())  # [T_local, H]

        # Base fc1 dgrad → grad_permute, accumulate both half-LoRA contributions.
        grad_permute = group_gemm_same_nk(
            a=grad_fc1_output,
            b=fc1_1_2_weight,
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_b=False,
        )
        grad_permute = grad_permute + grad_permute_gate + grad_permute_up

        # Base fc1 wgrad.
        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_output,
                b=permute_tokens,
                c=grad_fc1_1_2_weight,
                cumsum_K=cumsum,
                max_K=max_t,
                transpose_a=True,
                transpose_b=False,
            )

        return (
            grad_permute,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_2_weight,  # fc1_1_2_weight
            grad_fc2_weight,  # fc2_weight
            grad_lora_a_gate,  # lora_a_gate
            grad_lora_b_gate,  # lora_b_gate
            grad_lora_a_up,  # lora_a_up
            grad_lora_b_up,  # lora_b_up
            grad_lora_a_down,  # lora_a_down
            grad_lora_b_down,  # lora_b_down
            None,  # lora_scale_gate
            None,  # lora_scale_up
            None,  # lora_scale_down
        )


def group_gemm_fused_lora_moe_forward(
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
    """Triton grouped-gemm fused MoE forward with shared seed-style two-LoRA (Mode 2).

    Args:
        num_experts: number of experts ``E`` in this MoE layer (global on EP).
        routing_weights: ``[B*S, topk]`` per-(token, slot) routing weights.
        selected_experts: ``[B*S, topk]`` per-(token, slot) selected expert ids
            (global ids on EP — ``preprocess`` / ``token_pre_all2all`` route
            them to the owning rank).
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight (``E_local``
            on EP since the experts module is local-sliced).
        fc2_weight: ``[E, H, I]`` down base weight (likewise local-sliced on EP).
        lora_a_gate / lora_b_gate: shared LoRA pair on the gate half
            (``[r, H]`` / ``[I, r]``).
        lora_a_up / lora_b_up: shared LoRA pair on the up half
            (``[r, H]`` / ``[I, r]``).
        lora_a_down / lora_b_down: shared LoRA pair on down
            (``[r, I]`` / ``[H, r]``).
        lora_scale_gate / lora_scale_up / lora_scale_down: per-spec scaling.

    Returns:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.

    Branches:
        * Non-EP: dispatches to :class:`MergedFc1TritonFusedLoRAMoeExpertFunction`.
        * EP: delegates to :func:`veomni.distributed.moe.dispatch_to_ep_class`
          with :class:`EPMergedFc1SharedLoRAGroupGemm` (mirrors the non-LoRA
          EP fused path).
    """
    if get_parallel_state().ep_enabled:
        # Lazy import — keeps non-EP imports free of distributed deps so
        # eager-only / single-rank tests don't pay the cost.
        from .....distributed.moe import dispatch_to_ep_class

        return dispatch_to_ep_class(
            EPMergedFc1SharedLoRAGroupGemm,
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_2_weight,
            fc2_weight,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
            lora_scale_gate,
            lora_scale_up,
            lora_scale_down,
        )
    return MergedFc1TritonFusedLoRAMoeExpertFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        lora_a_gate,
        lora_b_gate,
        lora_a_up,
        lora_b_up,
        lora_a_down,
        lora_b_down,
        lora_scale_gate,
        lora_scale_up,
        lora_scale_down,
    )


def wrapper(
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor,
    fc1_1_2_weight: Tensor,
    fc2_weight: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
    *,
    num_experts: int,
    lora_scale_gate: float,
    lora_scale_up: float,
    lora_scale_down: float,
) -> Tensor:
    """Shared Triton fused LoRA MoE."""
    return group_gemm_fused_lora_moe_forward(
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
