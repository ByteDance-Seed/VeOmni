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

"""independent MoE-LoRA Triton implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from .....distributed.parallel_state import get_parallel_state
from ...moe_experts.shared.dispatch import expert_histogram, moe_gather, moe_scatter
from ...moe_experts.shared.group_gemm import group_gemm_same_mn, group_gemm_same_nk


def _per_expert_lora_half_forward(
    *,
    scatter_output: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    cumsum: torch.Tensor,
    max_t: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One per-expert LoRA forward chain (one of gate / up / down).

    Forward: ``tmp = scatter_output @ A_e.T``, ``delta = tmp @ B_e.T * scale``.
    Returns ``(tmp, delta)`` so the caller can save ``tmp`` for backward
    without re-running the first matmul.
    """
    tmp = group_gemm_same_nk(
        a=scatter_output,
        b=lora_a,
        cumsum_M=cumsum,
        max_M=max_t,
        transpose_b=True,
    )
    delta = group_gemm_same_nk(
        a=tmp,
        b=lora_b,
        cumsum_M=cumsum,
        max_M=max_t,
        transpose_b=True,
    )
    return tmp, delta * scale


def _per_expert_lora_half_backward(
    *,
    grad_delta: torch.Tensor,
    inp: torch.Tensor,
    tmp: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    cumsum: torch.Tensor,
    max_t: int,
    scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Closed-form backward for one per-expert LoRA half (gate / up / down).

    Mirrors the forward in :func:`_per_expert_lora_half_forward`. Returns
    ``(grad_lora_a, grad_lora_b, grad_inp_lora_contrib)`` where
    ``grad_inp_lora_contrib`` is the LoRA branch contribution to the
    *input* dgrad (the caller adds it to the base fc1 / fc2 dgrad).
    Returns ``None`` for grads of LoRA parameters that don't require grad
    (consistent with autograd convention).
    """
    grad_tmp = group_gemm_same_nk(
        a=grad_delta,
        b=lora_b,
        cumsum_M=cumsum,
        max_M=max_t,
        transpose_b=False,
    )
    grad_tmp = grad_tmp * scale

    grad_lora_b = None
    if lora_b.requires_grad:
        grad_lora_b = torch.empty_like(lora_b)
        group_gemm_same_mn(
            a=grad_delta,
            b=tmp,
            c=grad_lora_b,
            cumsum_K=cumsum,
            max_K=max_t,
            transpose_a=True,
            transpose_b=False,
        )
        grad_lora_b.mul_(scale)

    grad_lora_a = None
    if lora_a.requires_grad:
        grad_lora_a = torch.empty_like(lora_a)
        group_gemm_same_mn(
            a=grad_tmp,
            b=inp,
            c=grad_lora_a,
            cumsum_K=cumsum,
            max_K=max_t,
            transpose_a=True,
            transpose_b=False,
        )

    grad_inp_lora = group_gemm_same_nk(
        a=grad_tmp,
        b=lora_a,
        cumsum_M=cumsum,
        max_M=max_t,
        transpose_b=False,
    )
    return grad_lora_a, grad_lora_b, grad_inp_lora


class MergedFc1IndependentTritonFusedLoRAMoeExpertFunction(torch.autograd.Function):
    """Fused MoE forward + independent per-expert seed-style two-LoRA on gate_up (Mode 1), non-EP.

    Differs from :class:`MergedFc1TritonFusedLoRAMoeExpertFunction` only in
    that the LoRA tensors carry a leading expert dim, so each ``A``/``B``
    matmul is itself a grouped-gemm using the same boundaries
    (``cumsum_M`` / ``cumsum_K``) as the base ``fc1`` / ``fc2``. The
    seed-style two-LoRA layout (one rank-r adapter per gate / up half)
    is preserved end-to-end.

    Inputs (forward):
        num_experts: ``E``, the global expert count for this layer.
        gate_weights: ``[B*S, topk]`` routing weights per (token, slot).
        expert_index: ``[B*S, topk]`` selected expert ids per (token, slot).
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E, H, I]`` down base weight.
        lora_a_gate / lora_b_gate: per-expert LoRA pair on the gate half
            (``[E, r, H]`` / ``[E, I, r]``).
        lora_a_up / lora_b_up: per-expert LoRA pair on the up half
            (``[E, r, H]`` / ``[E, I, r]``).
        lora_a_down / lora_b_down: per-expert LoRA pair on down
            (``[E, r, I]`` / ``[E, H, r]``).
        lora_scale_gate / lora_scale_up / lora_scale_down: per-spec scaling
            (``alpha / r`` or ``alpha / sqrt(r)`` for rsLoRA).

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

        # Per-expert LoRA on each half — independent rank-r adapters,
        # both run as grouped-gemm chains and concatenated into [T, 2I]
        # before the add into ``fc1_output`` (LoRA must enter the
        # pre-activation linear).
        tmp_gate, delta_gate = _per_expert_lora_half_forward(
            scatter_output=scatter_output,
            lora_a=lora_a_gate,
            lora_b=lora_b_gate,
            cumsum=cumsum_t,
            max_t=max_t,
            scale=lora_scale_gate,
        )  # [T, r], [T, I]
        tmp_up, delta_up = _per_expert_lora_half_forward(
            scatter_output=scatter_output,
            lora_a=lora_a_up,
            lora_b=lora_b_up,
            cumsum=cumsum_t,
            max_t=max_t,
            scale=lora_scale_up,
        )  # [T, r], [T, I]
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

        # LoRA fc2 delta on down (per-expert).
        tmp_down = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=lora_a_down,  # [E, r, I] → N=r, K=I
            cumsum_M=cumsum_t,
            max_M=max_t,
            transpose_b=True,
        )  # [T, r]
        lora_delta_down = group_gemm_same_nk(
            a=tmp_down,
            b=lora_b_down,  # [E, H, r] → N=H, K=r
            cumsum_M=cumsum_t,
            max_M=max_t,
            transpose_b=True,
        )  # [T, H]
        lora_delta_down = lora_delta_down * lora_scale_down
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

        # ---- LoRA fc2 backward (per-expert closed form). ----------------
        grad_lora_a_down, grad_lora_b_down, grad_fc1_weighted_output_lora = _per_expert_lora_half_backward(
            grad_delta=grad_fc2_output,
            inp=fc1_weighted_output,
            tmp=tmp_down,
            lora_a=lora_a_down,
            lora_b=lora_b_down,
            cumsum=cumsum_t,
            max_t=max_t,
            scale=scale_down,
        )

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
        # Same chunk-then-contiguous trick as the shared variant — the views
        # from chunk(2, dim=-1) are non-contig and we run wgrad transposes
        # on them.
        grad_delta_gate, grad_delta_up = grad_fc1_output.chunk(2, dim=-1)
        grad_delta_gate = grad_delta_gate.contiguous()
        grad_delta_up = grad_delta_up.contiguous()

        grad_lora_a_gate, grad_lora_b_gate, grad_scatter_output_gate = _per_expert_lora_half_backward(
            grad_delta=grad_delta_gate,
            inp=scatter_output,
            tmp=tmp_gate,
            lora_a=lora_a_gate,
            lora_b=lora_b_gate,
            cumsum=cumsum_t,
            max_t=max_t,
            scale=scale_gate,
        )
        grad_lora_a_up, grad_lora_b_up, grad_scatter_output_up = _per_expert_lora_half_backward(
            grad_delta=grad_delta_up,
            inp=scatter_output,
            tmp=tmp_up,
            lora_a=lora_a_up,
            lora_b=lora_b_up,
            cumsum=cumsum_t,
            max_t=max_t,
            scale=scale_up,
        )

        # MoE step 4 (base) — single dgrad for merged fc1, accumulated
        # with both half-LoRA contributions.
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


class EPMergedFc1IndependentLoRAGroupGemm(torch.autograd.Function):
    """EP fused MoE forward + independent per-expert seed-style two-LoRA on gate_up (Mode 1).

    Differs from :class:`EPMergedFc1SharedLoRAGroupGemm` only in that the
    LoRA tensors carry a leading local-expert dim, so each ``A``/``B``
    matmul becomes a grouped-gemm using the same ``cumsum`` boundaries as
    the base ``fc1`` / ``fc2``. Reuses the
    :func:`_per_expert_lora_half_forward` /
    :func:`_per_expert_lora_half_backward` helpers shared with the non-EP
    Mode 1 class so the per-half (gate / up / down) chain stays in lock-step
    with the non-EP path.

    Inputs (forward):
        permute_tokens: ``[T_local, H]`` permuted hidden states (one row
            per (token, slot) routed to a local expert).
        cumsum: ``[E_local]`` cumulative count of tokens per local expert.
        fc1_1_2_weight: ``[E_local, 2I, H]`` fused gate+up base weight.
        fc2_weight: ``[E_local, H, I]`` down base weight.
        lora_a_gate / lora_b_gate: per-local-expert LoRA pair on gate
            (``[E_local, r, H]`` / ``[E_local, I, r]``).
        lora_a_up / lora_b_up: per-local-expert LoRA pair on up
            (``[E_local, r, H]`` / ``[E_local, I, r]``).
        lora_a_down / lora_b_down: per-local-expert LoRA pair on down
            (``[E_local, r, I]`` / ``[E_local, H, r]``).
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

        # Per-local-expert LoRA on each half — independent rank-r adapters,
        # grouped-gemm chains, concatenated into [T_local, 2I] before the
        # add (LoRA must enter pre-activation).
        tmp_gate, delta_gate = _per_expert_lora_half_forward(
            scatter_output=permute_tokens,
            lora_a=lora_a_gate,
            lora_b=lora_b_gate,
            cumsum=cumsum,
            max_t=max_t,
            scale=lora_scale_gate,
        )
        tmp_up, delta_up = _per_expert_lora_half_forward(
            scatter_output=permute_tokens,
            lora_a=lora_a_up,
            lora_b=lora_b_up,
            cumsum=cumsum,
            max_t=max_t,
            scale=lora_scale_up,
        )
        fc1_output = fc1_output + torch.cat([delta_gate, delta_up], dim=-1)

        # silu(gate) * up — no routing-weight scaling.
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

        # Per-local-expert LoRA delta on down.
        tmp_down = group_gemm_same_nk(
            a=fc1_act,
            b=lora_a_down,  # [E_local, r, I] → N=r, K=I
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_b=True,
        )  # [T_local, r]
        lora_delta_down = group_gemm_same_nk(
            a=tmp_down,
            b=lora_b_down,  # [E_local, H, r] → N=H, K=r
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_b=True,
        )  # [T_local, H]
        lora_delta_down = lora_delta_down * lora_scale_down
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

        # ---- LoRA fc2 backward (per-local-expert closed form). ----------
        grad_lora_a_down, grad_lora_b_down, grad_fc1_act_lora = _per_expert_lora_half_backward(
            grad_delta=grad_output,
            inp=fc1_act,
            tmp=tmp_down,
            lora_a=lora_a_down,
            lora_b=lora_b_down,
            cumsum=cumsum,
            max_t=max_t,
            scale=scale_down,
        )

        # Base fc2 dgrad + LoRA contribution.
        grad_fc1_act = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=max_t,
            transpose_b=False,
        )
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

        # silu chain.
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        grad_fc1_1_activation = grad_fc1_act * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_act
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)  # [T_local, 2I]

        # ---- LoRA fc1 backward (per-half, per-local-expert closed form). ----
        grad_delta_gate, grad_delta_up = grad_fc1_output.chunk(2, dim=-1)
        grad_delta_gate = grad_delta_gate.contiguous()
        grad_delta_up = grad_delta_up.contiguous()

        grad_lora_a_gate, grad_lora_b_gate, grad_permute_gate = _per_expert_lora_half_backward(
            grad_delta=grad_delta_gate,
            inp=permute_tokens,
            tmp=tmp_gate,
            lora_a=lora_a_gate,
            lora_b=lora_b_gate,
            cumsum=cumsum,
            max_t=max_t,
            scale=scale_gate,
        )
        grad_lora_a_up, grad_lora_b_up, grad_permute_up = _per_expert_lora_half_backward(
            grad_delta=grad_delta_up,
            inp=permute_tokens,
            tmp=tmp_up,
            lora_a=lora_a_up,
            lora_b=lora_b_up,
            cumsum=cumsum,
            max_t=max_t,
            scale=scale_up,
        )

        # Base fc1 dgrad + both half-LoRA contributions.
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


def group_gemm_fused_independent_lora_moe_forward(
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
    """Triton grouped-gemm fused MoE forward with independent per-expert seed-style two-LoRA (Mode 1).

    Same shape contract as :func:`group_gemm_fused_lora_moe_forward` except
    that every LoRA tensor carries a leading expert dim. Under EP that dim
    is ``E_local = E / ep_size`` (the LoRA wrapper allocates per the local
    experts slice it sees on the rank).

    Args:
        num_experts: number of experts ``E`` in this MoE layer (global on EP).
        routing_weights: ``[B*S, topk]`` per-(token, slot) routing weights.
        selected_experts: ``[B*S, topk]`` per-(token, slot) selected expert ids.
        hidden_states: ``[B, S, H]`` (or ``[N, H]``) input activations.
        fc1_1_2_weight: ``[E, 2I, H]`` fused gate+up base weight (``E_local`` on EP).
        fc2_weight: ``[E, H, I]`` down base weight (``E_local`` on EP).
        lora_a_gate / lora_b_gate: per-expert LoRA pair on gate
            (``[E, r, H]`` / ``[E, I, r]``).
        lora_a_up / lora_b_up: per-expert LoRA pair on up
            (``[E, r, H]`` / ``[E, I, r]``).
        lora_a_down / lora_b_down: per-expert LoRA pair on down
            (``[E, r, I]`` / ``[E, H, r]``).
        lora_scale_gate / lora_scale_up / lora_scale_down: per-spec scaling.

    Returns:
        ``[B, S, H]`` (or ``[N, H]``) — same shape as ``hidden_states``.

    Branches:
        * Non-EP: dispatches to :class:`MergedFc1IndependentTritonFusedLoRAMoeExpertFunction`.
        * EP: delegates to :func:`veomni.distributed.moe.dispatch_to_ep_class`
          with :class:`EPMergedFc1IndependentLoRAGroupGemm` (mirrors the
          non-LoRA EP fused path).
    """
    if get_parallel_state().ep_enabled:
        from .....distributed.moe import dispatch_to_ep_class

        return dispatch_to_ep_class(
            EPMergedFc1IndependentLoRAGroupGemm,
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
    return MergedFc1IndependentTritonFusedLoRAMoeExpertFunction.apply(
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
    """Independent Triton fused LoRA MoE."""
    return group_gemm_fused_independent_lora_moe_forward(
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
