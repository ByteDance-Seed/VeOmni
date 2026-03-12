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

"""Fused Triton kernel for the MoE load balancing auxiliary loss.

Fuses softmax + top-k selection + accumulation into a single GPU kernel,
eliminating the large ``[N, top_k, num_experts]`` one-hot intermediate tensor.
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _lb_loss_fwd_kernel(
    gate_logits_ptr,  # [N, E]
    mask_weights_ptr,  # [N] per-token weight (or unused when HAS_MASK=False)
    expert_count_ptr,  # [E] output accumulator
    router_prob_sum_ptr,  # [E] output accumulator
    stride_logits_row,  # stride of gate_logits along dim-0
    N,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    expert_offs = tl.arange(0, BLOCK_E)
    emask = expert_offs < E

    # Optional per-token mask weight
    if HAS_MASK:
        w = tl.load(mask_weights_ptr + row_idx).to(tl.float32)
        if w == 0.0:
            return
    else:
        w = 1.0

    # Load gate logits for this token
    row_start = row_idx * stride_logits_row
    logits = tl.load(gate_logits_ptr + row_start + expert_offs, mask=emask, other=float("-inf")).to(tl.float32)

    # ---- Online softmax ----
    max_val = tl.max(logits, axis=0)
    logits_shifted = logits - max_val
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Accumulate weighted router probability sums (all experts).
    # We use atomic_add because all N rows (one per program) reduce into a
    # shared [E] buffer.  Contention is low in practice: E is small (60–128)
    # so each atomic targets a different cache-line, and the hardware coalesces
    # concurrent writes from threads in the same warp.
    tl.atomic_add(router_prob_sum_ptr + expert_offs, w * probs, mask=emask)

    # ---- Top-k selection with expert count accumulation ----
    probs_for_topk = tl.where(emask, probs, float("-inf"))
    for _k in range(TOP_K):
        # Find the expert with the highest probability
        max_prob = tl.max(probs_for_topk, axis=0)
        is_max = probs_for_topk == max_prob
        # Tie-break: pick the lowest expert index
        candidate = tl.where(is_max, expert_offs, BLOCK_E)
        winner_idx = tl.min(candidate, axis=0)
        # Accumulate weighted expert count.
        # Scalar atomic — only TOP_K atomics per row, negligible overhead.
        tl.atomic_add(expert_count_ptr + winner_idx, w)
        # Mask out the winner for the next iteration
        probs_for_topk = tl.where(expert_offs == winner_idx, float("-inf"), probs_for_topk)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _lb_loss_bwd_kernel(
    gate_logits_ptr,  # [N, E] input (re-read for softmax recomputation)
    expert_count_ptr,  # [E] from forward
    mask_weights_ptr,  # [N] per-token weight (or unused)
    grad_logits_ptr,  # [N, E] output gradient
    grad_scale,  # scalar: upstream_grad * E / total_weight^2
    stride_logits_row,
    stride_grad_row,
    N,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Backward pass: compute d(loss)/d(gate_logits).

    Derivation (no-mask case, analogous for masked):
        loss = E / N^2 * sum_e( count_e * sum_n softmax(logits_n)[e] )
        d(loss)/d(logits_n[j])
            = E / N^2 * softmax_n[j] * (count[j] - dot(count, softmax_n))
    """
    row_idx = tl.program_id(0)
    expert_offs = tl.arange(0, BLOCK_E)
    emask = expert_offs < E

    if HAS_MASK:
        w = tl.load(mask_weights_ptr + row_idx).to(tl.float32)
        if w == 0.0:
            tl.store(grad_logits_ptr + row_idx * stride_grad_row + expert_offs, 0.0, mask=emask)
            return
    else:
        w = 1.0

    # Recompute softmax
    row_start = row_idx * stride_logits_row
    logits = tl.load(gate_logits_ptr + row_start + expert_offs, mask=emask, other=float("-inf")).to(tl.float32)
    max_val = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_val)
    probs = exp_logits / tl.sum(exp_logits, axis=0)

    # Load expert counts
    counts = tl.load(expert_count_ptr + expert_offs, mask=emask, other=0.0).to(tl.float32)

    # grad = grad_scale * w * probs * (counts - dot(counts, probs))
    dot_cs = tl.sum(counts * probs, axis=0)
    grad = grad_scale * w * probs * (counts - dot_cs)

    # No atomics in backward — each row writes to its own output row.
    grad_row_start = row_idx * stride_grad_row
    tl.store(grad_logits_ptr + grad_row_start + expert_offs, grad, mask=emask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class _FusedLoadBalancingLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        concatenated_gate_logits: torch.Tensor,
        num_experts: int,
        top_k: int,
        mask_weights: Optional[torch.Tensor],
        total_weight: float,
    ) -> torch.Tensor:
        N, E = concatenated_gate_logits.shape
        device = concatenated_gate_logits.device

        expert_count = torch.zeros(E, device=device, dtype=torch.float32)
        router_prob_sum = torch.zeros(E, device=device, dtype=torch.float32)

        BLOCK_E = triton.next_power_of_2(E)
        has_mask = mask_weights is not None
        # Use a dummy pointer when no mask; kernel will not access it.
        mask_ptr = mask_weights if has_mask else expert_count  # unused

        _lb_loss_fwd_kernel[(N,)](
            concatenated_gate_logits,
            mask_ptr,
            expert_count,
            router_prob_sum,
            concatenated_gate_logits.stride(0),
            N,
            E=E,
            TOP_K=top_k,
            BLOCK_E=BLOCK_E,
            HAS_MASK=has_mask,
        )

        # loss = E * dot(expert_count, router_prob_sum) / total_weight^2
        loss = torch.dot(expert_count, router_prob_sum) * (E / (total_weight * total_weight))

        # Save for backward
        ctx.save_for_backward(
            concatenated_gate_logits, expert_count, mask_weights if has_mask else torch.empty(0, device=device)
        )
        ctx.total_weight = total_weight
        ctx.has_mask = has_mask
        ctx.E = E
        ctx.N = N

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_logits, expert_count, mask_weights = ctx.saved_tensors
        N, E = ctx.N, ctx.E
        has_mask = ctx.has_mask
        total_weight = ctx.total_weight

        grad_logits = torch.empty_like(gate_logits, dtype=torch.float32)
        BLOCK_E = triton.next_power_of_2(E)
        grad_scale = grad_output.item() * E / (total_weight * total_weight)

        mask_ptr = mask_weights if has_mask else gate_logits  # dummy

        _lb_loss_bwd_kernel[(N,)](
            gate_logits,
            expert_count,
            mask_ptr,
            grad_logits,
            grad_scale,
            gate_logits.stride(0),
            grad_logits.stride(0),
            N,
            E=E,
            BLOCK_E=BLOCK_E,
            HAS_MASK=has_mask,
        )

        return grad_logits.to(gate_logits.dtype), None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_balancing_loss_triton(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """Fused Triton load balancing loss.

    Fuses softmax + top-k + accumulation into a single kernel, avoiding
    the ``[N, top_k, num_experts]`` one-hot intermediate tensor.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    ).contiguous()

    N, E = concatenated_gate_logits.shape
    assert E == num_experts, f"gate_logits last dim ({E}) != num_experts ({num_experts})"

    if attention_mask is not None:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = N // (batch_size * sequence_length)
        mask_weights = (
            attention_mask[None, :, :]
            .expand(num_hidden_layers, batch_size, sequence_length)
            .reshape(-1)
            .to(compute_device, dtype=torch.float32)
            .contiguous()
        )
        total_weight = mask_weights.sum().item()
        if total_weight == 0:
            return torch.tensor(0.0, device=compute_device)
    else:
        mask_weights = None
        total_weight = float(N)

    return _FusedLoadBalancingLoss.apply(concatenated_gate_logits, num_experts, top_k, mask_weights, total_weight)
