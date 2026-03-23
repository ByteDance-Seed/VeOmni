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

The forward kernel uses a two-pass tiled reduction to avoid ``atomic_add``,
ensuring deterministic results across runs.
"""

from typing import Optional, Union

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel — tiled reduction (no atomics)
# ---------------------------------------------------------------------------


@triton.jit
def _lb_loss_fwd_kernel(
    gate_logits_ptr,  # [N, E]
    mask_weights_ptr,  # [N] per-token weight (or unused when HAS_MASK=False)
    expert_count_ptr,  # [num_blocks, E] partial sums output
    router_prob_sum_ptr,  # [num_blocks, E] partial sums output
    stride_logits_row,  # stride of gate_logits along dim-0
    stride_count_row,  # stride of expert_count along dim-0
    stride_prob_row,  # stride of router_prob_sum along dim-0
    N,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_start = block_idx * BLOCK_N
    expert_offs = tl.arange(0, BLOCK_E)
    emask = expert_offs < E

    # Local accumulators in registers — no cross-block sharing needed.
    local_count = tl.zeros([BLOCK_E], dtype=tl.float32)
    local_prob_sum = tl.zeros([BLOCK_E], dtype=tl.float32)

    for row_offset in range(BLOCK_N):
        row_idx = row_start + row_offset
        if row_idx < N:
            # Optional per-token mask weight
            if HAS_MASK:
                w = tl.load(mask_weights_ptr + row_idx).to(tl.float32)
            else:
                w = 1.0

            if w != 0.0:
                # Load gate logits for this token and upcast to float32 for stable softmax.
                row_ptr = row_idx * stride_logits_row
                logits = tl.load(gate_logits_ptr + row_ptr + expert_offs, mask=emask, other=float("-inf")).to(
                    tl.float32
                )

                # ---- Online softmax ----
                max_val = tl.max(logits, axis=0)
                logits_shifted = logits - max_val
                exp_logits = tl.exp(logits_shifted)
                sum_exp = tl.sum(exp_logits, axis=0)
                probs = exp_logits / sum_exp

                # Accumulate weighted router probability sums (all experts).
                local_prob_sum += w * probs

                # ---- Top-k selection with expert count accumulation ----
                probs_for_topk = tl.where(emask, probs, float("-inf"))
                for _k in range(TOP_K):
                    max_prob = tl.max(probs_for_topk, axis=0)
                    is_max = probs_for_topk == max_prob
                    candidate = tl.where(is_max, expert_offs, BLOCK_E)
                    winner_idx = tl.min(candidate, axis=0)
                    # Accumulate into local register — no atomics needed.
                    local_count += tl.where(expert_offs == winner_idx, w, 0.0)
                    probs_for_topk = tl.where(expert_offs == winner_idx, float("-inf"), probs_for_topk)

    # Write partial sums to this block's row — each block writes to its own row.
    out_offset = block_idx * stride_count_row + expert_offs
    tl.store(expert_count_ptr + out_offset, local_count, mask=emask)
    out_offset_prob = block_idx * stride_prob_row + expert_offs
    tl.store(router_prob_sum_ptr + out_offset_prob, local_prob_sum, mask=emask)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _lb_loss_bwd_kernel(
    gate_logits_ptr,  # [N, E] input (re-read for softmax recomputation)
    expert_count_ptr,  # [E] from forward
    mask_weights_ptr,  # [N] per-token weight (or unused)
    grad_logits_ptr,  # [N, E] output gradient
    grad_scale_ptr,  # [1] scalar tensor: upstream_grad * E / total_weight^2
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
    grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
    dot_cs = tl.sum(counts * probs, axis=0)
    grad = grad_scale * w * probs * (counts - dot_cs)

    # No atomics in backward — each row writes to its own output row.
    grad_row_start = row_idx * stride_grad_row
    tl.store(grad_logits_ptr + grad_row_start + expert_offs, grad, mask=emask)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

BLOCK_N = 256


class _FusedLoadBalancingLoss(torch.autograd.Function):
    """Autograd wrapper for the fused load balancing loss kernels.

    Dtype handling:
        - ``concatenated_gate_logits`` can be any floating-point dtype
          (float16, bfloat16, float32). The Triton kernels cast inputs to
          float32 internally for numerical stability (softmax).
        - ``mask_weights`` must be float32 (pre-cast by the caller).
        - All intermediate accumulators (``expert_count``, ``router_prob_sum``)
          are allocated in float32.
        - The output ``loss`` is float32.
        - Backward produces float32 gradients and casts them back to the
          input dtype before returning.
    """

    @staticmethod
    def forward(
        ctx,
        concatenated_gate_logits: torch.Tensor,
        num_experts: int,
        top_k: int,
        mask_weights: Optional[torch.Tensor],
        total_weight: torch.Tensor,
    ) -> torch.Tensor:
        N, E = concatenated_gate_logits.shape
        device = concatenated_gate_logits.device

        num_blocks = triton.cdiv(N, BLOCK_N)
        BLOCK_E = triton.next_power_of_2(E)
        has_mask = mask_weights is not None

        # Partial-sum buffers: each block writes to its own row — no atomics.
        partial_expert_count = torch.zeros(num_blocks, E, device=device, dtype=torch.float32)
        partial_router_prob_sum = torch.zeros(num_blocks, E, device=device, dtype=torch.float32)

        # Use a dummy pointer when no mask; kernel will not access it.
        mask_ptr = mask_weights if has_mask else partial_expert_count  # unused

        _lb_loss_fwd_kernel[(num_blocks,)](
            concatenated_gate_logits,
            mask_ptr,
            partial_expert_count,
            partial_router_prob_sum,
            concatenated_gate_logits.stride(0),
            partial_expert_count.stride(0),
            partial_router_prob_sum.stride(0),
            N,
            E=E,
            TOP_K=top_k,
            BLOCK_E=BLOCK_E,
            BLOCK_N=BLOCK_N,
            HAS_MASK=has_mask,
        )

        # Reduce partial sums across blocks.
        expert_count = partial_expert_count.sum(0)
        router_prob_sum = partial_router_prob_sum.sum(0)

        # loss = E * dot(expert_count, router_prob_sum) / total_weight^2
        loss = torch.dot(expert_count, router_prob_sum) * (E / (total_weight * total_weight))

        # Save for backward
        ctx.save_for_backward(
            concatenated_gate_logits,
            expert_count,
            mask_weights if has_mask else torch.empty(0, device=device),
            total_weight,
        )
        ctx.has_mask = has_mask
        ctx.E = E
        ctx.N = N

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_logits, expert_count, mask_weights, total_weight = ctx.saved_tensors
        N, E = ctx.N, ctx.E
        has_mask = ctx.has_mask

        # Compute gradients in float32 for precision; cast to input dtype at the end.
        grad_logits = torch.empty_like(gate_logits, dtype=torch.float32)
        BLOCK_E = triton.next_power_of_2(E)
        grad_scale = grad_output * E / (total_weight * total_weight)

        mask_ptr = mask_weights if has_mask else gate_logits  # dummy

        _lb_loss_bwd_kernel[(N,)](
            gate_logits,
            expert_count,
            mask_ptr,
            grad_logits,
            grad_scale.contiguous(),
            gate_logits.stride(0),
            grad_logits.stride(0),
            N,
            E=E,
            BLOCK_E=BLOCK_E,
            HAS_MASK=has_mask,
        )

        # Cast gradients back to the original input dtype (e.g. bfloat16).
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
    """Fused Triton load balancing loss for Mixture-of-Experts models.

    Computes the auxiliary load balancing loss from the Switch Transformer paper
    (Fedus et al., 2021; https://arxiv.org/abs/2101.03961), equations (4)-(6)::

        loss = num_experts * sum_e(f_e * P_e)

    where ``f_e`` is the fraction of tokens routed to expert *e* and ``P_e`` is
    the average router probability assigned to expert *e* across all tokens.

    This implementation fuses softmax + top-k selection + accumulation into a
    single GPU kernel, eliminating the large ``[N, top_k, num_experts]`` one-hot
    intermediate tensor. The forward kernel uses a tiled reduction (no atomics)
    for deterministic results.

    Args:
        gate_logits: Tuple of per-layer gate logits, each shaped
            ``[batch_size * seq_len, num_experts]``. May be float16, bfloat16,
            or float32; all internal computation is performed in float32 for
            numerical stability. Returns ``0`` if ``None`` or not a tuple.
        num_experts: Total number of experts ``E``.
        top_k: Number of experts selected per token.
        attention_mask: Optional ``[batch_size, seq_len]`` binary mask where
            ``1`` indicates a real token and ``0`` indicates padding. When
            provided, padded tokens are excluded from both the expert count
            and the router probability sum. Named ``attention_mask`` (rather
            than ``loss_mask``) for compatibility with the HuggingFace
            ``load_balancing_loss_func`` API.

    Returns:
        Scalar float32 loss tensor, or ``0`` when *gate_logits* is ``None``
        or not a tuple.
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
        num_hidden_layers = len(gate_logits)
        expected_tokens = num_hidden_layers * batch_size * sequence_length
        if N != expected_tokens:
            raise ValueError(
                f"Mismatch between gate_logits total tokens ({N}) and attention_mask shape. "
                f"Expected {num_hidden_layers} * {batch_size} * {sequence_length} = {expected_tokens} tokens, "
                f"but got {N}."
            )
        mask_weights = (
            attention_mask[None, :, :]
            .expand(num_hidden_layers, batch_size, sequence_length)
            .reshape(-1)
            .to(compute_device, dtype=torch.float32)
            .contiguous()
        )
        total_weight = mask_weights.sum()
        if total_weight == 0:
            return torch.tensor(0.0, device=compute_device)
    else:
        mask_weights = None
        total_weight = torch.tensor(float(N), device=compute_device)

    return _FusedLoadBalancingLoss.apply(concatenated_gate_logits, num_experts, top_k, mask_weights, total_weight)
