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

"""standard load-balancing loss Triton impl."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .....registry import SavedState
from .....triton_cache import cached_triton_kernel
from . import eager as _eager


BLOCK_N = 256


@cached_triton_kernel
def _lb_loss_fwd_kernel():
    """Compile and cache the Triton load-balancing forward kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        gate_logits_ptr,
        mask_weights_ptr,
        expert_count_ptr,
        router_prob_sum_ptr,
        stride_logits_row,
        stride_count_row,
        stride_prob_row,
        N,
        E: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """Tiled softmax + top-k + accumulate. Each block writes its own partial sums."""
        block_idx = tl.program_id(0)
        row_start = block_idx * BLOCK_N
        expert_offs = tl.arange(0, BLOCK_E)
        emask = expert_offs < E

        local_count = tl.zeros([BLOCK_E], dtype=tl.float32)
        local_prob_sum = tl.zeros([BLOCK_E], dtype=tl.float32)

        for row_offset in range(BLOCK_N):
            row_idx = row_start + row_offset
            if row_idx < N:
                if HAS_MASK:
                    w = tl.load(mask_weights_ptr + row_idx).to(tl.float32)
                else:
                    w = 1.0

                if w != 0.0:
                    row_ptr = row_idx * stride_logits_row
                    logits = tl.load(gate_logits_ptr + row_ptr + expert_offs, mask=emask, other=float("-inf")).to(
                        tl.float32
                    )

                    max_val = tl.max(logits, axis=0)
                    logits_shifted = logits - max_val
                    exp_logits = tl.exp(logits_shifted)
                    sum_exp = tl.sum(exp_logits, axis=0)
                    probs = exp_logits / sum_exp

                    local_prob_sum += w * probs

                    probs_for_topk = tl.where(emask, probs, float("-inf"))
                    for _k in range(TOP_K):
                        max_prob = tl.max(probs_for_topk, axis=0)
                        is_max = probs_for_topk == max_prob
                        candidate = tl.where(is_max, expert_offs, BLOCK_E)
                        winner_idx = tl.min(candidate, axis=0)
                        local_count += tl.where(expert_offs == winner_idx, w, 0.0)
                        probs_for_topk = tl.where(expert_offs == winner_idx, float("-inf"), probs_for_topk)

        out_offset = block_idx * stride_count_row + expert_offs
        tl.store(expert_count_ptr + out_offset, local_count, mask=emask)
        out_offset_prob = block_idx * stride_prob_row + expert_offs
        tl.store(router_prob_sum_ptr + out_offset_prob, local_prob_sum, mask=emask)

    return kernel


@cached_triton_kernel
def _lb_loss_bwd_kernel():
    """Compile and cache the Triton load-balancing backward kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        gate_logits_ptr,
        expert_count_ptr,
        mask_weights_ptr,
        grad_logits_ptr,
        grad_scale_ptr,
        stride_logits_row,
        stride_grad_row,
        N,
        E: tl.constexpr,
        BLOCK_E: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        """``d(loss)/d(logits_n[j]) = scale * w * softmax_n[j] * (count[j] - dot(count, softmax_n))``."""
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

        row_start = row_idx * stride_logits_row
        logits = tl.load(gate_logits_ptr + row_start + expert_offs, mask=emask, other=float("-inf")).to(tl.float32)
        max_val = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_val)
        probs = exp_logits / tl.sum(exp_logits, axis=0)

        counts = tl.load(expert_count_ptr + expert_offs, mask=emask, other=0.0).to(tl.float32)
        grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
        dot_cs = tl.sum(counts * probs, axis=0)
        grad = grad_scale * w * probs * (counts - dot_cs)

        grad_row_start = row_idx * stride_grad_row
        tl.store(grad_logits_ptr + grad_row_start + expert_offs, grad, mask=emask)

    return kernel


@dataclass(frozen=True)
class _Meta:
    """``top_k`` plus whether ``attention_mask`` had any values."""

    top_k: int
    has_mask: bool


def forward(gate_logits: Tensor, attention_mask: Tensor, *, top_k: int) -> tuple[Tensor, SavedState]:
    """Fused Triton load-balancing loss on stacked layer logits.

    Empty ``attention_mask`` means every token counts. Top-k counts are
    constants in backward.
    """
    import triton

    num_layers, _num_tokens, num_experts = gate_logits.shape
    concatenated = gate_logits.reshape(-1, num_experts).contiguous()
    token_count, _ = concatenated.shape
    device = concatenated.device
    has_mask = attention_mask.numel() > 0

    if has_mask:
        batch_size, seq_len = attention_mask.shape
        mask_weights = (
            attention_mask.to(device=device, dtype=torch.float32)
            .expand(num_layers, batch_size, seq_len)
            .reshape(-1)
            .contiguous()
        )
        total_weight = mask_weights.sum()
        if total_weight == 0:
            output, saved = _eager.forward(gate_logits, attention_mask, top_k=top_k)
            return output, SavedState(saved.tensors, _Meta(top_k, True))
    else:
        mask_weights = None
        total_weight = torch.tensor(float(token_count), device=device)

    num_blocks = triton.cdiv(token_count, BLOCK_N)
    block_e = triton.next_power_of_2(num_experts)
    partial_expert_count = torch.zeros(num_blocks, num_experts, device=device, dtype=torch.float32)
    partial_router_prob_sum = torch.zeros(num_blocks, num_experts, device=device, dtype=torch.float32)
    mask_ptr = mask_weights if has_mask else partial_expert_count

    _lb_loss_fwd_kernel()[(num_blocks,)](
        concatenated,
        mask_ptr,
        partial_expert_count,
        partial_router_prob_sum,
        concatenated.stride(0),
        partial_expert_count.stride(0),
        partial_router_prob_sum.stride(0),
        token_count,
        E=num_experts,
        TOP_K=top_k,
        BLOCK_E=block_e,
        BLOCK_N=BLOCK_N,
        HAS_MASK=has_mask,
    )

    expert_count = partial_expert_count.sum(0)
    router_prob_sum = partial_router_prob_sum.sum(0)
    loss = torch.dot(expert_count, router_prob_sum) * (num_experts / (total_weight * total_weight))
    return loss, SavedState((gate_logits, attention_mask, expert_count, total_weight), _Meta(top_k, has_mask))


def backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor, None]:
    """Return ``(grad_gate_logits, None)``. Mask is not differentiated."""
    import triton

    meta = saved.metadata
    assert isinstance(meta, _Meta)
    gate_logits, attention_mask, expert_count, total_weight = saved.tensors
    if total_weight == 0:
        return torch.zeros_like(gate_logits), None

    num_layers, _num_tokens, num_experts = gate_logits.shape
    concatenated = gate_logits.reshape(-1, num_experts).contiguous()
    token_count, _ = concatenated.shape
    grad_logits = torch.empty_like(concatenated, dtype=torch.float32)
    block_e = triton.next_power_of_2(num_experts)
    grad_scale = grad_output * num_experts / (total_weight * total_weight)

    if meta.has_mask:
        batch_size, seq_len = attention_mask.shape
        mask_ptr = (
            attention_mask.to(device=gate_logits.device, dtype=torch.float32)
            .expand(num_layers, batch_size, seq_len)
            .reshape(-1)
            .contiguous()
        )
    else:
        mask_ptr = concatenated

    _lb_loss_bwd_kernel()[(token_count,)](
        concatenated,
        expert_count,
        mask_ptr,
        grad_logits,
        grad_scale.contiguous(),
        concatenated.stride(0),
        grad_logits.stride(0),
        token_count,
        E=num_experts,
        BLOCK_E=block_e,
        HAS_MASK=meta.has_mask,
    )
    return grad_logits.to(dtype=gate_logits.dtype).view_as(gate_logits), None
