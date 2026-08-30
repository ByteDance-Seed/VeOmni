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

"""MoE experts eager vs fused / HF references, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from tests.kernels.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    MOE_FUSED_ATOL,
    MOE_FUSED_GRAD_ATOL,
    MOE_FUSED_GRAD_RTOL,
    MOE_FUSED_RTOL,
)
from veomni.kernels import resolve_kernel
from veomni.kernels._kernels.moe_experts.standard.torch_npu import _fc1_weight
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from veomni.utils.import_utils import is_fused_moe_available, is_quack_gemm_available


def _empty(device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.empty(0, device=device, dtype=dtype)


def _clone(tensor: Tensor) -> Tensor:
    return tensor.detach().requires_grad_(True)


def _route(num_tokens: int, num_experts: int, top_k: int, device: torch.device | str, dtype: torch.dtype):
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return routing_weights.to(dtype), selected_experts


def _standard_fused_ref(
    hidden: Tensor,
    routing: Tensor,
    selected: Tensor,
    fc1_1: Tensor,
    fc1_2: Tensor,
    fc2: Tensor,
    *,
    num_experts: int,
    swiglu_limit: float | None = None,
) -> Tensor:
    """Fused operator-order reference for ``moe_experts`` ``standard``.

    Routing weights scale the SwiGLU intermediate, then ``fc2``. Triton
    and Quack non-EP rows use this order. NPU applies routing after fc2.
    """
    output = torch.zeros_like(hidden)
    expert_mask = F.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden[token_idx]
        gate = F.linear(x, fc1_1[idx])
        up = F.linear(x, fc1_2[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.silu(gate) * up
        y = y * routing[token_idx, top_k_pos, None]
        y = F.linear(y, fc2[idx])
        output.index_add_(0, token_idx, y.to(output.dtype))
    return output


def _gpt_oss_hf_loop(
    hidden: Tensor,
    routing: Tensor,
    selected: Tensor,
    gate_up: Tensor,
    gate_up_bias: Tensor,
    down: Tensor,
    down_bias: Tensor,
    *,
    num_experts: int,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> tuple[Tensor, GptOssExperts]:
    """Call HuggingFace ``GptOssExperts.forward``.

    Installed class: ``transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts``
    (``forward`` and ``_apply_gate``).

    Source:
    https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py

    Returns ``(output, experts)`` so weight grads are read from the HF module.
    """
    experts = GptOssExperts(
        GptOssConfig(
            hidden_size=hidden.shape[-1],
            intermediate_size=down.shape[1],
            num_local_experts=num_experts,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    experts.gate_up_proj = nn.Parameter(gate_up)
    experts.gate_up_proj_bias = nn.Parameter(gate_up_bias)
    experts.down_proj = nn.Parameter(down)
    experts.down_proj_bias = nn.Parameter(down_bias)
    experts.alpha = alpha
    experts.limit = limit
    return experts(hidden, selected, routing), experts


def test_eager_matches_fused_reference():
    torch.manual_seed(0)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 8, 4, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    fc1_1 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_2 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim)

    hidden_h, routing_h, fc1_1_h, fc1_2_h, fc2_h = map(_clone, (hidden, routing, fc1_1, fc1_2, fc2))
    out_h = _standard_fused_ref(hidden_h, routing_h, selected, fc1_1_h, fc1_2_h, fc2_h, num_experts=num_experts)

    hidden_e, routing_e, fc1_1_e, fc1_2_e, fc2_e = map(_clone, (hidden, routing, fc1_1, fc1_2, fc2))
    out_e = resolve_kernel("moe_experts", "standard", "eager").wrapper(
        hidden_e,
        routing_e,
        selected,
        fc1_1_e,
        fc1_2_e,
        fc2_e,
        _empty(hidden.device),
        num_experts=num_experts,
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(hidden_e.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(routing_e.grad, routing_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc1_1_e.grad, fc1_1_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc1_2_e.grad, fc1_2_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc2_e.grad, fc2_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_merged_matches_split():
    torch.manual_seed(1)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 6, 3, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    fc1_1 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_2 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_12 = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim)
    wrapper = resolve_kernel("moe_experts", "standard", "eager").wrapper

    hidden_s, routing_s, fc1_1_s, fc1_2_s, fc2_s = map(_clone, (hidden, routing, fc1_1, fc1_2, fc2))
    out_s = wrapper(
        hidden_s, routing_s, selected, fc1_1_s, fc1_2_s, fc2_s, _empty(hidden.device), num_experts=num_experts
    )

    hidden_m, routing_m, fc1_12_m, fc2_m = map(_clone, (hidden, routing, fc1_12, fc2))
    out_m = wrapper(
        hidden_m,
        routing_m,
        selected,
        _empty(hidden.device),
        _empty(hidden.device),
        fc2_m,
        fc1_12_m,
        num_experts=num_experts,
    )
    assert torch.allclose(out_s, out_m, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_s)
    out_s.backward(go)
    out_m.backward(go)
    assert torch.allclose(hidden_s.grad, hidden_m.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(routing_s.grad, routing_m.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(
        torch.cat([fc1_1_s.grad, fc1_2_s.grad], dim=1),
        fc1_12_m.grad,
        atol=EAGER_GRAD_ATOL,
        rtol=EAGER_GRAD_RTOL,
    )
    assert torch.allclose(fc2_s.grad, fc2_m.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_rejects_both_or_neither_fc1():
    hidden = torch.randn(4, 8)
    routing = torch.ones(4, 1)
    selected = torch.zeros(4, 1, dtype=torch.long)
    fc2 = torch.randn(2, 8, 4)
    wrapper = resolve_kernel("moe_experts", "standard", "eager").wrapper
    with pytest.raises(ValueError, match="either split"):
        wrapper(hidden, routing, selected, _empty("cpu"), _empty("cpu"), fc2, _empty("cpu"), num_experts=2)
    with pytest.raises(ValueError, match="either split"):
        wrapper(
            hidden,
            routing,
            selected,
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            fc2,
            torch.randn(2, 8, 8),
            num_experts=2,
        )


def test_gpt_oss_eager_matches_hf_loop():
    torch.manual_seed(2)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 8, 3, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    gate_up = torch.randn(num_experts, hidden_dim, 2 * ffn_dim)
    gate_up_b = torch.randn(num_experts, 2 * ffn_dim)
    down = torch.randn(num_experts, ffn_dim, hidden_dim)
    down_b = torch.randn(num_experts, hidden_dim)
    alpha, limit = 1.702, 7.0

    hidden_h, routing_h, gu_h, gub_h, dn_h, dnb_h = map(_clone, (hidden, routing, gate_up, gate_up_b, down, down_b))
    out_h, experts_h = _gpt_oss_hf_loop(
        hidden_h,
        routing_h,
        selected,
        gu_h,
        gub_h,
        dn_h,
        dnb_h,
        num_experts=num_experts,
        alpha=alpha,
        limit=limit,
    )

    hidden_e, routing_e, gu_e, gub_e, dn_e, dnb_e = map(_clone, (hidden, routing, gate_up, gate_up_b, down, down_b))
    out_e = resolve_kernel("moe_experts", "gpt_oss", "eager").wrapper(
        hidden_e, routing_e, selected, gu_e, gub_e, dn_e, dnb_e, num_experts=num_experts, alpha=alpha, limit=limit
    )
    assert torch.allclose(out_e, out_h, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(hidden_e.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(routing_e.grad, routing_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(gu_e.grad, experts_h.gate_up_proj.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(gub_e.grad, experts_h.gate_up_proj_bias.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(dn_e.grad, experts_h.down_proj.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(dnb_e.grad, experts_h.down_proj_bias.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def _run_fused_vs_eager(
    impl: str,
    *,
    swiglu_limit: float | None = None,
    merged: bool = False,
    shape: tuple[int, int, int, int, int] = (16, 4, 32, 16, 2),
    selected: Tensor | None = None,
    routing: Tensor | None = None,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = torch.device("npu" if impl == "torch_npu" else "cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = shape
    hidden = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    if routing is None or selected is None:
        routing, selected = _route(num_tokens, num_experts, top_k, device, dtype)
    fc1_1 = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2 = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_12 = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    empty = _empty(device, dtype)

    eager = resolve_kernel("moe_experts", "standard", "eager").wrapper
    other = resolve_kernel("moe_experts", "standard", impl).wrapper
    kwargs = {"num_experts": num_experts, "swiglu_limit": swiglu_limit}
    if merged:
        hidden_e, routing_e, fc1_12_e, fc2_e = map(_clone, (hidden, routing, fc1_12, fc2))
        hidden_o, routing_o, fc1_12_o, fc2_o = map(_clone, (hidden, routing, fc1_12, fc2))
        out_e = eager(hidden_e, routing_e, selected, empty, empty, fc2_e, fc1_12_e, **kwargs)
        out_o = other(hidden_o, routing_o, selected, empty, empty, fc2_o, fc1_12_o, **kwargs)
    else:
        hidden_e, routing_e, fc1_1_e, fc1_2_e, fc2_e = map(_clone, (hidden, routing, fc1_1, fc1_2, fc2))
        hidden_o, routing_o, fc1_1_o, fc1_2_o, fc2_o = map(_clone, (hidden, routing, fc1_1, fc1_2, fc2))
        out_e = eager(hidden_e, routing_e, selected, fc1_1_e, fc1_2_e, fc2_e, empty, **kwargs)
        out_o = other(hidden_o, routing_o, selected, fc1_1_o, fc1_2_o, fc2_o, empty, **kwargs)
    assert torch.allclose(out_e.float(), out_o.float(), atol=MOE_FUSED_ATOL, rtol=MOE_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(
        hidden_e.grad.float(), hidden_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
    )
    assert torch.allclose(
        routing_e.grad.float(), routing_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
    )
    assert torch.allclose(fc2_e.grad.float(), fc2_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL)
    if merged:
        assert torch.allclose(
            fc1_12_e.grad.float(), fc1_12_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
        )
    else:
        assert torch.allclose(
            fc1_1_e.grad.float(), fc1_1_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
        )
        assert torch.allclose(
            fc1_2_e.grad.float(), fc1_2_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
        )


def test_npu_fc1_layout_matches_eager_contract():
    with pytest.raises(ValueError, match="either split"):
        _fc1_weight(None, None, None)
    with pytest.raises(ValueError, match="either split"):
        _fc1_weight(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 8, 8))
    with pytest.raises(ValueError, match="both fc1_1_weight and fc1_2_weight"):
        _fc1_weight(torch.randn(2, 4, 8), None, None)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager():
    _run_fused_vs_eager("triton")


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager_merged():
    _run_fused_vs_eager("triton", merged=True)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager_swiglu_limit():
    _run_fused_vs_eager("triton", swiglu_limit=1.0)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager_merged_swiglu_limit():
    _run_fused_vs_eager("triton", merged=True, swiglu_limit=1.0)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager_duplicate_expert():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, top_k = 256, 2
    selected = torch.zeros(num_tokens, top_k, device=device, dtype=torch.long)
    routing = torch.full((num_tokens, top_k), 0.75, device=device, dtype=dtype)
    _run_fused_vs_eager(
        "triton",
        swiglu_limit=10.0,
        shape=(num_tokens, 4, 128, 64, top_k),
        selected=selected,
        routing=routing,
        seed=7,
    )
    _run_fused_vs_eager(
        "triton",
        merged=True,
        swiglu_limit=10.0,
        shape=(num_tokens, 4, 128, 64, top_k),
        selected=selected,
        routing=routing,
        seed=7,
    )


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(), reason="triton fused MoE needs CUDA + triton"
)
def test_triton_matches_eager_larger_gpu():
    _run_fused_vs_eager("triton", shape=(128, 16, 256, 128, 4), seed=11)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="quack fused MoE needs SM90+")
def test_quack_matches_eager():
    _run_fused_vs_eager("quack")


@pytest.mark.skipif(not is_quack_gemm_available(), reason="quack fused MoE needs SM90+")
def test_quack_matches_eager_merged():
    _run_fused_vs_eager("quack", merged=True)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="quack fused MoE needs SM90+")
def test_quack_matches_eager_swiglu_limit():
    _run_fused_vs_eager("quack", swiglu_limit=1.0)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="gpt_oss quack needs SM90+")
def test_gpt_oss_quack_matches_eager():
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 16, 4, 32, 16, 2
    hidden = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing, selected = _route(num_tokens, num_experts, top_k, device, dtype)
    gate_up = 0.1 * torch.randn(num_experts, hidden_dim, 2 * ffn_dim, device=device, dtype=dtype)
    gate_up_b = 0.1 * torch.randn(num_experts, 2 * ffn_dim, device=device, dtype=dtype)
    down = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    down_b = 0.1 * torch.randn(num_experts, hidden_dim, device=device, dtype=dtype)
    eager = resolve_kernel("moe_experts", "gpt_oss", "eager").wrapper
    other = resolve_kernel("moe_experts", "gpt_oss", "quack").wrapper
    hidden_e, routing_e, gu_e, gub_e, dn_e, dnb_e = map(_clone, (hidden, routing, gate_up, gate_up_b, down, down_b))
    hidden_o, routing_o, gu_o, gub_o, dn_o, dnb_o = map(_clone, (hidden, routing, gate_up, gate_up_b, down, down_b))
    kwargs = {"num_experts": num_experts, "alpha": 1.702, "limit": 7.0}
    out_e = eager(hidden_e, routing_e, selected, gu_e, gub_e, dn_e, dnb_e, **kwargs)
    out_o = other(hidden_o, routing_o, selected, gu_o, gub_o, dn_o, dnb_o, **kwargs)
    assert torch.allclose(out_e.float(), out_o.float(), atol=MOE_FUSED_ATOL, rtol=MOE_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(
        hidden_e.grad.float(), hidden_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
    )
    assert torch.allclose(
        routing_e.grad.float(), routing_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL
    )
    assert torch.allclose(gu_e.grad.float(), gu_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL)
    assert torch.allclose(gub_e.grad.float(), gub_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL)
    assert torch.allclose(dn_e.grad.float(), dn_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL)
    assert torch.allclose(dnb_e.grad.float(), dnb_o.grad.float(), atol=MOE_FUSED_GRAD_ATOL, rtol=MOE_FUSED_GRAD_RTOL)


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU fused MoE needs torch_npu")
def test_npu_matches_eager():
    _run_fused_vs_eager("torch_npu")
