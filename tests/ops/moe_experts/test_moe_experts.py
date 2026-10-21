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

"""MoE experts eager vs fused / HF references, and fused impls vs eager."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from transformers import GptOssConfig, Qwen3MoeConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from tests.ops.moe_experts.reference import standard_ep_reference, standard_fused_reference
from tests.ops.tol import (
    EAGER_ATOL,
    EAGER_GRAD_ATOL,
    EAGER_GRAD_RTOL,
    EAGER_RTOL,
    MOE_FUSED_ATOL,
    MOE_FUSED_GRAD_FC1_ATOL,
    MOE_FUSED_GRAD_FC1_RTOL,
    MOE_FUSED_GRAD_FC2_ATOL,
    MOE_FUSED_GRAD_FC2_RTOL,
    MOE_FUSED_GRAD_HIDDEN_ATOL,
    MOE_FUSED_GRAD_HIDDEN_RTOL,
    MOE_FUSED_PRODUCTION_PRE_SM90_GRAD_ROUTING_ATOL,
    MOE_FUSED_PRODUCTION_PRE_SM90_GRAD_ROUTING_RTOL,
    MOE_FUSED_RTOL,
    MOE_FUSED_SWIGLU_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC1_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC1_RTOL,
    MOE_FUSED_SWIGLU_GRAD_FC2_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC2_RTOL,
    MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL,
    MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL,
    MOE_FUSED_SWIGLU_RTOL,
    MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
    MOE_SPLIT_MERGED_GRAD_HIDDEN_RTOL,
)
from tests.ops.utils import assert_close_with_error, assert_reference_signal, make_grad_leaf, require_nvidia_cuda
from veomni.ops import resolve_op
from veomni.ops.kernels.moe_experts.shared.indices import build_moe_indices
from veomni.ops.kernels.moe_experts.standard import triton as triton_moe_module
from veomni.ops.kernels.moe_experts.standard.npu import _fc1_weight
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_MLU_AVAILABLE,
    IS_NPU_AVAILABLE,
    is_sm90_or_above,
)
from veomni.utils.import_utils import is_fused_moe_available, is_quack_gemm_available


def _empty(device: torch.device | str, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.empty(0, device=device, dtype=dtype)


def _route(num_tokens: int, num_experts: int, top_k: int, device: torch.device | str, dtype: torch.dtype):
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    return routing_weights.to(dtype), selected_experts


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


def _qwen3_moe_hf_experts(
    hidden: Tensor,
    routing: Tensor,
    selected: Tensor,
    gate_up: Tensor,
    down: Tensor,
    *,
    num_experts: int,
) -> tuple[Tensor, Qwen3MoeExperts]:
    """Call HuggingFace ``Qwen3MoeExperts.forward``.

    Installed class: ``transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts``.

    HF scales routing after ``down_proj``, the same order as our eager row.
    """
    config = Qwen3MoeConfig(
        hidden_size=hidden.shape[-1],
        moe_intermediate_size=down.shape[-1],
        num_experts=num_experts,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config._experts_implementation = "eager"
    experts = Qwen3MoeExperts(config)
    experts.gate_up_proj = nn.Parameter(gate_up)
    experts.down_proj = nn.Parameter(down)
    return experts(hidden, selected, routing), experts


def test_standard_eager_matches_hf_qwen3_moe_experts():
    torch.manual_seed(0)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 8, 4, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    fc1_1 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_2 = torch.randn(num_experts, ffn_dim, hidden_dim)
    gate_up = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim)

    hidden_h, routing_h, gu_h, fc2_h = map(make_grad_leaf, (hidden, routing, gate_up, fc2))
    out_h, experts_h = _qwen3_moe_hf_experts(hidden_h, routing_h, selected, gu_h, fc2_h, num_experts=num_experts)

    hidden_e, routing_e, gu_e, fc2_e = map(make_grad_leaf, (hidden, routing, gate_up, fc2))
    out_e = resolve_op("moe_experts", "standard", "eager").wrapper(
        hidden_e,
        routing_e,
        selected,
        _empty(hidden.device),
        _empty(hidden.device),
        fc2_e,
        gu_e,
        num_experts=num_experts,
    )
    assert torch.equal(out_e, out_h)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.equal(hidden_e.grad, hidden_h.grad)
    assert torch.equal(routing_e.grad, routing_h.grad)
    assert torch.equal(gu_e.grad, experts_h.gate_up_proj.grad)
    assert torch.equal(fc2_e.grad, experts_h.down_proj.grad)


def test_eager_matches_fused_reference():
    """Fused rows scale routing before ``fc2``; eager after. Equal up to rounding."""
    torch.manual_seed(0)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 8, 4, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    fc1_1 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_2 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim)

    hidden_h, routing_h, fc1_1_h, fc1_2_h, fc2_h = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
    out_h = standard_fused_reference(
        hidden_h,
        routing_h,
        selected,
        fc1_1_h,
        fc1_2_h,
        fc2_h,
        num_experts=num_experts,
    )

    hidden_e, routing_e, fc1_1_e, fc1_2_e, fc2_e = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
    out_e = resolve_op("moe_experts", "standard", "eager").wrapper(
        hidden_e,
        routing_e,
        selected,
        fc1_1_e,
        fc1_2_e,
        fc2_e,
        _empty(hidden.device),
        num_experts=num_experts,
    )
    assert torch.allclose(out_e, out_h, atol=1e-5, rtol=1e-5)

    go = torch.randn_like(out_e)
    out_h.backward(go)
    out_e.backward(go)
    assert torch.allclose(hidden_e.grad, hidden_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(routing_e.grad, routing_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc1_1_e.grad, fc1_1_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc1_2_e.grad, fc1_2_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(fc2_e.grad, fc2_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_eager_keeps_compute_dtype_when_routing_is_fp32():
    """DSV3-style router scores are fp32 while FSDP2 compute is bf16.

    Routing scales the ``fc2`` output, so batch-invariant ``F.linear`` still
    sees compute-dtype operands, and the result returns in compute dtype.
    """
    require_nvidia_cuda()
    from veomni.ops.batch_invariant import set_batch_invariant_mode

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 6, 3, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing, selected = _route(num_tokens, num_experts, top_k, device, torch.float32)
    fc1_12 = torch.randn(num_experts, 2 * ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    empty = _empty(device, dtype)
    wrapper = resolve_op("moe_experts", "standard", "eager").wrapper

    with set_batch_invariant_mode(True):
        out = wrapper(hidden, routing, selected, empty, empty, fc2, fc1_12, num_experts=num_experts)

    assert out.dtype == dtype
    assert out.shape == hidden.shape


def test_fused_triton_keeps_compute_dtype_when_routing_is_fp32():
    """DSV3-style router scores are fp32 while FSDP2 compute is bf16."""
    require_nvidia_cuda()
    if not is_fused_moe_available():
        pytest.skip("fused MoE kernel is unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 6, 3, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing, selected = _route(num_tokens, num_experts, top_k, device, torch.float32)
    fc1_12 = torch.randn(num_experts, 2 * ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    empty = _empty(device, dtype)
    wrapper = resolve_op("moe_experts", "standard", "fused_triton").wrapper

    out = wrapper(hidden, routing, selected, empty, empty, fc2, fc1_12, num_experts=num_experts)

    assert out.dtype == dtype
    assert out.shape == hidden.shape


def test_eager_merged_matches_split():
    torch.manual_seed(1)
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 6, 3, 16, 8, 2
    hidden = torch.randn(num_tokens, hidden_dim)
    routing, selected = _route(num_tokens, num_experts, top_k, hidden.device, hidden.dtype)
    fc1_1 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_2 = torch.randn(num_experts, ffn_dim, hidden_dim)
    fc1_12 = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = torch.randn(num_experts, hidden_dim, ffn_dim)
    wrapper = resolve_op("moe_experts", "standard", "eager").wrapper

    hidden_s, routing_s, fc1_1_s, fc1_2_s, fc2_s = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
    out_s = wrapper(
        hidden_s, routing_s, selected, fc1_1_s, fc1_2_s, fc2_s, _empty(hidden.device), num_experts=num_experts
    )

    hidden_m, routing_m, fc1_12_m, fc2_m = map(make_grad_leaf, (hidden, routing, fc1_12, fc2))
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
    wrapper = resolve_op("moe_experts", "standard", "eager").wrapper
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

    hidden_h, routing_h, gu_h, gub_h, dn_h, dnb_h = map(
        make_grad_leaf, (hidden, routing, gate_up, gate_up_b, down, down_b)
    )
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

    hidden_e, routing_e, gu_e, gub_e, dn_e, dnb_e = map(
        make_grad_leaf, (hidden, routing, gate_up, gate_up_b, down, down_b)
    )
    out_e = resolve_op("moe_experts", "gpt_oss", "eager").wrapper(
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
    device: torch.device | None = None,
    data_scale: float = 0.4,
    require_active_clamp: bool = False,
):
    torch.manual_seed(seed)
    if device is None:
        if impl == "fused_npu":
            device = torch.device("npu")
        elif impl == "fused_mlu":
            device = torch.device("mlu")
        else:
            device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = shape
    hidden = data_scale * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    if routing is None or selected is None:
        routing, selected = _route(num_tokens, num_experts, top_k, device, dtype)
    fc1_1 = data_scale * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2 = data_scale * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_12 = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = data_scale * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    empty = _empty(device, dtype)

    eager = resolve_op("moe_experts", "standard", "eager").wrapper
    other = resolve_op("moe_experts", "standard", impl).wrapper
    kwargs = {"num_experts": num_experts, "swiglu_limit": swiglu_limit}
    if require_active_clamp:
        assert swiglu_limit is not None
        routed_hidden = hidden[:, None, :].expand(-1, top_k, -1)
        gate = torch.einsum("tkh,tkfh->tkf", routed_hidden, fc1_1[selected])
        up = torch.einsum("tkh,tkfh->tkf", routed_hidden, fc1_2[selected])
        assert (gate > swiglu_limit).any() or (up.abs() > swiglu_limit).any()
    if merged:
        hidden_e, routing_e, fc1_12_e, fc2_e = map(make_grad_leaf, (hidden, routing, fc1_12, fc2))
        hidden_o, routing_o, fc1_12_o, fc2_o = map(make_grad_leaf, (hidden, routing, fc1_12, fc2))
        out_e = eager(hidden_e, routing_e, selected, empty, empty, fc2_e, fc1_12_e, **kwargs)
        out_o = other(hidden_o, routing_o, selected, empty, empty, fc2_o, fc1_12_o, **kwargs)
    else:
        hidden_e, routing_e, fc1_1_e, fc1_2_e, fc2_e = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
        hidden_o, routing_o, fc1_1_o, fc1_2_o, fc2_o = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
        out_e = eager(hidden_e, routing_e, selected, fc1_1_e, fc1_2_e, fc2_e, empty, **kwargs)
        out_o = other(hidden_o, routing_o, selected, fc1_1_o, fc1_2_o, fc2_o, empty, **kwargs)
    if swiglu_limit is not None:
        fwd_atol, fwd_rtol = MOE_FUSED_SWIGLU_ATOL, MOE_FUSED_SWIGLU_RTOL
        hidden_atol, hidden_rtol = MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL, MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL
        fc1_atol, fc1_rtol = MOE_FUSED_SWIGLU_GRAD_FC1_ATOL, MOE_FUSED_SWIGLU_GRAD_FC1_RTOL
        fc2_atol, fc2_rtol = MOE_FUSED_SWIGLU_GRAD_FC2_ATOL, MOE_FUSED_SWIGLU_GRAD_FC2_RTOL
    else:
        fwd_atol, fwd_rtol = MOE_FUSED_ATOL, MOE_FUSED_RTOL
        hidden_atol, hidden_rtol = MOE_FUSED_GRAD_HIDDEN_ATOL, MOE_FUSED_GRAD_HIDDEN_RTOL
        fc1_atol, fc1_rtol = MOE_FUSED_GRAD_FC1_ATOL, MOE_FUSED_GRAD_FC1_RTOL
        fc2_atol, fc2_rtol = MOE_FUSED_GRAD_FC2_ATOL, MOE_FUSED_GRAD_FC2_RTOL
    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    reference_checks = [
        ("output", out_e, fwd_atol, fwd_rtol),
        ("hidden gradient", hidden_e.grad, hidden_atol, hidden_rtol),
        ("routing gradient", routing_e.grad, hidden_atol, hidden_rtol),
        ("fc2 gradient", fc2_e.grad, fc2_atol, fc2_rtol),
    ]
    if merged:
        reference_checks.append(("fc1 gradient", fc1_12_e.grad, fc1_atol, fc1_rtol))
    else:
        reference_checks.extend(
            (
                ("fc1_1 gradient", fc1_1_e.grad, fc1_atol, fc1_rtol),
                ("fc1_2 gradient", fc1_2_e.grad, fc1_atol, fc1_rtol),
            )
        )
    for name, reference, atol, rtol in reference_checks:
        assert_reference_signal(name, reference, atol, rtol)
    assert torch.allclose(out_e.float(), out_o.float(), atol=fwd_atol, rtol=fwd_rtol)
    assert torch.allclose(hidden_e.grad.float(), hidden_o.grad.float(), atol=hidden_atol, rtol=hidden_rtol)
    assert torch.allclose(routing_e.grad.float(), routing_o.grad.float(), atol=hidden_atol, rtol=hidden_rtol)
    assert torch.allclose(fc2_e.grad.float(), fc2_o.grad.float(), atol=fc2_atol, rtol=fc2_rtol)
    if merged:
        assert torch.allclose(fc1_12_e.grad.float(), fc1_12_o.grad.float(), atol=fc1_atol, rtol=fc1_rtol)
    else:
        assert torch.allclose(fc1_1_e.grad.float(), fc1_1_o.grad.float(), atol=fc1_atol, rtol=fc1_rtol)
        assert torch.allclose(fc1_2_e.grad.float(), fc1_2_o.grad.float(), atol=fc1_atol, rtol=fc1_rtol)


def _run_fused_three_way(
    impl: str,
    *,
    swiglu_limit: float | None = None,
    shape: tuple[int, int, int, int, int] = (16, 4, 32, 16, 2),
    selected: Tensor | None = None,
    routing: Tensor | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    data_scale: float = 0.1,
    require_active_clamp: bool = False,
    grad_routing_atol: float | None = None,
    grad_routing_rtol: float | None = None,
    assume_distinct_experts: bool = False,
):
    """Compare one fused implementation's split and merged layouts with eager math.

    Non-EP Triton uses ``standard_ep_reference`` (post-``fc2`` routing-weight
    order matching EP and Hugging Face eager), while Quack uses
    ``standard_fused_reference`` (pre-``fc2`` routing-weight order).
    """
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = shape
    hidden = data_scale * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    if routing is None or selected is None:
        routing, selected = _route(num_tokens, num_experts, top_k, device, dtype)
    fc1_1 = data_scale * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2 = data_scale * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_12 = torch.cat([fc1_1, fc1_2], dim=1).contiguous()
    fc2 = data_scale * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    empty = _empty(device, dtype)
    fused = resolve_op("moe_experts", "standard", impl).wrapper
    kwargs = {"num_experts": num_experts, "swiglu_limit": swiglu_limit}
    fused_kwargs = dict(kwargs)
    if impl == "fused_triton":
        fused_kwargs["assume_distinct_experts"] = assume_distinct_experts

    if require_active_clamp:
        assert swiglu_limit is not None
        routed_hidden = hidden[:, None, :].expand(-1, top_k, -1)
        gate = torch.einsum("tkh,tkfh->tkf", routed_hidden, fc1_1[selected])
        up = torch.einsum("tkh,tkfh->tkf", routed_hidden, fc1_2[selected])
        assert (gate > swiglu_limit).any() or (up.abs() > swiglu_limit).any()

    hidden_s, routing_s, fc1_1_s, fc1_2_s, fc2_s = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
    hidden_m, routing_m, fc1_12_m, fc2_m = map(make_grad_leaf, (hidden, routing, fc1_12, fc2))
    hidden_e, routing_e, fc1_1_e, fc1_2_e, fc2_e = map(make_grad_leaf, (hidden, routing, fc1_1, fc1_2, fc2))
    out_s = fused(hidden_s, routing_s, selected, fc1_1_s, fc1_2_s, fc2_s, empty, **fused_kwargs)
    out_m = fused(hidden_m, routing_m, selected, empty, empty, fc2_m, fc1_12_m, **fused_kwargs)
    if impl == "fused_triton":
        # Eleven upstream tensors plus the optional device tile prefix.
        assert len(out_m.grad_fn.saved_tensors) == 12
    reference_fn = standard_ep_reference if impl == "fused_triton" else standard_fused_reference
    out_e = reference_fn(hidden_e, routing_e, selected, fc1_1_e, fc1_2_e, fc2_e, **kwargs)
    torch.testing.assert_close(out_s, out_m, rtol=0, atol=0)

    go = torch.randn_like(out_s)
    out_s.backward(go)
    out_m.backward(go)
    out_e.backward(go)
    torch.testing.assert_close(fc2_s.grad, fc2_m.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([fc1_1_s.grad, fc1_2_s.grad], dim=1),
        fc1_12_m.grad,
        rtol=0,
        atol=0,
    )
    assert torch.allclose(
        hidden_s.grad.float(),
        hidden_m.grad.float(),
        atol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
        rtol=MOE_SPLIT_MERGED_GRAD_HIDDEN_RTOL,
    )
    assert torch.allclose(
        routing_s.grad.float(),
        routing_m.grad.float(),
        atol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
        rtol=MOE_SPLIT_MERGED_GRAD_HIDDEN_RTOL,
    )
    if swiglu_limit is not None:
        fwd_atol, fwd_rtol = MOE_FUSED_SWIGLU_ATOL, MOE_FUSED_SWIGLU_RTOL
        hidden_atol, hidden_rtol = MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL, MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL
        fc1_atol, fc1_rtol = MOE_FUSED_SWIGLU_GRAD_FC1_ATOL, MOE_FUSED_SWIGLU_GRAD_FC1_RTOL
        fc2_atol, fc2_rtol = MOE_FUSED_SWIGLU_GRAD_FC2_ATOL, MOE_FUSED_SWIGLU_GRAD_FC2_RTOL
    else:
        fwd_atol, fwd_rtol = MOE_FUSED_ATOL, MOE_FUSED_RTOL
        hidden_atol, hidden_rtol = MOE_FUSED_GRAD_HIDDEN_ATOL, MOE_FUSED_GRAD_HIDDEN_RTOL
        fc1_atol, fc1_rtol = MOE_FUSED_GRAD_FC1_ATOL, MOE_FUSED_GRAD_FC1_RTOL
        fc2_atol, fc2_rtol = MOE_FUSED_GRAD_FC2_ATOL, MOE_FUSED_GRAD_FC2_RTOL
    routing_atol = hidden_atol if grad_routing_atol is None else grad_routing_atol
    routing_rtol = hidden_rtol if grad_routing_rtol is None else grad_routing_rtol
    reference_checks = (
        ("output", out_e, fwd_atol, fwd_rtol),
        ("hidden gradient", hidden_e.grad, hidden_atol, hidden_rtol),
        ("routing gradient", routing_e.grad, routing_atol, routing_rtol),
        ("fc1_1 gradient", fc1_1_e.grad, fc1_atol, fc1_rtol),
        ("fc1_2 gradient", fc1_2_e.grad, fc1_atol, fc1_rtol),
        ("fc2 gradient", fc2_e.grad, fc2_atol, fc2_rtol),
    )
    for name, reference, atol, rtol in reference_checks:
        assert_reference_signal(name, reference, atol, rtol)
    assert_close_with_error("output", out_m.float(), out_e.float(), atol=fwd_atol, rtol=fwd_rtol)
    assert_close_with_error(
        "hidden gradient", hidden_m.grad.float(), hidden_e.grad.float(), atol=hidden_atol, rtol=hidden_rtol
    )
    assert_close_with_error(
        "routing gradient", routing_m.grad.float(), routing_e.grad.float(), atol=routing_atol, rtol=routing_rtol
    )
    assert_close_with_error("fc2 gradient", fc2_m.grad.float(), fc2_e.grad.float(), atol=fc2_atol, rtol=fc2_rtol)
    assert_close_with_error(
        "fc1 gradient",
        fc1_12_m.grad.float(),
        torch.cat([fc1_1_e.grad, fc1_2_e.grad], dim=1).float(),
        atol=fc1_atol,
        rtol=fc1_rtol,
    )


def test_reference_signal_rejects_all_zero_and_tiny_values():
    """NPU/MLU comparisons must fail when the reference cannot beat the budget."""
    zeros = torch.zeros(4, 4)
    with pytest.raises(AssertionError, match="signal is too small"):
        assert_reference_signal("zeros", zeros, atol=1e-2, rtol=1e-2)
    tiny = torch.full((4, 4), 1e-4)
    with pytest.raises(AssertionError, match="signal is too small"):
        assert_reference_signal("tiny", tiny, atol=1e-2, rtol=1e-2)
    assert_reference_signal("ones", torch.ones(4, 4), atol=1e-2, rtol=1e-2)


def test_npu_fc1_layout_matches_eager_contract():
    with pytest.raises(ValueError, match="either split"):
        _fc1_weight(None, None, None)
    with pytest.raises(ValueError, match="either split"):
        _fc1_weight(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 8, 8))
    with pytest.raises(ValueError, match="both fc1_1_weight and fc1_2_weight"):
        _fc1_weight(torch.randn(2, 4, 8), None, None)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
@pytest.mark.parametrize("swiglu_limit", (None, 1.0))
def test_triton_split_and_merged_match_eager(swiglu_limit: float | None):
    _run_fused_three_way(
        "fused_triton",
        swiglu_limit=swiglu_limit,
        data_scale=0.4,
        require_active_clamp=swiglu_limit is not None,
    )


@pytest.mark.parametrize(
    "device_type",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
                reason="triton fused MoE needs a GPU + triton",
            ),
        ),
        pytest.param(
            "mlu",
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE or not is_fused_moe_available(),
                reason="MLU Triton fused MoE needs torch_mlu + triton",
            ),
        ),
    ],
)
@pytest.mark.parametrize("fc1_trainable,fc2_trainable", [(False, True), (True, False), (False, False)])
def test_merged_fc1_backward_with_frozen_expert_weights(fc1_trainable: bool, fc2_trainable: bool, device_type: str):
    """Skipping an expert wgrad must not change input or routing gradients."""
    torch.manual_seed(7)
    device = torch.device(device_type)
    num_tokens, num_experts, hidden_dim, ffn_dim, topk = 32, 4, 64, 128, 2
    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    gate_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.bfloat16)
    expert_index = torch.randint(num_experts, (num_tokens, topk), device=device)
    fc1_weight = 0.1 * torch.randn(num_experts, 2 * ffn_dim, hidden_dim, device=device, dtype=torch.bfloat16)
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=torch.bfloat16)
    empty = _empty(device, torch.bfloat16)
    fused = resolve_op("moe_experts", "standard", "fused_triton").wrapper

    def run(train_fc1: bool, train_fc2: bool):
        """Execute the registered merged path with the selected trainable weights."""
        routed = gate_weights.detach().clone().requires_grad_(True)
        hidden = hidden_states.detach().clone().requires_grad_(True)
        fc1 = fc1_weight.detach().clone().requires_grad_(train_fc1)
        fc2 = fc2_weight.detach().clone().requires_grad_(train_fc2)
        output = fused(hidden, routed, expert_index, empty, empty, fc2, fc1, num_experts=num_experts)
        output.sum().backward()
        return routed.grad, hidden.grad, fc1.grad, fc2.grad

    reference = run(True, True)
    candidate = run(fc1_trainable, fc2_trainable)
    for actual, expected in zip(candidate[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    for actual, expected, trainable in zip(candidate[2:], reference[2:], (fc1_trainable, fc2_trainable), strict=True):
        if trainable:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        else:
            assert actual is None


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
def test_triton_split_and_merged_match_eager_duplicate_expert():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, top_k = 256, 2
    selected = torch.zeros(num_tokens, top_k, device=device, dtype=torch.long)
    routing = torch.full((num_tokens, top_k), 0.75, device=device, dtype=dtype)
    _run_fused_three_way(
        "fused_triton",
        swiglu_limit=10.0,
        shape=(num_tokens, 4, 128, 64, top_k),
        selected=selected,
        routing=routing,
        seed=7,
    )


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
def test_triton_split_and_merged_match_eager_larger_gpu():
    _run_fused_three_way("fused_triton", shape=(128, 16, 256, 128, 4), seed=11)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="quack fused MoE needs SM90+")
@pytest.mark.parametrize("swiglu_limit", (None, 1.0))
def test_quack_split_and_merged_match_eager(swiglu_limit: float | None):
    _run_fused_three_way(
        "fused_quack",
        swiglu_limit=swiglu_limit,
        data_scale=0.4,
        require_active_clamp=swiglu_limit is not None,
    )


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
    eager = resolve_op("moe_experts", "gpt_oss", "eager").wrapper
    other = resolve_op("moe_experts", "gpt_oss", "fused_quack").wrapper
    hidden_e, routing_e, gu_e, gub_e, dn_e, dnb_e = map(
        make_grad_leaf, (hidden, routing, gate_up, gate_up_b, down, down_b)
    )
    hidden_o, routing_o, gu_o, gub_o, dn_o, dnb_o = map(
        make_grad_leaf, (hidden, routing, gate_up, gate_up_b, down, down_b)
    )
    kwargs = {"num_experts": num_experts, "alpha": 1.702, "limit": 7.0}
    out_e = eager(hidden_e, routing_e, selected, gu_e, gub_e, dn_e, dnb_e, **kwargs)
    out_o = other(hidden_o, routing_o, selected, gu_o, gub_o, dn_o, dnb_o, **kwargs)
    assert torch.allclose(out_e.float(), out_o.float(), atol=MOE_FUSED_ATOL, rtol=MOE_FUSED_RTOL)

    go = torch.randn_like(out_e)
    out_e.backward(go)
    out_o.backward(go)
    assert torch.allclose(
        hidden_e.grad.float(),
        hidden_o.grad.float(),
        atol=MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL,
        rtol=MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL,
    )
    assert torch.allclose(
        routing_e.grad.float(),
        routing_o.grad.float(),
        atol=MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL,
        rtol=MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL,
    )
    assert torch.allclose(
        gu_e.grad.float(), gu_o.grad.float(), atol=MOE_FUSED_SWIGLU_GRAD_FC1_ATOL, rtol=MOE_FUSED_SWIGLU_GRAD_FC1_RTOL
    )
    assert torch.allclose(
        gub_e.grad.float(),
        gub_o.grad.float(),
        atol=MOE_FUSED_SWIGLU_GRAD_FC1_ATOL,
        rtol=MOE_FUSED_SWIGLU_GRAD_FC1_RTOL,
    )
    assert torch.allclose(
        dn_e.grad.float(), dn_o.grad.float(), atol=MOE_FUSED_SWIGLU_GRAD_FC2_ATOL, rtol=MOE_FUSED_SWIGLU_GRAD_FC2_RTOL
    )
    assert torch.allclose(
        dnb_e.grad.float(),
        dnb_o.grad.float(),
        atol=MOE_FUSED_SWIGLU_GRAD_FC2_ATOL,
        rtol=MOE_FUSED_SWIGLU_GRAD_FC2_RTOL,
    )


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
@pytest.mark.parametrize("swiglu_limit", (None, 1.0))
def test_fused_vs_eager_helper_has_useful_signal_on_cuda(swiglu_limit: float | None):
    """The NPU/MLU helper must keep a useful reference signal on the GPU path."""
    require_active_clamp = swiglu_limit is not None
    _run_fused_vs_eager("fused_triton", swiglu_limit=swiglu_limit, require_active_clamp=require_active_clamp)
    _run_fused_vs_eager(
        "fused_triton", swiglu_limit=swiglu_limit, merged=True, require_active_clamp=require_active_clamp
    )


@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="NPU fused MoE needs torch_npu")
@pytest.mark.parametrize("merged", (False, True), ids=("split", "merged"))
@pytest.mark.parametrize("swiglu_limit", (None, 1.0))
def test_npu_matches_eager(swiglu_limit: float | None, merged: bool):
    _run_fused_vs_eager(
        "fused_npu", swiglu_limit=swiglu_limit, merged=merged, require_active_clamp=swiglu_limit is not None
    )


@pytest.mark.skipif(not IS_MLU_AVAILABLE, reason="MLU fused MoE needs torch_mlu")
def test_mlu_matches_eager():
    _run_fused_vs_eager("fused_mlu")


@pytest.mark.skipif(not IS_MLU_AVAILABLE, reason="MLU Triton fused MoE needs torch_mlu")
def test_triton_matches_eager_on_mlu():
    _run_fused_vs_eager("fused_triton", device=torch.device("mlu"))


@pytest.mark.skipif(not IS_MLU_AVAILABLE, reason="MLU fused MoE needs torch_mlu")
def test_mlu_matches_eager_merged():
    _run_fused_vs_eager("fused_mlu", merged=True)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
@pytest.mark.parametrize("swiglu_limit", [7.0, 10.0])
def test_triton_split_and_merged_match_eager_swiglu_limit(swiglu_limit: float):
    _run_fused_three_way("fused_triton", swiglu_limit=swiglu_limit, shape=(128, 8, 512, 256, 2), seed=42)


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or not is_fused_moe_available(),
    reason="triton fused MoE needs a GPU + triton",
)
@pytest.mark.parametrize(
    "shape,seed",
    [
        ((512, 128, 2048, 768, 8), 0),
        ((256, 64, 2048, 1408, 6), 1),
    ],
)
def test_triton_split_and_merged_match_eager_production(shape: tuple[int, int, int, int, int], seed: int):
    kwargs = {}
    if not is_sm90_or_above():
        kwargs["grad_routing_atol"] = MOE_FUSED_PRODUCTION_PRE_SM90_GRAD_ROUTING_ATOL
        kwargs["grad_routing_rtol"] = MOE_FUSED_PRODUCTION_PRE_SM90_GRAD_ROUTING_RTOL
    _run_fused_three_way("fused_triton", shape=shape, seed=seed, **kwargs)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="quack fused MoE needs SM90+")
@pytest.mark.parametrize(
    "shape",
    [
        (64, 8, 256, 128, 2),
        (128, 128, 2048, 768, 8),
        (512, 128, 2048, 1024, 8),
        (1024, 64, 2048, 1408, 6),
    ],
)
def test_quack_split_and_merged_match_eager_production(shape: tuple[int, int, int, int, int]):
    _run_fused_three_way("fused_quack", shape=shape, seed=42)


def test_build_moe_indices_basic_example():
    expert_index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]])
    cu_seqlens_m, a_idx, scatter_index = build_moe_indices(expert_index, num_experts=3)
    assert cu_seqlens_m.tolist() == [0, 3, 6, 8]
    assert a_idx.tolist() == [0, 1, 3, 1, 2, 3, 0, 2]
    dummy_sorted = torch.arange(expert_index.numel(), dtype=torch.float32)
    gathered = dummy_sorted[scatter_index.flatten().long()]
    re_sorted = torch.empty_like(dummy_sorted)
    re_sorted[scatter_index.flatten().long()] = gathered
    assert torch.equal(re_sorted, dummy_sorted)


def test_build_moe_indices_all_same_expert():
    expert_index = torch.zeros(8, 1, dtype=torch.long)
    cu_seqlens_m, a_idx, _scatter_index = build_moe_indices(expert_index, num_experts=4)
    assert cu_seqlens_m.tolist() == [0, 8, 8, 8, 8]
    assert a_idx.tolist() == list(range(8))


@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature")
@pytest.mark.filterwarnings("ignore:Logical operators 'and' and 'or' are deprecated")
def test_build_moe_indices_cuda_does_not_synchronize(monkeypatch):
    """CUDA routing uses the fixed-size Triton histogram without a host sync."""
    require_nvidia_cuda("triton")
    expert_index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]], device="cuda", dtype=torch.int32)

    def reject_bincount(*_args, **_kwargs):
        raise AssertionError("CUDA routing must not use torch.bincount")

    monkeypatch.setattr(torch, "bincount", reject_bincount)
    build_moe_indices(expert_index, num_experts=3)  # compile before enabling the sync detector
    torch.cuda.synchronize()

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cu_seqlens_m, a_idx, scatter_index = build_moe_indices(expert_index, num_experts=3)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert cu_seqlens_m.cpu().tolist() == [0, 3, 6, 8]
    assert a_idx.cpu().tolist() == [0, 1, 3, 1, 2, 3, 0, 2]
    assert scatter_index.shape == expert_index.shape


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("compiling", True),
        ("capturing", True),
        ("eager", False),
        ("unknown", True),
        ("noncallable", True),
        ("ambiguous", True),
        ("query_error", True),
    ],
)
def test_merged_moe_shape_bound_capture_fallback(mode: str, expected: bool, monkeypatch: pytest.MonkeyPatch):
    """Select the original schedule during compilation or unknown capture state."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: mode == "compiling")

    if mode == "compiling":

        def unexpected_device_query():
            raise AssertionError("Compilation must not query a live device")

        monkeypatch.setattr(triton_moe_module, "get_torch_device", unexpected_device_query)
    elif mode == "unknown":
        monkeypatch.setattr(triton_moe_module, "get_torch_device", object)
    elif mode == "noncallable":

        class DeviceAPI:
            is_current_stream_capturing = False

        monkeypatch.setattr(triton_moe_module, "get_torch_device", DeviceAPI)
    else:

        class DeviceAPI:
            def is_current_stream_capturing(self):
                if mode == "query_error":
                    raise RuntimeError("Capture state unavailable")
                if mode == "ambiguous":
                    return None
                return mode == "capturing"

        monkeypatch.setattr(triton_moe_module, "get_torch_device", DeviceAPI)

    assert triton_moe_module._must_use_shape_derived_bound() is expected


@pytest.mark.parametrize(
    "routing_kind,assume_distinct_experts",
    [("distinct", False), ("distinct", True), ("duplicate", False), ("concentrated", False)],
)
def test_triton_merged_same_nk_compact_schedule(
    routing_kind: str, assume_distinct_experts: bool, monkeypatch: pytest.MonkeyPatch
):
    """Build one prefix and reuse it for all four merged forward/backward calls."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 128, 8, 2
    torch.manual_seed(42)
    if routing_kind == "duplicate":
        selected = torch.zeros(num_tokens, top_k, device=device, dtype=torch.long)
    elif routing_kind == "concentrated":
        selected = torch.arange(top_k, device=device).expand(num_tokens, top_k).contiguous()
    else:
        selected = torch.randn(num_tokens, num_experts, device=device).topk(top_k, dim=-1).indices
    routing = torch.full((num_tokens, top_k), 1 / top_k, device=device, dtype=torch.bfloat16)
    shape_bound = num_tokens if assume_distinct_experts else num_tokens * top_k
    nk_bounds, mn_bounds, schedules, prefixes = [], [], [], []
    original_nk, original_mn = gemm_module.group_gemm_same_nk, gemm_module.group_gemm_same_mn
    original_prefix = gemm_module.merged_same_nk_tile_prefix

    def record_prefix(*args, **kwargs):
        """Retain the prefix built by the actual native helper."""
        prefix = original_prefix(*args, **kwargs)
        prefixes.append(prefix)
        return prefix

    def record_nk(*args, **kwargs):
        """Record the shape fallback and optional schedule without changing them."""
        nk_bounds.append(kwargs["max_M"])
        schedules.append(kwargs.get("tile_cumsum"))
        return original_nk(*args, **kwargs)

    def record_mn(*args, **kwargs):
        """Record the total-row reduction bound for weight gradients."""
        mn_bounds.append(kwargs["max_K"])
        return original_mn(*args, **kwargs)

    monkeypatch.setattr(gemm_module, "group_gemm_same_nk", record_nk)
    monkeypatch.setattr(gemm_module, "group_gemm_same_mn", record_mn)
    monkeypatch.setattr(gemm_module, "merged_same_nk_tile_prefix", record_prefix)
    _run_fused_three_way(
        "fused_triton",
        shape=(num_tokens, num_experts, 128, 64, top_k),
        selected=selected,
        routing=routing,
        device=device,
        seed=7,
        assume_distinct_experts=assume_distinct_experts,
    )
    # The helper runs split forward, merged forward, split backward, merged backward.
    assert nk_bounds == [shape_bound] * 5 + [num_tokens * top_k] * 3 + [shape_bound] * 2
    assert mn_bounds == [num_tokens * top_k] * 5
    assert len(prefixes) == 1 and prefixes[0] is not None
    assert all(schedule is None for schedule in schedules[:3] + schedules[5:8])
    assert all(schedule.data_ptr() == prefixes[0].data_ptr() for schedule in schedules[3:5] + schedules[8:])
    counts = torch.bincount(selected.flatten(), minlength=num_experts)
    expected = torch.cumsum(torch.div(counts + 127, 128, rounding_mode="floor"), dim=0)
    torch.testing.assert_close(prefixes[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["cpu", "empty", "too_many_experts", "no_rows", "other_tile"])
def test_merged_same_nk_prefix_metadata_fallback(reason: str, monkeypatch: pytest.MonkeyPatch):
    """Unsupported metadata must return before launching or allocating a prefix."""
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    class Metadata:
        device = torch.device("cpu" if reason == "cpu" else "cuda")

        def numel(self):
            return 0 if reason == "empty" else 257 if reason == "too_many_experts" else 2

    if reason == "other_tile":
        monkeypatch.setitem(gemm_module._GROUP_GEMM_BLOCK, "BLOCK_M", 64)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("An unsupported prefix must not allocate a device tensor")

    monkeypatch.setattr(torch, "empty_like", unexpected_allocation)
    assert gemm_module.merged_same_nk_tile_prefix(Metadata(), 0 if reason == "no_rows" else 1) is None


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("counts", [(2147483646,), (2147483647,), (0, 2147483646, 1)])
def test_same_nk_prefix_near_int32_limit(counts: tuple[int, ...], dtype):
    """Large logical row counts need only a tiny prefix allocation for this check."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    cumsum = torch.tensor(counts, device="cuda", dtype=dtype).cumsum(0, dtype=dtype)
    prefix = gemm_module.merged_same_nk_tile_prefix(cumsum, sum(counts))
    expected, total_tiles = [], 0
    for count in counts:
        total_tiles += (count + 127) // 128
        expected.append(total_tiles)
    assert prefix is not None
    assert prefix.cpu().tolist() == expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "counts,block_m",
    [
        pytest.param((0, 1, 127, 128, 129, 257), 128, id="tile-boundaries"),
        pytest.param((256, 0, 0, 0), 128, id="concentrated"),
        pytest.param((0, 0, 0), 128, id="empty"),
        pytest.param((0,) * 255 + (129,), 128, id="max-experts"),
        pytest.param((0,) * 256 + (1,), 128, id="expert-fallback"),
        pytest.param((1, 129, 0), 64, id="tile-fallback"),
    ],
)
def test_same_nk_device_prefix_matches_grouped_matmul(counts: tuple[int, ...], block_m: int, dtype, monkeypatch):
    """Cover tile tails, zero-load experts and shape fallbacks without a host read."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    monkeypatch.setitem(gemm_module._GROUP_GEMM_BLOCK, "BLOCK_M", block_m)
    torch.manual_seed(42)
    device = torch.device("cuda")
    total, experts, hidden, output_dim = sum(counts), len(counts), 31, 145
    cumsum = torch.tensor(counts, device=device, dtype=dtype).cumsum(0, dtype=dtype)
    a = torch.randn(total, hidden, device=device, dtype=torch.bfloat16)
    weights = torch.randn(experts, output_dim, hidden, device=device, dtype=torch.bfloat16)

    def run():
        """Build a device prefix and consume it in the native grouped GEMM."""
        prefix = gemm_module.merged_same_nk_tile_prefix(cumsum, total)
        output = gemm_module.group_gemm_same_nk(
            a, weights, cumsum, max_M=max(counts), transpose_b=True, tile_cumsum=prefix
        )
        return prefix, output

    run()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        prefix, output = run()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    classic = gemm_module.group_gemm_same_nk(a, weights, cumsum, max_M=max(counts), transpose_b=True)
    torch.testing.assert_close(output, classic, rtol=0, atol=0)
    expected_outputs, expected_prefix = [], []
    start, tile_count = 0, 0
    for expert, count in enumerate(counts):
        # Accumulate independently in high precision before the output cast.
        reference = a[start : start + count].double() @ weights[expert].double().T
        expected_outputs.append(reference.to(a.dtype))
        start += count
        tile_count += (count + 127) // 128
        expected_prefix.append(tile_count)
    torch.testing.assert_close(output, torch.cat(expected_outputs), rtol=MOE_FUSED_RTOL, atol=MOE_FUSED_ATOL)
    if total and experts <= 256 and block_m == 128:
        assert prefix is not None
        assert prefix.cpu().tolist() == expected_prefix
    else:
        assert prefix is None


@pytest.mark.parametrize("reason", ["capture_unavailable", "unsupported_experts"])
def test_triton_merged_shape_fallback_does_not_synchronize(reason: str, monkeypatch: pytest.MonkeyPatch):
    """Run the actual fallback forward/backward with scalar synchronization forbidden."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    torch.manual_seed(42)
    device, dtype = torch.device("cuda"), torch.bfloat16
    tokens, experts, hidden, intermediate, top_k = 31, 257 if reason == "unsupported_experts" else 8, 128, 64, 2
    routes = torch.randn(tokens, experts, device=device).topk(top_k, dim=-1).indices
    gate = torch.full((tokens, top_k), 1 / top_k, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype, requires_grad=True)
    empty = _empty(device, dtype)
    fused = resolve_op("moe_experts", "standard", "fused_triton").wrapper
    bounds, schedules = [], []
    original_nk = gemm_module.group_gemm_same_nk

    def record_bound(*args, **kwargs):
        """Observe the real fallback without changing its computation."""
        bounds.append(kwargs["max_M"])
        schedules.append(kwargs.get("tile_cumsum"))
        return original_nk(*args, **kwargs)

    monkeypatch.setattr(gemm_module, "group_gemm_same_nk", record_bound)
    if reason == "capture_unavailable":
        monkeypatch.setattr(triton_moe_module, "get_torch_device", object)
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        output = fused(x, gate, routes, empty, empty, w2, w1, num_experts=experts)
        output.sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    assert bounds == [tokens * top_k] * 4
    assert schedules == [None] * 4
    assert torch.isfinite(output).all()
    for tensor in (x, gate, w1, w2):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_triton_merged_stream_capture_uses_shape_bound(monkeypatch: pytest.MonkeyPatch):
    """Captured forward and backward reuse the shape bound without a scalar read."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    torch.manual_seed(42)
    device = torch.device("cuda")
    tokens, experts, hidden, intermediate, top_k = 128, 8, 128, 64, 2
    routes = torch.randn(tokens, experts, device=device).topk(top_k, dim=-1).indices
    assert torch.bincount(routes.flatten(), minlength=experts).max().item() < tokens
    gate = torch.full((tokens, top_k), 1 / top_k, device=device, dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16, requires_grad=True)
    empty = _empty(device, torch.bfloat16)
    fused = resolve_op("moe_experts", "standard", "fused_triton").wrapper
    bounds, schedules = [], []
    original = gemm_module.group_gemm_same_nk

    def record_bound(*args, **kwargs):
        """Record capture bounds while keeping the native operation unchanged."""
        bounds.append(kwargs["max_M"])
        schedules.append(kwargs.get("tile_cumsum"))
        return original(*args, **kwargs)

    def forward():
        """Exercise the registered merged implementation under stream capture."""
        return fused(x, gate, routes, empty, empty, w2, w1, num_experts=experts, assume_distinct_experts=True)

    monkeypatch.setattr(gemm_module, "group_gemm_same_nk", record_bound)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    # Warm the shape-only specialization that the real capture will select.
    with monkeypatch.context() as warmup_patch:
        warmup_patch.setattr(triton_moe_module, "_must_use_shape_derived_bound", lambda: True)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                forward().sum().backward()
    warmup_stream.synchronize()
    x.grad = w1.grad = w2.grad = None
    bounds.clear()
    schedules.clear()

    def unexpected_prefix(*args, **kwargs):
        raise AssertionError("Capture must retain the original schedule")

    monkeypatch.setattr(gemm_module, "merged_same_nk_tile_prefix", unexpected_prefix)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
        output.sum().backward()
    assert bounds == [tokens] * 4
    assert schedules == [None] * 4
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    for tensor in (x, w1, w2):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
