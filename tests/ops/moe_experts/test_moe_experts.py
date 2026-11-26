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

from scripts.ci.accelerator.launch_plan import LaunchSpec
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
from veomni.ops.kernels.moe_experts.shared import scheduling as same_nk_scheduling
from veomni.ops.kernels.moe_experts.shared.indices import build_moe_indices
from veomni.ops.kernels.moe_experts.standard import triton as triton_module
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(0, 257), (5, 257), (17, 768)])
@pytest.mark.parametrize("swiglu_limit", [None, 1.0])
@pytest.mark.parametrize("grad_layout", ["contiguous", "transposed"])
def test_triton_merged_swiglu_preserves_staged_rounding(dtype, shape, swiglu_limit, grad_layout):
    """Match staged ATen values for merged views and strided upstream gradients."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared.swiglu import (
        apply_swiglu_clamp,
        merged_swiglu_backward,
        merged_swiglu_forward,
    )

    torch.manual_seed(919)
    rows, columns = shape
    merged = torch.randn(rows, 2 * columns, device="cuda", dtype=dtype)
    gate, up, gate_mask, up_mask = apply_swiglu_clamp(*merged.chunk(2, dim=-1), swiglu_limit)
    gate_before, up_before = gate.clone(), up.clone()
    expected = torch.ops.aten.silu(gate) * up
    actual = merged_swiglu_forward(gate, up)
    if grad_layout == "transposed":
        grad = torch.randn(columns, rows, device="cuda", dtype=dtype).T
        if rows:
            assert not grad.is_contiguous()
            assert grad.stride(1) > 1
    else:
        grad = torch.randn_like(gate)
    activation, packed_grad = merged_swiglu_backward(gate, up, grad, gate_mask, up_mask)
    expected_gate_grad = torch.ops.aten.silu_backward(grad * up, gate)
    expected_up_grad = torch.ops.aten.silu(gate) * grad
    if gate_mask is not None:
        expected_gate_grad.masked_fill_(~gate_mask, 0)
        expected_up_grad.masked_fill_(~up_mask, 0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(activation, expected, rtol=0, atol=0)
    torch.testing.assert_close(packed_grad, torch.cat([expected_gate_grad, expected_up_grad], dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    torch.testing.assert_close(up, up_before, rtol=0, atol=0)
    assert actual.shape == shape and actual.dtype == dtype and actual.is_contiguous()
    if rows:
        single_rounding = (torch.ops.aten.silu(gate.float()) * up.float()).to(dtype)
        assert (expected != single_rounding).any(), "Fixture must distinguish staged rounding"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("swiglu_limit", [None, 1.0])
def test_triton_merged_swiglu_matches_unfused_forward_and_backward(monkeypatch, dtype, swiglu_limit):
    """Compare the fused consumer with the unchanged staged pointwise path across non-EP and EP."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import swiglu
    from veomni.ops.kernels.moe_experts.shared.ep import EPMergedFc1GroupGemm

    torch.manual_seed(31)
    tokens, experts, hidden, intermediate, top_k = 23, 4, 64, 128, 2
    x = 0.4 * torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    gate, routes = _route(tokens, experts, top_k, "cuda", dtype)
    w1 = 0.4 * torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=dtype)
    w2 = 0.1 * torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype)
    fused = resolve_op("moe_experts", "standard", "fused_triton").wrapper
    empty = _empty("cuda", dtype)
    calls = []
    original = swiglu.merged_swiglu_forward

    def record(gate_input, up):
        calls.append(gate_input.shape)
        return original(gate_input, up)

    def run():
        inputs = [value.detach().clone().requires_grad_() for value in (x, gate, w1, w2)]
        output = fused(
            inputs[0],
            inputs[1],
            routes,
            empty,
            empty,
            inputs[3],
            inputs[2],
            num_experts=experts,
            swiglu_limit=swiglu_limit,
        )
        grads = torch.autograd.grad(output.float().square().mean(), inputs)
        return output, grads

    monkeypatch.setattr(swiglu, "merged_swiglu_forward", record)
    actual, actual_grads = run()
    assert calls == [(tokens * top_k, intermediate)]

    cumsum = torch.tensor([6, 12, 18, tokens], device="cuda", dtype=torch.int32)
    ep_inputs = [value.detach().clone().requires_grad_() for value in (x, w1, w2)]
    ep_out = EPMergedFc1GroupGemm.apply(ep_inputs[0], cumsum, ep_inputs[1], ep_inputs[2], swiglu_limit)
    ep_grads = torch.autograd.grad(ep_out.float().square().mean(), ep_inputs)
    assert calls == [(tokens * top_k, intermediate), (tokens, intermediate)]

    monkeypatch.setattr(swiglu, "merged_swiglu_forward", lambda gate_input, up: torch.ops.aten.silu(gate_input) * up)

    def staged_backward(gate_input, up, grad, gate_mask=None, up_mask=None):
        silu = torch.ops.aten.silu(gate_input)
        grad_gate = torch.ops.aten.silu_backward(grad * up, gate_input)
        grad_up = silu * grad
        if gate_mask is not None:
            grad_gate.masked_fill_(~gate_mask, 0)
            grad_up.masked_fill_(~up_mask, 0)
        return silu * up, torch.cat([grad_gate, grad_up], dim=-1)

    monkeypatch.setattr(swiglu, "merged_swiglu_backward", staged_backward)
    expected, expected_grads = run()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    ep_ref_inputs = [value.detach().clone().requires_grad_() for value in (x, w1, w2)]
    ep_ref_out = EPMergedFc1GroupGemm.apply(ep_ref_inputs[0], cumsum, ep_ref_inputs[1], ep_ref_inputs[2], swiglu_limit)
    ep_ref_grads = torch.autograd.grad(ep_ref_out.float().square().mean(), ep_ref_inputs)
    torch.testing.assert_close(ep_out, ep_ref_out, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(ep_grads, ep_ref_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_merged_swiglu_backward_cpu_preserves_clamp_gradients():
    """The non-NVIDIA fallback matches autograd through the original clamped inputs."""
    from veomni.ops.kernels.moe_experts.shared.swiglu import apply_swiglu_clamp, merged_swiglu_backward

    torch.manual_seed(3)
    gate = torch.randn(5, 33, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    grad = torch.randn_like(gate)
    clamped_gate, clamped_up, gate_mask, up_mask = apply_swiglu_clamp(gate, up, 0.5)
    expected = torch.ops.aten.silu(clamped_gate) * clamped_up
    expected_grads = torch.autograd.grad(expected, (gate, up), grad)
    activation, packed_grad = merged_swiglu_backward(clamped_gate, clamped_up, grad, gate_mask, up_mask)
    torch.testing.assert_close(activation, expected, rtol=0, atol=0)
    torch.testing.assert_close(packed_grad, torch.cat(expected_grads, dim=-1), rtol=0, atol=0)


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
@pytest.mark.parametrize("routing_dtype", (torch.bfloat16, torch.float32))
def test_gpt_oss_quack_matches_eager(routing_dtype: torch.dtype):
    """GPT-OSS Quack fused MoE matches eager forward and backward across routing dtypes."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, top_k = 16, 4, 32, 16, 2
    hidden = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing, selected = _route(num_tokens, num_experts, top_k, device, routing_dtype)
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
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_merged_moe_shape_bound_capture_fallback(mode: str, expected: bool, monkeypatch: pytest.MonkeyPatch):
    """Select the original schedule during compilation or unknown capture state."""
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: mode == "compiling")

    if mode == "compiling":

        def unexpected_device_query():
            raise AssertionError("Compilation must not query a live device")

        monkeypatch.setattr(same_nk_scheduling, "get_torch_device", unexpected_device_query)
    elif mode == "unknown":
        monkeypatch.setattr(same_nk_scheduling, "get_torch_device", object)
    elif mode == "noncallable":

        class DeviceAPI:
            is_current_stream_capturing = False

        monkeypatch.setattr(same_nk_scheduling, "get_torch_device", DeviceAPI)
    else:

        class DeviceAPI:
            def is_current_stream_capturing(self):
                if mode == "query_error":
                    raise RuntimeError("Capture state unavailable")
                if mode == "ambiguous":
                    return None
                return mode == "capturing"

        monkeypatch.setattr(same_nk_scheduling, "get_torch_device", DeviceAPI)

    assert same_nk_scheduling.must_use_shape_derived_bound() is expected


@pytest.mark.parametrize(
    "routing_kind,assume_distinct_experts",
    [("distinct", False), ("distinct", True), ("duplicate", False), ("concentrated", False)],
)
def test_triton_merged_same_nk_compact_schedule(
    routing_kind: str, assume_distinct_experts: bool, monkeypatch: pytest.MonkeyPatch
):
    """Build independent split/merged prefixes and reuse each during backward."""
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
    assert len(prefixes) == 2 and all(prefix is not None for prefix in prefixes)
    assert prefixes[0].data_ptr() != prefixes[1].data_ptr()
    assert all(schedule.data_ptr() == prefixes[0].data_ptr() for schedule in schedules[:3] + schedules[5:8])
    assert all(schedule.data_ptr() == prefixes[1].data_ptr() for schedule in schedules[3:5] + schedules[8:])
    counts = torch.bincount(selected.flatten(), minlength=num_experts)
    expected = torch.cumsum(torch.div(counts + 127, 128, rounding_mode="floor"), dim=0)
    for prefix in prefixes:
        torch.testing.assert_close(prefix, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["cpu", "rocm", "empty", "too_many_experts", "no_rows", "other_tile"])
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_merged_same_nk_prefix_metadata_fallback(reason: str, monkeypatch: pytest.MonkeyPatch):
    """Unsupported metadata must return before launching or allocating a prefix."""
    pytest.importorskip("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    monkeypatch.setattr(torch.version, "hip", "mock-hip" if reason == "rocm" else None)

    class Metadata:
        device = torch.device("cpu" if reason == "cpu" else "cuda")

        def numel(self):
            if reason == "empty":
                return 0
            if reason == "too_many_experts":
                return gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS + 1
            return 2

    if reason == "other_tile":
        monkeypatch.setitem(gemm_module._GROUP_GEMM_BLOCK, "BLOCK_M", 64)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("An unsupported prefix must not allocate a device tensor")

    monkeypatch.setattr(torch, "empty_like", unexpected_allocation)
    assert gemm_module.merged_same_nk_tile_prefix(Metadata(), 0 if reason == "no_rows" else 1) is None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_same_nk_expert_limit_warns_once_before_allocation(monkeypatch: pytest.MonkeyPatch, caplog):
    """An otherwise eligible oversized schedule explains its fallback once."""
    pytest.importorskip("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module
    from veomni.utils import logging

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("LOCAL_RANK", "0")
    test_logger = logging.get_logger(__name__ + ".expert_limit_warning")
    monkeypatch.setattr(gemm_module, "logger", test_logger)
    experts = gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS + 1

    class Metadata:
        device = torch.device("cuda")

        def numel(self):
            return experts

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("An oversized prefix must not allocate a device tensor")

    monkeypatch.setattr(torch, "empty_like", unexpected_allocation)
    with caplog.at_level("WARNING", logger=test_logger.name):
        for _ in range(2):
            assert gemm_module.merged_same_nk_tile_prefix(Metadata(), 1) is None
    records = [record for record in caplog.records if record.name == test_logger.name]
    assert len(records) == 1
    assert f"{experts} experts (> {gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS})" in records[0].getMessage()
    assert "shape-derived" in records[0].getMessage()


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
    "hidden,output_dim,transpose_b,value_dtype",
    [
        pytest.param(31, 145, True, torch.bfloat16, id="two-column-tail-bf16"),
        pytest.param(65, 17, False, torch.float16, id="one-column-tail-fp16"),
        pytest.param(33, 385, False, torch.bfloat16, id="four-column-tail-bf16"),
        pytest.param(128, 256, True, torch.float16, id="aligned-columns-fp16"),
    ],
)
@pytest.mark.parametrize(
    "counts,block_m",
    [
        pytest.param((0, 1, 127, 128, 129, 257), 128, id="tile-boundaries"),
        pytest.param((256, 0, 0, 0), 128, id="concentrated"),
        pytest.param((0, 0, 0), 128, id="empty"),
        pytest.param("at_expert_limit", 128, id="max-experts"),
        pytest.param("above_expert_limit", 128, id="expert-fallback"),
        pytest.param((1, 129, 0), 64, id="tile-fallback"),
    ],
)
def test_same_nk_device_prefix_matches_grouped_matmul(
    counts: tuple[int, ...] | str,
    block_m: int,
    dtype,
    hidden,
    output_dim,
    transpose_b,
    value_dtype,
    monkeypatch,
):
    """Cover row/column tails, weight layouts and shape fallbacks without a host read."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    if counts == "at_expert_limit":
        counts = (0,) * (gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS - 1) + (129,)
    elif counts == "above_expert_limit":
        counts = (0,) * gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS + (1,)

    monkeypatch.setitem(gemm_module._GROUP_GEMM_BLOCK, "BLOCK_M", block_m)
    torch.manual_seed(42)
    device = torch.device("cuda")
    total, experts = sum(counts), len(counts)
    cumsum = torch.tensor(counts, device=device, dtype=dtype).cumsum(0, dtype=dtype)
    a = torch.randn(total, hidden, device=device, dtype=value_dtype)
    weight_shape = (experts, output_dim, hidden) if transpose_b else (experts, hidden, output_dim)
    weights = torch.randn(weight_shape, device=device, dtype=value_dtype)

    def run():
        """Build a device prefix and consume it in the native grouped GEMM."""
        prefix = gemm_module.merged_same_nk_tile_prefix(cumsum, total)
        output = gemm_module.group_gemm_same_nk(
            a, weights, cumsum, max_M=max(counts), transpose_b=transpose_b, tile_cumsum=prefix, tile_total_M=total
        )
        return prefix, output

    run()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        prefix, output = run()
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    classic = gemm_module.group_gemm_same_nk(a, weights, cumsum, max_M=max(counts), transpose_b=transpose_b)
    torch.testing.assert_close(output, classic, rtol=0, atol=0)
    expected_outputs, expected_prefix = [], []
    start, tile_count = 0, 0
    for expert, count in enumerate(counts):
        # Accumulate independently in high precision before the output cast.
        weight = weights[expert].double()
        reference = a[start : start + count].double() @ (weight.T if transpose_b else weight)
        expected_outputs.append(reference.to(a.dtype))
        start += count
        tile_count += (count + 127) // 128
        expected_prefix.append(tile_count)
    torch.testing.assert_close(output, torch.cat(expected_outputs), rtol=MOE_FUSED_RTOL, atol=MOE_FUSED_ATOL)
    if total and experts <= gemm_module._COMPACT_SCHEDULE_MAX_EXPERTS and block_m == 128:
        assert prefix is not None
        assert prefix.cpu().tolist() == expected_prefix
    else:
        assert prefix is None


@pytest.mark.parametrize("mode", ["accumulate", "silu_saved"])
def test_same_nk_compact_epilogue_matches_shape_schedule(mode):
    """N-fast tiles must preserve accumulated outputs and saved preactivations."""
    require_nvidia_cuda("triton")
    from veomni.ops.kernels.moe_experts.shared import group_gemm as gemm_module

    torch.manual_seed(42)
    device, dtype = torch.device("cuda"), torch.bfloat16
    counts = (129, 0, 1, 257)
    total, hidden, output_dim = sum(counts), 33, 385
    a = torch.randn(total, hidden, device=device, dtype=dtype)
    weights = torch.randn(len(counts), hidden, output_dim, device=device, dtype=dtype)
    cumsum = torch.tensor(counts, device=device, dtype=torch.int64).cumsum(0)
    prefix = gemm_module.merged_same_nk_tile_prefix(cumsum, total)
    assert prefix is not None
    # The existing tl.dot accumulator requires FP32 even with BF16 operands.
    initial = torch.randn(total, output_dim, device=device, dtype=torch.float32)

    def run(tile_prefix):
        return gemm_module.group_gemm_same_nk(
            a,
            weights,
            cumsum,
            max_M=max(counts),
            c=initial.clone() if mode == "accumulate" else None,
            activation=gemm_module.ActivationType.SILU if mode == "silu_saved" else None,
            save_activation=mode == "silu_saved",
            tile_cumsum=tile_prefix,
            tile_total_M=total if tile_prefix is not None else None,
        )

    torch.testing.assert_close(run(prefix), run(None), rtol=0, atol=0)


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
        monkeypatch.setattr(same_nk_scheduling, "get_torch_device", object)
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
        warmup_patch.setattr(same_nk_scheduling, "must_use_shape_derived_bound", lambda: True)
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["silu", "swish", "gelu", "gelu_pytorch_tanh"])
def test_the_eager_row_runs_the_configured_activation(activation):
    """The routed experts compute ``act(gate) * up`` for the model's activation.

    Before this, the fused rows hard-coded ``aten.silu`` and every model whose
    ``hidden_act`` was anything else fell back to a Python loop outside the family --
    which is why a quantizing selection had to be refused there. The activation is a
    call argument now, so the same row serves any of them.

    The oracle is ``ACT2FN[activation]``, not a restated formula: the exact-erf GELU
    and the tanh approximation are different functions.
    """
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.moe_experts.standard import eager

    torch.manual_seed(0)
    experts, hidden, intermediate, tokens, top_k = 3, 8, 16, 5, 2
    x = torch.randn(tokens, hidden, dtype=torch.float32, requires_grad=True)
    gate_w = torch.randn(experts, intermediate, hidden, dtype=torch.float32, requires_grad=True)
    up_w = torch.randn(experts, intermediate, hidden, dtype=torch.float32, requires_grad=True)
    down_w = torch.randn(experts, hidden, intermediate, dtype=torch.float32, requires_grad=True)
    routes = torch.randint(0, experts, (tokens, top_k))
    routing = torch.softmax(torch.randn(tokens, top_k), dim=-1)

    out = eager.wrapper(
        x,
        routing,
        routes,
        gate_w,
        up_w,
        down_w,
        gate_w.new_empty(0),
        num_experts=experts,
        activation=activation,
    )

    # Reference: the same loop with the transformers implementation.
    act = ACT2FN[activation]
    expected = torch.zeros(tokens, hidden)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(routes[token, slot])
            gate = x[token] @ gate_w[expert].T
            up = x[token] @ up_w[expert].T
            expected[token] += ((act(gate) * up) @ down_w[expert].T) * routing[token, slot]

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    out.sum().backward()
    for tensor in (x, gate_w, up_w, down_w):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.any(tensor.grad != 0)

    # The activation must actually change the result, or this proves nothing.
    # ``swish`` is HF's alias for SiLU, so it is expected to match rather than differ.
    if activation not in {"silu", "swish"}:
        other = eager.wrapper(
            x.detach(),
            routing,
            routes,
            gate_w.detach(),
            up_w.detach(),
            down_w.detach(),
            gate_w.new_empty(0),
            num_experts=experts,
            activation="silu",
        )
        assert not torch.equal(out.detach(), other)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    "impl",
    ["eager", "fused_quack", "fused_npu", "fused_mlu"],
)
def test_a_fused_row_delegates_a_non_silu_activation_to_eager(impl):
    """Rows whose kernel fuses SiLU agree with ``eager`` for every other activation.

    ``fused_quack``, ``fused_npu`` and ``fused_mlu`` compute ``act(gate) * up``
    inside their grouped GEMM, so another activation has no seam to reach. They hand
    that call to the ``eager`` row, which is the reference implementation and the same
    math models used before the activation became a call argument -- hence
    ``rtol=0, atol=0`` here rather than a loose tolerance. They lose nothing by it:
    those rows do bf16 math and ignore ``expert_dtype``.

    ``deepgemm`` is deliberately absent. Its GEMM is FP8, so handing the call
    away would skip the quantization and train the experts in bf16 under a config
    that said otherwise -- it refuses instead, covered below.

    This also pins the delegation's *placement*: it has to precede each row's
    hardware guard, or a non-SiLU model on the wrong device fails with a hardware
    error instead of running on the reference row.
    """
    from veomni.ops.kernels.moe_experts.standard import eager, mlu, npu, quack

    rows = {
        "eager": eager,
        "fused_quack": quack,
        "fused_npu": npu,
        "fused_mlu": mlu,
    }
    row = rows[impl]

    torch.manual_seed(0)
    experts, hidden, intermediate, tokens, top_k = 3, 8, 16, 5, 2
    x = torch.randn(tokens, hidden)
    gate_up = torch.randn(experts, 2 * intermediate, hidden)
    down = torch.randn(experts, hidden, intermediate)
    routes = torch.randint(0, experts, (tokens, top_k))
    routing = torch.softmax(torch.randn(tokens, top_k), dim=-1)
    empty = gate_up.new_empty(0)

    for activation in ("gelu", "gelu_pytorch_tanh"):
        got = row.wrapper(
            x,
            routing,
            routes,
            empty,
            empty,
            down,
            gate_up,
            num_experts=experts,
            activation=activation,
        )
        want = eager.wrapper(
            x,
            routing,
            routes,
            empty,
            empty,
            down,
            gate_up,
            num_experts=experts,
            activation=activation,
        )
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_every_ctx_attribute_a_backward_reads_is_assigned_in_forward():
    """A custom ``autograd.Function``'s ``ctx`` only carries what ``forward`` set.

    ``activation`` reaches the backward this way, and the two Triton Functions have
    separate ``ctx`` assignments -- so adding the argument to one and forgetting the
    other raises ``AttributeError`` only on the path that uses it. That happened
    here: the merged-fc1 backward read ``ctx.activation`` while only the split-fc1
    forward assigned it, and no CPU test reached the merged path.

    Static rather than runtime, so it covers both Functions without a GPU.
    """
    import ast
    import pathlib

    source = pathlib.Path(triton_module.__file__).read_text()
    missing = []
    checked = 0
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
        backward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"), None)
        if forward is None or backward is None:
            continue
        checked += 1
        assigned = {
            node.attr
            for node in ast.walk(forward)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and isinstance(node.ctx, ast.Store)
        }
        read = {
            node.attr
            for node in ast.walk(backward)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and isinstance(node.ctx, ast.Load)
        }
        missing.extend((cls.name, name) for name in read - assigned - {"saved_tensors", "needs_input_grad"})

    assert checked == 2, f"expected both Triton Functions, checked {checked}"
    assert not missing, f"backward reads an attribute its forward never sets: {missing}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_every_triton_function_backward_returns_one_gradient_per_argument():
    """A custom ``autograd.Function``'s backward must return one entry per argument.

    Adding ``activation`` to ``forward`` widened the argument list, and the backward
    tuples have to grow with it -- PyTorch reports the mismatch only at the first
    training step ("returned an incorrect number of gradients (expected 9, got 8)"),
    which on CPU never runs because these rows need a GPU. Two consecutive CI runs
    failed this way, so it is checked here statically instead.
    """
    import ast
    import pathlib

    from veomni.ops.kernels.moe_experts.shared import ep as ep_module

    mismatches = []
    checked = 0
    modules = (triton_module, ep_module)
    for cls in (
        node
        for module in modules
        for node in ast.parse(pathlib.Path(module.__file__).read_text()).body
        if isinstance(node, ast.ClassDef)
    ):
        forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
        backward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"), None)
        if forward is None or backward is None:
            continue
        checked += 1
        # ``ctx`` is not part of the tuple PyTorch compares against.
        arguments = len([argument for argument in forward.args.args if argument.arg != "ctx"])
        returned = next((node for node in ast.walk(backward) if isinstance(node, ast.Return)), None)
        entries = len(returned.value.elts) if isinstance(returned.value, ast.Tuple) else 1
        if arguments != entries:
            mismatches.append((cls.name, arguments, entries))

    assert checked == 4, f"expected both Triton and both EP Functions, checked {checked}"
    assert not mismatches, f"backward tuple does not match forward's arguments: {mismatches}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["relu", "mish", "quick_gelu"])
def test_the_eager_row_evaluates_an_activation_outside_the_whitelist(activation):
    """A config outside the four served names still runs on the reference row.

    ``veomni.ops.activation`` serves a closed set, and the fused rows refuse anything
    else -- but before the activation became a call argument, every model evaluated
    its own ``act_fn`` through ``merged_experts_act_fn_forward``. Dropping that made
    ``hidden_act: relu`` construct and then fail at the first forward, which is a
    regression for models that trained fine. ``act_fn`` restores the escape, the same
    way ``gated_mlp``'s ``eager`` row uses ``mlp.act_fn``.
    """
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.moe_experts.standard import eager

    torch.manual_seed(0)
    experts, hidden, intermediate, tokens, top_k = 3, 8, 16, 5, 2
    x = torch.randn(tokens, hidden)
    gate_up = torch.randn(experts, 2 * intermediate, hidden)
    down = torch.randn(experts, hidden, intermediate)
    routes = torch.randint(0, experts, (tokens, top_k))
    routing = torch.softmax(torch.randn(tokens, top_k), dim=-1)

    act_fn = ACT2FN[activation]
    got = eager.wrapper(
        x,
        routing,
        routes,
        gate_up.new_empty(0),
        gate_up.new_empty(0),
        down,
        gate_up,
        num_experts=experts,
        activation=activation,
    )

    expected = torch.zeros(tokens, hidden)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(routes[token, slot])
            gate = x[token] @ gate_up[expert][:intermediate].T
            up = x[token] @ gate_up[expert][intermediate:].T
            expected[token] += ((act_fn(gate) * up) @ down[expert].T) * routing[token, slot]

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    # A name transformers does not know is the only case with no answer at all.
    with pytest.raises(ValueError, match="no implementation for activation"):
        eager.wrapper(
            x,
            routing,
            routes,
            gate_up.new_empty(0),
            gate_up.new_empty(0),
            down,
            gate_up,
            num_experts=experts,
            activation="not_an_activation",
        )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_absent_clamp_masks_are_never_inverted():
    """The merged backward must not apply ``~`` to the empty placeholder tensor.

    ``forward`` saves ``torch.empty(0)`` where a clamp mask would go, so checking
    ``mask is not None`` is always true and ``~`` on a float tensor raises. The guard
    has to be ``swiglu_limit is not None``, as the split backward already does. This
    is a static check because the merged path needs a GPU.
    """
    import ast
    import pathlib

    source = pathlib.Path(triton_module.__file__).read_text()
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        comparisons = ast.unparse(node.test)
        body = ast.unparse(node)
        if "mask_fc1_" in comparisons and "is not None" in comparisons and "~mask_fc1_" in body:
            offenders.append(comparisons)
    assert not offenders, f"a clamp mask is tested with `is not None`, but forward saves an empty tensor: {offenders}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_a_quantizing_row_refuses_an_activation_it_cannot_evaluate():
    """Neither quantizing row may quietly fall back to the model dtype.

    ``qat`` and ``deepgemm`` round the expert GEMMs, so an activation they cannot
    evaluate has no correct answer: handing it to a reference row, or to the model's
    own ``act_fn``, would train the experts at full precision while the config claimed
    quantization. Both refuse, and both say which selection to use instead.

    ``moe_experts``' bf16 fused rows are the contrast: they delegate, because they
    round nothing and the reference path is exactly as precise.
    """

    from veomni.ops.kernels.moe_experts.standard import deepgemm, qat

    torch.manual_seed(0)
    experts, hidden, intermediate, tokens, top_k = 3, 128, 256, 4, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16)
    routes = torch.randint(0, experts, (tokens, top_k))
    routing = torch.softmax(torch.randn(tokens, top_k), dim=-1).to(torch.bfloat16)
    empty = gate_up.new_empty(0)

    for row, name in ((qat, "qat"), (deepgemm, "deepgemm")):
        with pytest.raises(ValueError, match=f"moe_implementation='{name}'"):
            row.wrapper(
                x,
                routing,
                routes,
                empty,
                empty,
                down,
                gate_up,
                num_experts=experts,
                activation="relu",
            )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_no_triton_function_forward_reads_the_activation_without_its_ctx_copy():
    """Every activation site in the Triton Functions goes through one helper.

    A row that evaluates the activation in `forward` with one expression and in
    `backward` with another produces a forward that computes SiLU and a backward that
    differentiates GELU -- a silently wrong training run, not a crash. The merged-fc1
    forward had exactly that shape. Routing every site through `_expert_activation`
    and the `ctx` copy of the same arguments makes the two halves agree by
    construction, so this pins the shape rather than re-testing the arithmetic.
    """
    import ast
    import pathlib

    source = pathlib.Path(triton_module.__file__).read_text()
    offenders = []
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        for method in ("forward", "backward"):
            function = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method), None)
            if function is None:
                continue
            # A direct call is fine inside an explicit ``== "silu"`` branch, which is
            # how the fused kernel is chosen; what is not fine is one side of the
            # function picking it and the other side not.
            guarded = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.If) or "silu" not in ast.unparse(node.test):
                    continue
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        guarded.add(id(inner))
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or id(node) in guarded:
                    continue
                callee = ast.unparse(node.func)
                if callee in {"apply_activation", "swiglu.merged_swiglu_forward", "swiglu.merged_swiglu_backward"}:
                    offenders.append(f"{cls.name}.{method} calls {callee} directly")

    assert not offenders, (
        "an activation is evaluated outside _expert_activation, so forward and backward "
        f"can pick different functions: {offenders}"
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_expert_parallel_functions_receive_the_activation():
    """The EP path is selected before any of the non-EP code, so it needs its own pass.

    `fused_triton` dispatches to `EPGroupGemm` / `EPMergedFc1GroupGemm` when expert
    parallelism is on, and those Functions are separate implementations with their own
    `forward` and `backward`. They were left on `aten.silu` while the non-EP path took
    the configured activation, which meant an EP run silently trained a different
    function than the same config without EP.
    """
    import ast
    import pathlib

    from veomni.ops.kernels.moe_experts.shared import ep as ep_module

    source = pathlib.Path(ep_module.__file__).read_text()
    checked = 0
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
        backward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"), None)
        if forward is None or backward is None:
            continue
        checked += 1
        arguments = [argument.arg for argument in forward.args.args]
        assert "activation" in arguments, f"{cls.name}.forward does not take the activation"
        body = ast.unparse(forward) + ast.unparse(backward)
        assert "_expert_activation" in body, f"{cls.name} evaluates the activation directly"
        assert "ctx.activation" in body, f"{cls.name}.backward does not read the saved act_fn"

    assert checked == 2, f"expected both EP Functions, checked {checked}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_every_backward_keeps_its_weight_gradient_inside_the_requires_grad_guard():
    """A ``requires_grad``-guarded wgrad must stay inside that guard.

    Adding a branch around the activation is easy to get wrong in a way that reads
    fine: the ``group_gemm_same_mn`` that produces ``grad_fc2_weight`` was left inside
    the *non*-SiLU branch, so the SiLU path -- every real training run -- skipped that
    weight's gradient entirely. Nothing raised; the loss simply diverged from the
    configuration that computes it correctly, which is what the expert-parallel
    alignment test caught.

    Static because these paths need a GPU and an EP group.
    """
    import ast
    import pathlib

    from veomni.ops.kernels.moe_experts.shared import ep as ep_module

    problems = []
    for module in (triton_module, ep_module):
        for cls in (
            n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n, ast.ClassDef)
        ):
            backward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"), None)
            if backward is None:
                continue
            # every `if <weight>.requires_grad:` must contain the GEMM writing that weight
            for node in ast.walk(backward):
                if not isinstance(node, ast.If):
                    continue
                test = ast.unparse(node.test)
                if "requires_grad" not in test:
                    continue
                body = ast.unparse(node)
                if "group_gemm" not in body:
                    problems.append(f"{cls.name}: `if {test}` computes no weight gradient")

    assert not problems, f"a weight gradient moved out of its guard: {problems}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["relu", "mish", "quick_gelu", "gelu", "silu"])
def test_the_shared_activation_backward_matches_autograd(activation):
    """A name outside the four served ones differentiates through the model's `act_fn`.

    `fused_triton` is the GPU default, so a `relu` config reaches it first. Forward
    worked through the `act_fn` fallback, but backward called
    `apply_activation_backward`, which raises for anything outside its four branches --
    so the run failed at the first training step instead of the first forward. This
    covers both routes through one helper.
    """
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.moe_experts.shared.activation import activation_backward

    torch.manual_seed(0)
    act_fn = ACT2FN[activation]
    hidden = torch.randn(256, dtype=torch.bfloat16)
    grad = torch.randn(256, dtype=torch.bfloat16)

    got = activation_backward(grad, hidden, activation)

    leaf = hidden.detach().clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(act_fn(leaf), leaf, grad)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_reference_row_owns_the_expert_parallel_dispatch():
    """A non-Swish activation under EP must be dispatched, not indexed by global id.

    `eager` is what every SiLU-fused row hands a non-Swish activation to, so it has to
    carry the expert-parallel path: under EP each rank holds a shard of the expert
    weights, and the per-expert loop indexes by global id -- reading past the shard or
    picking the wrong expert, silently. An earlier revision of this branch delegated
    EP traffic to `triton` instead, which made one row call another; the dispatch
    belongs to the row that runs the loop.
    """
    from transformers.activations import ACT2FN

    from veomni.ops.kernels.moe_experts.shared.act_fn_experts import merged_experts_act_fn_forward

    # Sharded expert weights with EP off: refuse rather than read the wrong expert.
    with pytest.raises(ValueError, match="expert-parallel sharded weights"):
        merged_experts_act_fn_forward(
            torch.randn(3, 4),
            torch.tensor([[3], [3], [3]]),
            torch.ones(3, 1),
            torch.randn(2, 8, 4),
            torch.randn(2, 4, 4),
            ACT2FN["gelu"],
            num_experts=4,
        )

    # And the fused rows delegate to a single row, which owns that dispatch.
    import pathlib

    for module in ("quack", "npu", "mlu"):
        source = (pathlib.Path(triton_module.__file__).parent / f"{module}.py").read_text()
        assert "_eager.wrapper(" in source, f"{module} does not delegate to the reference row"
        assert "_triton.wrapper(" not in source, f"{module} calls another row instead"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_every_backward_attribute_exists_on_the_ctx_its_forward_built():
    """A `ctx.<name>` a backward reads must be one the forward set, across modules.

    `ep.py` kept passing `ctx.act_fn` after that attribute was removed, so both EP
    Functions raised `AttributeError` on the first backward -- including the default
    SiLU path. The existing check only looked inside `triton.py`, which is how the
    copy in `ep.py` survived.

    Static because the EP path needs a distributed group and a GPU, and because the
    failure is a name, not a number.
    """
    import ast
    import pathlib

    from veomni.ops.kernels.moe_experts.shared import ep as ep_module

    missing = []
    checked = 0
    for module in (triton_module, ep_module):
        for cls in (
            n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n, ast.ClassDef)
        ):
            forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
            backward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"), None)
            if forward is None or backward is None:
                continue
            checked += 1
            assigned = {
                node.attr
                for node in ast.walk(forward)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ctx"
                and isinstance(node.ctx, ast.Store)
            }
            read = {
                node.attr
                for node in ast.walk(backward)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ctx"
                and isinstance(node.ctx, ast.Load)
            }
            missing.extend(f"{cls.name}.{name}" for name in read - assigned - {"saved_tensors", "needs_input_grad"})

    assert checked == 4, f"expected four Functions across both modules, checked {checked}"
    assert not missing, f"backward reads a ctx attribute its forward never sets: {missing}"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_expert_sharding_is_read_without_assuming_the_axis_is_named_ep():
    """A parallel state built with another extra axis must not raise here.

    ``ParallelState.ep_enabled`` indexes ``extra_parallel_sizes["ep"]`` unconditionally,
    and ``_init_parallel_state`` builds that dict from ``extra_parallel_names``. A state
    initialized with, say, ``("ple",)`` therefore has no ``"ep"`` key, and asking
    whether expert weights are sharded raised ``KeyError: 'ep'`` -- which took out an
    Ulysses test that never touches experts, because the guard runs on every call.

    The question this row needs answered is only "are experts sharded?", and a missing
    axis means no.
    """
    from types import SimpleNamespace
    from unittest import mock

    from veomni.ops.kernels.moe_experts.shared import act_fn_experts

    def with_sizes(sizes):
        return mock.patch(
            "veomni.distributed.moe.moe_layer.get_parallel_state",
            lambda: SimpleNamespace(extra_parallel_sizes=sizes),
        )

    with with_sizes({"ep": 4}):
        assert act_fn_experts.expert_parallel_size() == 4
    with with_sizes({"ep": 1}):
        assert act_fn_experts.expert_parallel_size() == 1
    with with_sizes({"ple": 2}):
        assert act_fn_experts.expert_parallel_size() == 1
    with with_sizes({}):
        assert act_fn_experts.expert_parallel_size() == 1


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_pytorch_tanh"])
def test_a_served_activation_differentiates_under_fullgraph_compile(activation):
    """The four served names compile; that is part of why they are the served set.

    `activation_backward` routes them through ATen backward ops, and a row built on it
    stays traceable under `torch.compile(fullgraph=True)`. A name outside the set has
    to differentiate an arbitrary transformers callable through a fresh
    `requires_grad_(True)` leaf, which Dynamo rejects -- a limit recorded in
    `veomni/ops/activation.py` and `constraints.md` rather than worked around,
    because no model here configures such a name.
    """
    import torch

    from veomni.ops.kernels.moe_experts.shared.activation import activation_backward

    class _Row(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, name):
            from veomni.ops.activation import resolve

            ctx.name = name
            ctx.save_for_backward(hidden)
            return resolve(name)(hidden)

        @staticmethod
        def backward(ctx, grad_output):
            (hidden,) = ctx.saved_tensors
            return activation_backward(grad_output, hidden, ctx.name), None

    hidden = torch.randn(64, dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(_Row.apply, fullgraph=True, backend="eager")
    compiled(hidden, activation).sum().backward()
    assert hidden.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_an_unserved_activation_is_the_documented_compile_limit():
    """The limit is real and pinned, so it is not mistaken for a regression later.

    If a future change makes these compile too, this test fails and the documentation
    in `activation.py` / `constraints.md` should be updated to match -- which is a
    better outcome than silently having prose that no longer describes the code.
    """
    import torch

    from veomni.ops.kernels.moe_experts.shared.activation import activation_backward

    class _Row(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, name):
            from veomni.ops.activation import resolve

            ctx.name = name
            ctx.save_for_backward(hidden)
            return resolve(name)(hidden)

        @staticmethod
        def backward(ctx, grad_output):
            (hidden,) = ctx.saved_tensors
            return activation_backward(grad_output, hidden, ctx.name), None

    hidden = torch.randn(64, dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(_Row.apply, fullgraph=True, backend="eager")
    with pytest.raises(Exception, match="requires_grad_|Unsupported"):
        compiled(hidden, "relu").sum().backward()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_dispatch_to_ep_class_calls_the_expert_math_in_cumsum_order():
    """Pin the ``(permute_tokens, cumsum, *weights)`` contract the rows rely on.

    Every fused row hands its per-rank math to `dispatch_to_ep_class`, and the shared
    helper decides what that argument order is. A swap there is invisible to the GPU
    parity test when both sides call the same helper -- both would be wrong
    identically. This stubs out the collectives and records what arrives.

    Also covers the two accepted shapes: an `autograd.Function` class (`.apply`) and a
    plain callable, which the Ascend and DeepGEMM rows pass because their grouped
    matmul is already a `Function`.
    """
    from unittest import mock

    import torch

    from veomni.distributed.moe import moe_layer

    seen = {}

    def record(tokens, cumsum, *rest):
        seen["args"] = (tokens, cumsum, *rest)
        return tokens

    counts = torch.tensor([3, 0, 5])
    stubs = {
        "preprocess": lambda **_: (
            [3],
            [3],
            counts,
            counts,
        ),
        "token_pre_all2all": lambda hidden_states, **_: (
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            None,
            None,
            tuple(hidden_states.shape),
        ),
        "tokens_post_all2all": lambda expert_outputs, **_: expert_outputs,
    }
    with (
        mock.patch.multiple(moe_layer, **stubs),
        mock.patch.object(moe_layer, "get_parallel_state", lambda: mock.MagicMock(ep_group=mock.MagicMock())),
    ):
        hidden = torch.randn(6, 4)
        weight = torch.randn(8, 4)

        # a plain callable, as the Ascend and DeepGEMM rows pass
        moe_layer.dispatch_to_ep_class(
            record, 2, torch.rand(6, 1), torch.zeros(6, 1, dtype=torch.long), hidden, weight
        )
        tokens, cumsum, *rest = seen["args"]
        assert rest[0] is weight
        assert torch.equal(cumsum, torch.cumsum(counts, dim=0)), "the helper hands over a cumsum"

        # and an autograd.Function class, as the Triton and LoRA rows pass
        class _Math(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tokens, cumsum, *rest):
                seen["class_args"] = (tokens, cumsum, *rest)
                return tokens

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output, None, None)

        moe_layer.dispatch_to_ep_class(_Math, 2, torch.rand(6, 1), torch.zeros(6, 1, dtype=torch.long), hidden, weight)
        assert seen["class_args"][2] is weight


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_eager_row_has_one_reference_loop_not_two():
    """`eager` must delegate, not carry a second copy of the expert loop.

    It used to hold an inline non-EP loop beside the shared one, chosen by
    ``if not supports(activation) or _experts_are_sharded()``. The inline copy indexed
    expert weights with *global* ids, so it was wrong under EP and could not serve an
    activation outside the whitelist -- two behaviours that the branch condition hid
    rather than expressed. Keeping only the delegated path means one implementation of
    "the reference row", and EP correctness is a property of the row rather than of
    which branch a caller happened to take.

    Static because the defect is structural: a second `for` loop over experts compiled
    and passed the numeric tests on the non-EP path.
    """
    import ast
    import pathlib

    from veomni.ops.kernels.moe_experts.standard import eager as eager_module

    source = pathlib.Path(eager_module.__file__).read_text()

    tree = ast.parse(source)
    wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "wrapper")
    loops = [node for node in ast.walk(wrapper) if isinstance(node, (ast.For, ast.AsyncFor))]
    assert not loops, "eager.wrapper runs the expert loop inline again; it must delegate to the shared one"

    def _callee_tail(node: ast.AST) -> str | None:
        """Last name of a call target, so an aliased import still counts as a call."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    calls = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)]
    assert any(_callee_tail(c.func) == "merged_experts_act_fn_forward" for c in calls), (
        "eager.wrapper must call merged_experts_act_fn_forward"
    )

    # The property the name claims is "one reference loop in the repository", so check
    # the module that actually owns it: exactly one loop over experts. A second copy
    # reintroduced there would not show up in eager.py at all.
    from veomni.ops.kernels.moe_experts.shared import act_fn_experts as shared_module

    shared_tree = ast.parse(pathlib.Path(shared_module.__file__).read_text())
    shared_loops = [
        node
        for node in ast.walk(shared_tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and any(isinstance(n, ast.Name) and n.id == "expert_idx" for n in ast.walk(node))
    ]
    assert len(shared_loops) == 2, (
        "act_fn_experts should hold exactly two per-expert loops (the EP math and the "
        f"non-EP loop), found {len(shared_loops)}"
    )


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_pytorch_tanh", "relu"])
def test_split_and_merged_fc1_agree_on_every_activation(activation):
    """Both weight layouts take the same path, so they must agree bit for bit.

    `eager` accepts a fused `gate_up_proj` or a `fc1_1`/`fc1_2` pair. Merging the two
    loops made the split layout a concatenation inside the shared loop; a mismatch
    there would silently train a different function depending on which layout a model
    happened to build.
    """
    from veomni.ops.kernels.moe_experts.standard import eager

    torch.manual_seed(0)
    experts, hidden, intermediate, tokens, top_k = 3, 8, 16, 5, 2
    routed = torch.randn(tokens, hidden)
    routes = torch.randint(0, experts, (tokens, top_k))
    routing = torch.softmax(torch.randn(tokens, top_k), dim=-1)
    down = torch.randn(experts, hidden, intermediate)
    gate_w = torch.randn(experts, intermediate, hidden)
    up_w = torch.randn(experts, intermediate, hidden)
    empty = gate_w.new_empty(0)

    merged = eager.wrapper(
        routed,
        routing,
        routes,
        empty,
        empty,
        down,
        torch.cat([gate_w, up_w], dim=1),
        num_experts=experts,
        activation=activation,
    )
    split = eager.wrapper(
        routed,
        routing,
        routes,
        gate_w,
        up_w,
        down,
        empty,
        num_experts=experts,
        activation=activation,
    )
    assert torch.equal(merged, split), f"{activation}: a layout change altered the result"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_the_reference_loop_keeps_an_empty_microbatch_on_the_graph():
    """Zero routed tokens must still produce a tensor with a `grad_fn`.

    The loop builds its accumulator with `new_zeros` + `index_add_`, so when the loop
    body never executes the result is detached and autograd skips the block. Under EP
    that skips a rank's all-to-all backward while the other ranks block. The EP variants
    guard with `+ tokens * 0`; the non-EP loop lost that guard when the two loops were
    merged, so this pins it.
    """
    from veomni.ops.kernels.moe_experts.standard import eager

    experts, hidden, intermediate, tokens = 3, 8, 16, 4
    routed = torch.randn(tokens, hidden, requires_grad=True)
    down = torch.randn(experts, hidden, intermediate, requires_grad=True)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, requires_grad=True)
    empty = gate_up.new_empty(0)

    # An empty microbatch: no token reaches any expert, so the loop body never runs.
    out = eager.wrapper(
        routed[:0],
        torch.ones(0, 1),
        torch.zeros(0, 1, dtype=torch.long),
        empty,
        empty,
        down,
        gate_up,
        num_experts=experts,
    )
    assert out.requires_grad, "an empty microbatch detached from the graph"
    assert out.grad_fn is not None

    out.sum().backward()
    assert routed.grad is not None, "the input got no gradient, so the block was skipped"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("ep_size", [2, 4])
def test_gpt_oss_eager_ep_math_matches_the_hf_loop(ep_size):
    """GPT-OSS's eager row dispatches under EP instead of indexing a shard.

    It had no EP branch at all: the loop read `gate_up_proj[idx]` with a global expert
    id against per-rank shards, so `gpt_oss + eager + ep_size>1` raised. The interleaved
    gate/up layout and the clamped activation are GPT-OSS's own, so the per-rank math
    stays in `gpt_oss/eager.py` and only the plumbing is shared.

    The oracle is HF's `GptOssExperts` itself, run once per local expert over the rows
    that expert owns. That pins the interleaved `[..., ::2]` slicing, which side of the
    clamp each half takes, and `alpha` -- a restated formula in the test would not.
    """
    from veomni.ops.kernels.moe_experts.gpt_oss import eager as gpt_oss_eager

    torch.manual_seed(0)
    experts, hidden, intermediate, rows = 4, 16, 32, 16
    routed = torch.randn(rows, hidden)
    # Right-multiplied layout, as the model builds it: [E, H, 2I] and [E, I, H].
    gate_up = torch.randn(experts, hidden, 2 * intermediate)
    gate_up_bias = torch.randn(experts, 2 * intermediate)
    down = torch.randn(experts, intermediate, hidden)
    down_bias = torch.randn(experts, hidden)

    # Expert 1 owns no rows, so its slice is zero-length and contributes nothing.
    cumsum = torch.tensor([5, 5, 11, 16], dtype=torch.long)

    expected = torch.empty_like(routed)
    row_starts = torch.cat([cumsum.new_zeros(1), cumsum[:-1]])
    units = torch.ones(1, 1)
    for e in range(experts):
        s, t = int(row_starts[e]), int(cumsum[e])
        if t == s:
            continue  # an empty expert: HF would be called with zero rows
        # One expert, routing weight 1.0, so HF returns that expert's contribution.
        oracle, _ = _gpt_oss_hf_loop(
            routed[s:t],
            units.expand(t - s, 1),
            torch.zeros((t - s, 1), dtype=torch.long),  # local id 0 in the 1-expert slice
            gate_up[e : e + 1],
            gate_up_bias[e : e + 1],
            down[e : e + 1],
            down_bias[e : e + 1],
            num_experts=1,
        )
        expected[s:t] = oracle

    per_rank = experts // ep_size
    pieces = [
        gpt_oss_eager._GptOssExpertMathEP.apply(
            routed[int(row_starts[rank * per_rank]) : int(cumsum[(rank + 1) * per_rank - 1])],
            cumsum[rank * per_rank : (rank + 1) * per_rank] - (cumsum[rank * per_rank - 1] if rank > 0 else 0),
            gate_up[rank * per_rank : (rank + 1) * per_rank],
            gate_up_bias[rank * per_rank : (rank + 1) * per_rank],
            down[rank * per_rank : (rank + 1) * per_rank],
            down_bias[rank * per_rank : (rank + 1) * per_rank],
            1.702,
            7.0,
        )
        for rank in range(ep_size)
    ]
    got = torch.cat(pieces, dim=0)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    # The empty expert contributed exactly nothing, and its rows stayed put.
    assert int(cumsum[1] - cumsum[0]) == 0
    torch.testing.assert_close(got[5:5], expected[5:5])


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_gpt_oss_eager_wrapper_reaches_the_ep_branch():
    """The EP branch of `wrapper` must actually be taken, not just reachable in theory.

    The numeric test above calls the math class directly, so it cannot catch a wrapper
    that forgets to dispatch -- which is the exact shape of the bug this fixed. Here the
    collectives are stubbed and the branch is asserted through its output.
    """
    from unittest import mock

    from veomni.distributed.moe import moe_layer
    from veomni.ops.kernels.moe_experts.gpt_oss import eager as gpt_oss_eager

    torch.manual_seed(0)
    experts, hidden, intermediate, rows = 4, 8, 16, 6
    called: dict[str, int] = {}

    def fake_dispatch(ep_class, num_experts, routing_weights, *args, **kwargs):
        """Record the dispatch and run the math on the local shard."""
        called["dispatch"] = num_experts
        called["routing_dtype"] = routing_weights.dtype
        return torch.zeros(rows, hidden)

    with mock.patch.object(gpt_oss_eager, "expert_parallel_size", lambda: 2):
        with mock.patch.object(moe_layer, "dispatch_to_ep_class", fake_dispatch):
            import veomni.distributed.moe as moe_pkg

            with mock.patch.object(moe_pkg, "dispatch_to_ep_class", fake_dispatch):
                out = gpt_oss_eager.wrapper(
                    torch.randn(rows, hidden, dtype=torch.bfloat16),
                    torch.ones(rows, 1, dtype=torch.float32),
                    torch.zeros(rows, 1, dtype=torch.long),
                    torch.randn(experts // 2, hidden, 2 * intermediate, dtype=torch.bfloat16),
                    torch.randn(experts // 2, 2 * intermediate, dtype=torch.bfloat16),
                    torch.randn(experts // 2, intermediate, hidden, dtype=torch.bfloat16),
                    torch.randn(experts // 2, hidden, dtype=torch.bfloat16),
                    num_experts=experts,
                )

    assert called.get("dispatch") == experts, "the EP branch was not taken"
    assert called["routing_dtype"] == torch.bfloat16, "routing weights were not cast before dispatch"
    assert out.shape == (rows, hidden)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_gpt_oss_eager_refuses_weights_sharded_by_the_wrong_factor():
    """An EP state whose weights are not actually sharded must name the problem."""
    from unittest import mock

    from veomni.ops.kernels.moe_experts.gpt_oss import eager as gpt_oss_eager

    hidden, intermediate, experts = 8, 16, 4
    with mock.patch(
        "veomni.ops.kernels.moe_experts.gpt_oss.eager.expert_parallel_size",
        lambda: 2,
    ):
        with pytest.raises(ValueError, match="not sharded as expected"):
            gpt_oss_eager.wrapper(
                torch.randn(4, hidden),
                torch.ones(4, 1),
                torch.zeros(4, 1, dtype=torch.long),
                torch.randn(experts, hidden, 2 * intermediate),
                torch.randn(experts, 2 * intermediate),
                torch.randn(experts, intermediate, hidden),
                torch.randn(experts, hidden),
                num_experts=experts,
            )
