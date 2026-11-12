# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EP-local grouped GEMM parity across split/merged layouts and a PyTorch oracle."""

from __future__ import annotations

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.ops.moe_experts.reference import standard_ep_reference
from tests.ops.tol import (
    MOE_EP_PRE_SM90_ATOL,
    MOE_EP_PRE_SM90_GRAD_FC1_ATOL,
    MOE_EP_PRE_SM90_GRAD_FC1_RTOL,
    MOE_EP_PRE_SM90_GRAD_FC2_ATOL,
    MOE_EP_PRE_SM90_GRAD_FC2_RTOL,
    MOE_EP_PRE_SM90_GRAD_ROUTING_ATOL,
    MOE_EP_PRE_SM90_GRAD_ROUTING_RTOL,
    MOE_EP_SM90_ATOL,
    MOE_EP_SM90_GRAD_FC1_ATOL,
    MOE_EP_SM90_GRAD_FC1_RTOL,
    MOE_EP_SM90_GRAD_FC2_ATOL,
    MOE_EP_SM90_GRAD_FC2_RTOL,
    MOE_FUSED_GRAD_HIDDEN_ATOL,
    MOE_FUSED_GRAD_HIDDEN_RTOL,
    MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL,
    MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL,
    MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
)
from tests.ops.utils import assert_close_with_error, assert_reference_signal
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.distributed.moe import EPGroupGemm, EPMergedFc1GroupGemm
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device, is_sm90_or_above
from veomni.utils.import_utils import is_fused_moe_available, is_quack_gemm_available


_EP_DISPATCH_PLAN = LaunchSpec(nproc=2, device="accelerator")


def _skip_if_unsupported():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for fused MoE EP tests.")
    if not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")


def _make_ep_inputs(num_tokens, num_experts, hidden_dim, ffn_dim, seed):
    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    tokens_per_expert = torch.full((num_experts,), num_tokens // num_experts, dtype=torch.int64)
    remainder = num_tokens - tokens_per_expert.sum().item()
    for i in range(remainder):
        tokens_per_expert[i] += 1
    total_tokens = tokens_per_expert.sum().item()
    cumsum = torch.cumsum(tokens_per_expert, dim=0).to(device)
    permute_tokens = 0.1 * torch.randn(total_tokens, hidden_dim, device=device, dtype=dtype)
    fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    return cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, fc1_1_2_weight, fc2_weight


def _scatter_tokens(hidden_states, selected_experts, num_experts):
    from veomni.ops.kernels.moe_experts.shared.dispatch import expert_histogram, moe_scatter

    splits = expert_histogram(selected_experts, num_experts)
    scatter_index = selected_experts.flatten().argsort(stable=True).argsort().int().view(selected_experts.shape)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    cumsum = torch.cumsum(splits, dim=0)
    return scatter_output, cumsum, scatter_index


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("swiglu_limit", [None, 0.15])
@pytest.mark.parametrize("trainable", ["all", "input", "weights"])
def test_ep_quack_retained_graph(merged, swiglu_limit, trainable):
    """Early saved-state release must preserve retained and fresh backwards."""
    if not IS_CUDA_AVAILABLE or not is_quack_gemm_available():
        pytest.skip("Quack requires CUDA SM90+.")

    from veomni.ops.kernels.moe_experts.standard.quack import EPMergedFc1QuackGroupGemm, EPQuackGroupGemm

    _, tokens, gate, up, merged_weight, down = _make_ep_inputs(65, 4, 256, 128, 20261024)
    # Include empty groups and a non-aligned token count.
    cumsum = torch.tensor([0, 17, 17, 65], device=tokens.device, dtype=torch.int64)
    tensors = (tokens, merged_weight, down) if merged else (tokens, gate, up, down)
    function = EPMergedFc1QuackGroupGemm if merged else EPQuackGroupGemm
    values = tuple(
        t.detach().requires_grad_(trainable != ("weights" if i == 0 else "input")) for i, t in enumerate(tensors)
    )
    leaves = tuple(t for t in values if t.requires_grad)
    output = function.apply(values[0], cumsum, *values[1:], swiglu_limit)
    upstream = torch.randn_like(output)
    first = torch.autograd.grad(output, leaves, upstream, retain_graph=True)
    second = torch.autograd.grad(output, leaves, upstream)
    fresh_output = function.apply(values[0], cumsum, *values[1:], swiglu_limit)
    fresh = torch.autograd.grad(fresh_output, leaves, upstream)
    for retained, released, independent in zip(first, second, fresh):
        torch.testing.assert_close(retained, released, rtol=0, atol=0)
        torch.testing.assert_close(retained, independent, rtol=0, atol=0)


def _gather_tokens(expert_output, scatter_index):
    from veomni.ops.kernels.moe_experts.shared.dispatch import moe_gather

    return moe_gather(expert_output, scatter_index)


def _scatter_routing_weights(routing_weights, scatter_index):
    reshaped = routing_weights.reshape(-1, 1)
    scattered = torch.empty_like(reshaped)
    scattered[scatter_index.flatten()] = reshaped
    return scattered


def _scatter_tokens_autograd(hidden_states, scatter_index):
    """Mirror the Triton scatter layout with differentiable PyTorch indexing."""
    topk = scatter_index.shape[1]
    sorted_to_assignment = scatter_index.flatten().argsort()
    return hidden_states.repeat_interleave(topk, dim=0)[sorted_to_assignment]


def _gather_tokens_autograd(expert_output, routing_weights, scatter_index):
    """Mirror weighted Triton gather while retaining the test's autograd graph."""
    num_tokens, topk = scatter_index.shape
    assignment_output = expert_output[scatter_index.flatten()]
    weighted_output = assignment_output * routing_weights.reshape(-1, 1)
    return weighted_output.view(num_tokens, topk, -1).sum(dim=1)


def _ep_atol() -> float:
    return MOE_EP_SM90_ATOL if is_sm90_or_above() else MOE_EP_PRE_SM90_ATOL


def _ep_gradient_tolerances(swiglu_limit):
    if swiglu_limit is not None:
        hidden_tol = (MOE_FUSED_SWIGLU_GRAD_HIDDEN_ATOL, MOE_FUSED_SWIGLU_GRAD_HIDDEN_RTOL)
    else:
        hidden_tol = (MOE_FUSED_GRAD_HIDDEN_ATOL, MOE_FUSED_GRAD_HIDDEN_RTOL)
    if is_sm90_or_above():
        fc1_tol = (MOE_EP_SM90_GRAD_FC1_ATOL, MOE_EP_SM90_GRAD_FC1_RTOL)
        fc2_tol = (MOE_EP_SM90_GRAD_FC2_ATOL, MOE_EP_SM90_GRAD_FC2_RTOL)
        routing_tol = hidden_tol
    else:
        fc1_tol = (MOE_EP_PRE_SM90_GRAD_FC1_ATOL, MOE_EP_PRE_SM90_GRAD_FC1_RTOL)
        fc2_tol = (MOE_EP_PRE_SM90_GRAD_FC2_ATOL, MOE_EP_PRE_SM90_GRAD_FC2_RTOL)
        routing_tol = (MOE_EP_PRE_SM90_GRAD_ROUTING_ATOL, MOE_EP_PRE_SM90_GRAD_ROUTING_RTOL)
    return hidden_tol, routing_tol, fc1_tol, fc2_tol


def _assert_ep_reference_grads(pairs, hidden_tol, routing_tol, fc1_tol, fc2_tol):
    """Require useful reference signal, then compare with recorded error."""
    budgets = {
        "hidden gradient": hidden_tol,
        "routing gradient": routing_tol,
        "fc1 gradient": fc1_tol,
        "fc1_1 gradient": fc1_tol,
        "fc1_2 gradient": fc1_tol,
        "fc2 gradient": fc2_tol,
    }
    for name, actual, expected in pairs:
        atol, rtol = budgets[name]
        assert_reference_signal(name, expected, atol, rtol)
        assert_close_with_error(name, actual, expected, atol=atol, rtol=rtol)


def test_ep_weight_grad_budgets_are_platform_specific():
    """EP weight grads must not reuse the generic fused relative budget."""
    from tests.ops.tol import MOE_FUSED_GRAD_FC1_ATOL, MOE_FUSED_GRAD_FC2_ATOL

    assert MOE_EP_SM90_GRAD_FC1_ATOL < MOE_FUSED_GRAD_FC1_ATOL
    assert MOE_EP_SM90_GRAD_FC2_ATOL < MOE_FUSED_GRAD_FC2_ATOL
    assert MOE_EP_SM90_GRAD_FC1_RTOL == 0
    assert MOE_EP_SM90_GRAD_FC2_RTOL == 0
    assert MOE_EP_PRE_SM90_GRAD_FC1_RTOL == 0
    assert MOE_EP_PRE_SM90_GRAD_FC2_RTOL == 0
    hidden_tol, routing_tol, fc1_tol, fc2_tol = _ep_gradient_tolerances(None)
    assert fc1_tol[1] == 0
    assert fc2_tol[1] == 0
    assert hidden_tol[0] == MOE_FUSED_GRAD_HIDDEN_ATOL
    if is_sm90_or_above():
        assert routing_tol == hidden_tol
    else:
        assert routing_tol[0] == MOE_EP_PRE_SM90_GRAD_ROUTING_ATOL


@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,seed",
    [
        (256, 8, 1024, 512, 0),
        (128, 4, 512, 256, 1),
    ],
)
def test_ep_split_vs_merged(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    seed: int,
    swiglu_limit: float | None,
):
    _skip_if_unsupported()
    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, fc1_1_2_weight, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    pt_split = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)
    out_split = EPGroupGemm.apply(pt_split, cumsum, fc1_1_split, fc1_2_split, fc2_split, swiglu_limit)
    grad_output = torch.randn_like(out_split)
    out_split.backward(grad_output)

    pt_merged = permute_tokens.clone().detach().requires_grad_(True)
    fc1_merged = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_merged = fc2_weight.clone().detach().requires_grad_(True)
    out_merged = EPMergedFc1GroupGemm.apply(pt_merged, cumsum, fc1_merged, fc2_merged, swiglu_limit)
    out_merged.backward(grad_output)

    torch.testing.assert_close(out_split, out_merged, rtol=0, atol=0)
    torch.testing.assert_close(fc2_split.grad, fc2_merged.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1),
        fc1_merged.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pt_split.grad,
        pt_merged.grad,
        rtol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
        atol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
    )


@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,seed",
    [
        (256, 8, 1024, 512, 0),
        (128, 4, 512, 256, 1),
    ],
)
def test_ep_quack_split_vs_merged(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    seed: int,
    swiglu_limit: float | None,
):
    _skip_if_unsupported()
    if not is_quack_gemm_available():
        pytest.skip("quack not available or GPU < SM90")

    from veomni.ops.kernels.moe_experts.standard.quack import EPMergedFc1QuackGroupGemm

    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, fc1_1_2_weight, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    pt_split = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)
    out_split = EPGroupGemm.apply(pt_split, cumsum, fc1_1_split, fc1_2_split, fc2_split, swiglu_limit)
    grad_output = torch.randn_like(out_split)
    out_split.backward(grad_output)

    pt_quack = permute_tokens.clone().detach().requires_grad_(True)
    fc1_quack = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_quack = fc2_weight.clone().detach().requires_grad_(True)
    out_quack = EPMergedFc1QuackGroupGemm.apply(pt_quack, cumsum, fc1_quack, fc2_quack, swiglu_limit)
    out_quack.backward(grad_output)

    torch.testing.assert_close(out_split, out_quack, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fc2_split.grad, fc2_quack.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1),
        fc1_quack.grad,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(pt_split.grad, pt_quack.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,seed",
    [
        (256, 8, 1024, 512, 0),
        (128, 4, 512, 256, 1),
    ],
)
def test_ep_quack_split(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    seed: int,
    swiglu_limit: float | None,
):
    _skip_if_unsupported()
    if not is_quack_gemm_available():
        pytest.skip("quack not available or GPU < SM90")

    from veomni.ops.kernels.moe_experts.standard.quack import EPQuackGroupGemm

    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, _, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    pt_triton = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_triton = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_triton = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_triton = fc2_weight.clone().detach().requires_grad_(True)
    out_triton = EPGroupGemm.apply(pt_triton, cumsum, fc1_1_triton, fc1_2_triton, fc2_triton, swiglu_limit)
    grad_output = torch.randn_like(out_triton)
    out_triton.backward(grad_output)

    pt_quack = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_quack = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_quack = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_quack = fc2_weight.clone().detach().requires_grad_(True)
    out_quack = EPQuackGroupGemm.apply(pt_quack, cumsum, fc1_1_quack, fc1_2_quack, fc2_quack, swiglu_limit)
    out_quack.backward(grad_output)

    torch.testing.assert_close(out_triton, out_quack, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fc2_triton.grad, fc2_quack.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fc1_1_triton.grad, fc1_1_quack.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(fc1_2_triton.grad, fc1_2_quack.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(pt_triton.grad, pt_quack.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,topk,seed",
    [
        (256, 8, 1024, 512, 2, 0),
        (128, 4, 512, 256, 2, 1),
        (256, 16, 1024, 512, 4, 2),
    ],
)
def test_ep_vs_non_ep(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
    swiglu_limit: float | None,
):
    _skip_if_unsupported()
    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
    routing_weights = routing_weights.to(dtype)
    fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)

    scatter_output, cumsum, scatter_index = _scatter_tokens(hidden_states, selected_experts, num_experts)
    scattered_gw = _scatter_routing_weights(routing_weights, scatter_index)
    out_eager = standard_ep_reference(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
    )
    ep_raw = EPGroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_weight.clone().detach(),
        fc1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
        swiglu_limit,
    )
    out_ep = _gather_tokens(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)
    atol = _ep_atol()
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=atol)

    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    routing_eager = routing_weights.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = standard_ep_reference(
        hs_eager,
        routing_eager,
        selected_experts,
        fc1_1_eager,
        fc1_2_eager,
        fc2_eager,
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
    )
    grad_output = torch.randn_like(out_e)
    out_e.backward(grad_output)

    hs_ep = hidden_states.clone().detach().requires_grad_(True)
    routing_ep = routing_weights.clone().detach().requires_grad_(True)
    pt_ep = _scatter_tokens_autograd(hs_ep, scatter_index)
    fc1_1_ep = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_ep = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPGroupGemm.apply(pt_ep, cumsum, fc1_1_ep, fc1_2_ep, fc2_ep, swiglu_limit)
    out_ep2 = _gather_tokens_autograd(ep_raw2, routing_ep, scatter_index)
    out_ep2.backward(grad_output)
    hidden_tol, routing_tol, fc1_tol, fc2_tol = _ep_gradient_tolerances(swiglu_limit)
    _assert_ep_reference_grads(
        (
            ("hidden gradient", hs_ep.grad, hs_eager.grad),
            ("routing gradient", routing_ep.grad, routing_eager.grad),
            ("fc2 gradient", fc2_ep.grad, fc2_eager.grad),
            ("fc1_1 gradient", fc1_1_ep.grad, fc1_1_eager.grad),
            ("fc1_2 gradient", fc1_2_ep.grad, fc1_2_eager.grad),
        ),
        hidden_tol,
        routing_tol,
        fc1_tol,
        fc2_tol,
    )


@pytest.mark.parametrize("swiglu_limit", [None, 7.0, 10.0])
@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,topk,seed",
    [
        (256, 8, 1024, 512, 2, 0),
        (128, 4, 512, 256, 2, 1),
    ],
)
def test_ep_merged_vs_non_ep(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
    swiglu_limit: float | None,
):
    _skip_if_unsupported()
    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
    routing_weights = routing_weights.to(dtype)
    fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)

    scatter_output, cumsum, scatter_index = _scatter_tokens(hidden_states, selected_experts, num_experts)
    scattered_gw = _scatter_routing_weights(routing_weights, scatter_index)
    out_eager = standard_ep_reference(
        hidden_states,
        routing_weights,
        selected_experts,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
    )
    ep_raw = EPMergedFc1GroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
        swiglu_limit,
    )
    out_ep = _gather_tokens(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)
    atol = _ep_atol()
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=atol)

    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    routing_eager = routing_weights.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = standard_ep_reference(
        hs_eager,
        routing_eager,
        selected_experts,
        fc1_1_eager,
        fc1_2_eager,
        fc2_eager,
        num_experts=num_experts,
        swiglu_limit=swiglu_limit,
    )
    grad_output = torch.randn_like(out_e)
    out_e.backward(grad_output)

    hs_ep = hidden_states.clone().detach().requires_grad_(True)
    routing_ep = routing_weights.clone().detach().requires_grad_(True)
    pt_ep = _scatter_tokens_autograd(hs_ep, scatter_index)
    fc1_merged_ep = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPMergedFc1GroupGemm.apply(pt_ep, cumsum, fc1_merged_ep, fc2_ep, swiglu_limit)
    out_ep2 = _gather_tokens_autograd(ep_raw2, routing_ep, scatter_index)
    out_ep2.backward(grad_output)
    hidden_tol, routing_tol, fc1_tol, fc2_tol = _ep_gradient_tolerances(swiglu_limit)
    _assert_ep_reference_grads(
        (
            ("hidden gradient", hs_ep.grad, hs_eager.grad),
            ("routing gradient", routing_ep.grad, routing_eager.grad),
            ("fc2 gradient", fc2_ep.grad, fc2_eager.grad),
            ("fc1 gradient", fc1_merged_ep.grad, torch.cat([fc1_1_eager.grad, fc1_2_eager.grad], dim=1)),
        ),
        hidden_tol,
        routing_tol,
        fc1_tol,
        fc2_tol,
    )


@pytest.mark.parametrize("swiglu_limit", [None, 7.0])
def test_fused_moe_routing_weights_match_ep_projection_order(swiglu_limit: float | None):
    """Non-EP fused MoE should apply routing weights after fc2, matching EP."""
    _skip_if_unsupported()
    from veomni.ops.kernels.moe_experts.standard.triton import (
        MergedFc1TritonFusedMoeExpertFunction,
        TritonFusedMoeExpertFunction,
    )

    torch.manual_seed(0)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, ffn_dim, topk = 64, 8, 256, 128, 2
    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
    routing_weights = routing_weights.to(dtype)
    fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)

    scatter_output, cumsum, scatter_index = _scatter_tokens(hidden_states, selected_experts, num_experts)
    grad_output = torch.randn_like(hidden_states)

    hs_split = hidden_states.clone().detach().requires_grad_(True)
    rw_split = routing_weights.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)
    out_split = TritonFusedMoeExpertFunction.apply(
        num_experts,
        rw_split,
        selected_experts,
        hs_split,
        fc1_1_split,
        fc1_2_split,
        fc2_split,
        swiglu_limit,
    )
    out_split.backward(grad_output)

    hs_merged = hidden_states.clone().detach().requires_grad_(True)
    rw_merged = routing_weights.clone().detach().requires_grad_(True)
    fc1_merged = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_merged = fc2_weight.clone().detach().requires_grad_(True)
    out_merged = MergedFc1TritonFusedMoeExpertFunction.apply(
        num_experts,
        rw_merged,
        selected_experts,
        hs_merged,
        fc1_merged,
        fc2_merged,
        swiglu_limit,
    )
    out_merged.backward(grad_output)

    hs_ep = hidden_states.clone().detach().requires_grad_(True)
    rw_ep = routing_weights.clone().detach().requires_grad_(True)
    pt_ep = _scatter_tokens_autograd(hs_ep, scatter_index)
    fc1_ep = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw = EPMergedFc1GroupGemm.apply(pt_ep, cumsum, fc1_ep, fc2_ep, swiglu_limit)
    out_ep = _gather_tokens_autograd(ep_raw, rw_ep, scatter_index)
    out_ep.backward(grad_output)

    torch.testing.assert_close(out_split, out_ep, rtol=0, atol=0)
    torch.testing.assert_close(out_merged, out_ep, rtol=0, atol=0)
    torch.testing.assert_close(rw_split.grad, rw_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(rw_merged.grad, rw_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(hs_merged.grad, hs_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(fc1_merged.grad, fc1_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(fc2_merged.grad, fc2_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1),
        fc1_ep.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(fc2_split.grad, fc2_ep.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        hs_split.grad,
        hs_ep.grad,
        rtol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
        atol=MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL,
    )


def _distributed_standard_ep_worker(rank, merged, endpoint):
    """Compare real token exchange and every gradient with a non-EP replica."""
    import torch.distributed as dist

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, use_parallel_state
    from veomni.ops import resolve_op

    world = _EP_DISPATCH_PLAN.nproc
    get_torch_device().set_device(rank)
    dist.init_process_group("nccl", store=endpoint.connect(), rank=rank, world_size=world)
    try:
        device = torch.device(get_device_type(), rank)
        generator = torch.Generator().manual_seed(20261105)
        tokens, experts, hidden, intermediate, top_k = 128, 4, 512, 256, 2

        def random(shape):
            return (0.1 * torch.randn(shape, generator=generator)).to(device=device, dtype=torch.bfloat16)

        inputs = random((world, tokens, hidden))[rank]
        gate = random((experts, intermediate, hidden))
        up = random((experts, intermediate, hidden))
        down = random((experts, hidden, intermediate))
        upstream = random((world, tokens, hidden))[rank]
        ids = torch.arange(tokens, device=device)
        routes = torch.stack(((ids + rank) % experts, (ids + rank + 1) % experts), dim=1)
        routing = torch.tensor([0.75, 0.25], device=device, dtype=torch.float32).expand(tokens, top_k).contiguous()
        full_weights = [torch.cat((gate, up), dim=1), down] if merged else [gate, up, down]
        reference_weights = [w.detach().clone().requires_grad_() for w in full_weights]
        local_experts = experts // world
        local_weights = [
            w[rank * local_experts : (rank + 1) * local_experts].detach().clone().requires_grad_()
            for w in full_weights
        ]
        reference_input = inputs.detach().clone().requires_grad_()
        actual_input = inputs.detach().clone().requires_grad_()
        reference_routing = routing.detach().clone().requires_grad_()
        actual_routing = routing.detach().clone().requires_grad_()
        empty = torch.empty(0, device=device, dtype=torch.bfloat16)
        fused = resolve_op("moe_experts", "standard", "fused_triton").wrapper

        def forward(x, weights, scores):
            if merged:
                return fused(x, scores, routes, empty, empty, weights[1], weights[0], num_experts=experts)
            return fused(x, scores, routes, weights[0], weights[1], weights[2], empty, num_experts=experts)

        nonep_state = _init_parallel_state(dp_size=world, device_type=get_device_type(), name="schedule_nonep")
        ep_state = _init_parallel_state(
            dp_size=world, device_type=get_device_type(), extra_parallel_sizes=(world,), name="schedule_ep"
        )
        with use_parallel_state(nonep_state):
            expected = forward(reference_input, reference_weights, reference_routing)
            reference_grads = torch.autograd.grad(
                expected, [reference_input, reference_routing, *reference_weights], upstream
            )
        with use_parallel_state(ep_state):
            actual = forward(actual_input, local_weights, actual_routing)
            actual_grads = torch.autograd.grad(actual, [actual_input, actual_routing, *local_weights], upstream)

        # Finish all collectives before any assertion can send a peer to cleanup.
        weight_pairs = []
        names = ["fc1 gradient", "fc2 gradient"] if merged else ["fc1_1 gradient", "fc1_2 gradient", "fc2 gradient"]
        for name, observed, full_grad in zip(names, actual_grads[2:], reference_grads[2:], strict=True):
            full = full_grad.float()
            dist.all_reduce(full, op=dist.ReduceOp.SUM)
            target = full[rank * local_experts : (rank + 1) * local_experts].to(observed.dtype)
            weight_pairs.append((name, observed, target))
        torch.testing.assert_close(actual, expected, rtol=0, atol=_ep_atol())
        _assert_ep_reference_grads(
            [
                ("hidden gradient", actual_grads[0], reference_grads[0]),
                ("routing gradient", actual_grads[1], reference_grads[1]),
                *weight_pairs,
            ],
            *_ep_gradient_tolerances(None),
        )
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_EP_DISPATCH_PLAN)
@pytest.mark.parametrize("merged", [False, True], ids=["split-fc1", "merged-fc1"])
def test_distributed_standard_ep_dispatch_matches_nonep(merged):
    """Exercise the compact expert schedule behind real all-to-all dispatch."""
    _skip_if_unsupported()
    if runner_device_count() < _EP_DISPATCH_PLAN.nproc:
        pytest.skip("Distributed EP dispatch requires its declared accelerator plan")
    spawn_with_rendezvous(_distributed_standard_ep_worker, args=(merged,), nprocs=_EP_DISPATCH_PLAN.nproc)
