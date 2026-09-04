# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EP-local grouped GEMM vs split/merged layouts and the non-EP fused path."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.tol import MOE_EP_PRE_SM90_ATOL, MOE_EP_SM90_ATOL, MOE_SPLIT_MERGED_GRAD_HIDDEN_ATOL
from veomni.distributed.moe import EPGroupGemm, EPMergedFc1GroupGemm
from veomni.kernels._kernels.moe_experts.shared.dispatch import expert_histogram, moe_gather, moe_scatter
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, is_sm90_or_above
from veomni.utils.import_utils import is_fused_moe_available, is_quack_gemm_available


def _skip_if_unsupported():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for fused MoE EP tests.")
    if not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")


def _eager_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Fused operator-order eager reference. Routing scales the SwiGLU intermediate."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        y = F.silu(gate) * up
        y = y * routing_weights[token_idx, top_k_pos, None]
        y = F.linear(y, fc2_weight[idx])
        output.index_add_(0, token_idx, y.to(output.dtype))
    return output


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
    splits = expert_histogram(selected_experts, num_experts)
    scatter_index = selected_experts.flatten().argsort(stable=True).argsort().int().view(selected_experts.shape)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    cumsum = torch.cumsum(splits, dim=0)
    return scatter_output, cumsum, scatter_index


def _scatter_routing_weights(routing_weights, scatter_index):
    reshaped = routing_weights.reshape(-1, 1)
    scattered = torch.empty_like(reshaped)
    scattered[scatter_index.flatten()] = reshaped
    return scattered


def _ep_atol() -> float:
    return MOE_EP_SM90_ATOL if is_sm90_or_above() else MOE_EP_PRE_SM90_ATOL


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

    from veomni.kernels._kernels.moe_experts.standard.quack import EPMergedFc1QuackGroupGemm

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

    from veomni.kernels._kernels.moe_experts.standard.quack import EPQuackGroupGemm

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
    out_eager = _eager_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit,
    )
    ep_raw = EPGroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_weight.clone().detach(),
        fc1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
        swiglu_limit,
    )
    out_ep = moe_gather(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)
    atol = _ep_atol()
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=atol)

    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = _eager_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hs_eager,
        fc1_1_eager,
        fc1_2_eager,
        fc2_eager,
        swiglu_limit,
    )
    out_e.sum().backward()

    pt_ep = scatter_output.clone().detach().requires_grad_(True)
    fc1_1_ep = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_ep = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPGroupGemm.apply(pt_ep, cumsum, fc1_1_ep, fc1_2_ep, fc2_ep, swiglu_limit)
    ep_raw2.backward(scattered_gw.expand_as(ep_raw2).contiguous())
    torch.testing.assert_close(fc2_eager.grad, fc2_ep.grad, rtol=0, atol=atol)
    torch.testing.assert_close(fc1_1_eager.grad, fc1_1_ep.grad, rtol=0, atol=atol)
    torch.testing.assert_close(fc1_2_eager.grad, fc1_2_ep.grad, rtol=0, atol=atol)


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
    out_eager = _eager_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit,
    )
    ep_raw = EPMergedFc1GroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
        swiglu_limit,
    )
    out_ep = moe_gather(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)
    atol = _ep_atol()
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=atol)

    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = _eager_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hs_eager,
        fc1_1_eager,
        fc1_2_eager,
        fc2_eager,
        swiglu_limit,
    )
    out_e.sum().backward()

    pt_ep = scatter_output.clone().detach().requires_grad_(True)
    fc1_merged_ep = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPMergedFc1GroupGemm.apply(pt_ep, cumsum, fc1_merged_ep, fc2_ep, swiglu_limit)
    ep_raw2.backward(scattered_gw.expand_as(ep_raw2).contiguous())
    torch.testing.assert_close(fc2_eager.grad, fc2_ep.grad, rtol=0, atol=atol)
    torch.testing.assert_close(
        torch.cat([fc1_1_eager.grad, fc1_2_eager.grad], dim=1),
        fc1_merged_ep.grad,
        rtol=0,
        atol=atol,
    )
