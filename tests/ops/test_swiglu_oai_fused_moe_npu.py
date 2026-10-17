# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch
import torch.nn.functional as F

from veomni.utils.device import IS_NPU_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="SwiGLU-OAI fused MoE parity requires torch_npu")


def _eager_moe(hidden, routing, selected, gate_up, down, *, limit=None, alpha=None):
    output = torch.zeros_like(hidden)
    expert_mask = F.one_hot(selected, num_classes=gate_up.shape[0]).permute(2, 1, 0)
    for expert_idx in range(gate_up.shape[0]):
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        gate, up = F.linear(hidden[token_idx], gate_up[expert_idx]).chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        if alpha is None:
            activated = F.silu(gate) * up
        else:
            # Same op order as MiniMaxM3VLExperts._apply_gate.
            activated = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
        current = F.linear(activated, down[expert_idx]) * routing[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current.to(output.dtype))
    return output


def _inputs(device, *, rank=0):
    dtype = torch.bfloat16
    tokens, experts, hidden_dim, intermediate_dim, top_k = 16, 4, 32, 24, 2
    torch.manual_seed(7)
    gate_up = 0.8 * torch.randn(experts, 2 * intermediate_dim, hidden_dim, device=device, dtype=dtype)
    down = 0.1 * torch.randn(experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
    torch.manual_seed(20 + rank)
    hidden = torch.randn(tokens, hidden_dim, device=device, dtype=dtype)
    selected = torch.stack(
        [
            (torch.arange(tokens, device=device) + rank) % experts,
            (torch.arange(tokens, device=device) + rank + 1) % experts,
        ],
        dim=-1,
    )
    routing = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).to(dtype)
    return hidden, routing, selected, gate_up, down


@pytest.mark.parametrize("limit,alpha", [(None, None), (0.7, 1.702)])
def test_npu_fused_moe_matches_eager_forward_backward(limit, alpha, monkeypatch):
    from veomni.ops.kernels.moe import npu_group_gemm

    monkeypatch.setattr(npu_group_gemm, "get_parallel_state", lambda: type("State", (), {"ep_enabled": False})())
    device = torch.device(get_device_type())
    hidden, routing, selected, gate_up, down = _inputs(device)

    def _leaves():
        return (
            hidden.detach().clone().requires_grad_(True),
            routing.detach().clone().requires_grad_(True),
            gate_up.detach().clone().requires_grad_(True),
            down.detach().clone().requires_grad_(True),
        )

    fused_hidden, fused_routing, fused_gate_up, fused_down = _leaves()
    fused_output = npu_group_gemm.npu_fused_moe_forward(
        gate_up.shape[0],
        fused_routing,
        selected,
        fused_hidden,
        None,
        None,
        fused_down,
        fused_gate_up,
        swiglu_limit=limit,
        swiglu_alpha=alpha,
    )
    grad_output = torch.randn_like(fused_output)
    fused_grads = torch.autograd.grad(
        fused_output, (fused_hidden, fused_routing, fused_gate_up, fused_down), grad_output
    )

    eager_hidden, eager_routing, eager_gate_up, eager_down = _leaves()
    eager_output = _eager_moe(
        eager_hidden,
        eager_routing,
        selected,
        eager_gate_up,
        eager_down,
        limit=limit,
        alpha=alpha,
    )
    eager_grads = torch.autograd.grad(
        eager_output, (eager_hidden, eager_routing, eager_gate_up, eager_down), grad_output
    )

    torch.testing.assert_close(fused_output, eager_output, rtol=5e-2, atol=5e-2)
    for fused_grad, eager_grad in zip(fused_grads, eager_grads, strict=True):
        torch.testing.assert_close(fused_grad, eager_grad, rtol=8e-2, atol=8e-2)


def _npu_ep_forward_backward_worker():
    import torch.distributed as dist

    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
    from veomni.ops.kernels.moe import npu_group_gemm

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    state = init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        device_type="npu",
        extra_parallel_sizes=(world_size,),
        extra_parallel_names=("ep",),
        extra_parallel_placement_innermost=(False,),
    )
    try:
        device = torch.device("npu", rank)
        hidden, routing, selected, full_gate_up, full_down = _inputs(device, rank=rank)
        num_experts = full_gate_up.shape[0]
        local_experts = num_experts // world_size
        expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
        local_gate_up = full_gate_up[expert_slice].detach().clone().requires_grad_(True)
        local_down = full_down[expert_slice].detach().clone().requires_grad_(True)
        hidden.requires_grad_(True)
        routing.requires_grad_(True)
        grad_output = torch.randn_like(hidden)

        fused_output = npu_group_gemm.npu_ep_fused_moe_forward(
            num_experts,
            routing,
            selected,
            hidden,
            None,
            None,
            local_down,
            local_gate_up,
            ep_group=state.ep_group,
            swiglu_limit=0.7,
            swiglu_alpha=1.702,
        )
        fused_grads = torch.autograd.grad(fused_output, (hidden, routing, local_gate_up, local_down), grad_output)

        eager_hidden = hidden.detach().clone().requires_grad_(True)
        eager_routing = routing.detach().clone().requires_grad_(True)
        eager_gate_up = full_gate_up.detach().clone().requires_grad_(True)
        eager_down = full_down.detach().clone().requires_grad_(True)
        eager_output = _eager_moe(
            eager_hidden,
            eager_routing,
            selected,
            eager_gate_up,
            eager_down,
            limit=0.7,
            alpha=1.702,
        )
        eager_grads = list(
            torch.autograd.grad(eager_output, (eager_hidden, eager_routing, eager_gate_up, eager_down), grad_output)
        )
        dist.all_reduce(eager_grads[2], group=state.ep_group)
        dist.all_reduce(eager_grads[3], group=state.ep_group)

        torch.testing.assert_close(fused_output, eager_output, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(fused_grads[0], eager_grads[0], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[1], eager_grads[1], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[2], eager_grads[2][expert_slice], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[3], eager_grads[3][expert_slice], rtol=8e-2, atol=8e-2)
    finally:
        clear_parallel_state()


def test_npu_swiglu_oai_ep2_matches_eager_forward_backward():
    if torch.npu.device_count() < 2:
        pytest.skip("SwiGLU-OAI EP parity requires two NPU devices.")

    from ..tools.launch_utils import torchrun

    torchrun(_npu_ep_forward_backward_worker, world_size=2)
