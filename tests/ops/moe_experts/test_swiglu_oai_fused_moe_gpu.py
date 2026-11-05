from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device
from veomni.utils.import_utils import is_fused_moe_available


_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")


def _skip_if_unsupported():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for SwiGLU-OAI fused MoE tests.")
    if not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")


def _eager_swiglu_oai_moe(
    hidden_states,
    routing_weights,
    selected_experts,
    gate_up_proj,
    down_proj,
    limit,
    alpha,
):
    output = torch.zeros_like(hidden_states)
    num_experts = gate_up_proj.shape[0]
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        gate, up = F.linear(hidden_states[token_idx], gate_up_proj[expert_idx]).chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        # Same op order as MiniMaxM3VLExperts._apply_gate / forward.
        activated = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
        current = F.linear(activated, down_proj[expert_idx]) * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current.to(output.dtype))
    return output


def test_triton_swiglu_oai_moe_matches_eager_forward_backward(monkeypatch):
    _skip_if_unsupported()
    from veomni.ops.kernels.moe_experts.standard import triton as group_gemm

    monkeypatch.setattr(group_gemm, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    torch.manual_seed(0)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_dim, intermediate_dim, top_k = 12, 4, 16, 8, 2
    limit, alpha = 0.7, 1.702

    hidden = torch.zeros(num_tokens, hidden_dim, device=device, dtype=dtype)
    hidden[:, 0] = 1
    hidden.requires_grad_(True)
    gate_up = torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim, device=device, dtype=dtype)
    gate_up[:, :intermediate_dim, 0] = torch.tensor(
        [-1.0, 0.2, 1.0, -0.4, 0.7, 1.2, -0.9, 0.1], device=device, dtype=dtype
    )
    gate_up[:, intermediate_dim:, 0] = torch.tensor(
        [-1.0, -0.7, 0.7, 1.0, -0.2, 0.3, 1.1, -1.2], device=device, dtype=dtype
    )
    gate_up.requires_grad_(True)
    down = (0.1 * torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)).requires_grad_(
        True
    )
    selected_experts = torch.stack(
        [
            torch.arange(num_tokens, device=device) % num_experts,
            (torch.arange(num_tokens, device=device) + 1) % num_experts,
        ],
        dim=-1,
    )
    routing_weights = (
        torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1).to(dtype).requires_grad_(True)
    )

    fused_output = group_gemm.group_gemm_fused_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden,
        fc1_1_weight=None,
        fc1_2_weight=None,
        fc2_weight=down,
        fc1_1_2_weight=gate_up,
        swiglu_limit=limit,
        swiglu_alpha=alpha,
    )
    grad_output = torch.randn_like(fused_output)
    fused_grads = torch.autograd.grad(fused_output, (hidden, routing_weights, gate_up, down), grad_output)

    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager_routing = routing_weights.detach().clone().requires_grad_(True)
    eager_gate_up = gate_up.detach().clone().requires_grad_(True)
    eager_down = down.detach().clone().requires_grad_(True)
    eager_output = _eager_swiglu_oai_moe(
        eager_hidden,
        eager_routing,
        selected_experts,
        eager_gate_up,
        eager_down,
        limit,
        alpha,
    )
    eager_grads = torch.autograd.grad(
        eager_output, (eager_hidden, eager_routing, eager_gate_up, eager_down), grad_output
    )

    torch.testing.assert_close(fused_output, eager_output, rtol=5e-2, atol=5e-2)
    for fused_grad, eager_grad in zip(fused_grads, eager_grads, strict=True):
        torch.testing.assert_close(fused_grad, eager_grad, rtol=8e-2, atol=8e-2)


def _ep_forward_backward_worker():
    import torch.distributed as dist

    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.ops.kernels.moe_experts.standard import triton as group_gemm

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        device_type=get_device_type(),
        extra_parallel_sizes=(world_size,),
        extra_parallel_names=("ep",),
        extra_parallel_placement_innermost=(False,),
    )
    try:
        device = torch.device(get_device_type(), rank)
        dtype = torch.bfloat16
        num_tokens, num_experts, hidden_dim, intermediate_dim, top_k = 8, 4, 16, 8, 2
        local_experts = num_experts // world_size
        limit, alpha = 0.7, 1.702

        torch.manual_seed(11)
        full_gate_up = 0.8 * torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, device=device, dtype=dtype)
        full_down = 0.1 * torch.randn(num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype)
        expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
        local_gate_up = full_gate_up[expert_slice].detach().clone().requires_grad_(True)
        local_down = full_down[expert_slice].detach().clone().requires_grad_(True)

        torch.manual_seed(100 + rank)
        hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype).requires_grad_(True)
        selected = torch.stack(
            [
                (torch.arange(num_tokens, device=device) + rank) % num_experts,
                (torch.arange(num_tokens, device=device) + rank + 1) % num_experts,
            ],
            dim=-1,
        )
        routing = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1).to(dtype).requires_grad_(True)
        grad_output = torch.randn_like(hidden)

        fused_output = group_gemm.group_gemm_fused_moe_forward(
            num_experts,
            routing,
            selected,
            hidden,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=local_down,
            fc1_1_2_weight=local_gate_up,
            swiglu_limit=limit,
            swiglu_alpha=alpha,
        )
        fused_grads = torch.autograd.grad(
            fused_output,
            (hidden, routing, local_gate_up, local_down),
            grad_output,
        )

        eager_hidden = hidden.detach().clone().requires_grad_(True)
        eager_routing = routing.detach().clone().requires_grad_(True)
        eager_gate_up = full_gate_up.detach().clone().requires_grad_(True)
        eager_down = full_down.detach().clone().requires_grad_(True)
        eager_output = _eager_swiglu_oai_moe(
            eager_hidden,
            eager_routing,
            selected,
            eager_gate_up,
            eager_down,
            limit,
            alpha,
        )
        eager_grads = list(
            torch.autograd.grad(
                eager_output,
                (eager_hidden, eager_routing, eager_gate_up, eager_down),
                grad_output,
            )
        )
        dist.all_reduce(eager_grads[2])
        dist.all_reduce(eager_grads[3])

        torch.testing.assert_close(fused_output, eager_output, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(fused_grads[0], eager_grads[0], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[1], eager_grads[1], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[2], eager_grads[2][expert_slice], rtol=8e-2, atol=8e-2)
        torch.testing.assert_close(fused_grads[3], eager_grads[3][expert_slice], rtol=8e-2, atol=8e-2)
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
def test_triton_swiglu_oai_ep2_matches_eager_forward_backward():
    _skip_if_unsupported()
    if get_torch_device().device_count() < 2:
        pytest.skip("SwiGLU-OAI EP parity requires two CUDA devices.")

    from tests.tools.launch_utils import torchrun

    torchrun(_ep_forward_backward_worker, world_size=_CI_ACCELERATOR_2.nproc)
