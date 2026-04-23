import pytest
import torch
import torch.nn.functional as F

from veomni.utils.device import get_device_type
from veomni.utils.import_utils import is_quack_gemm_available


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not is_quack_gemm_available(),
        reason="quack not available or GPU < SM90",
    ),
]


def _eager_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    """Reference eager MoE implementation for correctness comparison."""
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        idx = int(expert_idx[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[idx])
        x = hidden_states[token_idx]
        gate = F.linear(x, fc1_1_weight[idx])
        up = F.linear(x, fc1_2_weight[idx])
        y = F.linear(F.silu(gate) * up, fc2_weight[idx])
        y = y * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, y.to(output.dtype))

    return output


@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,topk",
    [
        (64, 8, 256, 128, 2),
        (128, 128, 2048, 768, 8),
        # Qwen3-MoE-like: 128 experts, top-8, hidden=2048, ffn=1024
        (512, 128, 2048, 1024, 8),
        # DeepSeek-V2-like: 64 experts, top-6, hidden=2048, ffn=1408
        (1024, 64, 2048, 1408, 6),
    ],
)
class TestQuackFusedMoe:
    """Test quack GEMM fused MoE against eager reference (forward + backward)."""

    def _make_inputs(self, num_tokens, num_experts, hidden_dim, ffn_dim, topk, device, dtype):
        torch.manual_seed(42)
        hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
        routing_weights = routing_weights.to(dtype)
        fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
        fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
        fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
        return hidden_states, routing_weights, selected_experts, fc1_1_weight, fc1_2_weight, fc2_weight

    def test_split_fc1(self, num_tokens, num_experts, hidden_dim, ffn_dim, topk):
        """Test split fc1 weights path against eager reference."""
        from veomni.ops.kernels.moe.quack_gemm import quack_gemm_fused_moe_forward

        device = torch.device(get_device_type())
        dtype = torch.bfloat16
        hidden_states, routing_weights, selected_experts, fc1_1_weight, fc1_2_weight, fc2_weight = self._make_inputs(
            num_tokens, num_experts, hidden_dim, ffn_dim, topk, device, dtype
        )

        # Quack forward + backward
        hs_q = hidden_states.clone().detach().requires_grad_(True)
        fc1_1_q = fc1_1_weight.clone().detach().requires_grad_(True)
        fc1_2_q = fc1_2_weight.clone().detach().requires_grad_(True)
        fc2_q = fc2_weight.clone().detach().requires_grad_(True)

        out_q = quack_gemm_fused_moe_forward(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hs_q,
            fc1_1_weight=fc1_1_q,
            fc1_2_weight=fc1_2_q,
            fc2_weight=fc2_q,
        )
        out_q.sum().backward()

        # Eager forward + backward
        hs_e = hidden_states.clone().detach().requires_grad_(True)
        fc1_1_e = fc1_1_weight.clone().detach().requires_grad_(True)
        fc1_2_e = fc1_2_weight.clone().detach().requires_grad_(True)
        fc2_e = fc2_weight.clone().detach().requires_grad_(True)

        out_e = _eager_moe_forward(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hs_e,
            fc1_1_weight=fc1_1_e,
            fc1_2_weight=fc1_2_e,
            fc2_weight=fc2_e,
        )
        out_e.sum().backward()

        # Forward comparison
        torch.testing.assert_close(out_q, out_e, rtol=1e-2, atol=1e-2)

        # Backward comparison
        torch.testing.assert_close(hs_q.grad, hs_e.grad, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(fc1_1_q.grad, fc1_1_e.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(fc1_2_q.grad, fc1_2_e.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(fc2_q.grad, fc2_e.grad, rtol=1e-2, atol=1e-2)

    def test_merged_fc1(self, num_tokens, num_experts, hidden_dim, ffn_dim, topk):
        """Test merged fc1_1_2 weights path against eager reference."""
        from veomni.ops.kernels.moe.quack_gemm import quack_gemm_fused_moe_forward

        device = torch.device(get_device_type())
        dtype = torch.bfloat16
        hidden_states, routing_weights, selected_experts, fc1_1_weight, fc1_2_weight, fc2_weight = self._make_inputs(
            num_tokens, num_experts, hidden_dim, ffn_dim, topk, device, dtype
        )

        fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()

        # Quack forward + backward
        hs_q = hidden_states.clone().detach().requires_grad_(True)
        fc1_q = fc1_1_2_weight.clone().detach().requires_grad_(True)
        fc2_q = fc2_weight.clone().detach().requires_grad_(True)

        out_q = quack_gemm_fused_moe_forward(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hs_q,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=fc2_q,
            fc1_1_2_weight=fc1_q,
        )
        out_q.sum().backward()

        # Eager forward + backward
        hs_e = hidden_states.clone().detach().requires_grad_(True)
        fc1_1_e = fc1_1_weight.clone().detach().requires_grad_(True)
        fc1_2_e = fc1_2_weight.clone().detach().requires_grad_(True)
        fc2_e = fc2_weight.clone().detach().requires_grad_(True)

        out_e = _eager_moe_forward(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hs_e,
            fc1_1_weight=fc1_1_e,
            fc1_2_weight=fc1_2_e,
            fc2_weight=fc2_e,
        )
        out_e.sum().backward()

        # Forward comparison
        torch.testing.assert_close(out_q, out_e, rtol=1e-2, atol=1e-2)

        # Backward comparison
        torch.testing.assert_close(hs_q.grad, hs_e.grad, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(fc2_q.grad, fc2_e.grad, rtol=1e-2, atol=1e-2)
        fc1_eager_grad = torch.cat([fc1_1_e.grad, fc1_2_e.grad], dim=1)
        torch.testing.assert_close(fc1_q.grad, fc1_eager_grad, rtol=3e-2, atol=3e-2)


class TestBuildMoeIndices:
    """Unit tests for _build_moe_indices with concrete examples."""

    def test_basic_example(self):
        """Verify cu_seqlens_m, A_idx, and scatter_index on a small hand-crafted input.

        Setup: 4 tokens, 3 experts, topk=2
            expert_index = [[0, 2],   # token 0 -> experts 0, 2
                            [1, 0],   # token 1 -> experts 1, 0
                            [2, 1],   # token 2 -> experts 2, 1
                            [0, 1]]   # token 3 -> experts 0, 1

        Flat expert assignments: [0, 2, 1, 0, 2, 1, 0, 1]
        Sorted by expert (stable): expert 0 appears at flat indices 0, 3, 6
                                    expert 1 appears at flat indices 2, 5, 7
                                    expert 2 appears at flat indices 1, 4

        Expected:
            cu_seqlens_m = [0, 3, 6, 8]
            A_idx (token indices) = [0, 1, 3, 1, 2, 3, 0, 2]  (flat_idx // topk)
            scatter_index: inverse of sorted_order, reshaped to [4, 2]
        """
        from veomni.ops.kernels.moe.quack_gemm import _build_moe_indices

        device = torch.device(get_device_type())
        expert_index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]], device=device)
        num_experts = 3

        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(expert_index, num_experts)

        # cu_seqlens_m: cumulative counts [0, 3, 6, 8]
        assert cu_seqlens_m.tolist() == [0, 3, 6, 8]

        # A_idx: token index for each expert-sorted position
        assert A_idx.tolist() == [0, 1, 3, 1, 2, 3, 0, 2]

        # scatter_index round-trip: gathering from expert-sorted output by scatter_index
        # should recover the original token order
        T, topk = expert_index.shape
        dummy_sorted = torch.arange(T * topk, device=device, dtype=torch.float32)
        gathered = dummy_sorted[scatter_index.flatten().long()]
        # Re-scattering and re-gathering should be identity
        re_sorted = torch.empty_like(dummy_sorted)
        re_sorted[scatter_index.flatten().long()] = gathered
        assert torch.equal(re_sorted, dummy_sorted)

    def test_all_same_expert(self):
        """All tokens routed to the same expert."""
        from veomni.ops.kernels.moe.quack_gemm import _build_moe_indices

        device = torch.device(get_device_type())
        expert_index = torch.zeros(8, 1, dtype=torch.long, device=device)
        num_experts = 4

        cu_seqlens_m, A_idx, scatter_index = _build_moe_indices(expert_index, num_experts)

        assert cu_seqlens_m.tolist() == [0, 8, 8, 8, 8]
        assert A_idx.tolist() == list(range(8))
