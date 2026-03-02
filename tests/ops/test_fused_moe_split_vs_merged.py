import pytest
import torch
import torch.nn.functional as F

from veomni.ops import fused_moe
from veomni.ops.fused_moe import fused_moe_forward
from veomni.ops.fused_moe.torch_fused_moe import torch_fused_moe_forward


def _skip_if_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused MoE split/merged parity test.")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm is not available in this torch build.")


def _eager_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
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
    "num_tokens,num_experts,hidden_dim,ffn_dim,topk,seed",
    [
        (61, 8, 128, 256, 2, 0),
        (96, 16, 128, 384, 2, 1),
    ],
)
def test_fused_moe_split_and_merged_match_eager(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_if_unsupported()

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing_weights = torch.randn(num_tokens, topk, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int64)
    fc1_1_weight = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc2_weight = torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)

    monkeypatch.setattr(fused_moe, "_fused_moe_forward", torch_fused_moe_forward)

    out_split = fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=fc2_weight,
    )
    out_merged = fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=None,
        fc1_2_weight=None,
        fc2_weight=fc2_weight,
        fc1_1_2_weight=fc1_1_2_weight,
    )
    out_eager = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=fc2_weight,
    )

    torch.testing.assert_close(out_split, out_merged, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_split, out_eager, rtol=2e-2, atol=2e-2)
