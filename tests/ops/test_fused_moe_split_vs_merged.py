import pytest
import torch
import torch.nn.functional as F

from veomni.ops import fused_moe
from veomni.ops.fused_moe import fused_moe_forward
from veomni.ops.fused_moe.group_gemm import group_gemm_fused_moe_forward
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type
from veomni.utils.import_utils import is_fused_moe_available


def _skip_if_unsupported():
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for fused MoE split/merged parity test.")
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
        # Qwen3-30B-A3B config: num_experts=16, top_k=2, hidden=2048, moe_intermediate=768
        (512, 128, 2048, 768, 8, 0),
        # DeepSeek V3 671B config: n_routed_experts=256, top_k=8, hidden=7168, moe_intermediate=2048
        (256, 256, 7168, 2048, 8, 1),
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
    device = torch.device(get_device_type())
    dtype = torch.bfloat16

    hidden_states = 0.1 * torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    # Match real router behavior: top-k experts from a softmax distribution.
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), topk, dim=-1)
    routing_weights = routing_weights.to(dtype)
    fc1_1_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = 0.1 * torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1).contiguous()
    fc2_weight = 0.1 * torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)

    # Use group_gemm backend directly.
    monkeypatch.setattr(fused_moe, "_fused_moe_forward", group_gemm_fused_moe_forward)

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

    torch.testing.assert_close(out_split, out_merged, rtol=0, atol=0)
    torch.testing.assert_close(out_split, out_eager, rtol=2e-3, atol=2e-3)
