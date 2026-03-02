import pytest
import torch
import torch.nn.functional as F

from veomni.ops import fused_moe
from veomni.ops.fused_moe import fused_moe_forward
from veomni.ops.fused_moe.torch_fused_moe import torch_fused_moe_forward
from veomni.utils.import_utils import is_fused_moe_available


def _skip_if_unsupported(moe_kernel: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused MoE split/merged parity test.")
    if moe_kernel == "triton" and not is_fused_moe_available():
        pytest.skip("Triton fused MoE is not available in this environment.")
    if moe_kernel == "torch" and not hasattr(torch, "_grouped_mm"):
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
    "moe_kernel,num_tokens,num_experts,hidden_dim,ffn_dim,topk,seed",
    [
        ("triton", 61, 8, 128, 256, 2, 0),
        ("triton", 96, 16, 128, 384, 2, 1),
        ("torch", 61, 8, 128, 256, 2, 0),
        ("torch", 96, 16, 128, 384, 2, 1),
    ],
)
def test_fused_moe_split_and_merged_match_eager(
    moe_kernel: str,
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_if_unsupported(moe_kernel)

    torch.manual_seed(seed)
    device = torch.device("cuda")
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

    if moe_kernel == "triton":
        from veomni.ops.fused_moe import group_gemm as group_gemm_module

        # Force fused_moe_forward -> group_gemm backend and force Triton branch first.
        monkeypatch.setattr(fused_moe, "_fused_moe_forward", group_gemm_module.group_gemm_fused_moe_forward)
        monkeypatch.setattr(group_gemm_module, "get_device_capability", lambda: (0, 0))
    elif moe_kernel == "torch":
        monkeypatch.setattr(fused_moe, "_fused_moe_forward", torch_fused_moe_forward)
    else:
        raise ValueError(f"Unsupported moe_kernel: {moe_kernel}")

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
