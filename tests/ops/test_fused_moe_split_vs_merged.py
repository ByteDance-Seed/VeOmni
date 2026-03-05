import pytest
import torch
import torch.nn.functional as F

from veomni.ops import fused_moe
from veomni.ops.fused_moe import fused_moe_forward
from veomni.ops.fused_moe.group_gemm import group_gemm_fused_moe_forward
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device
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
        # Qwen3-30B-A3B config: num_experts=128, top_k=8, hidden=2048, moe_intermediate=768
        (512, 128, 2048, 768, 8, 0),
        # Moonlight-16B-A3B config: n_routed_experts=64, top_k=6, hidden=2048, moe_intermediate=1408
        (256, 64, 2048, 1408, 6, 1),
    ],
)
def test_fused_moe_split_vs_merged(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify split and merged fc1 paths match in forward/backward, and both approximate eager."""
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

    monkeypatch.setattr(fused_moe, "_fused_moe_forward", group_gemm_fused_moe_forward)

    # --- Split fc1 forward + backward with memory profiling ---
    hs_split = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)

    get_torch_device().reset_peak_memory_stats(device)
    get_torch_device().synchronize(device)
    mem_before_split = get_torch_device().memory_allocated(device)
    out_split = fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hs_split,
        fc1_1_weight=fc1_1_split,
        fc1_2_weight=fc1_2_split,
        fc2_weight=fc2_split,
    )
    get_torch_device().synchronize(device)
    peak_split = get_torch_device().max_memory_allocated(device) - mem_before_split
    out_split.sum().backward()

    # --- Merged fc1 forward + backward with memory profiling ---
    hs_merged = hidden_states.clone().detach().requires_grad_(True)
    fc1_merged = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_merged = fc2_weight.clone().detach().requires_grad_(True)

    get_torch_device().reset_peak_memory_stats(device)
    get_torch_device().synchronize(device)
    mem_before_merged = get_torch_device().memory_allocated(device)
    out_merged = fused_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hs_merged,
        fc1_1_weight=None,
        fc1_2_weight=None,
        fc2_weight=fc2_merged,
        fc1_1_2_weight=fc1_merged,
    )
    get_torch_device().synchronize(device)
    peak_merged = get_torch_device().max_memory_allocated(device) - mem_before_merged
    out_merged.sum().backward()

    # --- Eager forward + backward ---
    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)

    out_eager = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hs_eager,
        fc1_1_weight=fc1_1_eager,
        fc1_2_weight=fc1_2_eager,
        fc2_weight=fc2_eager,
    )
    out_eager.sum().backward()

    # Split vs merged forward: bitwise identical (output columns are independent)
    torch.testing.assert_close(out_split, out_merged, rtol=0, atol=0)

    # Split vs merged backward: approximate match because the dgrad step
    # accumulates over 2I elements (merged) vs two sums of I elements (split),
    # producing different bf16 rounding.
    # TODO: make merged fc1 backward has higher accuracy
    torch.testing.assert_close(hs_split.grad, hs_merged.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(fc2_split.grad, fc2_merged.grad, rtol=0, atol=0)
    fc1_split_grad = torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1)
    torch.testing.assert_close(fc1_split_grad, fc1_merged.grad, rtol=0, atol=0)

    # Fused vs eager: approximate match
    torch.testing.assert_close(out_merged, out_eager, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hs_merged.grad, hs_eager.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(fc2_merged.grad, fc2_eager.grad, rtol=1e-2, atol=1e-2)
    fc1_eager_grad = torch.cat([fc1_1_eager.grad, fc1_2_eager.grad], dim=1)
    torch.testing.assert_close(fc1_merged.grad, fc1_eager_grad, rtol=3e-2, atol=3e-2)

    # Memory profiling
    peak_diff_mb = (peak_split - peak_merged) / (1024 * 1024)
    print(
        f"\n[Memory] experts={num_experts} hidden={hidden_dim} ffn={ffn_dim} tokens={num_tokens} topk={topk}"
        f"\n  split peak:  {peak_split / (1024 * 1024):.1f} MiB"
        f"\n  merged peak: {peak_merged / (1024 * 1024):.1f} MiB"
        f"\n  diff (split - merged): {peak_diff_mb:+.1f} MiB"
    )
