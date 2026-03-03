import pytest
import torch


def _skip_if_unsupported():
    from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_capability

    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA is required for fused MoE tests.")
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm is not available in this torch build.")
    capability = get_device_capability()
    if capability is None or capability[0] < 8:
        pytest.skip("torch._grouped_mm requires sm >= 8.")


def _build_inputs(num_tokens: int, num_experts: int, hidden_dim: int, ffn_dim: int, topk: int, seed: int):
    from veomni.utils.device import get_device_type

    torch.manual_seed(seed)
    device = torch.device(get_device_type())
    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    routing_weights = torch.randn(num_tokens, topk, device=device, dtype=dtype)
    selected_experts = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int64)

    fc1_1_weight = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc1_2_weight = torch.randn(num_experts, ffn_dim, hidden_dim, device=device, dtype=dtype)
    fc2_weight = torch.randn(num_experts, hidden_dim, ffn_dim, device=device, dtype=dtype)
    fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)

    return (
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight,
    )


@pytest.mark.parametrize(
    "num_tokens,num_experts,hidden_dim,ffn_dim,topk,seed",
    [
        (61, 8, 128, 256, 2, 0),
        (96, 16, 128, 384, 2, 1),
        (128, 4, 64, 128, 1, 42),
    ],
)
def test_fused_gate_up_matches_split(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    topk: int,
    seed: int,
):
    _skip_if_unsupported()

    from veomni.ops.fused_moe.torch_fused_moe import torch_fused_moe_forward

    (
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight,
    ) = _build_inputs(num_tokens, num_experts, hidden_dim, ffn_dim, topk, seed)

    # --- split path (2 grouped_mm for gate + up) ---
    rw_split = routing_weights.clone().detach().requires_grad_(True)
    hs_split = hidden_states.clone().detach().requires_grad_(True)
    w11_split = fc1_1_weight.clone().detach().requires_grad_(True)
    w12_split = fc1_2_weight.clone().detach().requires_grad_(True)
    w2_split = fc2_weight.clone().detach().requires_grad_(True)

    out_split = torch_fused_moe_forward(
        num_experts,
        rw_split,
        selected_experts,
        hs_split,
        w11_split,
        w12_split,
        w2_split,
    )
    loss_split = out_split.float().pow(2).mean()
    loss_split.backward()

    # --- fused gate+up path (1 grouped_mm for gate+up) ---
    rw_fused = routing_weights.clone().detach().requires_grad_(True)
    hs_fused = hidden_states.clone().detach().requires_grad_(True)
    w112_fused = fc1_1_2_weight.clone().detach().requires_grad_(True)
    w2_fused = fc2_weight.clone().detach().requires_grad_(True)

    out_fused = torch_fused_moe_forward(
        num_experts,
        rw_fused,
        selected_experts,
        hs_fused,
        # split weights unused when fc1_1_2_weight is provided; pass dummy values
        # that satisfy the signature but won't be touched.
        fc1_1_weight,
        fc1_2_weight,
        w2_fused,
        fc1_1_2_weight=w112_fused,
    )
    loss_fused = out_fused.float().pow(2).mean()
    loss_fused.backward()

    # --- forward parity (bit-exact: same K per output column) ---
    torch.testing.assert_close(out_split, out_fused, rtol=0, atol=0)

    # --- backward parity ---
    # Weight and routing grads have identical accumulation patterns → bit-exact.
    torch.testing.assert_close(rw_split.grad, rw_fused.grad, rtol=0, atol=0)
    torch.testing.assert_close(w2_split.grad, w2_fused.grad, rtol=0, atol=0)

    w11_w12_grad_split = torch.cat([w11_split.grad, w12_split.grad], dim=1)
    torch.testing.assert_close(w11_w12_grad_split, w112_fused.grad, rtol=0, atol=0)

    # grad_hidden_states: the fused path accumulates over K=2*ffn_dim in one
    # grouped_mm backward, while the split path does two with K=ffn_dim and
    # adds.  Different bf16 accumulation order → large element-wise diffs.
    # Use relative L2 norm instead of element-wise comparison.
    hs_diff = (hs_split.grad - hs_fused.grad).float().norm()
    hs_ref = hs_split.grad.float().norm()
    rel_l2 = hs_diff / hs_ref
    assert rel_l2 < 0.01, f"grad_hidden_states relative L2 error {rel_l2:.4f} exceeds 1%"
