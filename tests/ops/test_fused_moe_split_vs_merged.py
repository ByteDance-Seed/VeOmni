import pytest
import torch
import torch.nn.functional as F

from veomni.distributed.moe.moe_layer import EPGroupGemm, EPMergedFc1GroupGemm
from veomni.ops.kernels import moe as fused_moe
from veomni.ops.kernels.moe import fused_moe_forward
from veomni.ops.kernels.moe._kernels.kernel.moe import expert_histogram, moe_gather, moe_scatter
from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device, is_sm90_or_above
from veomni.utils.import_utils import is_fused_moe_available, is_quack_gemm_available


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


def _make_ep_inputs(num_tokens, num_experts, hidden_dim, ffn_dim, seed):
    """Create synthetic EP test inputs: permute_tokens, cumsum, and weights."""
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
):
    """Verify EPGroupGemm (split) and EPMergedFc1GroupGemm (merged) produce identical results."""
    _skip_if_unsupported()

    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, fc1_1_2_weight, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    # --- Split path ---
    pt_split = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)

    out_split = EPGroupGemm.apply(pt_split, cumsum, fc1_1_split, fc1_2_split, fc2_split)
    # Use a contiguous grad tensor; .sum().backward() produces non-contiguous expand grads
    grad_output = torch.randn_like(out_split)
    out_split.backward(grad_output)

    # --- Merged path ---
    pt_merged = permute_tokens.clone().detach().requires_grad_(True)
    fc1_merged = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_merged = fc2_weight.clone().detach().requires_grad_(True)

    out_merged = EPMergedFc1GroupGemm.apply(pt_merged, cumsum, fc1_merged, fc2_merged)
    out_merged.backward(grad_output)

    # Forward: bitwise identical
    torch.testing.assert_close(out_split, out_merged, rtol=0, atol=0)

    # Backward: fc2 weight grad bitwise identical
    torch.testing.assert_close(fc2_split.grad, fc2_merged.grad, rtol=0, atol=0)

    # Backward: fc1 weight grad bitwise identical
    fc1_split_grad = torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1)
    torch.testing.assert_close(fc1_split_grad, fc1_merged.grad, rtol=0, atol=0)

    # Backward: hidden grad approximate match (bf16 accumulation differences)
    torch.testing.assert_close(pt_split.grad, pt_merged.grad, rtol=3e-2, atol=3e-2)


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
):
    """Verify EPMergedFc1QuackGroupGemm matches EPGroupGemm (triton split) in forward/backward."""
    _skip_if_unsupported()
    if not is_quack_gemm_available():
        pytest.skip("quack not available or GPU < SM90")

    from veomni.ops.kernels.moe.quack_gemm import EPMergedFc1QuackGroupGemm

    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, fc1_1_2_weight, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    # --- Triton split path (reference) ---
    pt_split = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_split = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_split = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_split = fc2_weight.clone().detach().requires_grad_(True)

    out_split = EPGroupGemm.apply(pt_split, cumsum, fc1_1_split, fc1_2_split, fc2_split)
    grad_output = torch.randn_like(out_split)
    out_split.backward(grad_output)

    # --- Quack merged path ---
    pt_quack = permute_tokens.clone().detach().requires_grad_(True)
    fc1_quack = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_quack = fc2_weight.clone().detach().requires_grad_(True)

    out_quack = EPMergedFc1QuackGroupGemm.apply(pt_quack, cumsum, fc1_quack, fc2_quack)
    out_quack.backward(grad_output)

    # Forward: approximate match (different GEMM backends)
    torch.testing.assert_close(out_split, out_quack, rtol=1e-2, atol=1e-2)

    # Backward: fc2 weight grad
    torch.testing.assert_close(fc2_split.grad, fc2_quack.grad, rtol=1e-2, atol=1e-2)

    # Backward: fc1 weight grad
    fc1_split_grad = torch.cat([fc1_1_split.grad, fc1_2_split.grad], dim=1)
    torch.testing.assert_close(fc1_split_grad, fc1_quack.grad, rtol=3e-2, atol=3e-2)

    # Backward: hidden grad
    torch.testing.assert_close(pt_split.grad, pt_quack.grad, rtol=3e-2, atol=3e-2)


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
):
    """Verify EPQuackGroupGemm (quack split) matches EPGroupGemm (triton split) in forward/backward."""
    _skip_if_unsupported()
    if not is_quack_gemm_available():
        pytest.skip("quack not available or GPU < SM90")

    from veomni.ops.kernels.moe.quack_gemm import EPQuackGroupGemm

    cumsum, permute_tokens, fc1_1_weight, fc1_2_weight, _, fc2_weight = _make_ep_inputs(
        num_tokens, num_experts, hidden_dim, ffn_dim, seed
    )

    # --- Triton split path (reference) ---
    pt_triton = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_triton = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_triton = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_triton = fc2_weight.clone().detach().requires_grad_(True)

    out_triton = EPGroupGemm.apply(pt_triton, cumsum, fc1_1_triton, fc1_2_triton, fc2_triton)
    grad_output = torch.randn_like(out_triton)
    out_triton.backward(grad_output)

    # --- Quack split path ---
    pt_quack = permute_tokens.clone().detach().requires_grad_(True)
    fc1_1_quack = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_quack = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_quack = fc2_weight.clone().detach().requires_grad_(True)

    out_quack = EPQuackGroupGemm.apply(pt_quack, cumsum, fc1_1_quack, fc1_2_quack, fc2_quack)
    out_quack.backward(grad_output)

    # Forward: approximate match (different GEMM backends)
    torch.testing.assert_close(out_triton, out_quack, rtol=1e-2, atol=1e-2)

    # Backward: weight grads
    torch.testing.assert_close(fc2_triton.grad, fc2_quack.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fc1_1_triton.grad, fc1_1_quack.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(fc1_2_triton.grad, fc1_2_quack.grad, rtol=3e-2, atol=3e-2)

    # Backward: hidden grad
    torch.testing.assert_close(pt_triton.grad, pt_quack.grad, rtol=3e-2, atol=3e-2)


def _scatter_tokens(hidden_states, selected_experts, num_experts):
    """Scatter tokens by expert and return (scatter_output, cumsum, scatter_index, scattered_gw_fn).

    Mirrors the token-sorting logic in TritonFusedMoeExpertFunction, allowing
    the EP autograd functions to be tested without a real distributed process group.
    """
    splits = expert_histogram(selected_experts, num_experts)
    scatter_index = selected_experts.flatten().argsort(stable=True).argsort().int().view(selected_experts.shape)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    cumsum = torch.cumsum(splits, dim=0)
    return scatter_output, cumsum, scatter_index


def _scatter_routing_weights(routing_weights, scatter_index):
    """Reorder routing weights into expert-sorted order matching scatter_output."""
    reshaped = routing_weights.reshape(-1, 1)
    scattered = torch.empty_like(reshaped)
    scattered[scatter_index.flatten()] = reshaped
    return scattered


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
):
    """Verify EP autograd functions produce the same output as the non-EP eager path.

    The EP path (EPGroupGemm) computes fc2(silu(fc1_1(x)) * fc1_2(x)) per expert,
    then applies routing weights externally.  The non-EP path applies routing weights
    before fc2.  Since fc2 is linear, w * fc2(x) == fc2(w * x), so both should match
    up to bf16 rounding differences.

    Forward comparison uses the full scatter → EP gemm → routing weight → gather pipeline.
    Backward comparison uses scattered routing weights as grad_output to EPGroupGemm,
    which is equivalent to backprop through sum(routing_weight * ep_output).
    """
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

    # Scatter tokens by expert (shared preprocessing)
    scatter_output, cumsum, scatter_index = _scatter_tokens(hidden_states, selected_experts, num_experts)
    scattered_gw = _scatter_routing_weights(routing_weights, scatter_index)

    # --- Forward: eager reference vs EP ---
    out_eager = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=fc2_weight,
    )

    ep_raw = EPGroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_weight.clone().detach(),
        fc1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
    )
    out_ep = moe_gather(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)

    # SM90+ (Hopper): bitwise for topk=2, atol<=1.95e-3 for topk=4.
    # Pre-SM90 (e.g. L20/Ada): group_gemm has larger rounding diffs, atol<=1.56e-2 observed.
    fwd_atol = 4e-3 if is_sm90_or_above() else 3.2e-2
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=fwd_atol)

    # --- Backward: weight grads via EPGroupGemm with scattered routing weights as grad ---
    # d/d(ep_raw) of sum(gather(w * ep_raw)) = w (broadcast), so passing scattered_gw
    # as grad_output to EPGroupGemm.backward produces the same weight grads as eager.
    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hs_eager,
        fc1_1_weight=fc1_1_eager,
        fc1_2_weight=fc1_2_eager,
        fc2_weight=fc2_eager,
    )
    out_e.sum().backward()

    pt_ep = scatter_output.clone().detach().requires_grad_(True)
    fc1_1_ep = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_ep = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPGroupGemm.apply(pt_ep, cumsum, fc1_1_ep, fc1_2_ep, fc2_ep)
    ep_raw2.backward(scattered_gw.expand_as(ep_raw2).contiguous())

    # SM90+: bitwise; pre-SM90 (L20 CI): atol<=1.56e-2 observed.
    fc2_atol = 0 if is_sm90_or_above() else 3.2e-2
    torch.testing.assert_close(fc2_eager.grad, fc2_ep.grad, rtol=0, atol=fc2_atol)
    # SM90+: atol<=1.95e-3; pre-SM90: larger bf16 accumulation rounding.
    fc1_atol = 4e-3 if is_sm90_or_above() else 3.2e-2
    torch.testing.assert_close(fc1_1_eager.grad, fc1_1_ep.grad, rtol=0, atol=fc1_atol)
    torch.testing.assert_close(fc1_2_eager.grad, fc1_2_ep.grad, rtol=0, atol=fc1_atol)


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
):
    """Verify EPMergedFc1GroupGemm produces the same output as the non-EP eager path."""
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

    # Scatter tokens by expert
    scatter_output, cumsum, scatter_index = _scatter_tokens(hidden_states, selected_experts, num_experts)
    scattered_gw = _scatter_routing_weights(routing_weights, scatter_index)

    # --- Forward: eager reference vs EP merged ---
    out_eager = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=fc1_1_weight,
        fc1_2_weight=fc1_2_weight,
        fc2_weight=fc2_weight,
    )

    ep_raw = EPMergedFc1GroupGemm.apply(
        scatter_output.clone().detach(),
        cumsum,
        fc1_1_2_weight.clone().detach(),
        fc2_weight.clone().detach(),
    )
    out_ep = moe_gather(ep_raw * scattered_gw, scatter_index).reshape(hidden_states.shape)

    # SM90+: bitwise; pre-SM90 (L20 CI): larger rounding diffs.
    fwd_atol = 0 if is_sm90_or_above() else 3.2e-2
    torch.testing.assert_close(out_eager, out_ep, rtol=0, atol=fwd_atol)

    # --- Backward: weight grads ---
    hs_eager = hidden_states.clone().detach().requires_grad_(True)
    fc1_1_eager = fc1_1_weight.clone().detach().requires_grad_(True)
    fc1_2_eager = fc1_2_weight.clone().detach().requires_grad_(True)
    fc2_eager = fc2_weight.clone().detach().requires_grad_(True)
    out_e = _eager_moe_forward(
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hs_eager,
        fc1_1_weight=fc1_1_eager,
        fc1_2_weight=fc1_2_eager,
        fc2_weight=fc2_eager,
    )
    out_e.sum().backward()

    pt_ep = scatter_output.clone().detach().requires_grad_(True)
    fc1_merged_ep = fc1_1_2_weight.clone().detach().requires_grad_(True)
    fc2_ep = fc2_weight.clone().detach().requires_grad_(True)
    ep_raw2 = EPMergedFc1GroupGemm.apply(pt_ep, cumsum, fc1_merged_ep, fc2_ep)
    ep_raw2.backward(scattered_gw.expand_as(ep_raw2).contiguous())

    fc2_atol = 0 if is_sm90_or_above() else 3.2e-2
    torch.testing.assert_close(fc2_eager.grad, fc2_ep.grad, rtol=0, atol=fc2_atol)
    fc1_atol = 4e-3 if is_sm90_or_above() else 3.2e-2
    fc1_eager_grad = torch.cat([fc1_1_eager.grad, fc1_2_eager.grad], dim=1)
    torch.testing.assert_close(fc1_eager_grad, fc1_merged_ep.grad, rtol=0, atol=fc1_atol)
