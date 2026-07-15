from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as dsv4


def _deepseek_v4_experts_reference(
    *,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        gate_up = F.linear(hidden_states[token_idx], gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        current = F.linear(F.silu(gate) * up, down_proj[expert_idx])
        current = current * routing_weights[token_idx, top_k_pos, None]
        output.index_add_(0, token_idx, current.to(output.dtype))

    return output


def test_deepseek_v4_test_overrides_keep_eager_attention_and_expected_moe():
    from tests.tools.training_utils import resolve_ops_overrides
    from veomni.utils.import_utils import is_torch_npu_available

    overrides = resolve_ops_overrides("deepseek_v4")

    expected_moe = "eager" if is_torch_npu_available() else "fused_triton"
    assert "--model.ops_implementation.attn_implementation=eager" in overrides
    assert f"--model.ops_implementation.moe_implementation={expected_moe}" in overrides


def test_deepseek_v4_topk_router_uses_fp32_projection():
    config = SimpleNamespace(
        num_experts_per_tok=2,
        num_local_experts=4,
        hidden_size=8,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
    )
    router = dsv4.DeepseekV4TopKRouter(config).to(torch.bfloat16)
    router.use_fp32_projection = True
    hidden_states = torch.linspace(-1.0, 1.0, 24, dtype=torch.bfloat16).reshape(1, 3, 8)

    logits, weights, indices = router(hidden_states)
    expected_logits = F.linear(hidden_states.reshape(-1, 8).float(), router.weight.float())
    expected_scores = expected_logits.sigmoid()
    expected_indices = torch.topk(expected_scores, 2, dim=-1, sorted=False).indices
    expected_weights = expected_scores.gather(1, expected_indices)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_deepseek_v4_weighted_rms_norm_matches_official_fp32_multiply_order():
    norm = dsv4.DeepseekV4RMSNorm(32).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(42)
    hidden_states = torch.randn(2, 3, 32, generator=generator, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32, generator=generator, dtype=torch.bfloat16))

    normalized = hidden_states.float()
    variance = normalized.square().mean(-1, keepdim=True)
    normalized *= torch.rsqrt(variance + norm.variance_epsilon)
    expected = (norm.weight.float() * normalized).to(hidden_states.dtype)
    old_cast_order = norm.weight * normalized.to(hidden_states.dtype)

    actual = norm(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, old_cast_order)


def test_deepseek_v4_fused_moe_receives_merged_weights_and_swiglu_limit(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=3,
        hidden_size=5,
        intermediate_size=7,
        hidden_act="silu",
        swiglu_limit=6.5,
    )
    experts = dsv4.DeepseekV4Experts(config)

    gate = torch.arange(
        config.num_local_experts * config.intermediate_size * config.hidden_size,
        dtype=torch.float32,
    ).reshape(config.num_local_experts, config.intermediate_size, config.hidden_size)
    gate = gate.mul_(0.01).add_(0.1)
    up = torch.arange(gate.numel(), dtype=torch.float32).reshape_as(gate).mul_(0.02).sub_(0.2)
    down = torch.arange(
        config.num_local_experts * config.hidden_size * config.intermediate_size,
        dtype=torch.float32,
    ).reshape(config.num_local_experts, config.hidden_size, config.intermediate_size)
    down = down.mul_(0.015).sub_(0.05)

    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.cat([gate, up], dim=1))
        experts.down_proj.copy_(down)

    hidden_states = torch.linspace(-0.7, 0.8, steps=4 * config.hidden_size).reshape(4, config.hidden_size)
    selected_experts = torch.tensor(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 2],
        ],
        dtype=torch.long,
    )
    top_k_weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.8, 0.2],
        ],
        dtype=torch.float64,
    )

    class _FusedSlot:
        use_non_eager_impl = True

    captured = {}

    def fake_fused_moe_forward(
        *,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight=None,
        swiglu_limit=None,
    ):
        captured.update(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            fc1_1_weight=fc1_1_weight,
            fc1_2_weight=fc1_2_weight,
            fc2_weight=fc2_weight,
            fc1_1_2_weight=fc1_1_2_weight,
            swiglu_limit=swiglu_limit,
        )
        return _deepseek_v4_experts_reference(
            num_experts=num_experts,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_states,
            gate_up_proj=fc1_1_2_weight,
            down_proj=fc2_weight,
            swiglu_limit=swiglu_limit,
        )

    monkeypatch.setattr(dsv4, "veomni_moe_experts_forward", _FusedSlot())
    monkeypatch.setattr(dsv4, "fused_moe_forward", fake_fused_moe_forward)

    actual = experts(hidden_states, selected_experts, top_k_weights)
    expected = _deepseek_v4_experts_reference(
        num_experts=config.num_local_experts,
        routing_weights=top_k_weights.to(hidden_states.dtype),
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_up_proj=experts.gate_up_proj,
        down_proj=experts.down_proj,
        swiglu_limit=config.swiglu_limit,
    )

    assert captured["num_experts"] == config.num_local_experts
    assert captured["fc1_1_weight"] is None
    assert captured["fc1_2_weight"] is None
    assert captured["fc2_weight"] is experts.down_proj
    assert captured["fc1_1_2_weight"] is experts.gate_up_proj
    assert captured["swiglu_limit"] == config.swiglu_limit
    assert captured["routing_weights"].dtype == hidden_states.dtype
    torch.testing.assert_close(captured["routing_weights"], top_k_weights.to(hidden_states.dtype), rtol=0, atol=0)
    torch.testing.assert_close(captured["fc1_1_2_weight"][:, : config.intermediate_size], gate, rtol=0, atol=0)
    torch.testing.assert_close(captured["fc1_1_2_weight"][:, config.intermediate_size :], up, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepseek_v4_ondemand_fp4_uses_bf16_weights_and_quantizes_both_gemms(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=32,
        hidden_act="silu",
        swiglu_limit=6.0,
        expert_dtype="fp4",
    )
    experts = dsv4.DeepseekV4Experts(config).to(torch.bfloat16).eval()
    experts.use_ondemand_fp4 = True
    hidden_states = torch.randn(5, 32, dtype=torch.bfloat16)
    selected_experts = torch.tensor([[0], [1], [0], [1], [0]])
    routing_weights = torch.tensor([[0.8], [0.7], [0.6], [0.5], [0.4]])
    calls = {"act": 0, "weight": 0, "gemm": 0}

    def fake_act_quant(x, **_kwargs):
        calls["act"] += 1
        return x, torch.ones(*x.shape[:-1], 1)

    def fake_fp4_act_quant(weight):
        calls["weight"] += 1
        return weight, torch.ones(*weight.shape[:-1], 1)

    def fake_fp4_gemm(x, _x_scale, weight, _weight_scale, _scale_dtype=torch.float8_e8m0fnu):
        calls["gemm"] += 1
        return F.linear(x, weight)

    monkeypatch.setattr(dsv4, "act_quant", fake_act_quant)
    monkeypatch.setattr(dsv4, "fp4_act_quant", fake_fp4_act_quant)
    monkeypatch.setattr(dsv4, "fp4_gemm", fake_fp4_gemm)
    monkeypatch.setattr(dsv4, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False, ep_rank=0))

    actual = experts(hidden_states, selected_experts, routing_weights)
    expected = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_idx in range(2):
        token_idx, top_k_pos = torch.where(selected_experts == expert_idx)
        gate_up = F.linear(hidden_states[token_idx], experts.gate_up_proj[expert_idx]).float()
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate.clamp(max=6.0)) * up.clamp(min=-6.0, max=6.0)
        intermediate *= routing_weights[token_idx, top_k_pos, None]
        current = F.linear(intermediate.to(torch.bfloat16), experts.down_proj[expert_idx])
        expected.index_add_(0, token_idx, current.float())

    assert calls == {"act": 4, "weight": 2, "gemm": 4}
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)

    monkeypatch.setattr(
        dsv4,
        "get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=True, ep_rank=0, ep_group=None),
    )
    with pytest.raises(RuntimeError, match="requires replicated inputs"):
        experts(hidden_states, selected_experts, routing_weights)
