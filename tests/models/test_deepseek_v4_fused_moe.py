from types import SimpleNamespace

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

    is_npu = is_torch_npu_available()
    expected_moe = "eager" if is_npu else "fused_triton"
    assert "--model.ops_implementation.attn_implementation=eager" in overrides
    assert f"--model.ops_implementation.moe_implementation={expected_moe}" in overrides
    assert "--model.ops_implementation.rotary_pos_emb_implementation=eager" in overrides
    if is_npu:
        assert "--model.ops_implementation.rms_norm_implementation=eager" in overrides
        assert "--model.ops_implementation.swiglu_mlp_implementation=eager" in overrides
    else:
        assert not any("rms_norm_implementation" in override for override in overrides)
        assert not any("swiglu_mlp_implementation" in override for override in overrides)


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
