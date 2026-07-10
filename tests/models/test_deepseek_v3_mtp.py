import pytest
import torch

from veomni.models.transformers.deepseek_v3.checkpoint_tensor_converter import (
    DeepseekV3CheckpointTensorConverter,
    convert_deepseek_v3_fqn_to_index_mapping,
)


def test_mtp_shared_checkpoint_aliases_are_redirected_to_canonical_parameters():
    converter = DeepseekV3CheckpointTensorConverter(num_experts=2)
    aliases = {
        "model.layers.61.embed_tokens.weight": 4,
        "model.layers.61.shared_head.norm.weight": 4,
        "model.layers.61.shared_head.head.weight": 5,
    }

    expected_names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for name, expected_name in zip(aliases, expected_names):
        assert converter.can_handle(name)
        converted = converter.convert(name, torch.empty(1))
        assert converted is not None
        assert converted.name == expected_name

    assert convert_deepseek_v3_fqn_to_index_mapping(aliases) == {
        "model.embed_tokens.weight": 4,
        "model.norm.weight": 4,
        "lm_head.weight": 5,
    }


@pytest.mark.parametrize(
    ("sp_enabled", "labels"),
    [
        (False, [10, 11, 12, -100, 21]),
        (True, [11, 12, -100, 21, -100]),
    ],
)
def test_mtp_shift_respects_packed_boundaries(monkeypatch, sp_enabled, labels):
    from veomni.models.transformers.deepseek_v3.generated import patched_modeling_deepseek_v3_gpu as modeling

    group = object() if sp_enabled else None
    monkeypatch.setattr(modeling, "get_unified_sequence_parallel_group", lambda: group)
    monkeypatch.setattr(modeling, "gather_outputs", lambda tensor, gather_dim: tensor)
    monkeypatch.setattr(modeling, "slice_input_tensor", lambda tensor, dim, padding: tensor)

    input_ids = torch.tensor([[10, 11, 12, 20, 21]])
    labels = torch.tensor([labels])
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    future_input_ids, mtp_labels = modeling._shift_mtp_inputs(input_ids, labels, position_ids, depth=1)

    torch.testing.assert_close(future_input_ids, torch.tensor([[11, 12, 20, 21, 0]]))
    torch.testing.assert_close(mtp_labels, torch.tensor([[12, -100, -100, -100, -100]]))


def test_mtp_module_uses_official_checkpoint_keys():
    from transformers import DeepseekV3Config

    from veomni.models.transformers.deepseek_v3.generated import patched_modeling_deepseek_v3_gpu as modeling

    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=None,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        num_nextn_predict_layers=1,
    )
    model = modeling.DeepseekV3ForCausalLM(config)
    state_keys = set(model.state_dict())

    expected_prefix = "model.layers.1."
    assert expected_prefix + "enorm.weight" in state_keys
    assert expected_prefix + "hnorm.weight" in state_keys
    assert expected_prefix + "eh_proj.weight" in state_keys
    assert expected_prefix + "embed_tokens.weight" not in state_keys
    assert expected_prefix + "shared_head.norm.weight" not in state_keys
    assert expected_prefix + "shared_head.head.weight" not in state_keys
    target_classes = set(model._no_split_modules)
    fsdp_block_names = {name for name, module in model.named_modules() if module.__class__.__name__ in target_classes}
    assert "model.layers.1" in fsdp_block_names


def test_mtp_loss_backpropagates_to_predictor(monkeypatch):
    from transformers import DeepseekV3Config

    from veomni.models.transformers.deepseek_v3.generated import patched_modeling_deepseek_v3_gpu as modeling
    from veomni.ops import apply_ops_config

    from ..tools.training_utils import make_eager_ops_config

    apply_ops_config(make_eager_ops_config())
    parallel_state = type("ParallelState", (), {"sp_enabled": False})()
    monkeypatch.setattr("veomni.ops.kernels.cross_entropy.get_parallel_state", lambda: parallel_state)

    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=None,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        num_nextn_predict_layers=1,
        mtp_loss_weight=0.1,
    )
    config._attn_implementation = "eager"
    model = modeling.DeepseekV3ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)

    outputs = model(input_ids=input_ids, position_ids=position_ids, labels=input_ids, use_cache=False)
    outputs.loss.backward()

    predictor = model.model.layers[config.num_hidden_layers]
    assert predictor.eh_proj.weight.grad is not None
    assert torch.count_nonzero(predictor.eh_proj.weight.grad) > 0
