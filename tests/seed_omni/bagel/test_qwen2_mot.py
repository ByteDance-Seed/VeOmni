"""BAGEL Qwen2-MoT attention dispatch and QKV checkpoint schema."""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_NAME

from tests.seed_omni.bagel.helpers import config_cls, tiny_bagel_qwen2_cfg
from veomni.models.module_utils import load_model_weights
from veomni.models.seed_omni.modules.bagel.qwen2_mot import accelerated
from veomni.models.seed_omni.modules.bagel.qwen2_mot.checkpoint_conversion import (
    BagelQwen2MoTCheckpointTensorConverter,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.masking import (
    build_mot_attention_metadata,
    build_mot_block_mask,
)


def _flex_config() -> BagelQwen2MoTConfig:
    return BagelQwen2MoTConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attn_implementation="veomni_flex_attention_with_sp",
    )


def test_accelerated_training_attention_rejects_non_flex_backend() -> None:
    config = config_cls("bagel_qwen2_mot")(**tiny_bagel_qwen2_cfg(), attn_implementation="sdpa")
    attention = accelerated.BagelQwen2MoTAttentionAccelerated(config, layer_idx=0)
    sequence_length = 2
    metadata = build_mot_attention_metadata([[2]], [["causal"]], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="requires packed fused attention"):
        attention.forward_packed_train(
            packed_sequence=torch.randn(sequence_length, config.hidden_size),
            attention_mask=build_mot_block_mask(metadata),
            packed_position_cos=torch.ones(sequence_length, attention.head_dim),
            packed_position_sin=torch.zeros(sequence_length, attention.head_dim),
            packed_und_token_indexes=torch.arange(sequence_length),
            packed_gen_token_indexes=torch.empty(0, dtype=torch.long),
        )


def test_native_gqa_rejects_invalid_global_head_ratio_at_config_init() -> None:
    with pytest.raises(ValueError, match="query heads must be divisible"):
        config_cls("bagel_qwen2_mot")(
            **{
                **tiny_bagel_qwen2_cfg(),
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
            }
        )


def test_checkpoint_tensor_converter_combines_legacy_qkv_groups() -> None:
    converter = BagelQwen2MoTCheckpointTensorConverter()
    prefix = "model.layers.3.self_attn."
    tensors = {
        f"{prefix}q_proj.weight": torch.full((64, 64), 1.0),
        f"{prefix}k_proj.weight": torch.full((16, 64), 2.0),
        f"{prefix}v_proj.weight": torch.full((16, 64), 3.0),
        f"{prefix}q_proj_moe_gen.bias": torch.full((64,), 4.0),
        f"{prefix}k_proj_moe_gen.bias": torch.full((16,), 5.0),
        f"{prefix}v_proj_moe_gen.bias": torch.full((16,), 6.0),
    }

    converted = []
    for name in reversed(tensors):
        result = converter.convert(name, tensors[name])
        if result is not None:
            converted.append(result)
    assert converter.finalize() == []
    assert {result.name for result in converted} == {
        f"{prefix}qkv_proj_und.weight",
        f"{prefix}qkv_proj_gen.bias",
    }

    by_name = {result.name: result.tensor for result in converted}
    und_weight = by_name[f"{prefix}qkv_proj_und.weight"]
    gen_bias = by_name[f"{prefix}qkv_proj_gen.bias"]
    torch.testing.assert_close(und_weight[:64], tensors[f"{prefix}q_proj.weight"])
    torch.testing.assert_close(und_weight[64:80], tensors[f"{prefix}k_proj.weight"])
    torch.testing.assert_close(und_weight[80:], tensors[f"{prefix}v_proj.weight"])
    torch.testing.assert_close(gen_bias[:64], tensors[f"{prefix}q_proj_moe_gen.bias"])
    torch.testing.assert_close(gen_bias[64:80], tensors[f"{prefix}k_proj_moe_gen.bias"])
    torch.testing.assert_close(gen_bias[80:], tensors[f"{prefix}v_proj_moe_gen.bias"])


def test_accelerated_qkv_preserves_hf_checkpoint_schema_and_loaders(tmp_path) -> None:
    torch.manual_seed(3014)
    reference = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    internal_parameter_names = set(dict(reference.named_parameters()))
    assert "model.layers.0.self_attn.qkv_proj_und.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.q_proj.weight" not in internal_parameter_names
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" not in internal_parameter_names

    checkpoint_state = reference.state_dict()
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_state
    assert "model.layers.0.self_attn.k_proj.weight" in checkpoint_state
    assert "model.layers.0.self_attn.v_proj.weight" in checkpoint_state
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in checkpoint_state
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_state
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" not in checkpoint_state

    state_dict_loaded = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    state_dict_loaded.load_state_dict(checkpoint_state)
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        actual = getattr(state_dict_loaded.model.layers[0].self_attn, name)
        torch.testing.assert_close(actual.weight, expected.weight)
        torch.testing.assert_close(actual.bias, expected.bias)

    reference.save_pretrained(tmp_path)
    with safe_open(tmp_path / SAFE_WEIGHTS_NAME, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" not in checkpoint_keys

    hf_loaded = accelerated.BagelQwen2MoTAccelerated.from_pretrained(tmp_path)
    veomni_loaded = accelerated.BagelQwen2MoTAccelerated(_flex_config())
    load_model_weights(veomni_loaded, str(tmp_path), init_device="cpu")
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).bias, expected.bias)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).bias, expected.bias)
