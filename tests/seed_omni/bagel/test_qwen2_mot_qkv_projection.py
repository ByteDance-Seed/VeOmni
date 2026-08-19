"""QKV projection coverage for eager split and accelerated fused BAGEL MoT."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_NAME

from veomni.models.module_utils import load_model_weights
from veomni.models.seed_omni.modules.bagel.qwen2_mot.accelerated import (
    BagelQwen2MoTAccelerated,
    BagelQwen2MoTAttentionAccelerated,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.checkpoint_conversion import (
    BagelQwen2MoTCheckpointTensorConverter,
)
from veomni.models.seed_omni.modules.bagel.qwen2_mot.configuration import BagelQwen2MoTConfig
from veomni.models.seed_omni.modules.bagel.qwen2_mot.modeling import (
    BagelQwen2MoT,
    BagelQwen2MoTAttention,
)


def _eager_config() -> BagelQwen2MoTConfig:
    return BagelQwen2MoTConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attn_implementation="sdpa",
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


@pytest.mark.parametrize("is_gen", [False, True])
def test_eager_qkv_uses_split_linears(is_gen: bool) -> None:
    torch.manual_seed(3013)
    attention = BagelQwen2MoTAttention(_eager_config(), layer_idx=0)
    hidden = torch.randn(7, attention.hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    query_proj, key_proj, value_proj = (
        (attention.q_proj_moe_gen, attention.k_proj_moe_gen, attention.v_proj_moe_gen)
        if is_gen
        else (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    reference = torch.cat(
        (
            query_proj(reference_hidden),
            key_proj(reference_hidden),
            value_proj(reference_hidden),
        ),
        dim=-1,
    )
    eager = attention._project_qkv(hidden, is_gen=is_gen)
    eager_q, eager_k, eager_v = attention._project_qkv(hidden, is_gen=is_gen, split=True)
    assert torch.equal(eager, reference)
    assert torch.equal(torch.cat((eager_q, eager_k, eager_v), dim=-1), reference)
    torch.autograd.backward(eager, torch.ones_like(eager))
    torch.autograd.backward(reference, torch.ones_like(reference))
    assert hidden.grad is not None
    assert reference_hidden.grad is not None
    assert torch.equal(hidden.grad, reference_hidden.grad)


@pytest.mark.parametrize("projection_name", ["qkv_proj_und", "qkv_proj_gen"])
def test_accelerated_fused_qkv_matches_separate_projection_forward_backward(projection_name: str) -> None:
    torch.manual_seed(3011)
    attention = BagelQwen2MoTAttentionAccelerated(_flex_config(), layer_idx=0)
    projection = getattr(attention, projection_name)
    reference_weights = [
        tensor.detach().clone().requires_grad_()
        for tensor in projection.weight.split(attention.qkv_split_sizes, dim=0)
    ]
    reference_biases = [
        tensor.detach().clone().requires_grad_() for tensor in projection.bias.split(attention.qkv_split_sizes, dim=0)
    ]
    combined_input = torch.randn(7, attention.hidden_size, requires_grad=True)
    reference_input = combined_input.detach().clone().requires_grad_()

    combined_outputs = projection(combined_input).split(attention.qkv_split_sizes, dim=-1)
    reference_outputs = tuple(
        F.linear(reference_input, weight, bias)
        for weight, bias in zip(reference_weights, reference_biases, strict=True)
    )
    grad_outputs = tuple(torch.randn_like(output) for output in combined_outputs)
    torch.autograd.backward(combined_outputs, grad_outputs)
    torch.autograd.backward(reference_outputs, grad_outputs)

    for combined_output, reference_output in zip(combined_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(combined_output, reference_output)
    assert combined_input.grad is not None
    assert reference_input.grad is not None
    torch.testing.assert_close(combined_input.grad, reference_input.grad)
    assert projection.weight.grad is not None
    assert projection.bias.grad is not None
    for combined_grad, reference_weight in zip(
        projection.weight.grad.split(attention.qkv_split_sizes, dim=0),
        reference_weights,
        strict=True,
    ):
        assert reference_weight.grad is not None
        torch.testing.assert_close(combined_grad, reference_weight.grad)
    for combined_grad, reference_bias in zip(
        projection.bias.grad.split(attention.qkv_split_sizes, dim=0),
        reference_biases,
        strict=True,
    ):
        assert reference_bias.grad is not None
        torch.testing.assert_close(combined_grad, reference_bias.grad)


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


def test_eager_qkv_uses_native_split_checkpoint_schema(tmp_path) -> None:
    torch.manual_seed(3012)
    reference = BagelQwen2MoT(_eager_config())
    internal_parameter_names = set(dict(reference.named_parameters()))
    assert "model.layers.0.self_attn.q_proj.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.k_proj.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.v_proj.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.q_proj_moe_gen.weight" in internal_parameter_names
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in internal_parameter_names
    assert "model.layers.0.self_attn.qkv_proj_gen.weight" not in internal_parameter_names

    checkpoint_state = reference.state_dict()
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_state
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_state

    loaded = BagelQwen2MoT(_eager_config())
    loaded.load_state_dict(checkpoint_state)
    torch.testing.assert_close(
        loaded.model.layers[0].self_attn.q_proj.weight,
        reference.model.layers[0].self_attn.q_proj.weight,
    )

    reference.save_pretrained(tmp_path)
    with safe_open(tmp_path / SAFE_WEIGHTS_NAME, framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
    assert "model.layers.0.self_attn.q_proj.weight" in checkpoint_keys
    assert "model.layers.0.self_attn.qkv_proj_und.weight" not in checkpoint_keys

    hf_loaded = BagelQwen2MoT.from_pretrained(tmp_path)
    veomni_loaded = BagelQwen2MoT(_eager_config())
    load_model_weights(veomni_loaded, str(tmp_path), init_device="cpu")
    torch.testing.assert_close(
        hf_loaded.model.layers[0].self_attn.q_proj.weight,
        reference.model.layers[0].self_attn.q_proj.weight,
    )
    torch.testing.assert_close(
        veomni_loaded.model.layers[0].self_attn.q_proj.weight,
        reference.model.layers[0].self_attn.q_proj.weight,
    )


def test_accelerated_qkv_preserves_hf_checkpoint_schema_and_loaders(tmp_path) -> None:
    torch.manual_seed(3014)
    reference = BagelQwen2MoTAccelerated(_flex_config())
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

    state_dict_loaded = BagelQwen2MoTAccelerated(_flex_config())
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

    hf_loaded = BagelQwen2MoTAccelerated.from_pretrained(tmp_path)
    veomni_loaded = BagelQwen2MoTAccelerated(_flex_config())
    load_model_weights(veomni_loaded, str(tmp_path), init_device="cpu")
    for name in ("qkv_proj_und", "qkv_proj_gen"):
        expected = getattr(reference.model.layers[0].self_attn, name)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(hf_loaded.model.layers[0].self_attn, name).bias, expected.bias)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).weight, expected.weight)
        torch.testing.assert_close(getattr(veomni_loaded.model.layers[0].self_attn, name).bias, expected.bias)
