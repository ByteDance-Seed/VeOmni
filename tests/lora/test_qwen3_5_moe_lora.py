from pathlib import Path

import yaml

from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel, resolve_fused_moe_lora_targets
from veomni.models import build_foundation_model
from veomni.models.transformers.qwen3_5_moe import (
    register_qwen3_5_moe_modeling,
    register_qwen3_5_moe_text_modeling,
)

from ..tools.training_utils import make_eager_ops_config


_CONFIG_PATH = Path("configs/text/qwen3_5_moe_lora.yaml")
_TOY_CONFIG_PATH = "tests/toy_config/qwen3_5_moe_toy/config.json"
_DENSE_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
}
_SEMANTIC_EXPERT_TARGETS = {"gate_proj", "up_proj", "down_proj"}
_CONDITIONAL_EXPERT_PATTERNS = [
    "model.language_model.layers.*.mlp.experts.gate_up_proj",
    "model.language_model.layers.*.mlp.experts.down_proj",
]
_CAUSAL_EXPERT_PATTERNS = [
    "model.layers.*.mlp.experts.gate_up_proj",
    "model.layers.*.mlp.experts.down_proj",
]


def _production_lora_config():
    """Load the LoRA section from the production Qwen3.5-MoE config."""
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))["model"]["lora_config"]


def test_qwen3_5_moe_registers_semantic_expert_target_mapping_for_both_wrappers():
    """Verify semantic expert targets resolve for both Qwen3.5-MoE wrappers."""
    lora_modules = [*_DENSE_TARGETS, *_SEMANTIC_EXPERT_TARGETS]

    conditional_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    conditional_modules, conditional_parameters = conditional_cls._convert_lora_targets_to_parameters(
        None, lora_modules, []
    )
    causal_cls = register_qwen3_5_moe_text_modeling("Qwen3_5MoeForCausalLM")
    causal_modules, causal_parameters = causal_cls._convert_lora_targets_to_parameters(None, lora_modules, [])

    assert set(conditional_modules) == _DENSE_TARGETS
    assert conditional_parameters == _CONDITIONAL_EXPERT_PATTERNS
    assert set(causal_modules) == _DENSE_TARGETS
    assert causal_parameters == _CAUSAL_EXPERT_PATTERNS


def test_qwen3_5_moe_production_config_injects_all_targets_and_freezes_base_model():
    """Verify the production config injects every target and freezes base weights."""
    model = build_foundation_model(
        config_path=_TOY_CONFIG_PATH,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
    )
    resolved = resolve_fused_moe_lora_targets(model, _production_lora_config())

    assert set(resolved["lora_modules"]) == _DENSE_TARGETS
    assert resolved["target_parameters"] == _CONDITIONAL_EXPERT_PATTERNS

    wrapped = VeOmniLoraModel(model, VeOmniLoraConfig.from_yaml(resolved))
    dense_fqns = wrapped.base_model.wrapped_dense
    moe_fqns = wrapped.base_model.wrapped_moe

    for target in _DENSE_TARGETS:
        assert any(fqn.endswith(f".{target}") for fqn in dense_fqns), target
    assert moe_fqns

    trainable_names = {name for name, param in wrapped.named_parameters() if param.requires_grad}
    assert trainable_names
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable_names)

    expert_trainable_names = {name for name in trainable_names if ".mlp.experts." in name}
    for target in _SEMANTIC_EXPERT_TARGETS:
        assert any(f".experts.{target}.lora_" in name for name in expert_trainable_names), target

    for name, param in wrapped.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            assert not param.requires_grad, name
