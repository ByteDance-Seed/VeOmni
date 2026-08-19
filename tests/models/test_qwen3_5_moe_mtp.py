from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from veomni.distributed.utils import check_fqn_match
from veomni.models.checkpoint_tensor_loading import maybe_convert_checkpoint_tensor
from veomni.models.transformers.qwen3_5_moe import register_qwen3_5_moe_modeling
from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter
from veomni.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_npu as modeling
else:
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling


TOY_CONFIG = Path(__file__).parents[1] / "toy_config" / "qwen3_5_moe_toy"
NUM_EXPERTS = 4
HIDDEN_DIM = 8
INTERMEDIATE_DIM = 6


def _make_mtp_expert_key(expert: int, proj: str) -> str:
    return f"mtp.layers.0.mlp.experts.{expert}.{proj}.weight"


def _make_expert_tensor(proj: str, expert_id: int) -> torch.Tensor:
    shape = (HIDDEN_DIM, INTERMEDIATE_DIM) if proj == "down_proj" else (INTERMEDIATE_DIM, HIDDEN_DIM)
    offset = {"gate_proj": 0.0, "up_proj": 0.1, "down_proj": 0.2}[proj]
    return torch.full(shape, expert_id + offset)


def test_qwen3_5_moe_mtp_structure():
    text_config = AutoConfig.from_pretrained(TOY_CONFIG).text_config
    mtp = modeling.Qwen3_5MoeMTP(text_config)

    assert len(mtp.layers) == text_config.mtp_num_hidden_layers
    assert isinstance(mtp.layers[0], modeling.Qwen3_5MoeDecoderLayer)
    parameter_names = {name for name, _ in mtp.named_parameters()}
    assert "fc.weight" in parameter_names
    assert "layers.0.mlp.gate.weight" in parameter_names


def test_qwen3_5_moe_mtp_outputs_keep_auxiliary_fields():
    context = {"position_ids": torch.arange(4)}
    router_logits = (torch.zeros(4, 2),)
    model_output = modeling.Qwen3_5MoeMTPContextOutput(
        last_hidden_state=torch.zeros(1, 4, 8),
        router_logits=router_logits,
        mtp_context=context,
    )
    loss_dict = {"foundation_loss": torch.tensor(1.0), "mtp_loss": torch.tensor(0.5)}
    causal_output = modeling.Qwen3_5MoeCausalLMOutputWithLogProbs(loss_dict=loss_dict)

    assert model_output.mtp_context is context
    assert model_output.router_logits is router_logits
    assert causal_output.loss_dict == loss_dict


def test_qwen3_5_moe_parallel_plan_covers_mtp_experts():
    plan = get_parallel_plan()
    patterns = plan.extra_parallel_plan["ep"]
    no_shard_patterns = plan.extra_parallel_fsdp_no_shard_module["ep"]

    for prefix in ("model.language_model.layers.0", "mtp.layers.0"):
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts.gate_up_proj") for pattern in patterns)
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts.down_proj") for pattern in patterns)
        assert any(check_fqn_match(pattern, f"{prefix}.mlp.experts") for pattern in no_shard_patterns)


def test_qwen3_5_moe_registry_attaches_checkpoint_converter():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")

    assert callable(model_cls._create_checkpoint_tensor_converter)
    assert callable(model_cls._convert_fqn_to_index_mapping)


def test_qwen3_5_moe_reuses_checkpoint_converter_for_mtp_experts():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=object()))
    dispatched = {}

    assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
    assert converter.can_handle(_make_mtp_expert_key(0, "gate_proj"))

    trunk_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    trunk = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM)
    result = maybe_convert_checkpoint_tensor(trunk_key, trunk, converter)
    assert result is not None and result.name == trunk_key and result.tensor is trunk

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            result = maybe_convert_checkpoint_tensor(
                _make_mtp_expert_key(expert_id, proj),
                _make_expert_tensor(proj, expert_id),
                converter,
            )
            if result is not None:
                dispatched[result.name] = result.tensor

    assert converter.finalize() == []
    assert dispatched["mtp.layers.0.mlp.experts.gate_up_proj"].shape == (
        NUM_EXPERTS,
        2 * INTERMEDIATE_DIM,
        HIDDEN_DIM,
    )
    assert dispatched["mtp.layers.0.mlp.experts.down_proj"].shape == (
        NUM_EXPERTS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
    )


def test_qwen3_5_moe_checkpoint_converter_rejects_incomplete_mtp_experts():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=object()))
    converter.convert(_make_mtp_expert_key(0, "down_proj"), _make_expert_tensor("down_proj", 0))

    with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
        converter.finalize()


def test_qwen3_5_moe_checkpoint_converter_handles_trunk_without_mtp():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=NUM_EXPERTS))
    converter = model_cls._create_checkpoint_tensor_converter(SimpleNamespace(config=config, mtp=None))
    dispatched = {}

    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            key = f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj}.weight"
            result = maybe_convert_checkpoint_tensor(key, _make_expert_tensor(proj, expert_id), converter)
            if result is not None:
                dispatched[result.name] = result.tensor

    assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
    assert converter.finalize() == []
    assert set(dispatched) == {
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    }


def test_qwen3_5_moe_reuses_checkpoint_index_mapping():
    model_cls = register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")
    trunk_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    mapping = {trunk_key: 1}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert_id in range(NUM_EXPERTS):
            mapping[_make_mtp_expert_key(expert_id, proj)] = expert_id + 2

    converted = model_cls._convert_fqn_to_index_mapping(mapping)

    assert converted[trunk_key] == 1
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in converted
    assert "mtp.layers.0.mlp.experts.down_proj" in converted
    assert not any("experts.0." in key for key in converted)


def test_qwen3_5_moe_conditional_generation_builds_mtp(monkeypatch):
    config = AutoConfig.from_pretrained(TOY_CONFIG)
    config.text_config.mtp_loss_weight = 0.3
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    for slot_name in (
        "veomni_rms_norm",
        "veomni_moe_experts_forward",
        "veomni_causal_lm_loss",
        "veomni_load_balancing_loss",
        "veomni_rms_norm_gated",
        "veomni_causal_conv1d",
        "veomni_chunk_gated_delta_rule",
    ):
        getattr(modeling, slot_name).bind("eager")

    with torch.device("meta"):
        model = modeling.Qwen3_5MoeForConditionalGeneration(config)

    assert isinstance(model.mtp, modeling.Qwen3_5MoeMTP)
    assert model.model.language_model._veomni_mtp_enabled
    assert any(name.startswith("mtp.") for name, _ in model.named_parameters())
