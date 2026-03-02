import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from veomni.models.module_utils import load_model_weights
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import (
    Qwen3MoeV5CheckpointTensorConverter,
    create_qwen3_moe_checkpoint_tensor_converter,
)
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


def _build_qwen3_moe_config(num_experts: int, intermediate_size: int, hidden_size: int) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_hidden_layers=1,
        tie_word_embeddings=False,
    )


def _make_qwen3_expert_tensors(num_experts: int, intermediate_size: int, hidden_size: int):
    gate_by_expert = {}
    up_by_expert = {}
    down_by_expert = {}
    for expert_idx in range(num_experts):
        gate_by_expert[expert_idx] = torch.full(
            (intermediate_size, hidden_size), 10.0 + expert_idx, dtype=torch.float32
        )
        up_by_expert[expert_idx] = torch.full((intermediate_size, hidden_size), 20.0 + expert_idx, dtype=torch.float32)
        down_by_expert[expert_idx] = torch.full(
            (hidden_size, intermediate_size), 30.0 + expert_idx, dtype=torch.float32
        )
    return gate_by_expert, up_by_expert, down_by_expert


def _save_fake_sharded_safetensors(output_dir: Path, shard_dicts: list[dict[str, torch.Tensor]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    total_size = 0
    for shard_idx, shard_dict in enumerate(shard_dicts, start=1):
        shard_name = f"model-{shard_idx:05d}-of-{len(shard_dicts):05d}.safetensors"
        for key, tensor in shard_dict.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        save_file(shard_dict, str(output_dir / shard_name))

    index_path = output_dir / SAFE_WEIGHTS_INDEX_NAME
    index_payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")


class _DummyExperts(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))


class _DummyMLP(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.experts = _DummyExperts(
            num_experts=num_experts, intermediate_size=intermediate_size, hidden_size=hidden_size
        )


class _DummyLayer(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.mlp = _DummyMLP(num_experts=num_experts, intermediate_size=intermediate_size, hidden_size=hidden_size)


class _DummyBackbone(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DummyLayer(num_experts=num_experts, intermediate_size=intermediate_size, hidden_size=hidden_size)]
        )


class _DummyQwen3MoeV5Model(nn.Module):
    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        config = _build_qwen3_moe_config(
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        self.model = _DummyBackbone(
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        )
        self.config = config

    def _create_checkpoint_tensor_converter(self):
        return Qwen3MoeV5CheckpointTensorConverter(config=self.config)


def test_qwen3_moe_converter_accumulates_and_merges_with_out_of_order_inputs():
    num_experts, intermediate_size, hidden_size = 2, 3, 4
    gate_by_expert, up_by_expert, down_by_expert = _make_qwen3_expert_tensors(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )

    config = _build_qwen3_moe_config(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    converter = Qwen3MoeV5CheckpointTensorConverter(config=config)
    input_items = [
        ("model.layers.0.mlp.experts.1.down_proj.weight", down_by_expert[1]),
        ("model.layers.0.mlp.experts.1.gate_proj.weight", gate_by_expert[1]),
        ("model.layers.0.mlp.experts.0.up_proj.weight", up_by_expert[0]),
        ("model.layers.0.mlp.experts.0.down_proj.weight", down_by_expert[0]),
        ("model.layers.0.mlp.experts.1.up_proj.weight", up_by_expert[1]),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate_by_expert[0]),
    ]

    converted = []
    for key, tensor in input_items:
        out = converter.convert(key, tensor)
        if out is not None:
            converted.append((out.name, out.tensor))

    assert len(converted) == 2
    outputs = dict(converted)
    assert "model.layers.0.mlp.experts.gate_up_proj" in outputs
    assert "model.layers.0.mlp.experts.down_proj" in outputs

    expected_gate = torch.stack([gate_by_expert[0], gate_by_expert[1]], dim=0)
    expected_up = torch.stack([up_by_expert[0], up_by_expert[1]], dim=0)
    expected_gate_up = torch.cat([expected_gate, expected_up], dim=1)
    expected_down = torch.stack([down_by_expert[0], down_by_expert[1]], dim=0)

    torch.testing.assert_close(outputs["model.layers.0.mlp.experts.gate_up_proj"], expected_gate_up)
    torch.testing.assert_close(outputs["model.layers.0.mlp.experts.down_proj"], expected_down)


def test_load_model_weights_uses_registered_converter_for_qwen3_moe(tmp_path: Path):
    num_experts, intermediate_size, hidden_size = 2, 3, 4
    gate_by_expert, up_by_expert, down_by_expert = _make_qwen3_expert_tensors(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )

    shard_1 = {
        "model.layers.0.mlp.experts.1.gate_proj.weight": gate_by_expert[1],
        "model.layers.0.mlp.experts.0.down_proj.weight": down_by_expert[0],
        "model.layers.0.mlp.experts.0.up_proj.weight": up_by_expert[0],
    }
    shard_2 = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": gate_by_expert[0],
        "model.layers.0.mlp.experts.1.down_proj.weight": down_by_expert[1],
        "model.layers.0.mlp.experts.1.up_proj.weight": up_by_expert[1],
    }
    _save_fake_sharded_safetensors(tmp_path, [shard_1, shard_2])

    model = _DummyQwen3MoeV5Model(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    load_model_weights(model, str(tmp_path), init_device="cpu")

    expected_gate = torch.stack([gate_by_expert[0], gate_by_expert[1]], dim=0)
    expected_up = torch.stack([up_by_expert[0], up_by_expert[1]], dim=0)
    expected_gate_up = torch.cat([expected_gate, expected_up], dim=1)
    expected_down = torch.stack([down_by_expert[0], down_by_expert[1]], dim=0)

    loaded_gate_up = model.model.layers[0].mlp.experts.gate_up_proj.detach().cpu()
    loaded_down = model.model.layers[0].mlp.experts.down_proj.detach().cpu()
    torch.testing.assert_close(loaded_gate_up, expected_gate_up)
    torch.testing.assert_close(loaded_down, expected_down)


def test_qwen3_moe_converter_factory_is_version_gated():
    model = _DummyQwen3MoeV5Model(num_experts=2, intermediate_size=3, hidden_size=4)
    converter = create_qwen3_moe_checkpoint_tensor_converter(model)
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        assert isinstance(converter, Qwen3MoeV5CheckpointTensorConverter)
    else:
        assert converter is None
