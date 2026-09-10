# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import save as dcp_save

from veomni.models.transformers.deepseek_v4 import checkpoint_export as merge
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    DeepseekV4CheckpointTensorConverter,
    _dequantize_scaled_weight,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/merge_dcp_to_hf.py"
_spec = importlib.util.spec_from_file_location("dsv4_merge", _SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.mark.parametrize(
    "name,expert_dtype,scale_dtype",
    [
        ("v4-flash", torch.int8, torch.float8_e8m0fnu),
        ("v4-flash-base", torch.float8_e4m3fn, torch.float32),
    ],
)
def test_release_schema(name, expert_dtype, scale_dtype):
    target = merge.OutputFormat(name)
    assert len(target.schema) == 67612
    assert len(target.keys_by_shard) == 45
    assert not any(k.startswith("mtp.") for k in target.schema)
    assert target.schema["layers.0.ffn.experts.255.w1.weight"][0] == expert_dtype
    assert target.schema["layers.0.attn.wq_a.scale"][0] == scale_dtype
    assert target.schema["layers.0.ffn.gate.tid2eid"][0] == torch.int64
    assert target.schema["layers.0.hc_attn_fn"][0] == torch.float32
    assert "layers.2.attn.indexer.wq_b.weight" in target.schema
    assert "layers.3.attn.indexer.wq_b.weight" not in target.schema


def tiny_target():
    target = merge.OutputFormat.__new__(merge.OutputFormat)
    target.name = "v4-flash-base"
    target.config = {"n_routed_experts": 4, "expert_dtype": "fp8"}
    target.schema = {
        "layers.0.ffn.gate.tid2eid": (torch.int64, (4, 2)),
        "layers.0.attn_norm.weight": (torch.bfloat16, (4,)),
    }
    target.weight_map = dict.fromkeys(target.schema, "layer.safetensors")
    target.keys_by_shard = {"layer.safetensors": list(target.schema)}
    target.shard_metadata = {}
    return target


@pytest.mark.parametrize("workers", [1, 2])
def test_dcp_to_native_and_resume(tmp_path, monkeypatch, workers):
    target = tiny_target()
    source = {
        "model.model.layers.0.mlp.gate.tid2eid": torch.arange(8, dtype=torch.int64).reshape(4, 2),
        "model.model.layers.0.input_layernorm.weight": torch.arange(4, dtype=torch.float32),
        "model.model.mtp.0.ignored": torch.ones(1),
        "optimizer.ignored": torch.ones(1),
    }
    checkpoint = tmp_path / "dcp"
    output = tmp_path / "out"
    dcp_save(source, checkpoint_id=checkpoint)
    monkeypatch.setattr(merge, "OutputFormat", lambda name: target)
    commands = []

    def launch_worker(command):
        commands.append(command)
        assert Path(command[1]) == _SCRIPT
        with monkeypatch.context() as worker_patch:
            worker_patch.setattr(sys, "argv", command[1:])
            cli.main()
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(merge.subprocess, "Popen", launch_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--load-dir",
            str(checkpoint),
            "--save-dir",
            str(output),
            "--format",
            "v4-flash-base",
            "--device",
            "cpu",
            "--skip-assets",
            "--workers",
            str(workers),
        ],
    )
    cli.main()
    path = output / "layer.safetensors"
    result = load_file(path)
    torch.testing.assert_close(result["layers.0.ffn.gate.tid2eid"], source["model.model.layers.0.mlp.gate.tid2eid"])
    assert result["layers.0.attn_norm.weight"].dtype == torch.bfloat16
    index = json.loads((output / merge.INDEX_NAME).read_text())
    assert index["metadata"]["total_size"] == 72
    assert index["weight_map"] == target.weight_map
    assert len(commands) == (workers if workers > 1 else 0)
    before = path.stat().st_mtime_ns
    cli.main()
    assert path.stat().st_mtime_ns == before
    with path.open("r+b") as fh:
        fh.truncate(path.stat().st_size - 1)
    assert not merge.shard_is_complete(output, path.name, target)
    cli.main()
    assert merge.shard_is_complete(output, path.name, target)
    result["layers.0.attn_norm.weight"] = result["layers.0.attn_norm.weight"].float()
    save_file(result, path)
    assert not merge.shard_is_complete(output, path.name, target)


def test_incomplete_source_rejected(tmp_path):
    source = {"model.model.layers.0.input_layernorm.weight": torch.ones(4)}
    dcp_save(source, checkpoint_id=tmp_path)
    reader = merge.CachedFileSystemReader(tmp_path)
    metadata = reader.read_metadata()
    converter = DeepseekV4CheckpointTensorConverter(4)
    layers, _ = merge.group_source_keys(metadata, converter)
    with pytest.raises(ValueError, match="incomplete source"):
        merge.validate_source_group(metadata, layers[0], converter, tiny_target(), "layer.safetensors")


def test_expert_order_and_gate_up_split():
    # EP rank is outermost in expert-id order; DCP places the FSDP rank first.
    expected = torch.arange(16).reshape(8, 2, 1)
    stored = expected.reshape(2, 4, 1, 2, 1).transpose(0, 1).reshape(8, 2, 1)
    result = dict(merge.split_source_tensor("model.layers.0.mlp.experts.gate_up_proj", stored, 2, 4))
    for expert in range(8):
        assert result[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"].item() == 2 * expert
        assert result[f"model.layers.0.mlp.experts.{expert}.up_proj.weight"].item() == 2 * expert + 1


def test_reject_hidden_dim_sharding():
    meta = SimpleNamespace(
        state_dict_metadata={
            "experts": SimpleNamespace(
                size=(8, 4, 4), chunks=[SimpleNamespace(sizes=(4, 2, 4)), SimpleNamespace(sizes=(4, 2, 4))]
            )
        }
    )
    with pytest.raises(ValueError, match="sharded beyond the expert"):
        merge.expert_dim_shard_count(meta, "experts")


def test_assets_drop_mtp_and_replace_template(tmp_path):
    src = tmp_path / "tokenizer_config.json"
    src.write_text(json.dumps({"chat_template": "old", "tokenizer_class": "TokenizersBackend"}))
    output = tmp_path / "out"
    output.mkdir()
    (output / "generation_config.json").write_text("{}")
    template = merge.resolve_custom_template("{{ messages[0]['content'] }}")
    merge.write_assets(tiny_target(), output, {merge.TOKENIZER_CONFIG_NAME: str(src)}, template)
    assert json.loads((output / "config.json").read_text())["num_nextn_predict_layers"] == 0
    assert "chat_template" not in json.loads((output / merge.TOKENIZER_CONFIG_NAME).read_text())
    assert (output / merge.CHAT_TEMPLATE_NAME).read_text().strip() == template
    assert not (output / "generation_config.json").exists()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="DeepSeek quantization needs CUDA")
@pytest.mark.parametrize("format_name", ["v4-flash", "v4-flash-base"])
@pytest.mark.parametrize(
    "name", ["model.layers.0.mlp.experts.0.gate_proj.weight", "model.layers.0.self_attn.q_a_proj.weight"]
)
def test_gpu_quantization_round_trip(format_name, name):
    pytest.importorskip("tilelang")
    target = merge.OutputFormat(format_name)
    converter = DeepseekV4CheckpointTensorConverter(256)
    native = converter.export_name(name)
    torch.manual_seed(7)
    source = torch.randn(128, 128, device=get_device_type(), dtype=torch.bfloat16)
    original = source.clone()
    exported = dict(converter.export_tensor(name, source, target.weight_map, target.expert_dtype()))
    weight = exported[native]
    scale = exported[native.removesuffix(".weight") + ".scale"]
    assert weight.dtype == target.schema[native][0]
    assert scale.dtype == target.schema[native.removesuffix(".weight") + ".scale"][0]
    restored = _dequantize_scaled_weight(weight, scale, packed_fp4=weight.dtype == torch.int8)
    error = (restored - source.float()).norm() / source.float().norm()
    assert error < (0.2 if weight.dtype == torch.int8 else 0.05)
    torch.testing.assert_close(source, original)


def test_training_template_extensions():
    template = "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% break %}{% endfor %}"
    assert merge.resolve_custom_template(template) == template


@pytest.mark.parametrize("format_args", [[], ["--format", "hf"]])
def test_legacy_hf_cli_defaults(tmp_path, monkeypatch, format_args):
    checkpoint = tmp_path / "dcp"
    tensors = {
        "model.model.norm.weight": torch.arange(4, dtype=torch.float32),
        "model.model.table": torch.arange(6, dtype=torch.int64).reshape(3, 2),
        "optimizer.momentum": torch.ones(4),
    }
    dcp_save(tensors, checkpoint_id=checkpoint)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--load-dir", str(checkpoint), "--shard-size", "10", *format_args])
    cli.main()
    output = checkpoint / "hf_ckpt"
    index = json.loads((output / merge.INDEX_NAME).read_text())
    assert index["metadata"]["total_size"] == 56
    assert set(index["weight_map"]) == {"model.norm.weight", "model.table"}
    assert len(set(index["weight_map"].values())) == 2
    actual = {}
    for filename in set(index["weight_map"].values()):
        actual.update(load_file(output / filename))
    torch.testing.assert_close(actual["model.norm.weight"], tensors["model.model.norm.weight"].bfloat16())
    torch.testing.assert_close(actual["model.table"], tensors["model.model.table"])


@pytest.mark.parametrize("mode", ["auto", "lora"])
def test_legacy_lora_cli(tmp_path, monkeypatch, mode):
    checkpoint = tmp_path / "dcp"
    key = "base_model.model.layer.lora_A.default.weight"
    dcp_save(
        {"model." + key: torch.ones(2, 4), "model.model.layer.weight": torch.zeros(4, 4)}, checkpoint_id=checkpoint
    )
    adapter = tmp_path / "adapter.json"
    adapter.write_text('{"peft_type": "LORA", "r": 2}')
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT), "--load-dir", str(checkpoint), "--mode", mode, "--adapter-config-path", str(adapter)],
    )
    cli.main()
    output = checkpoint / "hf_ckpt"
    tensors = load_file(output / "adapter_model.safetensors")
    assert list(tensors) == [key]
    assert tensors[key].dtype == torch.bfloat16
    assert (output / "adapter_config.json").read_text() == adapter.read_text()
    assert not (output / merge.INDEX_NAME).exists()


def test_legacy_single_shard_pytorch_export(tmp_path):
    checkpoint = tmp_path / "dcp"
    dcp_save({"model.model.norm.weight": torch.ones(4)}, checkpoint_id=checkpoint)
    cli.save_model_weights(tmp_path / "out", checkpoint, safe_serialization=False, save_dtype=None)
    output = torch.load(tmp_path / "out/pytorch_model.bin", weights_only=True)
    assert output["model.norm.weight"].dtype == torch.float32
    assert not (tmp_path / "out/pytorch_model.bin.index.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--format", "hf", "--workers", "2"],
        ["--format", "v4-flash", "--mode", "lora"],
        ["--format", "v4-flash-base", "--shard-size", "123"],
    ],
)
def test_incompatible_cli_options_rejected(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--load-dir", "/unused", *options])
    with pytest.raises(SystemExit) as err:
        cli.main()
    assert err.value.code == 2
