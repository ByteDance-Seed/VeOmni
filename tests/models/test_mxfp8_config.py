# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for recipe persistence, hardware preflight and projection selection."""

import copy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.training_utils import make_eager_ops_config
from veomni.arguments import MixedPrecisionConfig, MXFP8Config, VeOmniArguments
from veomni.arguments.parser import _instantiate_recursive, parse_args, save_args
from veomni.distributed import mxfp8, torch_parallelize


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _model():
    root = nn.Module()
    root.config = Qwen3Config(hidden_size=64, intermediate_size=128)
    root.config.ops_implementation = {"qat_implementation": "none"}
    root.model = nn.Module()
    block = nn.Module()
    block.mlp = nn.ModuleDict(
        {
            "gate_proj": nn.Linear(64, 128, bias=False),
            "up_proj": nn.Linear(64, 128, bias=False),
            "down_proj": nn.Linear(128, 64, bias=False),
        }
    )
    root.model.layers = nn.ModuleList([block])
    root.lm_head = nn.Linear(64, 64, bias=False)
    return root


def _state(**overrides):
    return SimpleNamespace(**{"dp_mode": "fsdp2", "dp_shard_size": 2, **overrides})


@pytest.mark.parametrize("gather", [False, True])
def test_nested_recipe_yaml_cli_and_saved_job_config(tmp_path, monkeypatch, gather):
    path = tmp_path / "precision.yaml"
    path.write_text(f"mxfp8:\n  enable: true\n  enable_fsdp_all_gather: {str(gather).lower()}\n")
    monkeypatch.setattr(sys, "argv", ["test", str(path)])
    config = parse_args(MixedPrecisionConfig)
    assert config.mxfp8 == MXFP8Config(enable=True, enable_fsdp_all_gather=gather)
    save_args(config, str(tmp_path / "saved"))
    monkeypatch.setattr(sys, "argv", ["test", str(tmp_path / "saved" / "veomni_cli.yaml")])
    assert parse_args(MixedPrecisionConfig) == config
    monkeypatch.setattr(
        sys, "argv", ["test", str(path), "--mxfp8.enable_fsdp_all_gather", "false", "--mxfp8.enable", "false"]
    )
    assert not parse_args(MixedPrecisionConfig).mxfp8.enable


@pytest.mark.parametrize(
    "values",
    [
        {"enable": "false"},
        {"enable_fsdp_all_gather": 1},
        {"target_modules": "*"},
        {"target_modules": []},
        {"target_modules": [""]},
        {"enable_fsdp_all_gather": True},
    ],
)
def test_malformed_recipe_rejected_by_real_parser(values):
    with pytest.raises(ValueError):
        _instantiate_recursive(MixedPrecisionConfig, {"mxfp8": values})


def test_unset_and_explicit_disable_survive_hf_config_round_trip(tmp_path):
    model = _model()
    saved = MXFP8Config(enable=True, enable_fsdp_all_gather=True)
    model.config.veomni_mxfp8 = asdict(saved)
    model.config.save_pretrained(tmp_path)
    model.config = Qwen3Config.from_pretrained(tmp_path)
    assert mxfp8.resolve_mxfp8_config(model, MixedPrecisionConfig().mxfp8) == saved
    explicit = MXFP8Config(enable=True, target_modules=["model.layers.0.mlp.up_proj"])
    assert mxfp8.resolve_mxfp8_config(model, explicit) == explicit
    mxfp8.prepare_mxfp8(model, MXFP8Config(enable=False))
    model.config.save_pretrained(tmp_path)
    model.config = Qwen3Config.from_pretrained(tmp_path)
    assert not mxfp8.resolve_mxfp8_config(model, None).enable


@pytest.mark.parametrize("config", [None, MXFP8Config()])
def test_default_and_disabled_builder_never_require_optional_torchao(monkeypatch, config):
    model = _model()
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: _state(fsdp_enabled=True, tp_enabled=False))
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda **kwargs: kwargs["model"])
    result = torch_parallelize.build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(mxfp8=config),
        enable_gradient_checkpointing=False,
    )
    assert result is model and type(model.model.layers[0].mlp.up_proj) is nn.Linear


def test_explicit_disable_is_a_noop_for_non_hf_modules(monkeypatch):
    model = nn.Linear(32, 32)
    monkeypatch.setitem(sys.modules, "torchao", None)
    assert mxfp8.prepare_mxfp8(model, MXFP8Config()) is model
    assert not hasattr(model, "config")


@pytest.mark.parametrize("saved_recipe", [False, True])
def test_builder_inherits_recipe_and_converts_fp32_before_sharding(monkeypatch, saved_recipe):
    model = _model().bfloat16()
    config = MXFP8Config(enable=True)
    if saved_recipe:
        model.config.veomni_mxfp8 = asdict(config)
    events = []
    monkeypatch.setattr(mxfp8, "_validate_native_runtime", lambda: events.append("native"))

    def convert(module, recipe):
        assert recipe == config
        assert all(p.dtype == torch.float32 for p in module.parameters())
        events.append("convert")
        return module

    def shard(**kwargs):
        assert kwargs["model"].config.veomni_mxfp8 == asdict(config)
        events.append("shard/load")
        return kwargs["model"]

    monkeypatch.setattr(mxfp8, "_convert_linears", convert)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: _state(fsdp_enabled=True, tp_enabled=False))
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", shard)
    torch_parallelize.build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(mxfp8=None if saved_recipe else config),
        enable_gradient_checkpointing=False,
    )
    assert events == ["native", "convert", "shard/load"]


@pytest.mark.parametrize(
    "state",
    [
        _state(dp_mode="ddp"),
        _state(dp_replicate_enabled=True),
        _state(sp_enabled=True),
        _state(cp_enabled=True),
        _state(tp_enabled=True),
        _state(pp_enabled=True),
        _state(any_extra_parallel_enabled=True),
        _state(async_enabled=True),
    ],
)
def test_unsupported_topology_rejected_before_native_probe(monkeypatch, state):
    monkeypatch.setattr(mxfp8, "_validate_native_runtime", lambda: pytest.fail("unexpected native probe"))
    with pytest.raises(ValueError, match="requires FSDP2"):
        mxfp8.validate_mxfp8_training(_model(), MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True)), state)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compile_enabled": True},
        {"offload": True},
        {"activation_offload": True},
        {"is_peft_model": True},
        {"low_precision_reduce_scatter_comm": True},
        {"expert_weight_fp8_all_gather": True},
    ],
)
def test_unsupported_execution_rejected_before_native_probe(monkeypatch, kwargs):
    monkeypatch.setattr(mxfp8, "_validate_native_runtime", lambda: pytest.fail("unexpected native probe"))
    with pytest.raises(ValueError):
        mxfp8.validate_mxfp8_training(
            _model(), MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True)), _state(), **kwargs
        )


@pytest.mark.parametrize("target,match", [("lm_head", "supported Qwen3"), ("missing.*", "matched no modules")])
def test_explicit_targets_cannot_silently_fall_back(target, match):
    precision = MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True, target_modules=[target]))
    with pytest.raises(ValueError, match=match):
        mxfp8.validate_mxfp8_training(_model(), precision, _state())


def test_unaligned_shards_fail_before_conversion():
    precision = MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True, enable_fsdp_all_gather=True))
    with pytest.raises(ValueError, match="uniform row shards"):
        mxfp8.validate_mxfp8_training(_model(), precision, _state(dp_shard_size=4))


def test_config_owned_qat_is_rejected_without_module_attribute():
    model = _model()
    model.config.ops_implementation["qat_implementation"] = "fp8_blockwise"
    assert not hasattr(model, "qat_implementation")
    with pytest.raises(ValueError, match="combined with QAT"):
        mxfp8.validate_mxfp8_training(model, MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True)), _state())


def test_selected_shared_weight_is_rejected():
    model = _model()
    model.model.layers[0].mlp.up_proj.weight = model.model.layers[0].mlp.gate_proj.weight
    with pytest.raises(ValueError, match="shared weight"):
        mxfp8.validate_mxfp8_training(model, MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True)), _state())


@pytest.mark.parametrize("capability,should_probe", [((9, 0), False), ((10, 0), True)])
def test_native_preflight_checks_current_rank_device(monkeypatch, capability, should_probe):
    from veomni.utils import device

    seen = []
    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(
        device,
        "get_torch_device",
        lambda: SimpleNamespace(
            current_device=lambda: 3,
            get_device_capability=lambda index: seen.append(index) or capability,
        ),
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(mxfp8, "version", lambda name: "0.18.0")
    monkeypatch.setattr(mxfp8, "_probe_native_gemm", lambda index: seen.append(("probe", index)))
    if should_probe:
        mxfp8._validate_native_runtime()
        assert seen == [3, ("probe", 3)]
    else:
        with pytest.raises(RuntimeError, match="SM100"):
            mxfp8._validate_native_runtime()
        assert seen == [3]


def test_native_probe_failure_is_propagated_without_conversion(monkeypatch):
    model = _model().bfloat16()
    before = copy.deepcopy(model.state_dict())

    def fail():
        raise RuntimeError("native GEMM unavailable")

    monkeypatch.setattr(mxfp8, "_validate_native_runtime", fail)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: _state())
    with pytest.raises(RuntimeError, match="native GEMM unavailable"):
        torch_parallelize.build_parallelize_model(
            model, mixed_precision=MixedPrecisionConfig(mxfp8=MXFP8Config(enable=True))
        )
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_generated_qwen_mlp_calls_linear_subclass_forward(projection):
    from veomni.models.configuration_base import as_veomni_config
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3MLP
    from veomni.ops.kernels.swiglu_mlp import has_plain_linear_projections

    class ScaledLinear(nn.Linear):
        calls = 0

        def forward(self, value):
            self.calls += 1
            return super().forward(value) * 1.25

    config = as_veomni_config(Qwen3Config(hidden_size=32, intermediate_size=64))
    config._apply_ops_overrides(ops_implementation=make_eager_ops_config())
    mlp = Qwen3MLP(config)
    assert has_plain_linear_projections(mlp)
    original = getattr(mlp, projection)
    replacement = ScaledLinear(original.in_features, original.out_features, bias=False)
    replacement.load_state_dict(original.state_dict())
    setattr(mlp, projection, replacement)
    assert not has_plain_linear_projections(mlp)
    inputs = torch.randn(2, 17, 32, requires_grad=True)
    expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(inputs)) * mlp.up_proj(inputs))
    replacement.calls = 0
    actual = mlp(inputs)
    torch.testing.assert_close(actual, expected)
    assert replacement.calls == 1
    actual.sum().backward()
    assert inputs.grad is not None and all(p.grad is not None for p in mlp.parameters())


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_example_batch_size_scales_with_dp(monkeypatch, world_size):
    path = Path(__file__).resolve().parents[2] / "configs/text/qwen3_mxfp8.yaml"
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(sys, "argv", ["test", str(path)])
    args = parse_args(VeOmniArguments)
    assert args.train.global_batch_size == args.train.micro_batch_size * args.model.accelerator.dp_size
    assert args.train.global_batch_size == world_size
