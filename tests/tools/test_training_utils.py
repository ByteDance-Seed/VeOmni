"""Tests for shared training helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools import training_utils
from veomni.utils.device import MOE_TRITON_DEVICE_TYPES


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_dp_mean_without_distributed_initialization(monkeypatch):
    """Logging outside distributed training needs no model or process group."""
    monkeypatch.setattr(training_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_utils.dist, "is_initialized", lambda: False)
    assert training_utils.dp_mean(3.5, None) == 3.5


@pytest.mark.parametrize(("dp_group", "dp_size"), [(None, 2), (object(), 1)])
def test_dp_mean_without_multiple_dp_ranks(monkeypatch, dp_group, dp_size):
    """A missing or singleton DP group must not enter a collective."""
    monkeypatch.setattr(training_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_utils.dist, "is_initialized", lambda: True)

    def unexpected_reduce(*args, **kwargs):
        pytest.fail("a missing or singleton DP group must not reduce")

    monkeypatch.setattr(training_utils.dist, "all_reduce", unexpected_reduce)
    trainer = SimpleNamespace(
        model=SimpleNamespace(parallel_state=SimpleNamespace(dp_group=dp_group, dp_size=dp_size))
    )
    assert training_utils.dp_mean(3.5, trainer) == 3.5


def test_dp_mean_reduces_over_model_dp_group(monkeypatch):
    """Use the model's DP group and SUM/division, including on NPU backends."""
    dp_group = object()
    monkeypatch.setattr(training_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(training_utils.dist, "is_initialized", lambda: True)

    def reduce_sum(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group is dp_group
        assert tensor.device.type == "cpu"
        assert tensor.item() == 3.5
        tensor.add_(6.5)

    def group_size(group):
        assert group is dp_group
        return 2

    monkeypatch.setattr(training_utils.dist, "all_reduce", reduce_sum)
    monkeypatch.setattr(training_utils.dist, "get_world_size", group_size)
    trainer = SimpleNamespace(
        model=SimpleNamespace(parallel_state=SimpleNamespace(dp_group=dp_group, dp_size=2)), device="cpu"
    )
    assert training_utils.dp_mean(3.5, trainer) == 5.0


def test_training_command_uses_agent_bound_rendezvous():
    command = training_utils.build_torchrun_cmd("worker.py", "config", "weights", "data", "output", nproc=2)
    assert "--standalone" in command
    assert not any(arg.startswith(("--master_port", "--master-port")) for arg in command)


@pytest.mark.parametrize("previous_backend", [None, "veomni"])
def test_build_hf_reference_model_uses_models_and_restores_backend(monkeypatch, previous_backend):
    import veomni.models as models

    if previous_backend is None:
        monkeypatch.delenv("MODELING_BACKEND", raising=False)
    else:
        monkeypatch.setenv("MODELING_BACKEND", previous_backend)

    expected = object()
    captured = {}

    def fake_build_foundation_model(**kwargs):
        captured["backend"] = os.environ.get("MODELING_BACKEND")
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(models, "build_foundation_model", fake_build_foundation_model)

    actual = training_utils.build_hf_reference_model(
        "toy-config",
        torch_dtype="float32",
        init_device="cpu",
    )

    assert actual is expected
    assert captured["backend"] == "hf"
    assert captured["kwargs"]["config_path"] == "toy-config"
    assert captured["kwargs"]["weights_path"] is None
    assert captured["kwargs"]["torch_dtype"] == "float32"
    assert captured["kwargs"]["init_device"] == "cpu"
    assert captured["kwargs"]["ops_implementation"].attn_implementation == "eager"
    assert os.environ.get("MODELING_BACKEND") == previous_backend


def test_build_hf_reference_model_restores_backend_after_failure(monkeypatch):
    import veomni.models as models

    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    def fail_build_foundation_model(**_kwargs):
        assert os.environ["MODELING_BACKEND"] == "hf"
        raise RuntimeError("build failed")

    monkeypatch.setattr(models, "build_foundation_model", fail_build_foundation_model)

    with pytest.raises(RuntimeError, match="build failed"):
        training_utils.build_hf_reference_model(
            "toy-config",
            torch_dtype="float32",
            init_device="cpu",
        )

    assert os.environ["MODELING_BACKEND"] == "veomni"


def test_build_hf_reference_model_supports_an_unregistered_family(monkeypatch, tmp_path):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        architectures=["Qwen3ForCausalLM"],
        attn_implementation="eager",
    )
    config.save_pretrained(tmp_path)
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    model = training_utils.build_hf_reference_model(
        str(tmp_path),
        torch_dtype="float32",
        init_device="cpu",
    )

    assert model.config.model_type == "qwen3"
    assert model.__class__.__module__.startswith("transformers.models.qwen3")
    assert os.environ["MODELING_BACKEND"] == "veomni"


@pytest.mark.parametrize(
    ("device_type", "expected_moe"),
    [
        (MOE_TRITON_DEVICE_TYPES[0], "fused_triton"),
        ("npu", "fused_npu"),
    ],
)
def test_deepseek_v4_ops_overrides_follow_active_device(monkeypatch, device_type, expected_moe):
    monkeypatch.setattr(training_utils, "get_device_type", lambda: device_type)

    overrides = training_utils.resolve_ops_overrides("deepseek_v4")

    assert "--model.ops_implementation.attn_implementation=eager" in overrides
    assert f"--model.ops_implementation.moe_implementation={expected_moe}" in overrides


def test_deepseek_v3_gpu_keeps_default_moe(monkeypatch):
    monkeypatch.setattr(training_utils, "get_device_type", lambda: MOE_TRITON_DEVICE_TYPES[0])

    overrides = training_utils.resolve_ops_overrides("deepseek_v3")

    assert not any(flag.startswith("--model.ops_implementation.moe_implementation=") for flag in overrides)


def test_make_eager_ops_config_covers_every_kernel_field():
    from dataclasses import fields

    from veomni.arguments import OpsImplementationConfig

    selection = training_utils.make_eager_ops_config()
    for field in fields(OpsImplementationConfig):
        expected = "none" if field.name == "qat_implementation" else "eager"
        assert getattr(selection, field.name) == expected, field.name
