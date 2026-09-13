"""Tests for shared training helpers."""

from __future__ import annotations

import os

import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from tests.tools import training_utils
from veomni.utils.device import MOE_TRITON_DEVICE_TYPES


@pytest.mark.parametrize("previous_backend", [None, "veomni"])
def test_build_hf_reference_model_uses_models_kernel_and_restores_backend(monkeypatch, previous_backend):
    import veomni.models_kernel as models_kernel

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

    monkeypatch.setattr(models_kernel, "build_foundation_model", fake_build_foundation_model)

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
    import veomni.models_kernel as models_kernel

    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    def fail_build_foundation_model(**_kwargs):
        assert os.environ["MODELING_BACKEND"] == "hf"
        raise RuntimeError("build failed")

    monkeypatch.setattr(models_kernel, "build_foundation_model", fail_build_foundation_model)

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
