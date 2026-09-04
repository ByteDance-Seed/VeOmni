# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing limitations
# under the License.

"""models_kernel auto / registry construct helpers."""

from __future__ import annotations

import pytest
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from tests.models_kernel.compare import eager_kernels_config
from veomni.kernels.config import get_kernels_config, set_kernels_config
from veomni.models_kernel import (
    MODEL_CONFIG_REGISTRY,
    MODELING_REGISTRY,
    build_config,
    build_foundation_model,
    check_context_parallel_supported,
    check_model_build_prerequisites,
    get_model_class,
)


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        architectures=["Qwen3ForCausalLM"],
        attn_implementation="eager",
    )


def test_get_model_class_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown Modeling"):
        get_model_class(_tiny_qwen3_config())


def test_get_model_class_hf_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    from transformers import AutoModelForCausalLM

    assert get_model_class(_tiny_qwen3_config()) is AutoModelForCausalLM


def test_build_foundation_model_requires_kernels_config():
    previous = get_kernels_config()
    try:
        set_kernels_config(None)
        with pytest.raises(ValueError, match="kernels_implementation"):
            build_foundation_model(_tiny_qwen3_config())
    finally:
        set_kernels_config(previous)


def test_build_foundation_model_installs_kernels_config():
    previous = get_kernels_config()
    cfg = eager_kernels_config()
    try:
        set_kernels_config(None)
        with pytest.raises(ValueError, match="Unknown Modeling"):
            build_foundation_model(_tiny_qwen3_config(), kernels_implementation=cfg)
        assert get_kernels_config() is cfg
    finally:
        set_kernels_config(previous)


def test_modeling_registry_starts_without_qwen3():
    assert "qwen3" not in MODELING_REGISTRY.valid_keys()


def test_deepseek_v4_is_the_first_registered_model():
    assert "deepseek_v4" in MODEL_CONFIG_REGISTRY.valid_keys()
    assert "deepseek_v4" in MODELING_REGISTRY.valid_keys()


def test_get_model_class_returns_the_generated_dsv4_causal_lm():
    from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    config = DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
        architectures=["DeepseekV4ForCausalLM"],
    )
    model_cls = get_model_class(config)
    assert model_cls.__name__ == "DeepseekV4ForCausalLM"
    assert "models_kernel" in model_cls.__module__


def test_get_model_config_uses_the_registered_dsv4_subclass(tmp_path):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as UpstreamConfig

    from veomni.models_kernel.registry import get_model_config
    from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
    ).save_pretrained(tmp_path)
    config = get_model_config(str(tmp_path), dsa_indexer_loss=True, dsa_indexer_loss_coef=0.25)
    assert type(config) is DeepseekV4Config
    assert type(config) is not UpstreamConfig
    assert config.dsa_indexer_loss is True
    assert config.dsa_indexer_loss_coef == 0.25
    assert type(build_config(str(tmp_path), dsa_indexer_loss=True)) is DeepseekV4Config


def test_a_config_that_cannot_ask_for_the_objective_is_left_alone():
    from transformers import AutoConfig

    other = AutoConfig.for_model("llama", num_hidden_layers=2)
    assert not hasattr(other, "dsa_indexer_loss")
    assert not hasattr(other, "validate_build_prerequisites")
    check_model_build_prerequisites(other)


def test_the_generic_hook_reaches_the_model_that_implements_it():
    from veomni.models_kernel.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    previous = get_kernels_config()
    cfg = eager_kernels_config()
    cfg.dsa_indexer_implementation = "eager"
    set_kernels_config(cfg)
    try:
        config = DeepseekV4Config(
            num_hidden_layers=2,
            layer_types=["compressed_sparse_attention"] * 2,
            dsa_indexer_loss=True,
        )
        with pytest.raises(ValueError, match="dsa_indexer_implementation"):
            check_model_build_prerequisites(config)
    finally:
        set_kernels_config(previous)


def test_context_parallel_is_refused_on_npu(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    import veomni.models_kernel.auto as auto

    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True))
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: True)
    with pytest.raises(NotImplementedError, match="GPU-only"):
        check_context_parallel_supported(_tiny_qwen3_config())


def test_context_parallel_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch: pytest.MonkeyPatch):
    from veomni.distributed import parallel_state as parallel_state_module

    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)
    check_context_parallel_supported(_tiny_qwen3_config())
