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

from collections.abc import Callable
from dataclasses import dataclass

import pytest
from transformers import PretrainedConfig

from tests.models_kernel.compare import eager_ops_config
from tests.models_kernel.tiny_configs import (
    tiny_deepseek_v3_config as _tiny_deepseek_v3_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_deepseek_v4_config as _tiny_deepseek_v4_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_flux_config as _tiny_flux_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_gemma3_text_config as _tiny_gemma3_text_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_glm_moe_dsa_config as _tiny_glm_moe_dsa_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_gpt_oss_config as _tiny_gpt_oss_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_llama_config as _tiny_llama_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_5_omni_config as _tiny_qwen2_5_omni_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_5_omni_text_config as _tiny_qwen2_5_omni_text_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_5_omni_thinker_config as _tiny_qwen2_5_omni_thinker_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_5_vl_config as _tiny_qwen2_5_vl_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_config as _tiny_qwen2_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen2_vl_config as _tiny_qwen2_vl_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_5_config as _tiny_qwen3_5_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_5_moe_config as _tiny_qwen3_5_moe_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_5_moe_text_config as _tiny_qwen3_5_moe_text_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_5_text_config as _tiny_qwen3_5_text_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_config as _tiny_qwen3_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_moe_config as _tiny_qwen3_moe_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_omni_moe_config as _tiny_qwen3_omni_moe_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_omni_moe_text_config as _tiny_qwen3_omni_moe_text_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_omni_moe_thinker_config as _tiny_qwen3_omni_moe_thinker_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_vl_config as _tiny_qwen3_vl_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_qwen3_vl_moe_config as _tiny_qwen3_vl_moe_config,
)
from tests.models_kernel.tiny_configs import (
    tiny_seed_oss_config as _tiny_seed_oss_config,
)
from veomni.models_kernel import (
    MODEL_CONFIG_REGISTRY,
    MODEL_PROCESSOR_REGISTRY,
    MODELING_REGISTRY,
    build_config,
    build_foundation_model,
    check_context_parallel_supported,
    check_model_build_prerequisites,
    get_model_class,
)
from veomni.ops.config import get_ops_config, set_ops_config


class _UnregisteredConfig(PretrainedConfig):
    model_type = "unregistered_test_model"

    def __init__(self):
        super().__init__()
        self.architectures = ["UnregisteredForCausalLM"]


@dataclass(frozen=True)
class _ModelCase:
    model_type: str
    config_factory: Callable[[str], PretrainedConfig]
    architectures: tuple[str, ...]
    has_registered_config: bool = False
    registered_config_aliases: tuple[str, ...] = ()
    processor_class_name: str | None = None
    eager_op_path: str | None = "veomni_ce"


_MODEL_CASES = (
    _ModelCase(
        model_type="deepseek_v3",
        config_factory=_tiny_deepseek_v3_config,
        architectures=(
            "DeepseekV3ForCausalLM",
            "DeepseekV3ForSequenceClassification",
            "DeepseekV3ForTokenClassification",
            "DeepseekV3Model",
        ),
    ),
    _ModelCase(
        model_type="deepseek_v4",
        config_factory=_tiny_deepseek_v4_config,
        architectures=("DeepseekV4ForCausalLM", "DeepseekV4Model"),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="flux",
        config_factory=_tiny_flux_config,
        architectures=("FluxModel",),
        has_registered_config=True,
        eager_op_path="blocks.0.attn.norm_q_a.veomni_rms_norm",
    ),
    _ModelCase(
        model_type="gemma3_text",
        config_factory=_tiny_gemma3_text_config,
        architectures=("Gemma3ForCausalLM", "Gemma3TextModel"),
    ),
    _ModelCase(
        model_type="gpt_oss",
        config_factory=_tiny_gpt_oss_config,
        architectures=(
            "GptOssForCausalLM",
            "GptOssForSequenceClassification",
            "GptOssForTokenClassification",
            "GptOssModel",
        ),
    ),
    _ModelCase(
        model_type="glm_moe_dsa",
        config_factory=_tiny_glm_moe_dsa_config,
        architectures=("GlmMoeDsaForCausalLM", "GlmMoeDsaModel"),
    ),
    _ModelCase(
        model_type="llama",
        config_factory=_tiny_llama_config,
        architectures=(
            "LlamaForCausalLM",
            "LlamaForTokenClassification",
            "LlamaForSequenceClassification",
            "LlamaModel",
        ),
    ),
    _ModelCase(
        model_type="seed_oss",
        config_factory=_tiny_seed_oss_config,
        architectures=(
            "SeedOssForCausalLM",
            "SeedOssForQuestionAnswering",
            "SeedOssForSequenceClassification",
            "SeedOssForTokenClassification",
            "SeedOssModel",
        ),
    ),
    _ModelCase(
        model_type="qwen2",
        config_factory=_tiny_qwen2_config,
        architectures=(
            "Qwen2ForCausalLM",
            "Qwen2ForTokenClassification",
            "Qwen2ForSequenceClassification",
            "Qwen2ForQuestionAnswering",
            "Qwen2Model",
        ),
    ),
    _ModelCase(
        model_type="qwen2_vl",
        config_factory=_tiny_qwen2_vl_config,
        architectures=("Qwen2VLForConditionalGeneration", "Qwen2VLModel"),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="qwen2_5_vl",
        config_factory=_tiny_qwen2_5_vl_config,
        architectures=("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel"),
        has_registered_config=True,
        registered_config_aliases=("qwen2_5_vl_text",),
    ),
    _ModelCase(
        model_type="qwen2_5_omni",
        config_factory=_tiny_qwen2_5_omni_config,
        architectures=("Qwen2_5OmniForConditionalGeneration",),
        has_registered_config=True,
        processor_class_name="Qwen2_5OmniProcessor",
        eager_op_path="thinker.veomni_ce",
    ),
    _ModelCase(
        model_type="qwen2_5_omni_thinker",
        config_factory=_tiny_qwen2_5_omni_thinker_config,
        architectures=("Qwen2_5OmniThinkerForConditionalGeneration",),
    ),
    _ModelCase(
        model_type="qwen2_5_omni_text",
        config_factory=_tiny_qwen2_5_omni_text_config,
        architectures=("Qwen2_5OmniThinkerTextModel",),
        eager_op_path=None,
    ),
    _ModelCase(
        model_type="qwen3",
        config_factory=_tiny_qwen3_config,
        architectures=(
            "Qwen3ForCausalLM",
            "Qwen3ForTokenClassification",
            "Qwen3ForSequenceClassification",
            "Qwen3Model",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5",
        config_factory=_tiny_qwen3_5_config,
        architectures=(
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5ForTokenClassification",
            "Qwen3_5ForSequenceClassification",
            "Qwen3_5Model",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5_text",
        config_factory=_tiny_qwen3_5_text_config,
        architectures=(
            "Qwen3_5ForCausalLM",
            "Qwen3_5TextForSequenceClassification",
            "Qwen3_5TextModel",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5_moe",
        config_factory=_tiny_qwen3_5_moe_config,
        architectures=("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeModel"),
    ),
    _ModelCase(
        model_type="qwen3_5_moe_text",
        config_factory=_tiny_qwen3_5_moe_text_config,
        architectures=("Qwen3_5MoeForCausalLM", "Qwen3_5MoeTextModel"),
    ),
    _ModelCase(
        model_type="qwen3_moe",
        config_factory=_tiny_qwen3_moe_config,
        architectures=(
            "Qwen3MoeForCausalLM",
            "Qwen3MoeForTokenClassification",
            "Qwen3MoeForSequenceClassification",
            "Qwen3MoeForQuestionAnswering",
            "Qwen3MoeModel",
        ),
    ),
    _ModelCase(
        model_type="qwen3_omni_moe",
        config_factory=_tiny_qwen3_omni_moe_config,
        architectures=("Qwen3OmniMoeForConditionalGeneration",),
        has_registered_config=True,
        processor_class_name="Qwen3OmniMoeProcessor",
        eager_op_path="thinker.veomni_ce",
    ),
    _ModelCase(
        model_type="qwen3_omni_moe_thinker",
        config_factory=_tiny_qwen3_omni_moe_thinker_config,
        architectures=("Qwen3OmniMoeThinkerForConditionalGeneration",),
    ),
    _ModelCase(
        model_type="qwen3_omni_moe_text",
        config_factory=_tiny_qwen3_omni_moe_text_config,
        architectures=("Qwen3OmniMoeThinkerTextModel",),
        eager_op_path=None,
    ),
    _ModelCase(
        model_type="qwen3_vl",
        config_factory=_tiny_qwen3_vl_config,
        architectures=("Qwen3VLForConditionalGeneration", "Qwen3VLModel"),
    ),
    _ModelCase(
        model_type="qwen3_vl_moe",
        config_factory=_tiny_qwen3_vl_moe_config,
        architectures=(
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3VLMoeModel",
            "Qwen3VLMoeTextModel",
        ),
    ),
)

_ARCHITECTURE_CASES = tuple(
    pytest.param(model_case, architecture, id=f"{model_case.model_type}-{architecture}")
    for model_case in _MODEL_CASES
    for architecture in model_case.architectures
)


def test_get_model_class_unknown_type_raises():
    with pytest.raises(RuntimeError, match="unregistered_test_model.*not registered in veomni.models_kernel"):
        get_model_class(_UnregisteredConfig())


def test_get_model_class_hf_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    from transformers import AutoModelForCausalLM

    assert get_model_class(_tiny_qwen3_config()) is AutoModelForCausalLM


def test_build_foundation_model_requires_ops_config():
    previous = get_ops_config()
    try:
        set_ops_config(None)
        with pytest.raises(ValueError, match="ops_implementation"):
            build_foundation_model(_tiny_qwen3_config())
    finally:
        set_ops_config(previous)


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda model_case: model_case.model_type)
def test_build_foundation_model_constructs_registered_model(model_case: _ModelCase):
    previous = get_ops_config()
    cfg = eager_ops_config()
    try:
        set_ops_config(None)
        model = build_foundation_model(
            model_case.config_factory(model_case.architectures[0]),
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=cfg,
        )
        assert get_ops_config() is cfg
    finally:
        set_ops_config(previous)
    assert model.__class__.__name__ == model_case.architectures[0]
    if model_case.eager_op_path is not None:
        op = model
        for attribute in model_case.eager_op_path.split("."):
            op = getattr(op, attribute)
        assert op.impl == "eager"


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda model_case: model_case.model_type)
def test_model_registry_entries(model_case: _ModelCase):
    assert (model_case.model_type in MODEL_CONFIG_REGISTRY.valid_keys()) is model_case.has_registered_config
    assert set(model_case.registered_config_aliases) <= set(MODEL_CONFIG_REGISTRY.valid_keys())
    assert model_case.model_type in MODELING_REGISTRY.valid_keys()
    if model_case.processor_class_name is not None:
        assert model_case.processor_class_name in MODEL_PROCESSOR_REGISTRY.valid_keys()


@pytest.mark.parametrize(("model_case", "architecture"), _ARCHITECTURE_CASES)
def test_get_model_class_returns_registered_architecture(model_case: _ModelCase, architecture: str):
    model_cls = get_model_class(model_case.config_factory(architecture))
    assert model_cls.__name__ == architecture


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

    previous = get_ops_config()
    cfg = eager_ops_config()
    cfg.dsa_indexer_implementation = "eager"
    set_ops_config(cfg)
    try:
        config = DeepseekV4Config(
            num_hidden_layers=2,
            layer_types=["compressed_sparse_attention"] * 2,
            dsa_indexer_loss=True,
        )
        with pytest.raises(ValueError, match="dsa_indexer_implementation"):
            check_model_build_prerequisites(config)
    finally:
        set_ops_config(previous)


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
