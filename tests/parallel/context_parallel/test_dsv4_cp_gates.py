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
# See the License for the specific language governing permissions and
# limitations under the License.

"""The context-parallel model gate must not construct a parallel state.

``check_context_parallel_supported`` runs for every model and asks
``get_parallel_state()`` whether CP is on. In a process that never installed a
state, that accessor does not return ``None`` -- it *builds* a default
single-process ``ParallelState``, whose ``dp_size=1`` contradicts a multi-rank
world and so raises on the topology product check. That turned any multi-rank
process which builds a model without installing a state into a hard failure;
``tests/lora/test_moe_lora_ep2.py`` is one, spawning two ranks and calling
``build_foundation_model`` directly.

GPU CI caught this and the unit tests did not, because every one of them stubbed
the accessor. Hence the one thing worth pinning here is the case where nothing
is stubbed.

The DSA-versus-parallel-state gate has a matching trap: a native checkpoint
config is adapted into a generic class that keeps ``ops_config`` but not
``DeepseekV4Config.validate_build_prerequisites``, so the pair has to be judged
by the builder itself and exercised through ``build_foundation_model``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as NativeDSV4Config

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed import parallel_state as parallel_state_module
from veomni.models import auto as auto_module
from veomni.models.auto import build_config, check_context_parallel_supported


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _enable_npu_cp(monkeypatch):
    # The DSA-vs-parallel-state check now runs in the builder gate, so only
    # ``auto``'s copies of these three names matter.
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto_module, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True, global_rank=0))
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", True)


def _npu_ops_for_cp() -> dict:
    # The DSA selections ``eager`` and ``cann`` are both admitted under CP. The
    # tests below start from ``eager`` and swap a field to ``cann`` (the shipped
    # value) or to a rejected one.
    return {"dsa_indexer_implementation": "eager", "dsa_attention_implementation": "eager"}


def _installed_ops(ops_implementation: dict) -> OpsImplementationConfig:
    # The builder takes a resolved config, while the direct gate tests below go
    # through the model config's dict form.
    return OpsImplementationConfig(attn_implementation="eager", **ops_implementation)


def _dsv4_with_ops(ops_implementation: dict):
    from veomni.models.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
        ops_implementation=ops_implementation,
    )


def test_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    # A real two-rank world, which is what makes the default state invalid.
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)

    check_context_parallel_supported(build_config("tests/toy_config/qwen3_toy"))


def test_gate_rejects_context_parallel_on_npu(monkeypatch):
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        auto_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True),
    )
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", True)

    with pytest.raises(NotImplementedError, match="GPU-only"):
        check_context_parallel_supported(SimpleNamespace(model_type="qwen4_exp"))


def test_gate_allows_npu_dsv4_cp_with_eager_dsa_ops(monkeypatch):
    _enable_npu_cp(monkeypatch)

    check_context_parallel_supported(_dsv4_with_ops(_npu_ops_for_cp()))


@pytest.mark.parametrize("field", ["dsa_indexer_implementation", "dsa_attention_implementation"])
def test_gate_allows_npu_dsv4_cp_with_cann_dsa_ops(monkeypatch, field):
    """The CANN adapters take the query offset, so they serve CP's geometry too."""
    _enable_npu_cp(monkeypatch)
    ops = _npu_ops_for_cp()
    ops[field] = "cann"

    check_context_parallel_supported(_dsv4_with_ops(ops))


def test_gate_allows_npu_dsv4_cp_with_both_cann_dsa_ops(monkeypatch):
    _enable_npu_cp(monkeypatch)
    ops = dict.fromkeys(_npu_ops_for_cp(), "cann")

    check_context_parallel_supported(_dsv4_with_ops(ops))


@pytest.mark.parametrize("field", ["dsa_indexer_implementation", "dsa_attention_implementation"])
def test_gate_rejects_npu_dsv4_cp_with_a_non_cp_aware_dsa_operator(monkeypatch, field):
    _enable_npu_cp(monkeypatch)
    ops = _npu_ops_for_cp()
    ops[field] = "tilelang"

    with pytest.raises(NotImplementedError, match=field):
        check_context_parallel_supported(_dsv4_with_ops(ops))


def test_gate_rejects_a_native_config_adapted_by_the_builder(monkeypatch):
    """The failure the direct tests miss: the adapted class has no model hook."""
    _enable_npu_cp(monkeypatch)
    load_model = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(auto_module, "get_loader", lambda config: SimpleNamespace(load_model=load_model))
    ops = _npu_ops_for_cp()
    ops["dsa_attention_implementation"] = "tilelang"

    with pytest.raises(NotImplementedError, match="dsa_attention_implementation"):
        auto_module.build_foundation_model(
            NativeDSV4Config(), init_device="cpu", ops_implementation=_installed_ops(ops)
        )

    load_model.assert_not_called()


def test_gate_admits_a_native_config_adapted_by_the_builder(monkeypatch):
    _enable_npu_cp(monkeypatch)
    load_model = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(auto_module, "get_loader", lambda config: SimpleNamespace(load_model=load_model))

    model = auto_module.build_foundation_model(
        NativeDSV4Config(), init_device="cpu", ops_implementation=_installed_ops(_npu_ops_for_cp())
    )

    assert model is load_model.return_value
    load_model.assert_called_once()


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen4_exp"])
def test_gate_allows_context_parallel_models_on_gpu(monkeypatch, model_type):
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        auto_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True),
    )
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", False)

    check_context_parallel_supported(SimpleNamespace(model_type=model_type))


def test_gate_allows_seed_omni_module_config_under_cp(monkeypatch):
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
    from veomni.models.seed_omni.modules.module_configuration_base import OmniModuleConfig
    from veomni.models.seed_omni.modules.text_encoder.configuration import TextEncoderConfig

    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        auto_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True),
    )
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", False)

    # TextEncoderConfig explicitly opts in via _supports_context_parallel = True.
    check_context_parallel_supported(TextEncoderConfig(vocab_size=64, hidden_size=16))

    # A SeedOmni wrapper around a CP-supported inner model_type (e.g. deepseek_v4) is allowed.
    class _WrappedCPModuleConfig(OmniModuleConfig):
        model_type = "dummy_cp_llm"

        def __init__(self):
            self.text_config = SimpleNamespace(model_type="deepseek_v4")
            super().__init__()

    check_context_parallel_supported(_WrappedCPModuleConfig())

    # An unported OmniModuleConfig subclass is still rejected under cp_enabled=True.
    with pytest.raises(
        NotImplementedError, match="Context parallelism is not implemented for model type 'fake_module_a'"
    ):
        check_context_parallel_supported(FakeModuleAConfig(hidden_size=16))
