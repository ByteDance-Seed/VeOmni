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
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed import parallel_state as parallel_state_module
from veomni.models import auto as auto_module
from veomni.models.auto import build_config, check_context_parallel_supported


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    # A real two-rank world, which is what makes the default state invalid.
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)

    check_context_parallel_supported(build_config("tests/toy_config/qwen3_toy"))


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm_moe_dsa"])
def test_gate_rejects_context_parallel_on_npu(monkeypatch, model_type):
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        auto_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True),
    )
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", True)

    with pytest.raises(NotImplementedError, match="Ascend/NPU.*Qwen4-Exp only"):
        check_context_parallel_supported(SimpleNamespace(model_type=model_type))


def test_gate_allows_qwen4_exp_context_parallel_on_npu(monkeypatch):
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto_module, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True))
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", True)

    check_context_parallel_supported(SimpleNamespace(model_type="qwen4_exp"))


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


@pytest.mark.parametrize("is_npu,ulysses_size", [(True, 1), (False, 2)])
def test_seed_omni_cp_opt_in_does_not_bypass_device_or_hybrid_gate(monkeypatch, is_npu, ulysses_size):
    from veomni.models.seed_omni.modules.text_encoder.configuration import TextEncoderConfig

    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(
        auto_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, ulysses_size=ulysses_size),
    )
    monkeypatch.setattr(auto_module, "IS_NPU_AVAILABLE", is_npu)
    message = "Ascend/NPU.*Qwen4-Exp only" if is_npu else "Combined Ulysses/CP.*Qwen4-Exp only"
    with pytest.raises(NotImplementedError, match=message):
        check_context_parallel_supported(TextEncoderConfig(vocab_size=64, hidden_size=16))
