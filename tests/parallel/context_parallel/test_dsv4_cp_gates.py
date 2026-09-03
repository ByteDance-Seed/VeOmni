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

from veomni.distributed import parallel_state as parallel_state_module
from veomni.models import auto as auto_module
from veomni.models.auto import build_config, check_context_parallel_supported


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
    monkeypatch.setattr(auto_module, "is_torch_npu_available", lambda: True)

    with pytest.raises(NotImplementedError, match="GPU-only"):
        check_context_parallel_supported(SimpleNamespace(model_type="deepseek_v4"))


@pytest.fixture
def _cp_is_on(monkeypatch):
    """Context parallelism enabled on a GPU, which is where the model gate is the
    only thing left between a config and a run."""
    monkeypatch.setattr(auto_module, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto_module, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True))
    monkeypatch.setattr(auto_module, "is_torch_npu_available", lambda: False)


def test_the_declaring_model_is_admitted(_cp_is_on):
    """The declaration in ``veomni/models/transformers/deepseek_v4/__init__.py`` has to
    have been reached by the time the gate runs.

    Not a tautology: the registry is populated as an import side effect of the model
    package, and ``veomni/models/__init__.py`` imports ``transformers`` before
    ``auto`` for that to hold. Were the order ever reversed, or the package import
    made lazy, the model that *does* implement CP would be the one refused -- and
    every other test here would still pass.
    """
    check_context_parallel_supported(SimpleNamespace(model_type="deepseek_v4"))


@pytest.mark.parametrize("model_type", ["qwen3", "llama", None, "not_a_model_type"])
def test_an_undeclared_model_is_refused(_cp_is_on, model_type):
    """Absence has to mean refused, which is the whole safety property.

    Parametrised over an unported model, a plausible one, ``None`` and a name that
    was never a model type, because the query answers all four by the same path and a
    registry lookup that raised on an unknown key would turn the last three into a
    traceback about the registry rather than a message about CP.
    """
    with pytest.raises(NotImplementedError, match="Context parallelism is not implemented"):
        check_context_parallel_supported(SimpleNamespace(model_type=model_type))


def test_the_refusal_names_what_to_switch_to(_cp_is_on):
    """A refusal that names only what failed leaves the user to search for the model
    that would work. The enumeration is derived from the registry rather than
    restated, so it cannot drift from the gate's own answer."""
    with pytest.raises(NotImplementedError, match="only deepseek_v4 supports it"):
        check_context_parallel_supported(SimpleNamespace(model_type="qwen3"))


def test_a_capability_is_not_a_prefix_match(_cp_is_on):
    """The query is set membership, not a substring test.

    Cheap to pin and expensive to get wrong: a model declaring a longer capability
    that happens to contain ``context_parallel`` would otherwise be granted CP it
    never implemented.
    """
    from veomni.models.loader import MODEL_CAPABILITY_REGISTRY, model_supports

    # Not ``monkeypatch.setitem``: it saves the old value with ``dict.get``, and
    # ``Registry.__getitem__`` raises ``ValueError`` rather than ``KeyError`` on a
    # missing key, so the inherited ``Mapping.get`` propagates instead of returning
    # the default. Assigning writes the local override, which ``del`` undoes.
    MODEL_CAPABILITY_REGISTRY["toy"] = frozenset({"context_parallel_planned"})
    try:
        assert model_supports("toy", "context_parallel") is False

        with pytest.raises(NotImplementedError, match="Context parallelism is not implemented"):
            check_context_parallel_supported(SimpleNamespace(model_type="toy"))
    finally:
        del MODEL_CAPABILITY_REGISTRY["toy"]
