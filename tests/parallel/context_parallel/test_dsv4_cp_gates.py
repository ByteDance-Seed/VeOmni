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

from veomni.distributed import parallel_state as parallel_state_module
from veomni.models.auto import build_config, check_context_parallel_supported


def test_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    # A real two-rank world, which is what makes the default state invalid.
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)

    check_context_parallel_supported(build_config("tests/toy_config/qwen3_toy"))
