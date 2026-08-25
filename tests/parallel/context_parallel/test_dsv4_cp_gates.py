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

"""The three gates a context-parallel run has to pass, in the order it meets them.

1. ``TrainingArguments._validate_accelerator`` -- what a config file or CLI run
   goes through before any ``ParallelState`` exists.
2. ``ParallelState.__post_init__`` -- the same constraints again, because a state
   can also be constructed directly, which is how a per-module state under
   ``use_parallel_state`` is made.
3. ``check_context_parallel_supported`` -- the model allow-list. The first two
   gates are model-agnostic and admit ``cp_size > 1`` for anything, so this is
   the only one that can refuse a model whose forward has no CP path. Such a
   model would still be sharded by ``SequenceParallelCollator`` (which keys on
   ``sp_enabled``, true under CP alone) and would never gather the shards back
   (its forward keys on ``ulysses_enabled``, false under CP alone), attending
   only inside its own slice and training to a plausible loss curve while being
   silently wrong.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from veomni.arguments.arguments_types import AcceleratorConfig, TrainingArguments
from veomni.distributed.parallel_state import ParallelState
from veomni.models.auto import build_config, build_foundation_model


# --------------------------- gate 1: TrainingArguments ---------------------------


def test_accelerator_context_parallel_alone_is_accepted(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = TrainingArguments(accelerator=AcceleratorConfig(cp_size=2, ulysses_size=1))
    assert args.accelerator.cp_size == 2
    # dp_size = world_size // (pp * ulysses * cp * tp) = 2 // (1*1*2*1) = 1
    assert args.accelerator.dp_size == 1


@pytest.mark.parametrize("cp_size", [0, -1], ids=["zero", "negative"])
def test_accelerator_non_positive_context_parallel_size_is_rejected(monkeypatch, cp_size):
    """Neither non-positive value can reach the topology arithmetic.

    ``cp_size=0`` would make the world-size modulo raise ZeroDivisionError, and
    ``cp_size=-1`` would derive ``dp_size=-1`` and ``dp_shard_size=-1`` -- exactly
    the combination that then passes gate 2's product check, since the negatives
    cancel.
    """
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        TrainingArguments(accelerator=AcceleratorConfig(cp_size=cp_size))


def test_accelerator_hybrid_context_and_ulysses_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(NotImplementedError, match="ulysses_size"):
        TrainingArguments(accelerator=AcceleratorConfig(cp_size=2, ulysses_size=2))


# ----------------------------- gate 2: ParallelState -----------------------------


def test_context_parallel_alone_is_accepted(monkeypatch):
    # ParallelState.world_size reads torch.distributed directly; simulate a
    # 2-rank world so the world-size product check (cp_size=2, everything
    # else 1) is satisfied without a real process group. sp_enabled is True
    # here (cp_size > 1), which requires a non-None device_mesh -- a mock
    # stands in since nothing here touches the mesh itself.
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 2)
    state = ParallelState(dp_size=1, cp_size=2, ulysses_size=1, device_type="cpu", device_mesh=MagicMock())
    assert state.cp_size == 2
    assert state.cp_enabled


def test_context_parallel_is_disabled_at_size_one():
    state = ParallelState(dp_size=1, cp_size=1, ulysses_size=1, device_type="cpu")
    assert not state.cp_enabled


@pytest.mark.parametrize("cp_size", [0, -1], ids=["zero", "negative"])
def test_non_positive_context_parallel_size_is_rejected(monkeypatch, cp_size):
    """A negative cp_size survives the world-size product check by cancelling.

    ``dp_size=-1`` with ``cp_size=-1`` multiplies out to +1, so the product check
    would admit the topology and ``cp_enabled`` would then report False -- CP
    silently off, on a state that looks validated. ``dp_shard_size=-1`` is set for
    the same reason: it keeps the dp_replicate * dp_shard == dp_size check from
    being the thing that fires, so the rejection is attributable to the new guard.
    """
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 1)
    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        ParallelState(
            dp_size=-1,
            cp_size=cp_size,
            ulysses_size=1,
            dp_replicate_size=1,
            dp_shard_size=-1,
            device_type="cpu",
            device_mesh=MagicMock(),
        )


def test_hybrid_context_and_ulysses_is_rejected(monkeypatch):
    # cp_size=2 * ulysses_size=2 * (dp/tp/pp=1) == world_size=4, so the
    # rejection below is attributable to the new CP+Ulysses guard rather
    # than the pre-existing world-size product ValueError.
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 4)
    with pytest.raises(NotImplementedError, match="ulysses_size"):
        ParallelState(dp_size=1, cp_size=2, ulysses_size=2, device_type="cpu", device_mesh=MagicMock())


# --------------------------- gate 3: model allow-list ---------------------------


class _StubLoader:
    """Stands in for the real loader so the gate is the only thing under test.

    The gate runs before construction, which is also what makes the admitted
    case cheap: building a real DeepSeek-V4 would otherwise make it the most
    expensive test in this directory to assert nothing happened. Carrying no
    ``model_cls`` is deliberate -- ``build_foundation_model`` reads it with
    ``getattr(..., None)`` and skips the OpSlot binding.
    """

    def __init__(self) -> None:
        self.model = torch.nn.Identity()
        self.calls = 0

    def load_model(self, **kwargs) -> torch.nn.Module:
        self.calls += 1
        return self.model


def _build(config_path: str, cp_enabled: bool, loader: _StubLoader) -> None:
    """``build_foundation_model`` with the parallel state claiming CP and nothing built.

    The ops singleton is stubbed rather than installed: ``apply_ops_config``
    writes a process-global that would outlive the test.
    """
    parallel_state = SimpleNamespace(cp_enabled=cp_enabled, sp_enabled=cp_enabled, global_rank=0)
    ops_config = SimpleNamespace(attn_implementation="eager")
    with (
        patch("veomni.models.auto.get_parallel_state", return_value=parallel_state),
        # The gate asks this first, so that an uninitialized process answers "CP
        # off" rather than constructing a state. Stubbing a state means one is
        # installed as far as these tests are concerned.
        patch("veomni.models.auto.is_parallel_state_initialized", return_value=True),
        patch("veomni.models.auto.get_loader", return_value=loader),
        patch("veomni.ops.config.singleton.get_ops_config", return_value=ops_config),
    ):
        build_foundation_model(config_path=build_config(config_path), init_device="meta")


def test_context_parallel_admits_deepseek_v4():
    """The one model whose forward implements CP is built as usual."""
    loader = _StubLoader()
    _build("tests/toy_config/deepseek_v4_toy", cp_enabled=True, loader=loader)
    assert loader.calls == 1


def test_context_parallel_rejects_another_model_type():
    """Any other model type is refused, by name, before anything is constructed."""
    loader = _StubLoader()
    with pytest.raises(NotImplementedError, match="Context parallelism is not implemented") as raised:
        _build("tests/toy_config/qwen3_toy", cp_enabled=True, loader=loader)
    message = str(raised.value)
    assert "'qwen3'" in message, message
    # Naming the way out matters as much as the refusal: this fires on a
    # configuration that used to be refused by ``ParallelState`` itself.
    assert "cp_size=1" in message and "ulysses_size" in message, message
    assert loader.calls == 0, "the gate must refuse before the model is constructed"


def test_without_context_parallel_any_model_type_is_admitted():
    """The gate is keyed on CP, not on the model: ``cp_size=1`` changes nothing."""
    loader = _StubLoader()
    _build("tests/toy_config/qwen3_toy", cp_enabled=False, loader=loader)
    assert loader.calls == 1


def test_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch):
    """The gate must not *construct* a parallel state just to ask about CP.

    Every test above patches ``get_parallel_state``, which hides what the real one
    does when nothing installed a state: it builds a default single-process
    ``ParallelState``, whose ``dp_size=1`` contradicts a multi-rank world and
    raises on the topology product check. Since the gate runs for every model,
    that turned any multi-rank process which builds a model without installing a
    state into a hard failure -- as ``tests/lora/test_moe_lora_ep2.py`` did,
    spawning two ranks and calling ``build_foundation_model`` directly.
    """
    from veomni.distributed import parallel_state as parallel_state_module
    from veomni.models.auto import check_context_parallel_supported

    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    # A real two-rank world, which is what makes the default state invalid.
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)

    check_context_parallel_supported(build_config("tests/toy_config/qwen3_toy"))
