# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Pure-Python unit tests for ``veomni.trainer.omni.context`` (no GPU / distributed).

Covers the catalog + args-driven active-set wiring at setup, the composer
machinery (phase/mode gating, nesting order, provider-None skip, ExitStack unwind,
duplicate rejection), the built-in contexts' declared phases/modes, and the
built-in builders' wiring + build behavior (incl. the grad-accum reshard cascade).
"""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from veomni.distributed.offloading import custom_save_on_cpu
from veomni.trainer.omni import context
from veomni.trainer.omni.context import (
    VEOMNI_CONTEXT_CATALOG,
    register_veomni_context,
    setup_veomni_context,
    veomni_context,
)


# ── Fakes ───────────────────────────────────────────────────────────────────────


class _RecordingCM:
    """Context manager that appends ``(event, label)`` to a shared log."""

    def __init__(self, label, log, raise_on_enter=False):
        self.label = label
        self.log = log
        self.raise_on_enter = raise_on_enter

    def __enter__(self):
        if self.raise_on_enter:
            raise RuntimeError(f"boom:{self.label}")
        self.log.append(("enter", self.label))
        return self

    def __exit__(self, *exc):
        self.log.append(("exit", self.label))
        return False


class _FakeModuleTrainer:
    def __init__(self):
        self.reshard_calls = []

    def _model_reshard(self, reshard):
        self.reshard_calls.append(reshard)


def _args(*, offload=False, gc=False, batch_invariant=False):
    """Minimal stand-in for the nested OmniArguments the builders read."""
    return SimpleNamespace(
        train=SimpleNamespace(
            accelerator=SimpleNamespace(
                offload_config=SimpleNamespace(enable_activation=offload, activation_gpu_limit=0.0)
            ),
            gradient_checkpointing=SimpleNamespace(enable=gc),
            enable_batch_invariant_mode=batch_invariant,
        )
    )


def _module_trainers(n=2):
    return OrderedDict((f"m{i}", _FakeModuleTrainer()) for i in range(n))


@contextmanager
def _isolated_catalog():
    """Swap in an empty catalog + active set, restore both afterwards."""
    saved_catalog = OrderedDict(context.VEOMNI_CONTEXT_CATALOG)
    saved_active = OrderedDict(context._ACTIVE_CONTEXTS)
    context.VEOMNI_CONTEXT_CATALOG.clear()
    context._ACTIVE_CONTEXTS.clear()
    try:
        yield
    finally:
        context.VEOMNI_CONTEXT_CATALOG.clear()
        context.VEOMNI_CONTEXT_CATALOG.update(saved_catalog)
        context._ACTIVE_CONTEXTS.clear()
        context._ACTIVE_CONTEXTS.update(saved_active)


@contextmanager
def _saved_active():
    """Save/restore the active set (for tests that call setup against the real catalog)."""
    saved_active = OrderedDict(context._ACTIVE_CONTEXTS)
    try:
        yield
    finally:
        context._ACTIVE_CONTEXTS.clear()
        context._ACTIVE_CONTEXTS.update(saved_active)


# ── Composer machinery (through setup + the isolated catalog) ─────────────────────


def test_nesting_order_matches_catalog_order():
    with _isolated_catalog():
        log = []

        @register_veomni_context("a", phases=("forward",), modes=("train",))
        def _a(args, module_trainers):
            return lambda **kwargs: _RecordingCM("a", log)

        @register_veomni_context("b", phases=("forward",), modes=("train",))
        def _b(args, module_trainers):
            return lambda **kwargs: _RecordingCM("b", log)

        setup_veomni_context(_args())
        with veomni_context("forward", "train"):
            pass

        # catalog order = outer→inner on enter, reversed on exit
        assert log == [("enter", "a"), ("enter", "b"), ("exit", "b"), ("exit", "a")]


def test_phase_mode_gating_skips_non_matching_contexts():
    with _isolated_catalog():
        log = []

        @register_veomni_context("fwd_train", phases=("forward",), modes=("train",))
        def _ft(args, module_trainers):
            return lambda **kwargs: _RecordingCM("fwd_train", log)

        setup_veomni_context(_args())
        # matching phase + mode → entered
        with veomni_context("forward", "train"):
            pass
        # wrong phase → skipped
        with veomni_context("backward", "train"):
            pass
        # wrong mode → skipped
        with veomni_context("forward", "offline_cache"):
            pass

        assert log == [("enter", "fwd_train"), ("exit", "fwd_train")]


def test_builder_returning_none_is_not_wired():
    with _isolated_catalog():
        log = []

        @register_veomni_context("off", phases=("forward",), modes=("train",))
        def _off(args, module_trainers):
            return None  # args-driven: not wired into the run

        @register_veomni_context("on", phases=("forward",), modes=("train",))
        def _on(args, module_trainers):
            return lambda **kwargs: _RecordingCM("on", log)

        setup_veomni_context(_args())
        assert "off" not in context._ACTIVE_CONTEXTS
        assert "on" in context._ACTIVE_CONTEXTS
        with veomni_context("forward", "train"):
            pass
        assert log == [("enter", "on"), ("exit", "on")]


def test_provider_returning_none_is_skipped():
    with _isolated_catalog():
        log = []

        @register_veomni_context("skipped", phases=("forward",), modes=("train",))
        def _skipped(args, module_trainers):
            return lambda **kwargs: None  # wired, but skips this step

        @register_veomni_context("kept", phases=("forward",), modes=("train",))
        def _kept(args, module_trainers):
            return lambda **kwargs: _RecordingCM("kept", log)

        setup_veomni_context(_args())
        with veomni_context("forward", "train"):
            pass

        assert log == [("enter", "kept"), ("exit", "kept")]


def test_exitstack_unwinds_entered_contexts_on_error():
    with _isolated_catalog():
        log = []

        @register_veomni_context("a", phases=("forward",), modes=("train",))
        def _a(args, module_trainers):
            return lambda **kwargs: _RecordingCM("a", log)

        @register_veomni_context("b", phases=("forward",), modes=("train",))
        def _b(args, module_trainers):
            return lambda **kwargs: _RecordingCM("b", log, raise_on_enter=True)

        setup_veomni_context(_args())
        with pytest.raises(RuntimeError, match="boom:b"):
            with veomni_context("forward", "train"):
                pass

        # "a" was entered then unwound; "b" never entered (raised in __enter__).
        assert log == [("enter", "a"), ("exit", "a")]


def test_duplicate_registration_rejected():
    with _isolated_catalog():

        @register_veomni_context("dup", phases=("forward",), modes=("train",))
        def _first(args, module_trainers):
            return None

        with pytest.raises(ValueError, match="Duplicate"):

            @register_veomni_context("dup", phases=("forward",), modes=("train",))
            def _second(args, module_trainers):
                return None


def test_setup_passes_args_and_module_trainers_to_builders():
    with _isolated_catalog():
        seen = []

        @register_veomni_context("spy", phases=("forward",), modes=("train",))
        def _spy(args, module_trainers):
            seen.append((args, module_trainers))
            return None

        args = _args(batch_invariant=True)
        mts = _module_trainers()
        setup_veomni_context(args, mts)

        # the builder is called once at setup with the exact args + module_trainers.
        assert seen == [(args, mts)]


# ── Built-in contexts: declared phases / modes (gating metadata) ─────────────────


@pytest.mark.parametrize(
    ("name", "phases", "modes"),
    [
        ("no_grad", {"forward"}, {"offline_cache"}),
        ("activation_offloading_forward", {"forward"}, {"train", "offline_cache"}),
        ("activation_offloading_backward", {"backward"}, {"train", "offline_cache"}),
        ("batch_invariant", {"forward", "backward"}, {"train", "offline_cache"}),
        ("model_reshard", {"forward"}, {"train", "offline_cache"}),
    ],
)
def test_builtin_context_declared_phases_and_modes(name, phases, modes):
    entry = VEOMNI_CONTEXT_CATALOG[name]
    assert set(entry.phases) == phases
    assert set(entry.modes) == modes


# ── Built-in builders: wiring decision + provider build behavior ──────────────────


def test_no_grad_always_wired_and_builds_no_grad():
    provider = context._build_no_grad(_args())
    assert provider is not None
    assert isinstance(provider(), torch.no_grad)


def test_activation_offloading_wired_and_built_once_when_enabled():
    # offload on, grad-ckpt off → fwd is custom_save_on_cpu, bwd is nullcontext.
    fwd_provider = context._build_activation_offloading_forward(_args(offload=True, gc=False))
    bwd_provider = context._build_activation_offloading_backward(_args(offload=True, gc=False))
    assert isinstance(fwd_provider(), custom_save_on_cpu)
    assert isinstance(bwd_provider(), nullcontext)
    # built once at setup, reused every step (same instance across provider calls).
    assert fwd_provider() is fwd_provider()
    assert bwd_provider() is bwd_provider()


def test_activation_offloading_not_wired_when_disabled():
    assert context._build_activation_offloading_forward(_args(offload=False)) is None
    assert context._build_activation_offloading_backward(_args(offload=False)) is None


def test_batch_invariant_wired_only_when_enabled():
    provider = context._build_batch_invariant(_args(batch_invariant=True))
    assert provider is not None
    # generator CM (single-use), so a fresh instance is built each step.
    assert provider() is not provider()
    assert context._build_batch_invariant(_args(batch_invariant=False)) is None


def test_model_reshard_not_wired_without_module_trainers():
    assert context._build_model_reshard(_args(), {}) is None


def test_model_reshard_provider_absent_without_accumulation():
    provider = context._build_model_reshard(_args(), _module_trainers())
    assert provider is not None
    assert provider(micro_step=0, num_micro_steps=1) is None


@pytest.mark.parametrize(
    ("micro_step", "expected"),
    [
        (0, False),  # first micro-step keeps params gathered
        (2, True),  # last micro-step reshards
    ],
)
def test_model_reshard_cascades_bool_on_first_and_last(micro_step, expected):
    mts = _module_trainers(n=2)
    provider = context._build_model_reshard(_args(), mts)
    cm = provider(micro_step=micro_step, num_micro_steps=3)
    assert cm is not None
    with cm:
        pass
    for mt in mts.values():
        assert mt.reshard_calls == [expected]


def test_model_reshard_middle_step_toggles_nothing():
    mts = _module_trainers(n=2)
    provider = context._build_model_reshard(_args(), mts)
    cm = provider(micro_step=1, num_micro_steps=3)  # middle
    assert cm is not None
    with cm:
        pass
    for mt in mts.values():
        assert mt.reshard_calls == []


# ── End-to-end through the real catalog + setup ──────────────────────────────────


def test_setup_wires_only_enabled_contexts_from_args():
    with _saved_active():
        setup_veomni_context(_args(offload=False, batch_invariant=False), _module_trainers())
        active = context._ACTIVE_CONTEXTS
        assert "activation_offloading_forward" not in active
        assert "activation_offloading_backward" not in active
        assert "batch_invariant" not in active
        # reshard is wired (module-trainers present); no_grad always wired.
        assert "model_reshard" in active
        assert "no_grad" in active

        setup_veomni_context(_args(offload=True, batch_invariant=True), _module_trainers())
        active = context._ACTIVE_CONTEXTS
        assert "activation_offloading_forward" in active
        assert "activation_offloading_backward" in active
        assert "batch_invariant" in active


def test_composer_reshard_cascade_on_train_forward():
    with _saved_active():
        mts = _module_trainers(n=2)  # offloading + batch-invariant off
        setup_veomni_context(_args(), mts)
        with veomni_context("forward", "train", micro_step=0, num_micro_steps=3):
            pass
        for mt in mts.values():
            assert mt.reshard_calls == [False]


def test_setup_all_off_wires_only_no_grad():
    with _saved_active():
        # everything off, no module-trainers → only the always-on no_grad is wired.
        setup_veomni_context(_args(offload=False, batch_invariant=False))
        # only no_grad is wired; reshard/offloading/batch-invariant are absent.
        assert set(context._ACTIVE_CONTEXTS) == {"no_grad"}
