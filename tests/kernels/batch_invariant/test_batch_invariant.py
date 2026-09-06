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

"""Lifecycle tests for the process-wide batch-invariant ATen patch."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from veomni.kernels.batch_invariant import patch as batch_patch


class _FakeLibrary:
    instances: list[_FakeLibrary] = []
    fail_on_impl: int | None = None

    def __init__(self, namespace: str, kind: str):
        self.namespace = namespace
        self.kind = kind
        self.implementations = []
        self.destroyed = False
        self.instances.append(self)

    def impl(self, op_name, implementation, dispatch_key):
        self.implementations.append((op_name, implementation, dispatch_key))
        if len(self.implementations) == self.fail_on_impl:
            raise RuntimeError("registration failed")

    def _destroy(self):
        self.destroyed = True


@pytest.fixture
def fake_library(monkeypatch):
    batch_patch.disable_batch_invariant_mode()
    _FakeLibrary.instances.clear()
    _FakeLibrary.fail_on_impl = None

    implementations = tuple(
        (name, object()) for name in ("aten::mm", "aten::addmm", "aten::_log_softmax", "aten::mean.dim")
    )
    monkeypatch.setattr(batch_patch, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(batch_patch, "_batch_invariant_implementations", lambda: implementations)
    monkeypatch.setattr(batch_patch.torch.library, "Library", _FakeLibrary)
    monkeypatch.setattr(
        batch_patch.torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="test_accelerator"),
    )

    yield implementations

    batch_patch.disable_batch_invariant_mode()


def test_import_does_not_load_triton_implementations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import veomni.kernels.batch_invariant; "
                "assert 'veomni.kernels.batch_invariant.triton' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_enable_is_idempotent_and_registers_all_implementations(fake_library):
    batch_patch.enable_batch_invariant_mode()
    batch_patch.enable_batch_invariant_mode()

    assert batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 1
    library = _FakeLibrary.instances[0]
    assert library.namespace == "aten"
    assert library.kind == "IMPL"
    assert library.implementations == [
        (name, implementation, "TEST_ACCELERATOR") for name, implementation in fake_library
    ]

    batch_patch.disable_batch_invariant_mode()
    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert library.destroyed


def test_context_restores_state_after_exception(fake_library):
    with pytest.raises(RuntimeError, match="body failed"):
        with batch_patch.set_batch_invariant_mode():
            assert batch_patch.is_batch_invariant_mode_enabled()
            raise RuntimeError("body failed")

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert _FakeLibrary.instances[0].destroyed


def test_nested_enabled_context_reuses_outer_install(fake_library):
    with batch_patch.set_batch_invariant_mode():
        outer_library = _FakeLibrary.instances[0]
        with batch_patch.set_batch_invariant_mode(True):
            assert batch_patch.is_batch_invariant_mode_enabled()
            assert len(_FakeLibrary.instances) == 1
        assert batch_patch.is_batch_invariant_mode_enabled()
        assert not outer_library.destroyed

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert outer_library.destroyed


def test_nested_disabled_context_restores_outer_install(fake_library):
    with batch_patch.set_batch_invariant_mode(True):
        outer_library = _FakeLibrary.instances[0]
        with batch_patch.set_batch_invariant_mode(False):
            assert not batch_patch.is_batch_invariant_mode_enabled()
            assert outer_library.destroyed

        assert batch_patch.is_batch_invariant_mode_enabled()
        restored_library = _FakeLibrary.instances[-1]
        assert restored_library is not outer_library
        assert not restored_library.destroyed

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert restored_library.destroyed


def test_context_restores_manual_enable(fake_library):
    batch_patch.enable_batch_invariant_mode()
    manual_library = _FakeLibrary.instances[0]

    with batch_patch.set_batch_invariant_mode(False):
        assert not batch_patch.is_batch_invariant_mode_enabled()
        assert manual_library.destroyed

    assert batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 2


def test_context_is_noop_without_cuda(fake_library, monkeypatch):
    monkeypatch.setattr(batch_patch, "IS_CUDA_AVAILABLE", False)

    with batch_patch.set_batch_invariant_mode(True):
        assert not batch_patch.is_batch_invariant_mode_enabled()

    assert _FakeLibrary.instances == []


def test_failed_registration_destroys_partial_library(fake_library):
    _FakeLibrary.fail_on_impl = 3

    with pytest.raises(RuntimeError, match="registration failed"):
        batch_patch.enable_batch_invariant_mode()

    assert not batch_patch.is_batch_invariant_mode_enabled()
    assert len(_FakeLibrary.instances) == 1
    assert _FakeLibrary.instances[0].destroyed
