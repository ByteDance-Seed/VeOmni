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

"""Shared fixtures for hardware-independent ``models_kernel`` consume tests."""

from __future__ import annotations

import pytest

from veomni.ops import VeomniOp
from veomni.ops import registry as op_registry
from veomni.ops.platform import NvidiaGpuPlatform


@pytest.fixture
def available_nvidia_ops(monkeypatch):
    """Make static NVIDIA/package gates pass without executing a GPU kernel."""
    previous_handles = dict(VeomniOp._intern)
    VeomniOp._intern.clear()
    monkeypatch.setattr(op_registry, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(NvidiaGpuPlatform, "matches", lambda self: True)
    monkeypatch.setattr(op_registry, "is_package_available", lambda _package: True)
    yield
    VeomniOp._intern.clear()
    VeomniOp._intern.update(previous_handles)
