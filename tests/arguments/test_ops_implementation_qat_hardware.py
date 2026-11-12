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

"""``qat_implementation='fp8_blockwise'`` is accepted exactly where an FP8 quantizer runs."""

from __future__ import annotations

import pytest
import torch

import veomni.utils.device as device
from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import OpsImplementationConfig


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _set_gpu(monkeypatch: pytest.MonkeyPatch, *, hip: str | None, arch: str | None, capability: int) -> None:
    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(device, "get_rocm_gcn_arch", lambda _device=None: arch)
    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda _device=None: capability)


@pytest.mark.parametrize(
    ("hip", "arch", "capability"),
    ((None, None, 90), ("7.14", "gfx942", 0), ("7.14", "gfx950", 0)),
    ids=("sm90", "gfx942", "gfx950"),
)
def test_fp8_blockwise_is_accepted_where_an_fp8_quantizer_runs(monkeypatch, hip, arch, capability):
    _set_gpu(monkeypatch, hip=hip, arch=arch, capability=capability)
    assert OpsImplementationConfig(qat_implementation="fp8_blockwise").qat_implementation == "fp8_blockwise"


@pytest.mark.parametrize(
    ("hip", "arch", "capability"),
    ((None, None, 89), ("7.14", "gfx90a", 0)),
    ids=("sm89", "gfx90a"),
)
def test_fp8_blockwise_is_rejected_elsewhere(monkeypatch, hip, arch, capability):
    _set_gpu(monkeypatch, hip=hip, arch=arch, capability=capability)
    with pytest.raises(ValueError, match="gfx942/gfx950"):
        OpsImplementationConfig(qat_implementation="fp8_blockwise")
