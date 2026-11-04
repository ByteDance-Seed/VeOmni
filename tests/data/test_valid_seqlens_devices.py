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

"""Tail-segment extraction must agree on CPU and Ascend NPU."""

import json
import os
from pathlib import Path

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils.device import IS_NPU_AVAILABLE
from veomni.utils.seqlen_pos_transform_utils import valid_seqlens_from_cu_seqlens


_CASES = [
    ([0, 4, 7, 8, 9, 10], None, [4, 3]),
    ([0, 4, 5, 6, 7, 8], 3, [4, 1]),
    ([0, 4, 5], 0, [4, 1]),
    ([0, 1, 3], None, [1, 2]),
    ([0, 1, 2], None, []),
    ([0], None, []),
]


def _npu_test_eligible():
    """Honor the CI lease; on NPU CI leave device availability to the test assertion."""
    capabilities_file = os.environ.get("VEOMNI_CI_CAPABILITIES")
    if capabilities_file:
        return json.loads(Path(capabilities_file).read_text())["backend"] == "npu"
    return IS_NPU_AVAILABLE and torch.npu.is_available()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(("backend", "expected"), [("cpu", False), ("mlu", False), ("npu", True)])
def test_npu_eligibility_respects_ci_backend(monkeypatch, tmp_path, backend, expected):
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps({"backend": backend, "devices": []}))
    monkeypatch.setenv("VEOMNI_CI_CAPABILITIES", str(capabilities))
    assert _npu_test_eligible() is expected


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("cu,tail,expected", _CASES)
def test_valid_seqlens_cpu(cu, tail, expected):
    value = torch.tensor(cu, dtype=torch.int32)
    result = valid_seqlens_from_cu_seqlens(value, tail_padding_length=tail)
    assert result.tolist() == expected
    assert result.dtype == torch.int32


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
@pytest.mark.skipif(not _npu_test_eligible(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("cu,tail,expected", _CASES)
def test_valid_seqlens_ascend(cu, tail, expected):
    assert IS_NPU_AVAILABLE and torch.npu.is_available(), "Ascend CI must provide an available NPU"
    value = torch.tensor(cu, dtype=torch.int32, device="npu")
    result = valid_seqlens_from_cu_seqlens(value, tail_padding_length=tail)
    assert result.cpu().tolist() == expected
    assert result.dtype == torch.int32
