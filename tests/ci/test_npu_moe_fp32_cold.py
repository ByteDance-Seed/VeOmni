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

"""FSDP2 + expert-parallel optimizer/DCP cold-recovery regression on Ascend."""

import os

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ci._npu_moe_fp32_cold_worker import run
from veomni.utils.device import IS_NPU_AVAILABLE


def _assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    else:
        assert left == right


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.parametrize("ep", [1, 2])
def test_native_fsdp_expert_fp32_dcp_cold_resume(tmp_path, ep):
    if not IS_NPU_AVAILABLE or torch.npu.device_count() < 2:
        if os.environ.get("VEOMNI_REQUIRE_NPU_TESTS") == "1":
            pytest.fail("VEOMNI_REQUIRE_NPU_TESTS=1 requires two native NPU devices")
        pytest.skip("requires two native Ascend devices")
    from tests.tools.launch_utils import spawn_with_rendezvous

    for phase in ("fresh", "resume"):
        spawn_with_rendezvous(run, args=(phase, str(tmp_path), ep), nprocs=2)
    for rank in range(2):
        fresh = torch.load(tmp_path / f"ep{ep}-fresh-rank{rank}.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(tmp_path / f"ep{ep}-resume-rank{rank}.pt", map_location="cpu", weights_only=True)
        _assert_equal(fresh, resumed)
