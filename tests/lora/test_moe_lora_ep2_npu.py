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

"""NPU entry points for the shared MoE-LoRA EP=2 integration coverage.

The original suite is kept unchanged as the GPU contract. These wrappers reuse
its trainer harness and assertions while selecting the Ascend fused MoE backend.
"""

import pytest

from veomni.utils.device import IS_NPU_AVAILABLE

from . import test_moe_lora_ep2 as _base


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="MoE-LoRA EP=2 NPU tests require torch_npu")

toy_base_dir = _base.toy_base_dir

_FUSED_NPU_OPS_OVERRIDE = "--model.ops_implementation.moe_implementation=fused_npu"


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_ep2_trainer_integration_npu(tmp_path, toy_base_dir, mode, monkeypatch):
    monkeypatch.setattr(_base, "_FUSED_OPS_OVERRIDE", _FUSED_NPU_OPS_OVERRIDE)
    _base.test_ep2_trainer_integration(tmp_path, toy_base_dir, mode)


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_moe_lora_ep_save_load_parallel_align_npu(tmp_path, toy_base_dir, mode, monkeypatch):
    monkeypatch.setattr(_base, "_FUSED_OPS_OVERRIDE", _FUSED_NPU_OPS_OVERRIDE)
    _base.test_moe_lora_ep_save_load_parallel_align(tmp_path, toy_base_dir, mode)


def test_independent_reset_lora_parameters_under_ep_shard_npu():
    _base.test_independent_reset_lora_parameters_under_ep_shard()
