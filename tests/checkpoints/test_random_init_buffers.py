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

"""Config-derived buffers must survive random initialization from meta."""

import pytest
import torch
from torch import nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.torch_parallelize import _materialize_and_load_weights
from veomni.models.checkpoint.weights import init_empty_weights
from veomni.utils.device import get_device_id, get_device_type, get_torch_device


class BufferModel(nn.Module):
    def __init__(self, overwrite=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2))
        self.register_buffer("persistent_value", torch.tensor([2.0, 5.0]))
        self.register_buffer("runtime_value", torch.tensor([3.0, 7.0]), persistent=False)
        self.register_buffer("optional_value", None)
        self.child = nn.Module()
        self.child.register_buffer("persistent_value", torch.tensor([11.0, 13.0]))
        self.overwrite = overwrite
        self.init_calls = 0

    def init_weights(self):
        self.init_calls += 1
        with torch.no_grad():
            self.weight.fill_(19)
            if self.overwrite:
                self.child.persistent_value.fill_(23)


def _check_materialization(device, *, overwrite, skip_hf):
    with init_empty_weights():
        model = BufferModel(overwrite=overwrite)
    _materialize_and_load_weights(
        model,
        None,
        str(device),
        should_skip_hf_weight_load=skip_hf,
        is_peft_model=False,
        adapter_path=None,
        broadcast_from_rank0=False,
    )
    assert model.weight.device.type == device.type
    assert model.persistent_value.device.type == device.type
    assert model.runtime_value.device.type == device.type
    assert model.child.persistent_value.device.type == device.type
    assert model.optional_value is None
    torch.testing.assert_close(model.runtime_value.cpu(), torch.tensor([3.0, 7.0]), rtol=0, atol=0)
    if skip_hf:
        # A later DCP load is authoritative for persistent buffers and params.
        assert model.init_calls == 0
    else:
        assert model.init_calls == 1
        torch.testing.assert_close(model.weight.cpu(), torch.full((2,), 19.0), rtol=0, atol=0)
        torch.testing.assert_close(model.persistent_value.cpu(), torch.tensor([2.0, 5.0]), rtol=0, atol=0)
        child = [23.0, 23.0] if overwrite else [11.0, 13.0]
        torch.testing.assert_close(model.child.persistent_value.cpu(), torch.tensor(child), rtol=0, atol=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("skip_hf", [False, True])
def test_random_init_and_resume_buffer_materialization_cpu(overwrite, skip_hf):
    _check_materialization(torch.device("cpu"), overwrite=overwrite, skip_hf=skip_hf)


@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator"))
def test_random_init_and_resume_buffer_materialization_accelerator():
    device_type = get_device_type()
    if device_type == "cpu":
        pytest.skip("requires an accelerator runtime")
    assert get_torch_device().is_available(), "selected accelerator runtime is unavailable"
    device = torch.device(device_type, get_device_id())
    for overwrite in (False, True):
        for skip_hf in (False, True):
            _check_materialization(device, overwrite=overwrite, skip_hf=skip_hf)
