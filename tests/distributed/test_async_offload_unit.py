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

"""CPU-only unit tests for MindSpeed-style async activation offload helpers."""

import pytest
import torch.nn as nn

from veomni.arguments.arguments_types import OffloadConfig
from veomni.distributed.async_offload import GetCnt, get_offload_modules
from veomni.utils.module_match import module_name_match


def test_offload_config_requires_modules_when_async_enabled():
    with pytest.raises(ValueError, match="activation_offload_modules"):
        OffloadConfig(enable_async_activation=True)

    cfg = OffloadConfig(
        enable_async_activation=True,
        activation_offload_modules=["model.layers.{*}"],
    )
    assert cfg.enable_async_activation is True
    assert cfg.activation_offload_modules == ["model.layers.{*}"]


def test_module_name_match_glob():
    assert module_name_match("model.layers.*", "model.layers.0")
    assert module_name_match("model.layers.*", "model.layers.12")
    assert not module_name_match("model.layers.*", "model.norm")


def test_get_cnt_unique_keys_across_second_pass():
    """Second forward over the same layer indices must not collide keys."""
    cnt = GetCnt()
    first = [cnt.get_cnt(i)[0] for i in range(3)]
    second = [cnt.get_cnt(i)[0] for i in range(3)]
    assert first == ["0_0", "1_0", "2_0"]
    assert second == ["0_1", "1_1", "2_1"]
    assert set(first).isdisjoint(second)


def test_get_offload_modules_brace_star_expands_sequential():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    model = Toy()
    matched = get_offload_modules(model, ["model.layers.{*}"])
    names = [item[0] for item in matched]
    assert names == ["model.layers.0", "model.layers.1", "model.layers.2"]
    # depth field is rewritten to total offload layer count
    assert all(item[-1] == 3 for item in matched)
    assert [item[2] for item in matched] == [0, 1, 2]
