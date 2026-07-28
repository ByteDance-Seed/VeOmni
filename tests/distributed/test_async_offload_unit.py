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
from veomni.distributed.async_offload import GetCnt, apply_async_activation_offload, get_offload_modules


def test_offload_config_allows_no_modules_for_auto_discovery():
    cfg = OffloadConfig(enable_async_activation=True)
    assert cfg.enable_async_activation is True
    assert cfg.activation_offload_modules == []


def test_offload_config_rejects_sync_and_async_activation_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OffloadConfig(
            enable_activation=True,
            enable_async_activation=True,
            activation_offload_modules=["model.layers.{*}"],
        )


def test_get_offload_modules_glob_is_segment_aware():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    matched = get_offload_modules(Toy(), ["model.layers.*"])

    assert [item[0] for item in matched] == ["model.layers.0", "model.layers.1"]


def test_get_cnt_unique_keys_across_second_pass():
    """Second forward over the same layer indices must not collide keys."""
    cnt = GetCnt()
    first = [cnt.get_cnt(i)[0] for i in range(3)]
    second = [cnt.get_cnt(i)[0] for i in range(3)]
    assert first == ["0_0", "1_0", "2_0"]
    assert second == ["0_1", "1_1", "2_1"]
    assert set(first).isdisjoint(second)


def test_get_prefetch_keys_use_previous_offloaded_layer_and_its_tensor_count():
    cnt = GetCnt()
    cnt.get_cnt(2)
    cnt.get_cnt(2)
    cnt.get_cnt(5)

    assert cnt.get_prefetch_keys(5) == ["2_0", "2_1"]


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


def test_apply_async_activation_offload_rejects_unmatched_patterns():
    model = nn.Sequential(nn.Linear(4, 4))

    with pytest.raises(ValueError, match="did not match any model modules"):
        apply_async_activation_offload(model, ["model.layers.{*}"])


def test_apply_async_activation_offload_auto_discovers_no_split_modules():
    class DecoderLayer(nn.Module):
        pass

    class Toy(nn.Module):
        _no_split_modules = ["DecoderLayer"]

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])

    model = Toy()
    apply_async_activation_offload(model, [])

    assert [layer._veomni_offload_layer_idx for layer in model.layers] == [0, 1]
    assert all(layer._veomni_offload_depth == 2 for layer in model.layers)


@pytest.mark.parametrize(
    "no_split_modules, error_match",
    [
        (None, "requires model._no_split_modules"),
        (["MissingDecoderLayer"], "did not match any modules"),
    ],
)
def test_apply_async_activation_offload_auto_discovery_fails_closed(no_split_modules, error_match):
    model = nn.Sequential(nn.Linear(4, 4))
    if no_split_modules is not None:
        model._no_split_modules = no_split_modules

    with pytest.raises(ValueError, match=error_match):
        apply_async_activation_offload(model, [])
