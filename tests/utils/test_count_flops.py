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

from unittest.mock import patch

import pytest
from transformers import AutoConfig

from veomni.utils.count_flops import VeomniFlopsCounter


@pytest.fixture(autouse=True)
def mock_device_flops():
    with patch("veomni.utils.count_flops.get_device_flops", return_value=1000.0):
        yield


@pytest.fixture
def qwen3_5_counter():
    config = AutoConfig.from_pretrained("tests/toy_config/qwen3_5_toy")
    return VeomniFlopsCounter(config)


@pytest.fixture
def qwen3_5_moe_counter():
    config = AutoConfig.from_pretrained("tests/toy_config/qwen3_5_moe_toy")
    return VeomniFlopsCounter(config)


class TestQwen35Flops:
    def test_text_only(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops > 0

    def test_with_vit(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        text_flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        vit_flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        assert vit_flops > text_flops

    def test_numerical(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops == pytest.approx(136.375009542144, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        assert flops == pytest.approx(138.606242955264, rel=1e-9)


class TestQwen35MoeFlops:
    def test_text_only(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops > 0

    def test_with_vit(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        text_flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        vit_flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        assert vit_flops > text_flops

    def test_numerical(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops == pytest.approx(29.08363948032, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        assert flops == pytest.approx(31.249643077632, rel=1e-9)
