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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from veomni.utils.count_flops import VeomniFlopsCounter


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _load_toy_config(config_dir):
    with Path(config_dir, "config.json").open(encoding="utf-8") as fp:
        return _to_namespace(json.load(fp))


@pytest.fixture(autouse=True)
def mock_device_flops():
    with patch("veomni.utils.count_flops.get_device_flops", return_value=1000.0):
        yield


@pytest.fixture
def qwen3_5_counter():
    config = _load_toy_config("tests/toy_config/qwen3_5_toy")
    return VeomniFlopsCounter(config)


@pytest.fixture
def qwen3_5_moe_counter():
    config = _load_toy_config("tests/toy_config/qwen3_5_moe_toy")
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
        # Embedding lookup is not a matmul; only lm_head contributes vocab_size * hidden_size.
        assert flops == pytest.approx(105.419032756224, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        # Embedding lookup is not a matmul; only lm_head contributes vocab_size * hidden_size.
        assert flops == pytest.approx(107.650266169344, rel=1e-9)


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
        # Embedding lookup is not a matmul; only lm_head contributes vocab_size * hidden_size.
        assert flops == pytest.approx(16.68192141312, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        # Embedding lookup is not a matmul; only lm_head contributes vocab_size * hidden_size.
        assert flops == pytest.approx(18.847925010432, rel=1e-9)
