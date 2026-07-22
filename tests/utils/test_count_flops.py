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
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from veomni.utils.count_flops import VeomniFlopsCounter, get_device_flops


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _load_toy_config(config_dir):
    with Path(config_dir, "config.json").open(encoding="utf-8") as fp:
        return _to_namespace(json.load(fp))


@pytest.fixture
def mock_device_flops():
    with patch("veomni.utils.count_flops.get_device_flops", return_value=1000.0):
        yield


def test_b300_device_flops():
    with patch("veomni.utils.count_flops.get_device_name", return_value="NVIDIA B300"):
        assert get_device_flops() == 2250.0


def test_gb300_device_flops():
    with patch("veomni.utils.count_flops.get_device_name", return_value="NVIDIA GB300"):
        assert get_device_flops() == 2500.0


@pytest.fixture
def qwen3_5_counter():
    config = _load_toy_config("tests/toy_config/qwen3_5_toy")
    return VeomniFlopsCounter(config)


@pytest.fixture
def qwen3_5_moe_counter():
    config = _load_toy_config("tests/toy_config/qwen3_5_moe_toy")
    return VeomniFlopsCounter(config)


@pytest.fixture
def gpt_oss_config():
    return _load_toy_config("tests/toy_config/gpt_oss_toy")


@pytest.fixture
def gpt_oss_counter(gpt_oss_config):
    return VeomniFlopsCounter(gpt_oss_config)


@pytest.fixture
def deepseek_v4_config():
    config = _load_toy_config("tests/toy_config/deepseek_v4_toy")
    config.compress_rates = vars(config.compress_rates)
    config.layer_types = [
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    return config


@pytest.fixture
def deepseek_v4_counter(deepseek_v4_config):
    return VeomniFlopsCounter(deepseek_v4_config)


class TestQwen35Flops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

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
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

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


class TestGptOssFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_numerical(self, gpt_oss_counter):
        batch_seqlens = [12, 5]
        flops, promised_flops = gpt_oss_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops == pytest.approx(0.000326931456, rel=1e-9)
        assert promised_flops == 1000.0

    def test_sliding_attention_reduces_quadratic_flops(self, gpt_oss_config):
        batch_seqlens = [12, 5]
        mixed_counter = VeomniFlopsCounter(gpt_oss_config)
        mixed_flops, _ = mixed_counter.estimate_flops(batch_seqlens, delta_time=1.0)

        full_config = deepcopy(gpt_oss_config)
        full_config.layer_types = ["full_attention"] * full_config.num_hidden_layers
        full_counter = VeomniFlopsCounter(full_config)
        full_flops, _ = full_counter.estimate_flops(batch_seqlens, delta_time=1.0)

        assert full_flops > mixed_flops


class TestDeepseekV4Flops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_numerical(self, deepseek_v4_counter):
        flops, promised_flops = deepseek_v4_counter.estimate_flops([12, 5], delta_time=1.0)

        assert flops == pytest.approx(0.000264658944, rel=1e-9)
        assert promised_flops == 1000.0

    def test_csa_topk_caps_main_attention_but_not_indexer(self, deepseek_v4_config):
        batch_seqlens = [256]
        baseline_flops, _ = VeomniFlopsCounter(deepseek_v4_config).estimate_flops(batch_seqlens, delta_time=1.0)

        smaller_topk_config = deepcopy(deepseek_v4_config)
        smaller_topk_config.index_topk = 4
        smaller_topk_flops, _ = VeomniFlopsCounter(smaller_topk_config).estimate_flops(batch_seqlens, delta_time=1.0)

        assert smaller_topk_flops < baseline_flops

    def test_shared_experts_scale_moe_flops(self, deepseek_v4_config):
        batch_seqlens = [64]
        baseline_flops, _ = VeomniFlopsCounter(deepseek_v4_config).estimate_flops(batch_seqlens, delta_time=1.0)

        more_shared_config = deepcopy(deepseek_v4_config)
        more_shared_config.n_shared_experts = deepseek_v4_config.n_shared_experts + 1
        more_shared_flops, _ = VeomniFlopsCounter(more_shared_config).estimate_flops(batch_seqlens, delta_time=1.0)

        assert more_shared_flops > baseline_flops

    def test_hca_compression_rate_reduces_attention_flops(self, deepseek_v4_config):
        batch_seqlens = [256]
        baseline_flops, _ = VeomniFlopsCounter(deepseek_v4_config).estimate_flops(batch_seqlens, delta_time=1.0)

        compressed_config = deepcopy(deepseek_v4_config)
        compressed_config.compress_rates["heavily_compressed_attention"] = 64
        compressed_flops, _ = VeomniFlopsCounter(compressed_config).estimate_flops(batch_seqlens, delta_time=1.0)

        assert compressed_flops < baseline_flops
