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

from veomni.lora.config import LORA_MODULES_BY_MODEL_TYPE, VeOmniLoraConfig
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


def _lora_config(rank, target_modules=None, target_parameters=None, moe_mode=None):
    return VeOmniLoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        target_parameters=target_parameters,
        moe_mode=moe_mode,
    )


def _default_lora_config(config, rank=8):
    return _lora_config(rank, list(LORA_MODULES_BY_MODEL_TYPE[config.model_type]))


ROUTED_EXPERT_TARGETS = ["*.mlp.experts.gate_up_proj", "*.mlp.experts.down_proj"]


def _routed_lora_config(rank, target_modules=None, moe_mode="independent"):
    return _lora_config(rank, target_modules, ROUTED_EXPERT_TARGETS, moe_mode)


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
def qwen3_config():
    return _load_toy_config("tests/toy_config/qwen3_toy")


@pytest.fixture
def qwen3_counter(qwen3_config):
    return VeomniFlopsCounter(qwen3_config)


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
        assert flops == pytest.approx(106.965220982784, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        # Embedding lookup is not a matmul; only lm_head contributes vocab_size * hidden_size.
        assert flops == pytest.approx(109.196454395904, rel=1e-9)


class TestQwen35LoraFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    @staticmethod
    def _expected_flops(config, batch_seqlens, delta_time, lora_rank, lora_modules):
        text_config = config.text_config if hasattr(config, "text_config") else config
        tokens_sum = sum(batch_seqlens)
        hidden_size = text_config.hidden_size
        head_dim = getattr(
            text_config,
            "head_dim",
            hidden_size // text_config.num_attention_heads,
        )
        q_size = text_config.num_attention_heads * head_dim
        kv_size = text_config.num_key_value_heads * head_dim
        linear_k_size = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        linear_v_size = text_config.linear_num_value_heads * text_config.linear_value_head_dim
        num_full_layers = sum(layer_type == "full_attention" for layer_type in text_config.layer_types)
        num_linear_layers = sum(layer_type == "linear_attention" for layer_type in text_config.layer_types)

        module_shapes_and_counts = {
            "q_proj": ((hidden_size, 2 * q_size), num_full_layers),
            "k_proj": ((hidden_size, kv_size), num_full_layers),
            "v_proj": ((hidden_size, kv_size), num_full_layers),
            "o_proj": ((q_size, hidden_size), num_full_layers),
            "in_proj_qkv": ((hidden_size, 2 * linear_k_size + linear_v_size), num_linear_layers),
            "in_proj_z": ((hidden_size, linear_v_size), num_linear_layers),
            "in_proj_b": ((hidden_size, text_config.linear_num_value_heads), num_linear_layers),
            "in_proj_a": ((hidden_size, text_config.linear_num_value_heads), num_linear_layers),
            "out_proj": ((linear_v_size, hidden_size), num_linear_layers),
            "gate_proj": ((hidden_size, text_config.intermediate_size), text_config.num_hidden_layers),
            "up_proj": ((hidden_size, text_config.intermediate_size), text_config.num_hidden_layers),
            "down_proj": ((text_config.intermediate_size, hidden_size), text_config.num_hidden_layers),
        }

        full_attn_params = hidden_size * (2 * q_size + 2 * kv_size) + q_size * hidden_size
        gdn_params = hidden_size * (2 * linear_k_size + 3 * linear_v_size + 2 * text_config.linear_num_value_heads)
        gdn_params += text_config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)
        mlp_params = hidden_size * text_config.intermediate_size * 3 * text_config.num_hidden_layers
        lm_head_params = hidden_size * text_config.vocab_size
        base_params = full_attn_params * num_full_layers + gdn_params * num_linear_layers + mlp_params + lm_head_params

        lora_params = 0
        for module_name in lora_modules:
            (in_features, out_features), layer_count = module_shapes_and_counts[module_name]
            lora_params += lora_rank * (in_features + out_features) * layer_count
        linear_flops = (4 * base_params + 6 * lora_params) * tokens_sum

        attention_flops = (
            12
            * sum(seqlen * seqlen for seqlen in batch_seqlens)
            * head_dim
            * text_config.num_attention_heads
            * num_full_layers
        )
        gdn_flops = (
            15
            * text_config.linear_key_head_dim
            * text_config.linear_value_head_dim
            * text_config.linear_num_value_heads
            * tokens_sum
            * num_linear_layers
        )
        return (linear_flops + attention_flops + gdn_flops) / delta_time / 1e12

    @pytest.mark.parametrize(
        "lora_modules",
        [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
            ["gate_proj", "up_proj", "down_proj"],
        ],
    )
    def test_selected_lora_modules(self, qwen3_5_counter, lora_modules):
        batch_seqlens = [12, 5]
        flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(8, lora_modules),
        )

        expected = self._expected_flops(qwen3_5_counter.config, batch_seqlens, 2.0, 8, lora_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_default_lora_modules(self, qwen3_5_counter):
        batch_seqlens = [12, 5]
        flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_default_lora_config(qwen3_5_counter.config),
        )

        default_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        expected = self._expected_flops(qwen3_5_counter.config, batch_seqlens, 2.0, 8, default_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_standalone_text_model_type(self, qwen3_5_counter):
        text_config = deepcopy(qwen3_5_counter.config.text_config)
        counter = VeomniFlopsCounter(text_config)
        lora_modules = ["q_proj", "in_proj_qkv", "gate_proj"]

        flops, _ = counter.estimate_flops(
            [12, 5],
            delta_time=2.0,
            lora_config=_lora_config(8, lora_modules),
        )

        expected = self._expected_flops(text_config, [12, 5], 2.0, 8, lora_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_vision_inputs_use_frozen_vit_cost(self, qwen3_5_counter):
        batch_seqlens = [12, 5]
        images_seqlens = [16]
        full_text, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=2.0)
        full_vl, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            images_seqlens=images_seqlens,
        )
        lora_config = _lora_config(8, ["q_proj", "in_proj_qkv", "gate_proj"])
        lora_text, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=lora_config,
        )
        lora_vl, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=lora_config,
            images_seqlens=images_seqlens,
        )

        # Decoder-only targets leave the vision tower frozen and detached.
        assert lora_vl - lora_text == pytest.approx((full_vl - full_text) / 3, rel=1e-9)

    def test_vision_target_counts_backward_and_adapter_flops(self, qwen3_5_counter):
        batch_seqlens = [12, 5]
        images_seqlens = [16]
        rank = 8
        text_flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=1.0,
            lora_config=_lora_config(rank, ["qkv"]),
        )
        vl_flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=1.0,
            lora_config=_lora_config(rank, ["qkv"]),
            images_seqlens=images_seqlens,
        )

        vision = qwen3_5_counter.config.vision_config
        tokens_sum = sum(images_seqlens)
        dim = vision.hidden_size
        merger_hidden_size = dim * vision.spatial_merge_size**2
        patch_embed_params = (
            dim * vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
        )
        block_params = dim * (2 * vision.intermediate_size + 4 * dim) * vision.depth
        merger_params = merger_hidden_size * (merger_hidden_size + vision.out_hidden_size)
        adaptable_base_params = block_params + merger_params
        lora_params = rank * (dim + 3 * dim) * vision.depth
        linear_flops = (2 * patch_embed_params + 4 * adaptable_base_params + 6 * lora_params) * tokens_sum
        attention_flops = (
            12
            * sum(seqlen * seqlen for seqlen in images_seqlens)
            * (dim // vision.num_heads)
            * vision.num_heads
            * vision.depth
        )

        assert vl_flops - text_flops == pytest.approx((linear_flops + attention_flops) / 1e12, rel=1e-9)

    def test_empty_image_sequences_skip_vision(self, qwen3_5_counter):
        lora_config = _lora_config(8, ["linear_fc1"])
        text_flops, _ = qwen3_5_counter.estimate_flops([12, 5], 1.0, lora_config=lora_config)
        empty_image_flops, _ = qwen3_5_counter.estimate_flops(
            [12, 5],
            1.0,
            lora_config=lora_config,
            images_seqlens=[],
        )

        assert empty_image_flops == text_flops


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
        text_config = qwen3_5_moe_counter.config.text_config
        shared_expert_gate_flops = (
            6 * text_config.hidden_size * text_config.num_hidden_layers * sum(batch_seqlens) / 1e12
        )
        # The embedding lookup is excluded. The shared-expert scalar gate is
        # an ordinary trainable linear and follows the FFT factor-six convention.
        assert flops == pytest.approx(16.888079843328 + shared_expert_gate_flops, rel=1e-9)

    def test_numerical_with_vit(self, qwen3_5_moe_counter):
        batch_seqlens = [1024, 1024, 1024, 1024]
        flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0, images_seqlens=[256, 512])
        text_config = qwen3_5_moe_counter.config.text_config
        shared_expert_gate_flops = (
            6 * text_config.hidden_size * text_config.num_hidden_layers * sum(batch_seqlens) / 1e12
        )
        assert flops == pytest.approx(19.05408344064 + shared_expert_gate_flops, rel=1e-9)


class TestQwen3Flops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_uses_explicit_head_dim_for_projection_shapes(self, qwen3_counter):
        config = qwen3_counter.config
        batch_seqlens = [12, 5]
        tokens_sum = sum(batch_seqlens)
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim

        mlp_N = config.hidden_size * config.intermediate_size * 3
        attn_linear_N = config.hidden_size * (2 * q_size + 2 * kv_size)
        lm_head_N = config.hidden_size * config.vocab_size
        dense_N = (mlp_N + attn_linear_N) * config.num_hidden_layers + lm_head_N
        expected_flops = 6 * dense_N * tokens_sum
        expected_flops += (
            12
            * sum(seqlen * seqlen for seqlen in batch_seqlens)
            * config.head_dim
            * config.num_attention_heads
            * config.num_hidden_layers
        )

        flops, _ = qwen3_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        assert flops == pytest.approx(expected_flops / 1e12, rel=1e-9)

    @pytest.mark.parametrize(
        "lora_modules",
        [
            ["q_proj", "in_proj_qkv", "out_proj"],
            ["gate_proj", "up_proj", "down_proj"],
        ],
    )
    def test_lora_modules(self, qwen3_5_moe_counter, lora_modules):
        batch_seqlens = [12, 5]
        full_flops, _ = qwen3_5_moe_counter.estimate_flops(batch_seqlens, delta_time=1.0)
        lora_flops, _ = qwen3_5_moe_counter.estimate_flops(
            batch_seqlens,
            delta_time=1.0,
            lora_config=_lora_config(8, lora_modules),
        )

        assert 0 < lora_flops < full_flops

    def test_standalone_moe_text_model_type(self, qwen3_5_moe_counter):
        text_config = deepcopy(qwen3_5_moe_counter.config.text_config)
        counter = VeomniFlopsCounter(text_config)

        flops, _ = counter.estimate_flops(
            [12, 5],
            delta_time=1.0,
            lora_config=_default_lora_config(text_config),
        )

        assert flops > 0

    def test_routed_expert_lora_scales_with_topk(self, qwen3_5_moe_counter):
        def rank_delta(config):
            counter = VeomniFlopsCounter(config)
            rank4, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(4))
            rank8, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(8))
            return rank8 - rank4

        topk1_config = deepcopy(qwen3_5_moe_counter.config)
        topk1_config.text_config.num_experts_per_tok = 1

        actual_delta = rank_delta(qwen3_5_moe_counter.config)
        text_config = qwen3_5_moe_counter.config.text_config
        params_per_rank = (
            3
            * (text_config.hidden_size + text_config.moe_intermediate_size)
            * text_config.num_hidden_layers
            * text_config.num_experts_per_tok
        )
        expected_delta = 6 * (8 - 4) * params_per_rank * (12 + 5) / 1e12

        assert actual_delta == pytest.approx(
            text_config.num_experts_per_tok * rank_delta(topk1_config),
            rel=1e-9,
        )
        assert actual_delta == pytest.approx(expected_delta, rel=1e-9)


class TestAllQwenLoraFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_single_fused_moe_mlp_target_installs_all_adapters(self, qwen3_5_moe_counter):
        qwen3_moe = _load_toy_config("tests/toy_config/qwen3_moe_toy")
        qwen3_5_moe = qwen3_5_moe_counter.config
        qwen3_next = deepcopy(qwen3_5_moe.text_config)
        qwen3_next.model_type = "qwen3_next"

        for config in (qwen3_moe, qwen3_5_moe, qwen3_next):
            counter = VeomniFlopsCounter(config)
            all_mlp_flops, _ = counter.estimate_flops(
                [12, 5],
                1.0,
                lora_config=_routed_lora_config(8),
            )
            for target_parameter in ROUTED_EXPERT_TARGETS:
                single_target_flops, _ = counter.estimate_flops(
                    [12, 5],
                    1.0,
                    lora_config=_lora_config(8, target_parameters=[target_parameter], moe_mode="independent"),
                )
                assert single_target_flops == pytest.approx(all_mlp_flops, rel=1e-9)

    @pytest.mark.parametrize(
        "config_dir",
        [
            "qwen2vl_toy",
            "qwen25vl_toy",
            "qwen3vl_toy",
            "qwen3_moe_toy",
            "qwen3vlmoe_toy",
        ],
    )
    def test_new_qwen_family_support(self, config_dir):
        config = _load_toy_config(f"tests/toy_config/{config_dir}")
        counter = VeomniFlopsCounter(config)
        kwargs = {"images_seqlens": [16]} if hasattr(config, "vision_config") else {}

        full_flops, _ = counter.estimate_flops([12, 5], delta_time=1.0, **kwargs)
        lora_flops, _ = counter.estimate_flops(
            [12, 5],
            delta_time=1.0,
            lora_config=_default_lora_config(config),
            **kwargs,
        )

        assert 0 < lora_flops < full_flops

    def test_qwen3_moe_adapter_work_scales_with_topk(self):
        config = _load_toy_config("tests/toy_config/qwen3_moe_toy")

        def rank_delta(current_config):
            counter = VeomniFlopsCounter(current_config)
            rank4, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(4))
            rank8, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(8))
            return rank8 - rank4

        topk1_config = deepcopy(config)
        topk1_config.num_experts_per_tok = 1

        actual_delta = rank_delta(config)
        params_per_rank = (
            3
            * (config.hidden_size + config.moe_intermediate_size)
            * config.num_hidden_layers
            * config.num_experts_per_tok
        )
        expected_delta = 6 * (8 - 4) * params_per_rank * (12 + 5) / 1e12

        assert actual_delta == pytest.approx(2 * rank_delta(topk1_config), rel=1e-9)
        assert actual_delta == pytest.approx(expected_delta, rel=1e-9)

    def test_qwen3_next_projection_names(self, qwen3_5_moe_counter):
        config = deepcopy(qwen3_5_moe_counter.config.text_config)
        config.model_type = "qwen3_next"
        counter = VeomniFlopsCounter(config)
        modules = ["q_proj", "in_proj_qkvz", "in_proj_ba", "out_proj", "gate_proj"]

        rank4, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_lora_config(4, modules))
        rank8, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_lora_config(8, modules))

        full_layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)
        linear_layers = sum(layer_type == "linear_attention" for layer_type in config.layer_types)
        q_size = config.num_attention_heads * config.head_dim
        linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
        params_per_rank = (
            (config.hidden_size + 2 * q_size) * full_layers
            + (config.hidden_size + 2 * linear_k_size + 2 * linear_v_size) * linear_layers
            + (config.hidden_size + 2 * config.linear_num_value_heads) * linear_layers
            + (linear_v_size + config.hidden_size) * linear_layers
            + (config.hidden_size + config.shared_expert_intermediate_size) * config.num_hidden_layers
        )
        expected_delta = 6 * (8 - 4) * params_per_rank * (12 + 5) / 1e12

        assert rank8 - rank4 == pytest.approx(expected_delta, rel=1e-9)

    def test_qwen3_next_lora_uses_correct_gated_q_proj_base(self, qwen3_5_moe_counter):
        config = deepcopy(qwen3_5_moe_counter.config.text_config)
        config.model_type = "qwen3_next"
        counter = VeomniFlopsCounter(config)
        batch_seqlens = [12, 5]
        tokens_sum = sum(batch_seqlens)
        rank = 8

        attn_linear_N, full_layers, linear_layers, head_dim, num_heads = counter._compute_hybrid_attn_params(config)
        moe_N = (
            config.hidden_size * config.num_experts
            + config.hidden_size * config.moe_intermediate_size * config.num_experts_per_tok * 3
            + config.hidden_size * config.shared_expert_intermediate_size * 3
            + config.hidden_size
        ) * config.num_hidden_layers
        base_N = moe_N + attn_linear_N + config.hidden_size * config.vocab_size
        q_size = config.num_attention_heads * head_dim
        lora_N = rank * (config.hidden_size + 2 * q_size) * full_layers
        linear_flops = (4 * base_N + 6 * lora_N) * tokens_sum
        attention_flops = 12 * sum(seqlen * seqlen for seqlen in batch_seqlens) * head_dim * num_heads * full_layers
        recurrence_flops = counter._compute_gdn_recurrence_flops(config, tokens_sum, linear_layers)
        expected = (linear_flops + attention_flops + recurrence_flops) / 1e12

        actual, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(rank, ["q_proj"]),
        )

        assert actual == pytest.approx(expected, rel=1e-9)

    def test_qwen3_next_mlp_targets_routed_experts(self, qwen3_5_moe_counter):
        config = deepcopy(qwen3_5_moe_counter.config.text_config)
        config.model_type = "qwen3_next"

        def rank_delta(current_config):
            counter = VeomniFlopsCounter(current_config)
            rank4, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(4))
            rank8, _ = counter.estimate_flops([12, 5], 1.0, lora_config=_routed_lora_config(8))
            return rank8 - rank4

        topk1_config = deepcopy(config)
        topk1_config.num_experts_per_tok = 1

        assert rank_delta(config) == pytest.approx(
            config.num_experts_per_tok * rank_delta(topk1_config),
            rel=1e-9,
        )

    def test_shared_moe_mode_reuses_gate_and_up_adapters(self, qwen3_5_moe_counter):
        config = qwen3_5_moe_counter.config
        text_config = config.text_config
        batch_seqlens = [12, 5]

        rank4, _ = VeomniFlopsCounter(config).estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_routed_lora_config(4, moe_mode="shared"),
        )
        rank8, _ = VeomniFlopsCounter(config).estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_routed_lora_config(8, moe_mode="shared"),
        )

        params_per_rank = (
            (text_config.hidden_size + text_config.moe_intermediate_size)
            * text_config.num_hidden_layers
            * (2 + text_config.num_experts_per_tok)
        )
        expected_delta = 6 * (8 - 4) * params_per_rank * sum(batch_seqlens) / 1e12
        assert rank8 - rank4 == pytest.approx(expected_delta, rel=1e-9)

    def test_shared_and_routed_expert_adapters_are_additive(self, qwen3_5_moe_counter):
        counter = qwen3_5_moe_counter
        batch_seqlens = [12, 5]
        attention_modules = ["q_proj"]
        shared_modules = ["q_proj", "gate_proj", "up_proj", "down_proj"]

        attention, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(8, attention_modules),
        )
        shared, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(8, shared_modules),
        )
        routed, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_routed_lora_config(8, attention_modules),
        )
        combined, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_routed_lora_config(8, shared_modules),
        )

        assert combined - attention == pytest.approx((shared - attention) + (routed - attention), rel=1e-9)

    def test_qwen25_vl_matching_vision_mlp_modules(self):
        config = _load_toy_config("tests/toy_config/qwen25vl_toy")
        counter = VeomniFlopsCounter(config)
        batch_seqlens = [12, 5]
        images_seqlens = [16]
        modules = ["gate_proj", "up_proj", "down_proj"]
        rank = 8

        text_flops, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(rank, modules),
        )
        vl_flops, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(rank, modules),
            images_seqlens=images_seqlens,
        )

        vision = config.vision_config
        tokens_sum = sum(images_seqlens)
        dim = vision.hidden_size
        intermediate_size = vision.intermediate_size
        mlp_params = dim * intermediate_size * 3
        attention_params = dim * dim * 4
        merger_params = (vision.out_hidden_size + dim * vision.spatial_merge_size**2) * (
            dim * vision.spatial_merge_size**2
        )
        base_params = (mlp_params + attention_params) * vision.depth + merger_params
        lora_params = rank * (dim + intermediate_size) * 3 * vision.depth
        linear_flops = (4 * base_params + 6 * lora_params) * tokens_sum
        attention_flops = (
            12
            * sum(seqlen * seqlen for seqlen in images_seqlens)
            * (dim // vision.num_heads)
            * vision.num_heads
            * len(vision.fullatt_block_indexes)
        )
        expected_vision_flops = (linear_flops + attention_flops) / 1e12

        assert vl_flops - text_flops == pytest.approx(expected_vision_flops, rel=1e-9)

    def test_qwen3_vl_deepstack_merger_lora(self):
        config = _load_toy_config("tests/toy_config/qwen3vl_toy")
        counter = VeomniFlopsCounter(config)
        batch_seqlens = [12, 5]
        images_seqlens = [16]

        rank4_flops, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(4, ["linear_fc1"]),
            images_seqlens=images_seqlens,
        )
        rank8_flops, _ = counter.estimate_flops(
            batch_seqlens,
            1.0,
            lora_config=_lora_config(8, ["linear_fc1"]),
            images_seqlens=images_seqlens,
        )

        vision = config.vision_config
        merger_hidden_size = vision.hidden_size * vision.spatial_merge_size**2
        params_per_rank = (vision.hidden_size + vision.intermediate_size) * vision.depth + 2 * merger_hidden_size * (
            1 + len(vision.deepstack_visual_indexes)
        )
        expected_delta = 6 * (8 - 4) * params_per_rank * sum(images_seqlens) / 1e12

        assert rank8_flops - rank4_flops == pytest.approx(expected_delta, rel=1e-9)


class TestQwen2LoraFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    @staticmethod
    def _expected_flops(config, batch_seqlens, delta_time, lora_rank, lora_modules):
        tokens_sum = sum(batch_seqlens)
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        q_size = config.num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim

        module_shapes = {
            "q_proj": (hidden_size, q_size),
            "k_proj": (hidden_size, kv_size),
            "v_proj": (hidden_size, kv_size),
            "o_proj": (q_size, hidden_size),
            "gate_proj": (hidden_size, intermediate_size),
            "up_proj": (hidden_size, intermediate_size),
            "down_proj": (intermediate_size, hidden_size),
        }
        mlp_params = hidden_size * intermediate_size * 3
        attn_linear_params = hidden_size * (q_size + 2 * kv_size + q_size)
        lm_head_params = hidden_size * config.vocab_size
        base_linear_params = (mlp_params + attn_linear_params) * config.num_hidden_layers + lm_head_params
        lora_params = sum(lora_rank * sum(module_shapes[name]) for name in lora_modules)
        lora_params *= config.num_hidden_layers
        linear_flops = (4 * base_linear_params + 6 * lora_params) * tokens_sum

        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attention_flops = 12 * seqlen_square_sum * head_dim * config.num_attention_heads * config.num_hidden_layers
        return (linear_flops + attention_flops) / delta_time / 1e12

    def test_full_finetuning_unchanged(self, qwen3_counter, qwen3_config):
        batch_seqlens = [12, 5]
        flops, promised_flops = qwen3_counter.estimate_flops(batch_seqlens, delta_time=2.0)

        tokens_sum = sum(batch_seqlens)
        head_dim = getattr(
            qwen3_config,
            "head_dim",
            qwen3_config.hidden_size // qwen3_config.num_attention_heads,
        )
        q_size = qwen3_config.num_attention_heads * head_dim
        kv_size = qwen3_config.num_key_value_heads * head_dim
        linear_params = (
            qwen3_config.hidden_size
            * (3 * qwen3_config.intermediate_size + 2 * q_size + 2 * kv_size)
            * qwen3_config.num_hidden_layers
            + qwen3_config.hidden_size * qwen3_config.vocab_size
        )
        attention_flops = (
            12
            * sum(seqlen * seqlen for seqlen in batch_seqlens)
            * head_dim
            * qwen3_config.num_attention_heads
            * qwen3_config.num_hidden_layers
        )
        expected = (6 * linear_params * tokens_sum + attention_flops) / 2.0 / 1e12

        assert flops == pytest.approx(expected, rel=1e-9)
        assert promised_flops == 1000.0

    @pytest.mark.parametrize(
        "lora_modules",
        [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
        ],
    )
    def test_selected_lora_modules(self, qwen3_counter, qwen3_config, lora_modules):
        batch_seqlens = [12, 5]
        flops, _ = qwen3_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(8, lora_modules),
        )

        expected = self._expected_flops(qwen3_config, batch_seqlens, 2.0, 8, lora_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_default_lora_modules(self, qwen3_counter, qwen3_config):
        batch_seqlens = [12, 5]
        default_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        flops, _ = qwen3_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(8, default_modules),
        )
        expected = self._expected_flops(qwen3_config, batch_seqlens, 2.0, 8, default_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_frozen_lm_head_keeps_input_gradient_cost(self, qwen3_config):
        batch_seqlens = [12, 5]
        tokens_sum = sum(batch_seqlens)
        larger_vocab_config = deepcopy(qwen3_config)
        larger_vocab_config.vocab_size += 1

        full_flops, _ = VeomniFlopsCounter(qwen3_config).estimate_flops(batch_seqlens, delta_time=1.0)
        larger_full_flops, _ = VeomniFlopsCounter(larger_vocab_config).estimate_flops(
            batch_seqlens,
            delta_time=1.0,
        )
        lora_flops, _ = VeomniFlopsCounter(qwen3_config).estimate_flops(
            batch_seqlens,
            delta_time=1.0,
            lora_config=_default_lora_config(qwen3_config),
        )
        larger_lora_flops, _ = VeomniFlopsCounter(larger_vocab_config).estimate_flops(
            batch_seqlens,
            delta_time=1.0,
            lora_config=_default_lora_config(larger_vocab_config),
        )

        # A larger vocabulary adds one lm_head row. FFT computes forward, dX,
        # and dW; LoRA freezes the head but still computes forward and dX.
        one_head_row_flops = qwen3_config.hidden_size * tokens_sum / 1e12
        assert larger_full_flops - full_flops == pytest.approx(6 * one_head_row_flops, rel=1e-9)
        assert larger_lora_flops - lora_flops == pytest.approx(4 * one_head_row_flops, rel=1e-9)

    def test_qwen2_model_type(self, qwen3_config):
        qwen2_config = deepcopy(qwen3_config)
        qwen2_config.model_type = "qwen2"
        counter = VeomniFlopsCounter(qwen2_config)

        flops, _ = counter.estimate_flops(
            [12, 5],
            delta_time=2.0,
            lora_config=_default_lora_config(qwen2_config),
        )
        assert flops > 0

    def test_qwen3_explicit_head_dim_for_full_and_lora(self, qwen3_config):
        config = deepcopy(qwen3_config)
        config.head_dim = config.hidden_size // config.num_attention_heads * 2
        counter = VeomniFlopsCounter(config)
        batch_seqlens = [12, 5]
        lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        full_flops, _ = counter.estimate_flops(batch_seqlens, delta_time=2.0)
        lora_flops, _ = counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(8, lora_modules),
        )

        tokens_sum = sum(batch_seqlens)
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        linear_params = (
            config.hidden_size * (3 * config.intermediate_size + 2 * q_size + 2 * kv_size) * config.num_hidden_layers
            + config.hidden_size * config.vocab_size
        )
        attention_flops = (
            12
            * sum(seqlen * seqlen for seqlen in batch_seqlens)
            * config.head_dim
            * config.num_attention_heads
            * config.num_hidden_layers
        )
        expected_full = (6 * linear_params * tokens_sum + attention_flops) / 2.0 / 1e12
        expected_lora = self._expected_flops(config, batch_seqlens, 2.0, 8, lora_modules)

        assert full_flops == pytest.approx(expected_full, rel=1e-9)
        assert lora_flops == pytest.approx(expected_lora, rel=1e-9)

    @pytest.mark.parametrize(
        ("lora_config", "error_match"),
        [
            (_lora_config(8, ["q_proj", "q_proj"]), "must not contain duplicates"),
            (_lora_config(8, ["q_proj", "unknown_proj"]), "Unsupported qwen3"),
            (_lora_config(8, "q_proj"), "does not support regex-string"),
        ],
    )
    def test_invalid_lora_config(self, qwen3_counter, lora_config, error_match):
        with pytest.raises(ValueError, match=error_match):
            qwen3_counter.estimate_flops(
                [12, 5],
                delta_time=2.0,
                lora_config=lora_config,
            )

    def test_lora_config_requires_config_type(self, qwen3_counter):
        with pytest.raises(TypeError, match="VeOmniLoraConfig"):
            qwen3_counter.estimate_flops([12, 5], delta_time=2.0, lora_config={"r": 8})

    def test_loose_lora_arguments_are_not_part_of_api(self, qwen3_counter):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            qwen3_counter.estimate_flops([12, 5], delta_time=2.0, lora_rank=8)

    def test_non_moe_model_rejects_target_parameters(self, qwen3_counter):
        with pytest.raises(ValueError, match="not supported for non-MoE"):
            qwen3_counter.estimate_flops(
                [12, 5],
                delta_time=2.0,
                lora_config=_routed_lora_config(8),
            )

    def test_unsupported_target_parameter_is_rejected(self):
        counter = VeomniFlopsCounter(_load_toy_config("tests/toy_config/qwen3_moe_toy"))
        with pytest.raises(ValueError, match="fused routed-expert"):
            counter.estimate_flops(
                [12, 5],
                delta_time=2.0,
                lora_config=_lora_config(8, target_parameters=["*.mlp.experts.router_weight"]),
            )

    def test_non_flop_lora_fields_are_ignored(self, qwen3_counter):
        target_modules = ["q_proj"]
        baseline, _ = qwen3_counter.estimate_flops(
            [12, 5],
            delta_time=2.0,
            lora_config=_lora_config(8, target_modules),
        )
        ignored_fields, _ = qwen3_counter.estimate_flops(
            [12, 5],
            delta_time=2.0,
            lora_config=VeOmniLoraConfig(
                r=8,
                lora_alpha=256,
                target_modules=target_modules,
                exclude_modules=["q_proj"],
                lora_dropout=0.5,
                bias="all",
                use_rslora=True,
                init_lora_weights=False,
                rank_pattern={".*q_proj": 64},
                alpha_pattern={".*q_proj": 512},
            ),
        )

        assert ignored_fields == baseline

    def test_lora_rejects_unsupported_model_type(self, gpt_oss_counter):
        with pytest.raises(ValueError, match="supports Qwen model types"):
            gpt_oss_counter.estimate_flops(
                [12, 5],
                delta_time=2.0,
                lora_config=_lora_config(8, ["q_proj"]),
            )


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
