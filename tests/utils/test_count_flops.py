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

    def test_dense_hybrid_lora_arithmetic(self, qwen3_5_counter):
        batch_seqlens = [12, 5]
        lora_modules = [
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
        flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(8, lora_modules),
        )

        expected = self._expected_flops(qwen3_5_counter.config, batch_seqlens, 2.0, 8, lora_modules)
        assert flops == pytest.approx(expected, rel=1e-9)

    def test_vision_flops_follow_input_and_lora_targets(self, qwen3_5_counter):
        batch_seqlens = [12, 5]
        images_seqlens = [16]
        rank = 8

        full_text, _ = qwen3_5_counter.estimate_flops(batch_seqlens, delta_time=2.0)
        full_vl, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            images_seqlens=images_seqlens,
        )
        decoder_lora_config = _lora_config(rank, ["q_proj", "in_proj_qkv", "gate_proj"])
        lora_text, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=decoder_lora_config,
        )
        empty_image_flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=decoder_lora_config,
            images_seqlens=[],
        )
        lora_vl, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=decoder_lora_config,
            images_seqlens=images_seqlens,
        )

        # No vision tokens skip the ViT. Decoder-only targets leave it frozen
        # and detached, so only one forward pass (one third of FFT) remains.
        assert empty_image_flops == lora_text
        # Decoder-only targets leave the vision tower frozen and detached.
        assert lora_vl - lora_text == pytest.approx((full_vl - full_text) / 3, rel=1e-9)

        text_flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
            lora_config=_lora_config(rank, ["qkv"]),
        )
        vl_flops, _ = qwen3_5_counter.estimate_flops(
            batch_seqlens,
            delta_time=2.0,
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

        assert vl_flops - text_flops == pytest.approx((linear_flops + attention_flops) / 2.0 / 1e12, rel=1e-9)


class TestQwen35MoeFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

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


class TestAllQwenLoraFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_supported_qwen_family_dispatch(self, qwen3_5_moe_counter):
        configs = [
            _load_toy_config(f"tests/toy_config/{config_dir}")
            for config_dir in ("qwen2vl_toy", "qwen25vl_toy", "qwen3vl_toy", "qwen3_moe_toy", "qwen3vlmoe_toy")
        ]
        qwen3_next = deepcopy(qwen3_5_moe_counter.config.text_config)
        qwen3_next.model_type = "qwen3_next"
        configs.extend((qwen3_5_moe_counter.config, qwen3_5_moe_counter.config.text_config, qwen3_next))

        routed_moe_types = {"qwen3_moe", "qwen3_vl_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
        for config in configs:
            counter = VeomniFlopsCounter(config)
            kwargs = {"images_seqlens": [16]} if hasattr(config, "vision_config") else {}
            modules = list(LORA_MODULES_BY_MODEL_TYPE[config.model_type])
            make_config = _routed_lora_config if config.model_type in routed_moe_types else _lora_config

            full_flops, _ = counter.estimate_flops([12, 5], 1.0, **kwargs)
            rank4, _ = counter.estimate_flops([12, 5], 1.0, lora_config=make_config(4, modules), **kwargs)
            rank8, _ = counter.estimate_flops([12, 5], 1.0, lora_config=make_config(8, modules), **kwargs)

            assert 0 < rank4 < rank8 < full_flops, config.model_type

    def test_routed_moe_lora_modes_and_topk(self, qwen3_5_moe_counter):
        config = qwen3_5_moe_counter.config
        text_config = config.text_config
        batch_seqlens = [12, 5]

        for mode, adapter_uses in (
            ("independent", 3 * text_config.num_experts_per_tok),
            ("shared", 2 + text_config.num_experts_per_tok),
        ):
            rank4, _ = VeomniFlopsCounter(config).estimate_flops(
                batch_seqlens, 1.0, lora_config=_routed_lora_config(4, moe_mode=mode)
            )
            rank8, _ = VeomniFlopsCounter(config).estimate_flops(
                batch_seqlens, 1.0, lora_config=_routed_lora_config(8, moe_mode=mode)
            )
            params_per_rank = (
                (text_config.hidden_size + text_config.moe_intermediate_size)
                * text_config.num_hidden_layers
                * adapter_uses
            )
            expected_delta = 6 * (8 - 4) * params_per_rank * sum(batch_seqlens) / 1e12
            assert rank8 - rank4 == pytest.approx(expected_delta, rel=1e-9), mode

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


class TestLoraValidationFlops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_lora_validation_and_failure_behavior(self, qwen3_config, gpt_oss_config):
        invalid_cases = [
            (qwen3_config, _lora_config(8, ["q_proj", "q_proj"]), "must not contain duplicates"),
            (qwen3_config, _lora_config(8, ["unknown_proj"]), "Unsupported qwen3"),
            (qwen3_config, _lora_config(8, "q_proj"), "does not support regex-string"),
            (qwen3_config, {"r": 8}, "VeOmniLoraConfig"),
            (qwen3_config, _routed_lora_config(8), "not supported for non-MoE"),
            (
                _load_toy_config("tests/toy_config/qwen3_moe_toy"),
                _lora_config(8, target_parameters=["*.mlp.experts.router_weight"]),
                "fused routed-expert",
            ),
            (gpt_oss_config, _lora_config(8, ["q_proj"]), "supports Qwen model types"),
        ]
        for config, lora_config, error_match in invalid_cases:
            counter = VeomniFlopsCounter(config)
            with patch("veomni.utils.count_flops.logger.warning_rank0") as warning:
                flops, promised_flops = counter.estimate_flops([12, 5], 2.0, lora_config=lora_config)
            assert (flops, promised_flops) == (0, 1000.0), error_match
            assert error_match in warning.call_args.args[1]

        counter = VeomniFlopsCounter(qwen3_config)
        duplicate_warning = _lora_config(8, ["warning_once_unknown_proj"])
        with patch("veomni.utils.count_flops.logger.warning_rank0") as warning:
            counter.estimate_flops([12, 5], 2.0, lora_config=duplicate_warning)
            counter.estimate_flops([12, 5], 2.0, lora_config=duplicate_warning)
        warning.assert_called_once()

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            counter.estimate_flops([12, 5], 2.0, lora_rank=8)

        baseline, _ = counter.estimate_flops([12, 5], 2.0, lora_config=_lora_config(8, ["q_proj"]))
        ignored_fields, _ = counter.estimate_flops(
            [12, 5],
            2.0,
            lora_config=VeOmniLoraConfig(
                r=8,
                lora_alpha=256,
                target_modules=["q_proj"],
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

        def fail_estimation(*args, **kwargs):
            raise RuntimeError("estimator failure")

        counter.estimate_func["qwen3"] = fail_estimation
        with pytest.raises(RuntimeError, match="estimator failure"):
            counter.estimate_flops([12, 5], 2.0, lora_config=_lora_config(8, ["q_proj"]))


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


@pytest.fixture
def hunyuan_image_3_real_base_config():
    """The real HunyuanImage-3 Base geometry (83B total / ~12.2B active per token)."""
    return SimpleNamespace(
        model_type="hunyuan_image_3_moe",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        attention_head_dim=128,
        num_experts=64,
        moe_topk=8,
        moe_intermediate_size=3072,
        vocab_size=133120,
        component_policy={"lm_head": "absent"},
    )


class TestHunyuanImage3Flops:
    pytestmark = pytest.mark.usefixtures("mock_device_flops")

    def test_matches_hand_derived_active_params(self, hunyuan_image_3_real_base_config):
        """Pin the counter against the hand-derived closed form for HI3 Base.

        ``6 * active_params * tokens`` (~12.2B active params/token) plus a
        ``12 * S**2 * H * L`` attention term. If the counter drifts from it,
        every reported MFU number for this model silently shifts.
        """
        counter = VeomniFlopsCounter(hunyuan_image_3_real_base_config)
        seq_len, global_batch_size = 4138, 16
        batch_seqlens = [seq_len] * global_batch_size

        flops, _ = counter.estimate_flops(batch_seqlens, delta_time=1.0)

        active_params = 12.222201856e9  # 32 layers x (router + 9 experts + QKVO)
        tokens = seq_len * global_batch_size
        expected = 6 * active_params * tokens + 12 * tokens * seq_len * 4096 * 32
        assert flops * 1e12 == pytest.approx(expected, rel=1e-12)

    def test_absent_lm_head_is_not_counted(self, hunyuan_image_3_real_base_config):
        """T2I never builds ``lm_head``; billing its 133k-vocab matmul inflates MFU."""
        counter = VeomniFlopsCounter(hunyuan_image_3_real_base_config)
        with_head_config = deepcopy(hunyuan_image_3_real_base_config)
        with_head_config.component_policy = {"lm_head": "trainable"}
        with_head_counter = VeomniFlopsCounter(with_head_config)

        batch_seqlens = [4138]
        flops, _ = counter.estimate_flops(batch_seqlens, delta_time=1.0)
        with_head_flops, _ = with_head_counter.estimate_flops(batch_seqlens, delta_time=1.0)

        assert with_head_flops > flops
        lm_head_flops = 6 * 133120 * 4096 * 4138
        assert (with_head_flops - flops) * 1e12 == pytest.approx(lm_head_flops, rel=1e-12)
