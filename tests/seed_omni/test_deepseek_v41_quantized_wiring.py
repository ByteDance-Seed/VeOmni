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

"""Which projections hold the released quantized weight, and which deliberately do not.

The split is not a tuning knob, it is a property of the released checkpoint, so
these tests read as an inventory. The ones left in bf16 are the small
projections whose outputs are decisions rather than magnitudes -- a compressor
gate, an index key, the router -- and quantizing one of them does not shrink a
number slightly, it selects a different block or a different expert.

The failure this guards against is silent. A site that should carry a
``weight_scale`` and does not will load its fp8 bytes into a bf16 parameter and
produce garbage that still has the right shape.
"""

import math

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import (
    DeepseekV41Attention,
    DeepseekV41Indexer,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.engram_connection import DeepseekV41EngramConnection
from veomni.models.seed_omni.modules.deepseek_v41.llm.quantized import (
    QUANT_BLOCK_SIZE,
    SCALE_DTYPE,
    linear_quant_dtype,
)


#: ``quantization_config`` exactly as the released DeepSeek-V4.1-Flash declares it.
RELEASED_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}

HIDDEN = 64
N_LAYERS = 8


def make_config(**overrides) -> DeepseekV41LlmConfig:
    kwargs = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=48,
        o_lora_rank=16,
        o_groups=2,
        sliding_window=6,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=4,
        rms_norm_eps=1e-20,
        compress_ratios=[2] * N_LAYERS + [0, 0, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 4],
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        engram_layer_ids=[1],
        dspark_target_layer_ids=[5, 6, 7],
        max_position_embeddings=256,
    )
    kwargs.update(overrides)
    return DeepseekV41LlmConfig(**kwargs)


def scale_names(module: torch.nn.Module) -> set:
    return {name for name, _ in module.named_parameters() if name.endswith("weight_scale")}


class TestTheSwitchItself:
    def test_a_bf16_checkpoint_declares_no_scheme(self):
        assert linear_quant_dtype(make_config()) is None

    def test_the_native_checkpoint_asks_for_fp8(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        # fp8 and not fp4: fp4 is only ever the stacked routed experts, and a
        # stack of experts is not an nn.Linear.
        assert linear_quant_dtype(config) is torch.float8_e4m3fn

    def test_a_different_block_width_is_refused_rather_than_mispaired(self):
        config = make_config(weight_quantization={**RELEASED_QUANTIZATION, "weight_block_size": [128, 128]})
        with pytest.raises(ValueError, match="wrong scale"):
            linear_quant_dtype(config)

    def test_a_scale_format_with_a_mantissa_is_refused(self):
        config = make_config(weight_quantization={**RELEASED_QUANTIZATION, "scale_fmt": "e4m3"})
        with pytest.raises(ValueError, match="power of two"):
            linear_quant_dtype(config)


class TestAttentionInventory:
    """Layer 0 owns keys and an index, so it builds every projection there is."""

    def test_bf16_leaves_every_projection_an_ordinary_layer(self):
        attention = DeepseekV41Attention(make_config(), layer_idx=0)
        assert scale_names(attention) == set()
        assert attention.wq_a.weight.dtype is torch.float32

    def test_the_quantized_sites_are_exactly_the_ones_the_release_quantized(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        attention = DeepseekV41Attention(config, layer_idx=0)
        assert scale_names(attention) == {
            "wq_a.weight_scale",
            "wq_b.weight_scale",
            "wkv.weight_scale",
            "wo_b.weight_scale",
            "indexer.wq_b.weight_scale",
        }

    def test_the_projections_that_pick_rather_than_scale_stay_bf16(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        attention = DeepseekV41Attention(config, layer_idx=0)
        for name in ("compressor.wkv", "compressor.wgate", "indexer.wk", "indexer.weights_proj"):
            weight = attention.get_submodule(name).weight
            assert weight.dtype is torch.float32, f"{name} must not be quantized"

    def test_quantizing_does_not_move_a_single_projection(self):
        plain = DeepseekV41Attention(make_config(), layer_idx=0)
        quantized = DeepseekV41Attention(make_config(weight_quantization=RELEASED_QUANTIZATION), layer_idx=0)
        shapes = lambda m: {n: tuple(p.shape) for n, p in m.named_parameters() if n.endswith(".weight")}  # noqa: E731
        assert shapes(plain) == shapes(quantized)


class TestScaleShapes:
    """One scale per ``QUANT_BLOCK_SIZE`` square tile, which is what the kernels index."""

    @pytest.mark.parametrize("site", ["wq_a", "wq_b", "wkv", "wo_b"])
    def test_a_scale_covers_the_weight_in_blocks(self, site):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        attention = DeepseekV41Attention(config, layer_idx=0)
        layer = getattr(attention, site)
        out_features, in_features = layer.weight.shape
        assert tuple(layer.weight_scale.shape) == (
            math.ceil(out_features / QUANT_BLOCK_SIZE),
            math.ceil(in_features / QUANT_BLOCK_SIZE),
        )

    def test_a_scale_is_a_bare_exponent(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        indexer = DeepseekV41Indexer(config, layer_idx=0)
        assert indexer.wq_b.weight_scale.dtype is SCALE_DTYPE


class TestEngram:
    def test_the_engram_projection_follows_the_same_switch(self):
        plain = DeepseekV41EngramConnection(make_config(), layer_idx=1)
        assert scale_names(plain) == set()

        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        quantized = DeepseekV41EngramConnection(config, layer_idx=1)
        assert scale_names(quantized) == {"wkv.weight_scale"}
        assert quantized.wkv.weight.dtype is torch.float8_e4m3fn

    def test_the_gates_beside_it_stay_unquantized(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        engram = DeepseekV41EngramConnection(config, layer_idx=1)
        assert engram.q_weight.dtype is torch.float32
        assert engram.k_weight.dtype is torch.float32
