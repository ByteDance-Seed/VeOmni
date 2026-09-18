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

from veomni.models.seed_omni.modules.deepseek_v41.dequant import dequant_fp8_blockwise
from veomni.models.seed_omni.modules.deepseek_v41.llm.attention import (
    DeepseekV41Attention,
    DeepseekV41Indexer,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.engram_connection import DeepseekV41EngramConnection
from veomni.models.seed_omni.modules.deepseek_v41.llm.quantized import (
    QUANT_BLOCK_SIZE,
    SCALE_DTYPE,
    SCALE_FMT,
    DeepseekV41GroupedLinear,
    DeepseekV41QuantizedExperts,
    DeepseekV41SharedExpert,
    QuantizedEPExperts,
    expert_quant_dtype,
    linear_quant_dtype,
)
from veomni.models.seed_omni.modules.deepseek_v41.llm.reuse import DeepseekV4Experts, DeepseekV4GroupedLinear
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()
requires_sm90 = pytest.mark.skipif(
    not IS_CUDA_AVAILABLE or torch.version.hip is not None or get_gpu_compute_capability() < 90,
    reason="the quantized path runs on TileLang kernels needing an SM90 or later NVIDIA GPU",
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
        q_lora_rank=64,
        # Both must be whole numbers of 32-row scale tiles, as they are in the
        # release (q_lora_rank 1280, o_lora_rank 1024). The other V4.1 test
        # fixtures predate quantization and use sizes that are not.
        o_lora_rank=32,
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
            "wo_a.weight_scale",
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


class TestGroupedOutputProjection:
    """``wo_a`` holds one flat weight that its forward views as ``n_groups`` blocks."""

    def test_a_group_that_does_not_divide_into_scale_tiles_is_refused(self):
        with pytest.raises(ValueError, match="cannot be sliced per group"):
            DeepseekV41GroupedLinear(64, 2 * 48, n_groups=2, quant_dtype=torch.float8_e4m3fn)

    def test_the_scale_covers_the_flat_weight_not_a_block(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        wo_a = DeepseekV41Attention(config, layer_idx=0).wo_a
        out_features, in_per_group = wo_a.weight.shape
        assert tuple(wo_a.weight_scale.shape) == (
            math.ceil(out_features / QUANT_BLOCK_SIZE),
            math.ceil(in_per_group / QUANT_BLOCK_SIZE),
        )

    @requires_sm90
    def test_the_block_diagonal_forward_reproduces_the_bmm(self):
        """The looped per-group fp8 GEMMs must land where the bf16 ``bmm`` does.

        Both sides are given the same weight -- one as fp8 plus scale, the other
        dequantized from exactly that pair -- so the weight rounding cancels and
        what is left is the kernel's own error plus the activation's, which the
        quantized side takes and the reference does not.

        Slicing is the thing being tested. Handing a group the neighbouring
        group's scale rows would still produce a finite, right-shaped answer.
        """
        from veomni.models.seed_omni.modules.deepseek_v41.llm.quantized import act_quant

        torch.manual_seed(0)
        n_groups, in_per_group, out_per_group = 4, 128, 64
        shape = dict(in_features_per_group=in_per_group, out_features=n_groups * out_per_group, n_groups=n_groups)

        quantized = DeepseekV41GroupedLinear(**shape, quant_dtype=torch.float8_e4m3fn).to(DEVICE)
        reference_weight = torch.randn(n_groups * out_per_group, in_per_group, dtype=torch.bfloat16, device=DEVICE)
        fp8_weight, row_scale = act_quant(reference_weight * 0.1, QUANT_BLOCK_SIZE, SCALE_FMT, SCALE_DTYPE)
        tile_scale = row_scale[::QUANT_BLOCK_SIZE, :]
        with torch.no_grad():
            quantized.weight.copy_(fp8_weight)
            quantized.weight_scale.copy_(tile_scale)

        reference = DeepseekV4GroupedLinear(in_per_group, n_groups * out_per_group, n_groups=n_groups)
        reference = reference.to(device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            reference.weight.copy_(dequant_fp8_blockwise(fp8_weight, tile_scale, torch.bfloat16))

        x = torch.randn(6, n_groups, in_per_group, dtype=torch.bfloat16, device=DEVICE)
        expected = reference(x)
        actual = quantized(x)

        assert actual.shape == expected.shape
        # Scaled to the outputs at hand rather than absolute: the products here
        # are order 1, and an atol alone would pass a group reading the wrong
        # scale if that scale happened to be small.
        assert (actual - expected).abs().max() < 0.05 * expected.abs().max()


class TestSharedExpert:
    def test_all_three_projections_are_quantized(self):
        plain = DeepseekV41SharedExpert(make_config())
        assert scale_names(plain) == set()

        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        quantized = DeepseekV41SharedExpert(config, quant_dtype=linear_quant_dtype(config))
        assert scale_names(quantized) == {
            "gate_proj.weight_scale",
            "up_proj.weight_scale",
            "down_proj.weight_scale",
        }

    def test_quantizing_does_not_move_a_projection(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        plain = DeepseekV41SharedExpert(make_config())
        quantized = DeepseekV41SharedExpert(config, quant_dtype=linear_quant_dtype(config))
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert getattr(plain, name).weight.shape == getattr(quantized, name).weight.shape


class TestRoutedExperts:
    """The fp4 slabs -- the bulk of the parameters, and the only 4-bit weights."""

    def test_bf16_leaves_the_slabs_alone(self):
        experts = DeepseekV41QuantizedExperts(make_config())
        assert not experts.experts_are_quantized
        assert experts.gate_up_proj.dtype is torch.float32
        assert not hasattr(experts, "gate_up_proj_scale")

    def test_the_slabs_are_packed_two_values_to_a_byte(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        experts = DeepseekV41QuantizedExperts(config, quant_dtype=expert_quant_dtype(config))
        n_experts = config.n_routed_experts
        hidden, intermediate = config.hidden_size, config.moe_intermediate_size

        # int8 and half the columns: safetensors has no code for a 4-bit float.
        assert experts.gate_up_proj.dtype is torch.int8
        assert tuple(experts.gate_up_proj.shape) == (n_experts, 2 * intermediate, hidden // 2)
        assert tuple(experts.down_proj.shape) == (n_experts, hidden, intermediate // 2)

    def test_a_scale_is_per_row_not_per_tile(self):
        """Experts do not share a range with each other, so neither do their scales."""
        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        experts = DeepseekV41QuantizedExperts(config, quant_dtype=expert_quant_dtype(config))
        n_experts = config.n_routed_experts
        hidden, intermediate = config.hidden_size, config.moe_intermediate_size

        assert tuple(experts.gate_up_proj_scale.shape) == (
            n_experts,
            2 * intermediate,
            hidden // QUANT_BLOCK_SIZE,
        )
        assert tuple(experts.down_proj_scale.shape) == (n_experts, hidden, intermediate // QUANT_BLOCK_SIZE)
        assert experts.gate_up_proj_scale.dtype is SCALE_DTYPE

    def test_an_fp8_expert_release_is_refused_rather_than_misread(self):
        config = make_config(weight_quantization=RELEASED_QUANTIZATION, expert_dtype="fp8")
        with pytest.raises(ValueError, match="different scale layout"):
            expert_quant_dtype(config)

    @requires_sm90
    def test_the_routed_forward_matches_its_dequantized_slabs(self):
        """Routing, gating and weighting must survive the slabs going to 4 bits.

        The reference is the generated eager loop over bf16 weights dequantized
        from the very slabs under test, so the weight rounding cancels and what
        is left is the activation's, which only the quantized side pays.

        The bounds come from measuring one quantized GEMM in isolation: about
        2.7% relative L2, all of it the activation. This chains two of them with
        a SwiGLU between, and lands near 5%, which is that error accumulating
        rather than anything going wrong.

        The relative L2 is the bound that matters, not the max. It is what
        catches the mistake worth catching -- an expert or a scale block read
        from the wrong offset puts it at order 1, nowhere near 8%, however
        ordinary the largest single deviation might look.
        """
        from veomni.ops.kernels.deepseek_v4 import fp4_act_quant

        torch.manual_seed(0)
        config = make_config(weight_quantization=RELEASED_QUANTIZATION, n_routed_experts=4, num_experts_per_tok=2)
        quantized = DeepseekV41QuantizedExperts(config, quant_dtype=expert_quant_dtype(config)).to(DEVICE)
        reference = DeepseekV4Experts(config).to(device=DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            for name in ("gate_up_proj", "down_proj"):
                plain = torch.randn(getattr(reference, name).shape, dtype=torch.bfloat16, device=DEVICE) * 0.1
                packed, scale = fp4_act_quant(plain, QUANT_BLOCK_SIZE)
                getattr(quantized, name).copy_(packed.view(torch.int8))
                getattr(quantized, f"{name}_scale").copy_(scale)
                getattr(reference, name).copy_(fp4_act_quant(plain, QUANT_BLOCK_SIZE, dequant=True))

        n_tokens = 8
        hidden_states = torch.randn(n_tokens, config.hidden_size, dtype=torch.bfloat16, device=DEVICE)
        top_k_index = torch.randint(0, config.n_routed_experts, (n_tokens, 2), device=DEVICE)
        top_k_weights = torch.rand(n_tokens, 2, dtype=torch.bfloat16, device=DEVICE)

        expected = reference(hidden_states, top_k_index, top_k_weights).float()
        actual = quantized(hidden_states, top_k_index, top_k_weights).float()

        assert actual.shape == expected.shape
        assert (actual - expected).abs().max() < 0.10 * expected.abs().max()
        assert (actual - expected).norm() / expected.norm() < 0.08


class TestExpertParallelLocalHalf:
    """What ``dispatch_to_ep_class`` calls between its two all-to-alls.

    Exercised directly rather than through a process group: the all-to-all is
    VeOmni's and already tested, while the part that is new here is reading each
    local expert's slice out of one permuted buffer.
    """

    @requires_sm90
    def test_each_expert_gets_exactly_its_own_tokens(self):
        from veomni.ops.kernels.deepseek_v4 import fp4_act_quant

        torch.manual_seed(0)
        n_local_experts, hidden, intermediate = 3, 64, 96
        counts = [2, 0, 5]  # an expert with no tokens is the case worth covering
        cumsum = torch.tensor(counts, device=DEVICE).cumsum(0)

        def slab(out_features, in_features):
            plain = torch.randn(n_local_experts, out_features, in_features, dtype=torch.bfloat16, device=DEVICE) * 0.1
            packed, scale = fp4_act_quant(plain, QUANT_BLOCK_SIZE)
            return packed.view(torch.int8), scale, fp4_act_quant(plain, QUANT_BLOCK_SIZE, dequant=True)

        gate_up, gate_up_scale, gate_up_ref = slab(2 * intermediate, hidden)
        down, down_scale, down_ref = slab(hidden, intermediate)

        tokens = torch.randn(sum(counts), hidden, dtype=torch.bfloat16, device=DEVICE)
        limit = 7.0

        def apply_gate(gate_up_output):
            gate, up = gate_up_output.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

        actual = QuantizedEPExperts.apply(tokens, cumsum, gate_up, gate_up_scale, down, down_scale, apply_gate)

        # The reference walks the same boundaries with plain matmuls, so a slice
        # taken from the wrong expert shows up as a whole row being wrong rather
        # than as rounding.
        expected = torch.empty_like(actual)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            chunk = tokens[start : start + count]
            gated = apply_gate(chunk.float() @ gate_up_ref[expert].float().T)
            expected[start : start + count] = (gated @ down_ref[expert].float().T).to(actual.dtype)
            start += count

        assert actual.shape == (sum(counts), hidden)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.08


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
