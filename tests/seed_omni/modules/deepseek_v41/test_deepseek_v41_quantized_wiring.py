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

from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
    QUANT_BLOCK_SIZE,
    SCALE_DTYPE,
    DeepseekV41Attention,
    DeepseekV41EngramConnection,
    DeepseekV41GroupedLinear,
    DeepseekV41Indexer,
    DeepseekV41QuantizedExperts,
    DeepseekV41SharedExpert,
    expert_quant_dtype,
    linear_quant_dtype,
)
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

    def test_quantized_shared_expert_bypasses_plain_linear_gated_mlp_fusion(self):
        from veomni.ops.kernels.gated_mlp._module import plain_projections

        config = make_config(weight_quantization=RELEASED_QUANTIZATION)
        plain = DeepseekV41SharedExpert(make_config())
        quantized = DeepseekV41SharedExpert(config, quant_dtype=linear_quant_dtype(config))
        assert plain_projections(plain)
        assert not plain_projections(quantized)


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

    def test_quantizing_ops_force_bf16_storage_and_load_time_dequantization(self):
        """Selecting a quantizing training op (including via per-site ``OpsSelection``)
        on ``deepseek_v41_llm`` turns off quantized inference storage even when
        ``config.weight_quantization`` is present on disk."""
        from veomni.models.seed_omni.modules.deepseek_v41.checkpoint_tensor_converter import (
            DeepseekV41LlmConverter,
        )
        from veomni.models.seed_omni.modules.deepseek_v41.quant_scheme import quantized_storage_dtype

        for ops_impl in (
            {"moe_implementation": "deepgemm"},
            {"linear_implementation": "qat"},
            {"gated_mlp_implementation": "torchao"},
            {
                "linear_implementation": {
                    "name": "eager",
                    "overrides": [{"site": r"layers\.0\.q_a_proj", "name": "torchao", "kwargs": {"dtype": "mxfp4"}}],
                }
            },
        ):
            config = make_config(weight_quantization=RELEASED_QUANTIZATION, ops_implementation=ops_impl)
            assert quantized_storage_dtype(config) is None, ops_impl
            assert linear_quant_dtype(config) is None, ops_impl
            assert expert_quant_dtype(config) is None, ops_impl
            converter = DeepseekV41LlmConverter(config, dequantize=quantized_storage_dtype(config) is None)
            assert converter.dequantize is True

    def test_deepgemm_rejects_quantized_storage_and_coexists_with_qat_shared_expert(self, monkeypatch):
        from unittest.mock import Mock

        import veomni.models.seed_omni.modules.deepseek_v41.llm.modeling as quant_mod
        from veomni.ops.kernels.moe_experts.standard import deepgemm as moe_deepgemm

        monkeypatch.setattr(
            quant_mod,
            "VeomniOp",
            lambda _domain, _variant, impl, **_kw: Mock(
                impl=impl or "deepgemm",
                side_effect=moe_deepgemm.wrapper,
            ),
        )
        config_deepgemm = make_config(ops_implementation={"moe_implementation": "deepgemm"})
        experts_quant = DeepseekV41QuantizedExperts(config_deepgemm, quant_dtype=torch.float4_e2m1fn_x2)
        x = torch.randn(2, config_deepgemm.hidden_size, dtype=torch.bfloat16)
        top_k_idx = torch.zeros(2, config_deepgemm.num_experts_per_tok, dtype=torch.long)
        top_k_w = torch.ones(2, config_deepgemm.num_experts_per_tok, dtype=torch.bfloat16)
        # Pre-quantized FP4 experts fail fast when invoked under training MoE ops.
        with pytest.raises(ValueError, match="holds pre-quantized weights"):
            experts_quant(x, top_k_idx, top_k_w)

        # A ``qat`` shared expert beside ``deepgemm`` experts is legal -- they are
        # separate GEMMs.
        config_shared_qat = make_config(
            ops_implementation={"moe_implementation": "deepgemm", "gated_mlp_implementation": "qat"},
        )
        experts = DeepseekV41QuantizedExperts(config_shared_qat, quant_dtype=None)
        assert experts.veomni_moe.impl == "deepgemm"


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


class TestKvStorageRecipes:
    """Window / compressed / indexer use the released V4.1 recipes, not V4's FP8/64 NoPE helper."""

    def test_training_without_qat_is_a_no_op(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            round_compress_kv,
            round_index_act,
            round_window_kv,
        )

        kv = torch.randn(2, 3, 32, dtype=torch.bfloat16)
        torch.testing.assert_close(round_window_kv(kv, training=True), kv)
        torch.testing.assert_close(round_compress_kv(kv, training=True), kv)
        torch.testing.assert_close(round_index_act(kv, training=True), kv)

    def test_eval_rounds_window_kv_as_fp8_over_the_whole_head(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import WINDOW_KV_FP8_BLOCK, round_window_kv

        assert WINDOW_KV_FP8_BLOCK == 32
        # 32 is the official group and is *not* divisible by V4 QAT's 64-wide NoPE block.
        kv = torch.full((1, 2, 32), 100.0, dtype=torch.bfloat16)
        rounded = round_window_kv(kv, training=False)
        assert rounded.shape == kv.shape
        assert rounded.dtype == kv.dtype
        assert not torch.equal(rounded, kv)

    def test_eval_rounds_compressed_kv_as_fp4_groups_of_16(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            COMPRESS_KV_FP4_GROUP,
            round_compress_kv,
        )

        assert COMPRESS_KV_FP4_GROUP == 16
        # 16 matches compressed KV and would raise if this still used the indexer's 32.
        kv = torch.full((1, 2, 16), 100.0, dtype=torch.bfloat16)
        rounded = round_compress_kv(kv, "fp4", training=False)
        assert rounded.shape == kv.shape
        assert not torch.equal(rounded, kv)

    def test_eval_rounds_compressed_kv_as_fp8_when_compress_kv_dtype_is_fp8(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            WINDOW_KV_FP8_BLOCK,
            round_compress_kv,
            round_window_kv,
        )
        from veomni.ops import VeomniOp

        kv = torch.randn(1, 2, WINDOW_KV_FP8_BLOCK, dtype=torch.bfloat16, requires_grad=True)
        rounded_fp8 = round_compress_kv(kv, "fp8", training=True, op_fp8=VeomniOp("kv_round", "standard", "eager"))
        expected_fp8 = round_window_kv(kv.detach(), training=False)
        rounded_fp4 = round_compress_kv(kv.detach(), "fp4", training=False)
        torch.testing.assert_close(rounded_fp8, expected_fp8)
        assert not torch.equal(rounded_fp8, rounded_fp4)
        rounded_fp8.sum().backward()
        torch.testing.assert_close(kv.grad, torch.ones_like(kv))

    def test_attention_wires_compress_kv_dtype_from_config(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import DeepseekV41Attention

        cfg_fp4 = make_config(compress_kv_dtype="fp4")
        cfg_fp8 = make_config(compress_kv_dtype="fp8")
        assert DeepseekV41Attention(cfg_fp4, layer_idx=2).compress_kv_dtype == "fp4"
        assert DeepseekV41Attention(cfg_fp8, layer_idx=2).compress_kv_dtype == "fp8"

    def test_eval_rounds_indexer_as_fp4_groups_of_32(self):
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import INDEX_FP4_GROUP, round_index_act

        assert INDEX_FP4_GROUP == 32
        keys = torch.full((1, 4, 32), 50.0, dtype=torch.bfloat16)
        rounded = round_index_act(keys, training=False)
        assert rounded.shape == keys.shape
        assert not torch.equal(rounded, keys)


class TestV41QatAlignment:
    """V4.1 low-precision training is controlled per family and per site via VeOmniOps."""

    def test_the_recipe_comes_from_config_kwargs_not_the_call_site(self, monkeypatch):
        """The call site passes model semantics only; the format and block width are config.

        Two things must hold, and they pull in opposite directions:

        * The call site must **not** hard-code the QAT recipe. A `block_size` baked in
          here would be handed to every row, and `deepgemm` refuses any width but 128
          -- so an all-`deepgemm` config used to die at the first projection with
          "deepgemm supports only block_size=128, got 32". The same reasoning makes the
          routed experts pass `dtype=self.expert_dtype` (the checkpoint's on-disk
          format) rather than a QAT-only `"fp8"`.
        * The recipe must still reach the row. It does, through
          `OpsSelection.kwargs`, which `VeomniOp.__call__` merges *over* the call
          arguments -- so a config that sets `block_size: 32` wins over the row's
          default 128, and one that sets nothing gets the row's own recipe.

        This test pins the first half at the model's own call sites. The second half is
        `VeomniOp`'s contract and is pinned by the ops tests.
        """
        from unittest.mock import Mock

        import veomni.models.seed_omni.modules.deepseek_v41.llm.modeling as quant_mod

        gated_calls = []
        moe_calls = []
        linear_calls = []

        def fake_veomni_op(domain, variant, impl, *, site=None):
            if domain == "gated_mlp":
                return Mock(
                    impl="qat",
                    side_effect=lambda mlp, x, **kw: (gated_calls.append((site, kw)), x)[1],
                )
            if domain == "moe_experts":
                return Mock(
                    impl="qat",
                    side_effect=lambda h, *_a, **kw: (moe_calls.append((site, kw)), h)[1],
                )
            if domain == "linear":
                return Mock(
                    impl="qat",
                    side_effect=lambda mod, x, **kw: (linear_calls.append((site, kw)), x)[1],
                )
            return Mock(impl="eager")

        monkeypatch.setattr(quant_mod, "VeomniOp", fake_veomni_op)

        config = make_config(
            expert_dtype="fp4",
            ops_implementation={"gated_mlp_implementation": "qat", "moe_implementation": "qat"},
        )
        shared = DeepseekV41SharedExpert(config, quant_dtype=None, layer_idx=3).to(torch.bfloat16)
        x = torch.randn(2, 4, config.hidden_size, dtype=torch.bfloat16)
        shared(x)
        # No `block_size` here: the config's `OpsSelection.kwargs` supplies it.
        assert gated_calls == [
            ("layers.3.mlp", {"swiglu_limit": config.swiglu_limit, "activation": config.hidden_act})
        ]

        experts = DeepseekV41QuantizedExperts(config, quant_dtype=None, layer_idx=3).to(torch.bfloat16)
        top_k_idx = torch.zeros(8, 2, dtype=torch.long)
        top_k_w = torch.ones(8, 2, dtype=torch.float32) * 0.5
        experts(x.reshape(8, -1), top_k_idx, top_k_w)
        # `dtype` follows the checkpoint (`config.expert_dtype`), and no block width is
        # baked in -- a `deepgemm` MoE would reject one.
        assert moe_calls == [
            (
                "layers.3.experts",
                {
                    "num_experts": config.n_routed_experts,
                    "activation": config.hidden_act,
                    "swiglu_limit": config.swiglu_limit,
                    "assume_distinct_experts": False,
                    "dtype": config.expert_dtype,
                },
            )
        ]

        # And a dense projection, so all three quantizing families are covered.
        attn = DeepseekV41Attention(config, layer_idx=3).to(torch.bfloat16)
        quant_mod.dispatch_linear(attn.veomni_wq_a, attn.wq_a, x)
        assert linear_calls == [("layers.3.q_a_proj", {})]

    def test_deepgemm_wires_linear_gated_mlp_and_kv_round_ops(self):
        """Under an all-``deepgemm`` selection, dense/grouped linears, shared
        experts, and KV/indexer rounding all route through registered VeomniOps.

        Every family is named explicitly. A ``deepgemm`` MoE does not turn the dense
        projections into FP8 GEMMs, and it does not turn KV rounding on either.
        """
        from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            DeepseekV41Attention,
            resolve_module_op_impl,
        )

        full_cfg = DeepseekV41LlmConfig(
            weight_quantization=None,
            compress_kv_dtype="fp4",
            ops_implementation={
                "moe_implementation": "deepgemm",
                "linear_implementation": "deepgemm",
                "gated_mlp_implementation": "deepgemm",
            },
        )
        assert resolve_module_op_impl(full_cfg, "linear_implementation") == "deepgemm"
        assert resolve_module_op_impl(full_cfg, "gated_mlp_implementation") == "deepgemm"

        small_cfg = make_config(
            compress_kv_dtype="fp4",
            ops_implementation={
                "moe_implementation": "deepgemm",
                "kv_round_implementation": "eager",
            },
        )
        attn = DeepseekV41Attention(small_cfg, layer_idx=0)
        assert attn.veomni_kv_round_fp8.op == "kv_round" and attn.veomni_kv_round_fp8.variant == "standard"
        assert attn.veomni_kv_round_fp8.impl == "eager"
        assert attn.veomni_kv_round_compress_fp8.op == "kv_round" and attn.veomni_kv_round_compress_fp8.impl == "eager"
        assert attn.veomni_kv_round_fp4.op == "kv_round" and attn.veomni_kv_round_fp4.variant == "fp4"
        assert attn.veomni_kv_round_fp4.impl == "eager"
        assert attn.veomni_wq_a.op == "linear" and attn.veomni_wq_a.variant == "standard"
        assert attn.veomni_wo_a.op == "linear" and attn.veomni_wo_a.variant == "grouped"
        assert attn.indexer.veomni_kv_round_fp4.op == "kv_round" and attn.indexer.veomni_kv_round_fp4.variant == "fp4"
        assert attn.indexer.veomni_kv_round_fp4.impl == "eager"

    def test_a_deepgemm_moe_alone_does_not_select_the_dense_families(self):
        """Selecting only the MoE family leaves ``linear``/``gated_mlp`` alone."""
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import resolve_module_op_impl

        cfg = make_config(
            compress_kv_dtype="fp4",
            ops_implementation={"moe_implementation": "deepgemm"},
        )
        assert resolve_module_op_impl(cfg, "linear_implementation") != "deepgemm"
        assert resolve_module_op_impl(cfg, "gated_mlp_implementation") != "deepgemm"

    def test_an_all_qat_selection_resolves_every_gemm_family(self):
        """Each GEMM family selects ``qat`` independently."""
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import resolve_module_op_impl

        cfg = make_config(
            expert_dtype="fp8",
            compress_kv_dtype="fp4",
            ops_implementation={
                "linear_implementation": "qat",
                "gated_mlp_implementation": "qat",
                "moe_implementation": "qat",
            },
        )
        for field in ("linear_implementation", "gated_mlp_implementation", "moe_implementation"):
            assert resolve_module_op_impl(cfg, field) == "qat"

    def test_qat_resolves_linear_gated_mlp_and_moe_to_registered_qat_ops(self, monkeypatch):
        """With all three families selected as ``qat``, each resolves to ``'qat'``."""
        import veomni.ops.kernels.gated_mlp.standard.qat as gated_mlp_qat
        import veomni.ops.kernels.moe_experts.standard.qat as moe_qat
        from veomni.models.seed_omni.modules.deepseek_v41.llm.configuration import DeepseekV41LlmConfig
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import resolve_module_op_impl

        qat_cfg = DeepseekV41LlmConfig(
            weight_quantization=None,
            expert_dtype="fp8",
            compress_kv_dtype="fp4",
            ops_implementation={
                "linear_implementation": "qat",
                "gated_mlp_implementation": "qat",
                "moe_implementation": "qat",
            },
        )
        assert resolve_module_op_impl(qat_cfg, "linear_implementation") == "qat"
        assert resolve_module_op_impl(qat_cfg, "gated_mlp_implementation") == "qat"
        assert resolve_module_op_impl(qat_cfg, "moe_implementation") == "qat"

        # Verify gated_mlp/qat and moe_experts/qat wrappers honor block_size=32 and dtype
        gated_mlp_calls = []
        monkeypatch.setattr(
            gated_mlp_qat,
            "fp8_fake_quant_act",
            lambda t, block_size=128: (gated_mlp_calls.append(("act", block_size)), t)[1],
        )
        monkeypatch.setattr(
            gated_mlp_qat,
            "fp8_fake_quant_weight",
            lambda t, block_size=128: (gated_mlp_calls.append(("w", block_size)), t)[1],
        )
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        gw = torch.randn(96, 64, dtype=torch.bfloat16)
        uw = torch.randn(96, 64, dtype=torch.bfloat16)
        dw = torch.randn(64, 96, dtype=torch.bfloat16)
        empty = gw.new_empty(0)

        class _Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(64, 96, bias=False)
                self.up_proj = torch.nn.Linear(64, 96, bias=False)
                self.down_proj = torch.nn.Linear(96, 64, bias=False)

        mlp = _Mlp()
        mlp.gate_proj.weight = torch.nn.Parameter(gw)
        mlp.up_proj.weight = torch.nn.Parameter(uw)
        mlp.down_proj.weight = torch.nn.Parameter(dw)
        out = gated_mlp_qat.wrapper(mlp, x, swiglu_limit=7.0, block_size=32)
        assert out.shape == (2, 4, 64)
        assert gated_mlp_calls == [("act", 32), ("w", 32), ("w", 32), ("act", 32), ("w", 32)]

        moe_calls = []
        monkeypatch.setattr(
            moe_qat, "fp8_fake_quant_act", lambda t, block_size=128: (moe_calls.append(("act", block_size)), t)[1]
        )
        monkeypatch.setattr(
            moe_qat,
            "fp8_fake_quant_stacked_weight",
            lambda t, block_size=128: (moe_calls.append(("fp8_w", block_size)), t)[1],
        )
        monkeypatch.setattr(
            moe_qat.standard_triton,
            "wrapper",
            lambda h, *_a, **_kw: h,
        )
        h = torch.randn(8, 64, dtype=torch.bfloat16)
        rw = torch.ones(8, 2, dtype=torch.float32) * 0.5
        sel = torch.zeros(8, 2, dtype=torch.long)
        fc1 = torch.randn(2, 192, 64, dtype=torch.bfloat16)
        fc2 = torch.randn(2, 64, 96, dtype=torch.bfloat16)
        moe_out = moe_qat.wrapper(
            h, rw, sel, empty, empty, fc2, fc1, num_experts=2, swiglu_limit=7.0, block_size=32, dtype="fp8"
        )
        assert moe_out.shape == (8, 64)
        assert moe_calls == [("act", 32), ("fp8_w", 32), ("fp8_w", 32)]

    def test_explicit_kv_round_implementation_controls_storage_rounding_in_training(self):
        """``kv_round_implementation`` controls FP8/FP4 KV & indexer storage rounding
        in training directly via bound ``VeomniOp`` handles (including per-site rules)."""
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            DeepseekV41Attention,
            resolve_module_op_impl,
            round_compress_kv,
            round_index_act,
            round_window_kv,
        )

        bf16_cfg = make_config(
            compress_kv_dtype="fp4",
            ops_implementation={"moe_implementation": "eager", "kv_round_implementation": "eager"},
        )
        attn = DeepseekV41Attention(bf16_cfg, layer_idx=0)
        assert attn.veomni_kv_round_fp8.impl == "eager"
        assert attn.veomni_kv_round_fp4.impl == "eager"
        assert attn.indexer.veomni_kv_round_fp4.impl == "eager"

        kv = torch.full((1, 2, 32), 100.0, dtype=torch.bfloat16)
        assert not torch.equal(round_window_kv(kv, training=True, op=attn.veomni_kv_round_fp8), kv)
        assert not torch.equal(round_compress_kv(kv, "fp4", training=True, op_fp4=attn.veomni_kv_round_fp4), kv)
        assert not torch.equal(round_index_act(kv, training=True, op=attn.indexer.veomni_kv_round_fp4), kv)

        # Unset or explicit ``none`` must NOT round during training, while eval
        # still rounds to match deployment caches.
        for pure_ops in (
            {"moe_implementation": "eager"},
            {"moe_implementation": "eager", "kv_round_implementation": "none"},
        ):
            pure_bf16_cfg = make_config(compress_kv_dtype="fp4", ops_implementation=pure_ops)
            pure_attn = DeepseekV41Attention(pure_bf16_cfg, layer_idx=0)
            assert pure_attn.veomni_kv_round_fp8.impl == "none"
            assert pure_attn.veomni_kv_round_fp4.impl == "none"
            assert pure_attn.indexer.veomni_kv_round_fp4.impl == "none"
            assert torch.equal(round_window_kv(kv, training=True, op=pure_attn.veomni_kv_round_fp8), kv)
            assert torch.equal(round_compress_kv(kv, "fp4", training=True, op_fp4=pure_attn.veomni_kv_round_fp4), kv)
            assert torch.equal(round_index_act(kv, training=True, op=pure_attn.indexer.veomni_kv_round_fp4), kv)
            assert not torch.equal(round_window_kv(kv, training=False, op=pure_attn.veomni_kv_round_fp8), kv)

        # An explicit ``none`` stays the string ``none`` rather than collapsing to ``None``.
        explicit_none = make_config(ops_implementation={"kv_round_implementation": "none"})
        assert resolve_module_op_impl(explicit_none, "kv_round_implementation") == "none"
        unset = make_config(ops_implementation={})
        assert resolve_module_op_impl(unset, "kv_round_implementation") is None

    def test_per_site_ops_selection_controls_every_v41_site(self):
        """Per-site ``OpsSelection`` rules bind independent rows/kwargs across
        attention projections, indexer, engram, DSpark, shared MLP, experts, and kv_round."""
        from veomni.models.seed_omni.modules.deepseek_v41.llm.modeling import (
            DeepseekV41DecoderLayer,
            DeepseekV41EngramConnection,
            dispatch_linear,
        )
        from veomni.models.seed_omni.modules.deepseek_v41.mtp.modeling import DeepseekV41DSparkBlock
        from veomni.utils.import_utils import is_package_available

        if not is_package_available("torchao"):
            pytest.skip("torchao is required to bind torchao rows")

        cfg = make_config(
            hidden_size=64,
            head_dim=32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=2,
            num_attention_heads=4,
            index_n_heads=2,
            index_head_dim=16,
            moe_intermediate_size=64,
            n_routed_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            compress_ratios=[2, 1, 0],
            kv_source_layer_ids=[0, 1],
            index_source_layer_ids=[0, 1],
            candidate_source_layer_id=1,
            engram_layer_ids=[0, 1],
            dspark_target_layer_ids=[1],
            ops_implementation={
                "linear_implementation": {
                    "name": "eager",
                    "overrides": [
                        {"site": r"layers\.0\.(q_a_proj|engram_wkv)", "name": "torchao", "kwargs": {"dtype": "mxfp4"}},
                        {"site": r"layers\.2\.main_proj", "name": "torchao", "kwargs": {"dtype": "nvfp4"}},
                    ],
                },
                "gated_mlp_implementation": {
                    "name": "eager",
                    "overrides": [
                        {"site": r"layers\.1\.mlp", "name": "torchao", "kwargs": {"dtype": "mxfp4"}},
                    ],
                },
                "moe_implementation": "eager",
                "kv_round_implementation": {
                    "name": "none",
                    "overrides": [
                        {"site": r"layers\.0\.window_kv", "name": "eager"},
                        {"site": r"layers\.1\.indexer", "name": "eager"},
                    ],
                },
            },
        )
        layer0 = DeepseekV41DecoderLayer(cfg, layer_idx=0)
        layer1 = DeepseekV41DecoderLayer(cfg, layer_idx=1)
        engram0 = DeepseekV41EngramConnection(cfg, layer_idx=0)
        engram1 = DeepseekV41EngramConnection(cfg, layer_idx=1)
        dspark0 = DeepseekV41DSparkBlock(cfg, layer_idx=2)

        assert layer0.self_attn.veomni_wq_a.impl == "torchao"
        assert layer0.self_attn.veomni_wq_a.kwargs == {"dtype": "mxfp4"}
        assert layer0.self_attn.veomni_wkv.impl == "eager"
        assert layer0.self_attn.veomni_wo_a.impl == "eager"
        assert layer1.self_attn.veomni_wq_a.impl == "eager"
        assert engram0.veomni_wkv.impl == "torchao"
        assert engram1.veomni_wkv.impl == "eager"
        assert dspark0.veomni_main_proj.impl == "torchao"
        assert dspark0.veomni_main_proj.kwargs == {"dtype": "nvfp4"}

        assert layer0.mlp.shared_experts.veomni_gated_mlp.impl == "eager"
        assert layer1.mlp.shared_experts.veomni_gated_mlp.impl == "torchao"
        assert layer1.mlp.shared_experts.veomni_gated_mlp.kwargs == {"dtype": "mxfp4"}

        assert layer0.self_attn.veomni_kv_round_fp8.impl == "eager"
        assert layer0.self_attn.veomni_kv_round_fp4.impl == "none"
        assert layer0.self_attn.indexer.veomni_kv_round_fp4.impl == "none"
        assert layer1.self_attn.veomni_kv_round_fp8.impl == "none"
        assert layer1.self_attn.indexer.veomni_kv_round_fp4.impl == "eager"

        # Verify forward + backward through per-site dispatched projections and
        # DeepseekV41SharedExpert (torchao mxfp4) on BF16.
        layer0 = layer0.to(torch.bfloat16)
        layer1 = layer1.to(torch.bfloat16)
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16, requires_grad=True)
        q_a = dispatch_linear(layer0.self_attn.veomni_wq_a, layer0.self_attn.wq_a, x)
        kv_out = dispatch_linear(layer0.self_attn.veomni_wkv, layer0.self_attn.wkv, x)
        mlp_out = layer1.mlp.shared_experts(x)
        assert q_a.shape == (2, 4, 32) and kv_out.shape == (2, 4, 32) and mlp_out.shape == (2, 4, 64)
        (q_a.sum() + kv_out.sum() + mlp_out.sum()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert (
            layer0.self_attn.wq_a.weight.grad is not None and torch.isfinite(layer0.self_attn.wq_a.weight.grad).all()
        )
        assert layer1.mlp.shared_experts.down_proj.weight.grad is not None


def test_quantized_storage_fails_fast_under_training_ops():
    """Training ops refuse pre-quantized weight storage at runtime rather than silently casting."""
    eager_quant_cfg = make_config(
        weight_quantization=RELEASED_QUANTIZATION,
        ops_implementation={
            "linear_implementation": "eager",
            "gated_mlp_implementation": "eager",
            "moe_implementation": "eager",
        },
    )
    x = torch.randn(2, eager_quant_cfg.hidden_size, dtype=torch.bfloat16)
    shared = DeepseekV41SharedExpert(eager_quant_cfg, quant_dtype=linear_quant_dtype(eager_quant_cfg))
    with pytest.raises(ValueError, match="holds pre-quantized weights"):
        shared(x)

    grouped = DeepseekV41GroupedLinear(64, 128, n_groups=2, quant_dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="holds pre-quantized weights"):
        grouped(torch.randn(2, 2, 64, dtype=torch.bfloat16))

    experts = DeepseekV41QuantizedExperts(eager_quant_cfg, quant_dtype=expert_quant_dtype(eager_quant_cfg))
    top_k_idx = torch.zeros(2, eager_quant_cfg.num_experts_per_tok, dtype=torch.long)
    top_k_w = torch.ones(2, eager_quant_cfg.num_experts_per_tok, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="holds pre-quantized weights"):
        experts(x, top_k_idx, top_k_w)
