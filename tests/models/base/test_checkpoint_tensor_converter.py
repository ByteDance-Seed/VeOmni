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

"""
Tests for checkpoint tensor converters.

Tests the base protocol helpers (get_checkpoint_tensor_converter, maybe_convert_checkpoint_tensor)
and per-model converter implementations (e.g. Qwen3MoeCheckpointTensorConverter).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from veomni.models.checkpoint.convert import (
    ConvertedCheckpointTensor,
    checkpoint_converter_fused_expert_target,
    get_checkpoint_tensor_converter,
    get_fqn_to_index_mapping_converter,
    maybe_convert_checkpoint_tensor,
    split_moe_params,
)
from veomni.models.transformers.deepseek_v3.checkpoint_tensor_converter import (
    DeepseekV3CheckpointTensorConverter,
    create_deepseek_v3_checkpoint_tensor_converter,
)
from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import (
    DeepseekV4CheckpointTensorConverter,
    _dequantize_scaled_weight,
    _get_fp4_e2m1_table,
    convert_deepseek_v4_checkpoint_key,
    create_deepseek_v4_checkpoint_tensor_converter,
)
from veomni.models.transformers.glm_moe_dsa.checkpoint_tensor_converter import (
    GlmMoeDsaCheckpointTensorConverter,
    create_glm_moe_dsa_checkpoint_tensor_converter,
)
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import (
    Qwen3MoeCheckpointTensorConverter,
    create_qwen3_moe_checkpoint_tensor_converter,
)
from veomni.models.transformers.qwen3_omni_moe.checkpoint_tensor_converter import (
    Qwen3OmniMoeCheckpointTensorConverter,
    create_qwen3_omni_moe_checkpoint_tensor_converter,
)
from veomni.models.transformers.qwen3_vl_moe.checkpoint_tensor_converter import (
    Qwen3VLMoeCheckpointTensorConverter,
    create_qwen3_vl_moe_checkpoint_tensor_converter,
)


# ---------------------------------------------------------------------------
# Tests for base protocol helpers
# ---------------------------------------------------------------------------


class _DummyConverter:
    """Minimal converter that uppercases key names for testing."""

    def can_handle(self, name: str) -> bool:
        return name.startswith("handle_me")

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        return ConvertedCheckpointTensor(name=name.upper(), tensor=tensor)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        return []


class TestGetCheckpointTensorConverter:
    def test_returns_none_when_no_factory(self):
        model = torch.nn.Linear(4, 4)
        assert get_checkpoint_tensor_converter(model) is None

    def test_returns_converter_from_factory(self):
        model = torch.nn.Linear(4, 4)
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: _DummyConverter())
        converter = get_checkpoint_tensor_converter(model)
        assert converter is not None
        assert converter.can_handle("handle_me.foo")

    def test_ignores_non_callable_factory(self):
        model = torch.nn.Linear(4, 4)
        model._create_checkpoint_tensor_converter = "not_callable"
        assert get_checkpoint_tensor_converter(model) is None

    def test_ignores_factory_returning_none(self):
        model = torch.nn.Linear(4, 4)
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: None)
        assert get_checkpoint_tensor_converter(model) is None

    def test_ignores_invalid_converter(self):
        model = torch.nn.Linear(4, 4)
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: object())
        assert get_checkpoint_tensor_converter(model) is None


class TestMaybeConvertCheckpointTensor:
    def test_passthrough_when_no_converter(self):
        t = torch.randn(2, 3)
        result = maybe_convert_checkpoint_tensor("foo", t, converter=None)
        assert result is not None
        assert result.name == "foo"
        assert torch.equal(result.tensor, t)

    def test_passthrough_when_converter_cannot_handle(self):
        t = torch.randn(2, 3)
        converter = _DummyConverter()
        result = maybe_convert_checkpoint_tensor("other_key", t, converter)
        assert result is not None
        assert result.name == "other_key"

    def test_converts_when_converter_handles(self):
        t = torch.randn(2, 3)
        converter = _DummyConverter()
        result = maybe_convert_checkpoint_tensor("handle_me.weight", t, converter)
        assert result is not None
        assert result.name == "HANDLE_ME.WEIGHT"


# ---------------------------------------------------------------------------
# Tests for Qwen3MoeCheckpointTensorConverter
# ---------------------------------------------------------------------------


NUM_EXPERTS = 4
HIDDEN_DIM = 8
INTERMEDIATE_DIM = 6


def _make_expert_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def _make_expert_tensor(proj: str, expert_id: int) -> torch.Tensor:
    """Create a deterministic tensor for a given projection and expert id."""
    if proj == "down_proj":
        shape = (HIDDEN_DIM, INTERMEDIATE_DIM)
    else:
        shape = (INTERMEDIATE_DIM, HIDDEN_DIM)
    # Fill with expert_id + small offset per proj for easy verification
    offset = {"gate_proj": 0.0, "up_proj": 0.1, "down_proj": 0.2}[proj]
    return torch.full(shape, expert_id + offset)


class TestQwen3MoeConverterCanHandle:
    def setup_method(self):
        self.converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

    def test_matches_expert_keys(self):
        assert self.converter.can_handle("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert self.converter.can_handle("model.layers.3.mlp.experts.7.up_proj.weight")
        assert self.converter.can_handle("model.layers.10.mlp.experts.63.down_proj.weight")

    def test_rejects_non_expert_keys(self):
        assert not self.converter.can_handle("model.layers.0.self_attn.q_proj.weight")
        assert not self.converter.can_handle("model.layers.0.mlp.gate.weight")
        assert not self.converter.can_handle("model.layers.0.mlp.experts.gate_up_proj")
        assert not self.converter.can_handle("model.embed_tokens.weight")


class TestQwen3MoeConverterConvert:
    def _feed_all_experts(self, converter, layer: int, proj: str) -> List[Optional[ConvertedCheckpointTensor]]:
        """Feed all experts for a given layer and projection, return list of results."""
        results = []
        for expert_id in range(NUM_EXPERTS):
            key = _make_expert_key(layer, expert_id, proj)
            tensor = _make_expert_tensor(proj, expert_id)
            results.append(converter.convert(key, tensor))
        return results

    def test_buffers_until_all_experts_collected(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        # Feed first N-1 experts — should all return None
        for expert_id in range(NUM_EXPERTS - 1):
            key = _make_expert_key(0, expert_id, "down_proj")
            result = converter.convert(key, _make_expert_tensor("down_proj", expert_id))
            assert result is None, f"Expected None for expert {expert_id}, got {result}"

    def test_down_proj_emitted_after_all_experts(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        results = self._feed_all_experts(converter, layer=0, proj="down_proj")

        # First N-1 should be None, last should emit
        assert all(r is None for r in results[:-1])
        result = results[-1]
        assert result is not None
        assert result.name == "model.layers.0.mlp.experts.down_proj"
        assert result.tensor.shape == (NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM)

        # Verify each expert slice has the correct value
        for expert_id in range(NUM_EXPERTS):
            expected = _make_expert_tensor("down_proj", expert_id)
            assert torch.equal(result.tensor[expert_id], expected)

    def test_gate_up_merged_after_both_collected(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

        # Feed all gate_proj experts — should buffer (no up_proj yet)
        gate_results = self._feed_all_experts(converter, layer=0, proj="gate_proj")
        assert all(r is None for r in gate_results)

        # Feed all up_proj experts — last one should emit merged gate_up_proj
        up_results = self._feed_all_experts(converter, layer=0, proj="up_proj")
        assert all(r is None for r in up_results[:-1])
        result = up_results[-1]
        assert result is not None
        assert result.name == "model.layers.0.mlp.experts.gate_up_proj"
        assert result.tensor.shape == (NUM_EXPERTS, 2 * INTERMEDIATE_DIM, HIDDEN_DIM)

        # Verify: first half is gate, second half is up
        for expert_id in range(NUM_EXPERTS):
            gate_expected = _make_expert_tensor("gate_proj", expert_id)
            up_expected = _make_expert_tensor("up_proj", expert_id)
            assert torch.equal(result.tensor[expert_id, :INTERMEDIATE_DIM, :], gate_expected)
            assert torch.equal(result.tensor[expert_id, INTERMEDIATE_DIM:, :], up_expected)

    def test_up_before_gate_also_works(self):
        """gate_up merge should work regardless of which proj arrives first."""
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

        # Feed up_proj first, then gate_proj
        up_results = self._feed_all_experts(converter, layer=0, proj="up_proj")
        assert all(r is None for r in up_results)

        gate_results = self._feed_all_experts(converter, layer=0, proj="gate_proj")
        assert all(r is None for r in gate_results[:-1])
        result = gate_results[-1]
        assert result is not None
        assert result.name == "model.layers.0.mlp.experts.gate_up_proj"
        # gate is still first in the concat, up second
        for expert_id in range(NUM_EXPERTS):
            gate_expected = _make_expert_tensor("gate_proj", expert_id)
            up_expected = _make_expert_tensor("up_proj", expert_id)
            assert torch.equal(result.tensor[expert_id, :INTERMEDIATE_DIM, :], gate_expected)
            assert torch.equal(result.tensor[expert_id, INTERMEDIATE_DIM:, :], up_expected)

    def test_experts_out_of_order(self):
        """Experts can arrive in any order (e.g. from different shards)."""
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        order = [3, 1, 0, 2]
        results = []
        for expert_id in order:
            key = _make_expert_key(0, expert_id, "down_proj")
            results.append(converter.convert(key, _make_expert_tensor("down_proj", expert_id)))

        assert all(r is None for r in results[:-1])
        result = results[-1]
        assert result is not None
        # Stacking should still be in expert_id order [0, 1, 2, 3]
        for expert_id in range(NUM_EXPERTS):
            expected = _make_expert_tensor("down_proj", expert_id)
            assert torch.equal(result.tensor[expert_id], expected)

    def test_multiple_layers_independent(self):
        """Different layers are tracked independently."""
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

        # Feed layer 0 and layer 1 down_proj interleaved
        for expert_id in range(NUM_EXPERTS):
            key0 = _make_expert_key(0, expert_id, "down_proj")
            key1 = _make_expert_key(1, expert_id, "down_proj")
            r0 = converter.convert(key0, _make_expert_tensor("down_proj", expert_id))
            r1 = converter.convert(key1, _make_expert_tensor("down_proj", expert_id))

            if expert_id < NUM_EXPERTS - 1:
                assert r0 is None
                assert r1 is None
            else:
                assert r0 is not None
                assert r0.name == "model.layers.0.mlp.experts.down_proj"
                assert r1 is not None
                assert r1.name == "model.layers.1.mlp.experts.down_proj"

    def test_non_expert_key_returns_none(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        result = converter.convert("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4))
        assert result is None


class TestQwen3MoeConverterFinalize:
    def test_finalize_empty_when_all_flushed(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

        # Feed complete set for all 3 projections
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for expert_id in range(NUM_EXPERTS):
                key = _make_expert_key(0, expert_id, proj)
                converter.convert(key, _make_expert_tensor(proj, expert_id))

        results = converter.finalize()
        assert results == []

    def test_finalize_raises_on_incomplete_experts(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        # Feed only 2 of 4 experts
        for expert_id in range(2):
            key = _make_expert_key(0, expert_id, "down_proj")
            converter.convert(key, _make_expert_tensor("down_proj", expert_id))

        # finalize should raise because expert buffer is incomplete
        with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
            converter.finalize()

    def test_finalize_raises_on_unpaired_gate_up(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)
        # Feed all gate_proj but no up_proj — stacked buffer will be non-empty
        for expert_id in range(NUM_EXPERTS):
            key = _make_expert_key(0, expert_id, "gate_proj")
            converter.convert(key, _make_expert_tensor("gate_proj", expert_id))

        # finalize should raise because gate/up pair is incomplete
        with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
            converter.finalize()


_PER_EXPERT_CONVERTERS = pytest.mark.parametrize(
    "converter_cls",
    [
        Qwen3MoeCheckpointTensorConverter,
        Qwen3OmniMoeCheckpointTensorConverter,
        DeepseekV3CheckpointTensorConverter,
        GlmMoeDsaCheckpointTensorConverter,
    ],
)


@_PER_EXPERT_CONVERTERS
class TestPerExpertConverterExpertRange:
    def test_fused_expert_target_names_the_fused_param_and_expert_row(self, converter_cls):
        converter = converter_cls(num_experts=4)
        assert converter.fused_expert_target("model.layers.0.mlp.experts.3.up_proj.weight") == (
            "model.layers.0.mlp.experts.gate_up_proj",
            3,
        )
        assert converter.fused_expert_target("model.layers.0.mlp.experts.1.down_proj.weight") == (
            "model.layers.0.mlp.experts.down_proj",
            1,
        )
        assert converter.fused_expert_target("model.layers.0.mlp.experts.gate_up_proj") is None

    def test_helper_requires_both_expert_streaming_capabilities(self, converter_cls):
        converter = converter_cls(num_experts=4)
        name = "model.layers.0.mlp.experts.3.up_proj.weight"
        assert checkpoint_converter_fused_expert_target(converter, name) == (
            "model.layers.0.mlp.experts.gate_up_proj",
            3,
        )

        class _TargetOnly:
            can_handle = converter.can_handle
            fused_expert_target = converter.fused_expert_target

        assert checkpoint_converter_fused_expert_target(_TargetOnly(), name) is None
        assert checkpoint_converter_fused_expert_target(None, name) is None

    def test_range_converter_stacks_only_its_experts_as_local_rows(self, converter_cls):
        local = converter_cls(num_experts=4).for_expert_range(2, 2)
        assert type(local) is converter_cls
        tensors = {expert: torch.full((1, 2), float(expert)) for expert in (2, 3)}

        assert local.convert("model.layers.0.mlp.experts.2.down_proj.weight", tensors[2]) is None
        converted = local.convert("model.layers.0.mlp.experts.3.down_proj.weight", tensors[3])

        assert converted.name == "model.layers.0.mlp.experts.down_proj"
        torch.testing.assert_close(converted.tensor, torch.stack([tensors[2], tensors[3]]))
        assert local.finalize() == []

    def test_range_converter_rejects_experts_outside_its_range(self, converter_cls):
        local = converter_cls(num_experts=4).for_expert_range(2, 2)
        with pytest.raises(ValueError, match="outside this converter's range"):
            local.convert("model.layers.0.mlp.experts.1.down_proj.weight", torch.zeros(1, 2))

    def test_range_must_fit_in_the_converter(self, converter_cls):
        with pytest.raises(ValueError, match="outside this converter's 4 experts"):
            converter_cls(num_experts=4).for_expert_range(3, 2)

    def test_fp8_experts_raise_instead_of_loading_unscaled(self, converter_cls):
        converter = converter_cls(num_experts=4)
        fp8 = torch.zeros(1, 2).to(torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="does not dequantize"):
            converter.convert("model.layers.0.mlp.experts.0.down_proj.weight", fp8)

    def test_incomplete_checkpoint_error_names_the_model(self, converter_cls):
        converter = converter_cls(num_experts=4)
        converter.convert("model.layers.0.mlp.experts.0.down_proj.weight", torch.zeros(1, 2))
        with pytest.raises(RuntimeError, match=f"^{converter_cls.model_name} checkpoint converter: incomplete"):
            converter.finalize()


class TestQwen3MoeConverterFactory:
    def test_factory_creates_converter(self):
        model = SimpleNamespace(config=SimpleNamespace(num_experts=8))
        converter = create_qwen3_moe_checkpoint_tensor_converter(model)
        assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
        assert converter.num_experts == 8


class TestDeepseekV3ConverterFactory:
    def test_factory_reads_routed_expert_count(self):
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=8))
        converter = create_deepseek_v3_checkpoint_tensor_converter(model)
        assert isinstance(converter, DeepseekV3CheckpointTensorConverter)
        assert converter.num_experts == 8


class TestGlmMoeDsaConverter:
    def test_factory_reads_routed_expert_count(self):
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=8))
        converter = create_glm_moe_dsa_checkpoint_tensor_converter(model)
        assert isinstance(converter, GlmMoeDsaCheckpointTensorConverter)
        assert converter.num_experts == 8

    @pytest.fixture
    def eager_ops(self):
        from tests.tools.training_utils import make_eager_ops_config
        from veomni.ops.config import get_ops_config, set_ops_config

        previous = get_ops_config()
        set_ops_config(make_eager_ops_config())
        try:
            yield
        finally:
            set_ops_config(previous)

    def test_per_expert_checkpoint_loads_into_fused_experts(self, tmp_path, eager_ops):
        """The released GLM-5 layout: per-expert routed experts plus an unbuilt MTP layer."""
        from safetensors.torch import save_file

        from veomni.models.auto import build_foundation_model
        from veomni.models.checkpoint.weights import load_model_weights

        toy_config = Path(__file__).resolve().parents[2] / "toy_config" / "glm_moe_dsa_toy"
        model = build_foundation_model(
            config_path=str(toy_config), weights_path=None, torch_dtype="float32", init_device="cpu"
        )
        config = model.config
        generator = torch.Generator().manual_seed(0)
        expected = {name: torch.randn(param.shape, generator=generator) for name, param in model.state_dict().items()}

        on_disk = {}
        for name, tensor in expected.items():
            if name.endswith(".mlp.experts.gate_up_proj"):
                prefix = name.removesuffix(".gate_up_proj")
                gate, up = tensor.chunk(2, dim=1)
                for expert in range(tensor.shape[0]):
                    on_disk[f"{prefix}.{expert}.gate_proj.weight"] = gate[expert].contiguous()
                    on_disk[f"{prefix}.{expert}.up_proj.weight"] = up[expert].contiguous()
            elif name.endswith(".mlp.experts.down_proj"):
                prefix = name.removesuffix(".down_proj")
                for expert in range(tensor.shape[0]):
                    on_disk[f"{prefix}.{expert}.down_proj.weight"] = tensor[expert].contiguous()
            else:
                on_disk[name] = tensor
        assert not any(name.endswith((".gate_up_proj", ".experts.down_proj")) for name in on_disk)

        mtp_layer = config.num_hidden_layers
        inter, hidden = config.moe_intermediate_size, config.hidden_size
        for expert in range(config.n_routed_experts):
            prefix = f"model.layers.{mtp_layer}.mlp.experts.{expert}"
            on_disk[f"{prefix}.gate_proj.weight"] = torch.randn(inter, hidden, generator=generator)
            on_disk[f"{prefix}.up_proj.weight"] = torch.randn(inter, hidden, generator=generator)
            on_disk[f"{prefix}.down_proj.weight"] = torch.randn(hidden, inter, generator=generator)

        config.save_pretrained(tmp_path)
        save_file(on_disk, str(tmp_path / "model.safetensors"))
        loaded = build_foundation_model(
            config_path=str(tmp_path), weights_path=None, torch_dtype="float32", init_device="cpu"
        )
        load_model_weights(loaded, str(tmp_path), init_device="cpu")

        loaded_state = loaded.state_dict()
        assert loaded_state.keys() == expected.keys()
        mismatched = [name for name, tensor in loaded_state.items() if not torch.equal(tensor, expected[name])]
        assert mismatched == []


class TestQwen3MoeConverterIntegration:
    """Simulate a realistic checkpoint loading flow through maybe_convert_checkpoint_tensor."""

    def test_full_layer_conversion(self):
        converter = Qwen3MoeCheckpointTensorConverter(num_experts=NUM_EXPERTS)

        non_expert_keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate.weight",
        ]
        dispatched = {}

        # Non-expert keys pass through
        for key in non_expert_keys:
            t = torch.randn(4, 4)
            result = maybe_convert_checkpoint_tensor(key, t, converter)
            assert result is not None
            dispatched[result.name] = result.tensor

        # Expert keys: feed all 3 projections for all experts
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for expert_id in range(NUM_EXPERTS):
                key = _make_expert_key(0, expert_id, proj)
                t = _make_expert_tensor(proj, expert_id)
                result = maybe_convert_checkpoint_tensor(key, t, converter)
                if result is not None:
                    dispatched[result.name] = result.tensor

        # After finalize, nothing extra
        for result in converter.finalize():
            dispatched[result.name] = result.tensor

        # Verify all non-expert keys are present
        for key in non_expert_keys:
            assert key in dispatched

        # Verify fused expert keys are present
        assert "model.layers.0.mlp.experts.gate_up_proj" in dispatched
        assert "model.layers.0.mlp.experts.down_proj" in dispatched

        # Verify shapes
        assert dispatched["model.layers.0.mlp.experts.gate_up_proj"].shape == (
            NUM_EXPERTS,
            2 * INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )
        assert dispatched["model.layers.0.mlp.experts.down_proj"].shape == (
            NUM_EXPERTS,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )


# ---------------------------------------------------------------------------
# Tests for DeepseekV4CheckpointTensorConverter
# ---------------------------------------------------------------------------


def _make_deepseek_v4_expert_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def _make_deepseek_v4_expert_tensor(proj: str, expert_id: int) -> torch.Tensor:
    if proj in ("w2", "down_proj"):
        shape = (HIDDEN_DIM, INTERMEDIATE_DIM)
        offset = 0.2
    elif proj in ("w1", "gate_proj"):
        shape = (INTERMEDIATE_DIM, HIDDEN_DIM)
        offset = 0.0
    else:
        shape = (INTERMEDIATE_DIM, HIDDEN_DIM)
        offset = 0.1
    return torch.full(shape, expert_id + offset)


def _to_fp8(tensor: torch.Tensor) -> torch.Tensor:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    return tensor.to(fp8_dtype)


class TestDeepseekV4ConverterCanHandle:
    def setup_method(self):
        self.converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)

    @pytest.mark.parametrize(
        "proj",
        ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"],
    )
    def test_matches_raw_and_renamed_expert_keys(self, proj: str):
        assert self.converter.can_handle(_make_deepseek_v4_expert_key(0, 1, proj))

    def test_matches_scale_keys_for_fp8_scale_pairs(self):
        assert self.converter.can_handle("model.layers.0.attn.wkv.scale")
        assert self.converter.can_handle("model.layers.0.attn.wkv.weight_scale_inv")

    def test_prefixed_raw_weight_keys_are_name_handled(self):
        assert self.converter.can_handle("model.layers.0.attn.wkv.weight")

    def test_matches_quantized_weight_tensors(self):
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        assert self.converter.can_handle_tensor("model.layers.0.attn.wkv.weight", weight)

    def test_matches_raw_inference_checkpoint_names(self):
        assert self.converter.can_handle("embed.weight")
        assert self.converter.can_handle("layers.0.attn.wkv.scale")
        assert self.converter.can_handle_tensor("layers.0.attn.wkv.weight", torch.ones(4, 4))

    def test_rejects_non_tensor_keys(self):
        assert not self.converter.can_handle("model.layers.0.mlp.experts.gate_up_proj")

    @pytest.mark.parametrize("model_prefix", ("", "model."))
    def test_ignores_mtp_checkpoint_keys(self, model_prefix: str):
        stem = f"{model_prefix}mtp.0.attn.wkv"
        weight = torch.ones(4, 4, dtype=torch.int8)

        assert not self.converter.can_handle(f"{stem}.scale")
        assert not self.converter.can_handle_tensor(f"{stem}.weight", weight)
        result = maybe_convert_checkpoint_tensor(f"{stem}.weight", weight, self.converter)
        assert result is not None
        assert result.name == f"{stem}.weight"
        assert result.tensor is weight


class TestDeepseekV4ConverterConvert:
    @pytest.mark.parametrize("target_model_prefix", ["model.", ""])
    @pytest.mark.parametrize("source_model_prefix", ["model.", ""])
    @pytest.mark.parametrize(
        "source_suffix",
        [
            "attn.indexer.weights_proj.weight",
            "self_attn.compressor.indexer.weights_proj.weight",
            "self_attn.compressor.indexer.scorer.weights_proj.weight",
        ],
    )
    def test_scorer_checkpoint_keys_load_into_current_model(
        self, target_model_prefix, source_model_prefix, source_suffix
    ):
        source = f"{source_model_prefix}layers.2.{source_suffix}"
        target = f"{target_model_prefix}layers.2.self_attn.compressor.indexer.scorer.weights_proj.weight"
        weight = torch.randn(4, 8)
        converter = DeepseekV4CheckpointTensorConverter(
            num_experts=NUM_EXPERTS, target_model_prefix=target_model_prefix
        )

        result = maybe_convert_checkpoint_tensor(source, weight, converter)

        assert result.name == target
        assert result.tensor is weight
        assert maybe_convert_checkpoint_tensor(target, result.tensor, converter).name == target
        assert converter.finalize() == []

    @pytest.mark.parametrize("target_model_prefix", ["model.", ""])
    @pytest.mark.parametrize(
        "source",
        [
            "layers.2.attn.indexer.weights_proj.weight",
            # Reversing HF's ``.norm.`` -> ``.kv_norm.`` renaming used to turn these into ``attn.norm``.
            "layers.0.attn.kv_norm.weight",
            "layers.2.attn.indexer.compressor.norm.weight",
            # Nested compressor projections used to export with a literal ``\2`` in the name.
            "layers.2.attn.compressor.wkv.weight",
            "layers.2.attn.compressor.wgate.weight",
            "layers.2.attn.indexer.compressor.wkv.weight",
        ],
    )
    def test_export_round_trips_through_inference_format(self, monkeypatch, target_model_prefix, source):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module

        target = module.convert_deepseek_v4_checkpoint_key(source, target_model_prefix=target_model_prefix)
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        model = SimpleNamespace(config=SimpleNamespace(expert_dtype="fp8"), _veomni_fqn_to_index_mapping={source: 1})
        monkeypatch.setattr(module, "export_weights", lambda model: iter([(target, weight)]))
        converter = DeepseekV4CheckpointTensorConverter(
            num_experts=NUM_EXPERTS, target_model_prefix=target_model_prefix
        )

        exported = dict(converter.export_weights(model))

        assert set(exported) == {source}
        restored = maybe_convert_checkpoint_tensor(source, exported[source], converter)
        assert restored.name == target
        assert torch.equal(restored.tensor, weight)

    @pytest.mark.parametrize(
        ("expert_dtype", "scale_dtype"),
        [("fp8", torch.float32), ("fp4", torch.float8_e8m0fnu)],
    )
    @pytest.mark.parametrize("block_size", [128, 32])
    def test_fp8_export_rounds_scales_like_qat_and_respects_block_size(
        self, monkeypatch, expert_dtype, scale_dtype, block_size
    ):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module
        from veomni.models.transformers.deepseek_v4 import export_quant

        origin = "layers.0.attn.wq_a.weight"
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        calls: list[tuple[int, str | None, torch.dtype]] = []

        def fake_fp8_weight_quant(x, block_size, scale_fmt, scale_dtype):
            calls.append((block_size, scale_fmt, scale_dtype))
            return x, torch.ones(1, dtype=scale_dtype)

        model = SimpleNamespace(
            config=SimpleNamespace(
                quantization_config={"weight_block_size": [block_size, block_size], "expert_dtype": expert_dtype},
            ),
            _veomni_fqn_to_index_mapping={origin: 1, "layers.0.attn.wq_a.scale": 1},
        )
        target = module.convert_deepseek_v4_checkpoint_key(origin)
        monkeypatch.setattr(module, "export_weights", lambda model: iter([(target, weight)]))
        monkeypatch.setattr(export_quant, "fp8_weight_quant", fake_fp8_weight_quant)

        exported = dict(DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS).export_weights(model))

        assert set(exported) == {origin, "layers.0.attn.wq_a.scale"}
        assert calls == [(block_size, "ue8m0", scale_dtype)]

    def test_fp4_expert_export_uses_fp4_act_quant_and_fp8_for_shared(self, monkeypatch):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module
        from veomni.models.transformers.deepseek_v4 import export_quant

        expert_origin = "layers.0.ffn.experts.0.w1.weight"
        shared_origin = "layers.0.ffn.shared_experts.w1.weight"
        expert_target = "model.layers.0.mlp.experts.0.gate_proj.weight"
        shared_target = module.convert_deepseek_v4_checkpoint_key(shared_origin)
        expert_param = torch.randn(4, 32, dtype=torch.float32)
        shared_param = torch.randn(4, 32, dtype=torch.float32)

        fp4_calls: list[tuple[torch.dtype, int]] = []
        fp8_calls: list[tuple[torch.dtype, int, str | None, torch.dtype]] = []

        def fake_fp4_act_quant(x, block_size=32, dequant=False):
            assert dequant is False
            fp4_calls.append((x.dtype, block_size))
            packed = torch.ones((x.shape[0], x.shape[1] // 2), dtype=torch.uint8)
            scale = torch.ones((x.shape[0], x.shape[1] // block_size), dtype=torch.float8_e8m0fnu)
            return packed, scale

        def fake_fp8_weight_quant(x, block_size, scale_fmt, scale_dtype):
            fp8_calls.append((x.dtype, block_size, scale_fmt, scale_dtype))
            q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
            s = torch.ones((1, 1), dtype=scale_dtype)
            return q, s

        mapping = {
            expert_origin: 1,
            "layers.0.ffn.experts.0.w1.scale": 1,
            shared_origin: 1,
            "layers.0.ffn.shared_experts.w1.scale": 1,
        }
        model = SimpleNamespace(
            config=SimpleNamespace(
                expert_dtype="fp4",
                quantization_config=SimpleNamespace(
                    to_dict=lambda: {"weight_block_size": [32, 32], "expert_dtype": "fp4"}
                ),
            ),
            _veomni_fqn_to_index_mapping=mapping,
        )
        monkeypatch.setattr(
            module,
            "export_weights",
            lambda model: iter([(expert_target, expert_param), (shared_target, shared_param)]),
        )
        monkeypatch.setattr(export_quant, "fp4_act_quant", fake_fp4_act_quant)
        monkeypatch.setattr(export_quant, "fp8_weight_quant", fake_fp8_weight_quant)

        exported = dict(DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS).export_weights(model))

        assert fp4_calls == [(torch.bfloat16, 32)]
        assert fp8_calls == [(torch.bfloat16, 32, "ue8m0", torch.float8_e8m0fnu)]
        assert exported[expert_origin].dtype == torch.int8
        assert exported[expert_origin].shape == (4, 16)
        assert exported["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float8_e8m0fnu
        assert exported[shared_origin].dtype == torch.float8_e4m3fn
        assert exported["layers.0.ffn.shared_experts.w1.scale"].dtype == torch.float8_e8m0fnu

    def test_unquantized_weights_cast_to_bf16_while_non_weight_params_stay_fp32(self, monkeypatch):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module

        raw_to_dtype = {
            "embed.weight": torch.bfloat16,
            "head.weight": torch.bfloat16,
            "layers.0.attn_norm.weight": torch.bfloat16,
            "layers.0.ffn.gate.weight": torch.bfloat16,
            "layers.2.attn.indexer.weights_proj.weight": torch.bfloat16,
            "layers.0.hc_attn_base": torch.float32,
            "layers.0.attn.attn_sink": torch.float32,
            "layers.0.ffn.gate.bias": torch.float32,
        }
        mapping = dict.fromkeys(raw_to_dtype, 1)
        live_items = [
            (module.convert_deepseek_v4_checkpoint_key(k), torch.ones(4, dtype=torch.float32)) for k in raw_to_dtype
        ]
        model = SimpleNamespace(
            config=SimpleNamespace(),
            _veomni_fqn_to_index_mapping=mapping,
        )
        monkeypatch.setattr(module, "export_weights", lambda model: iter(live_items))

        exported = dict(DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS).export_weights(model))

        assert set(exported) == set(raw_to_dtype)
        for key, expected_dtype in raw_to_dtype.items():
            assert exported[key].dtype == expected_dtype, (
                f"{key}: got {exported[key].dtype}, expected {expected_dtype}"
            )

    def test_export_weights_synthesizes_release_mapping_when_none(self):
        from tests.models.tiny_configs import tiny_deepseek_v4_config
        from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import (
            _build_default_deepseek_v4_fqn_to_index_mapping,
        )
        from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
            DeepseekV4ForCausalLM,
        )

        cfg = tiny_deepseek_v4_config()
        model = DeepseekV4ForCausalLM(cfg).float()
        converter = DeepseekV4CheckpointTensorConverter(num_experts=cfg.n_routed_experts)

        exported = dict(converter.export_weights(model))

        assert getattr(model, "_veomni_fqn_to_index_mapping", None) is not None
        assert "embed.weight" in exported
        assert "head.weight" in exported
        assert "layers.0.ffn.experts.0.w1.weight" in exported
        assert "layers.0.hc_attn_base" in exported
        assert exported["embed.weight"].dtype == torch.bfloat16
        assert exported["layers.0.ffn.experts.0.w1.weight"].dtype == torch.bfloat16
        assert exported["layers.0.hc_attn_base"].dtype == torch.float32
        assert not any(k.endswith(".scale") for k in exported)

        # With quant_config and 32-divisible weights, .scale entries are synthesized for quantized projections
        cfg.quantization_config = {"quant_method": "fp8", "weight_block_size": [16, 16]}
        quant_mapping = _build_default_deepseek_v4_fqn_to_index_mapping(model)
        assert "layers.0.attn.wq_a.scale" in quant_mapping
        assert "layers.0.ffn.experts.0.w1.scale" in quant_mapping
        assert "embed.scale" not in quant_mapping
        assert "layers.0.attn.compressor.wkv.scale" not in quant_mapping
        assert "layers.3.attn.indexer.weights_proj.scale" not in quant_mapping

    def test_export_weights_skips_unmapped_tied_weights_and_handles_none_block_size(self, monkeypatch):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module

        mapping = {"embed.weight": 1}
        model = SimpleNamespace(
            config=SimpleNamespace(quantization_config={"weight_block_size": None}),
            _veomni_fqn_to_index_mapping=mapping,
        )
        monkeypatch.setattr(
            module,
            "export_weights",
            lambda m: iter(
                [
                    ("model.embed_tokens.weight", torch.ones(4, 4, dtype=torch.float32)),
                    ("lm_head.weight", torch.ones(4, 4, dtype=torch.float32)),
                ]
            ),
        )

        exported = dict(DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS).export_weights(model))

        assert set(exported) == {"embed.weight"}
        assert exported["embed.weight"].dtype == torch.bfloat16

    def test_export_weights_accepts_fused_expert_fqn_mapping(self, monkeypatch):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module

        mapping = {
            "model.layers.0.mlp.experts.gate_up_proj": 2,
            "model.layers.0.mlp.experts.down_proj": 3,
        }
        live_items = []
        for e in range(2):
            live_items.append((f"model.layers.0.mlp.experts.{e}.gate_proj.weight", torch.ones(4, 4)))
            live_items.append((f"model.layers.0.mlp.experts.{e}.up_proj.weight", torch.ones(4, 4)))
            live_items.append((f"model.layers.0.mlp.experts.{e}.down_proj.weight", torch.ones(4, 4)))
        model = SimpleNamespace(
            config=SimpleNamespace(),
            _veomni_fqn_to_index_mapping=mapping,
        )
        monkeypatch.setattr(module, "export_weights", lambda m: iter(live_items))

        exported = dict(DeepseekV4CheckpointTensorConverter(num_experts=2).export_weights(model))

        expected_keys = {f"layers.0.ffn.experts.{e}.{w}.weight" for e in range(2) for w in ("w1", "w2", "w3")}
        assert set(exported) == expected_keys

    def test_export_weights_should_quantize_skips_non_owner_quantization(self, monkeypatch):
        from veomni.models.transformers.deepseek_v4 import checkpoint_tensor_converter as module
        from veomni.models.transformers.deepseek_v4 import export_quant

        mapping = {
            "layers.0.attn.wq_a.weight": 1,
            "layers.0.attn.wq_a.scale": 1,
            "layers.0.attn.wkv.weight": 2,
            "layers.0.attn.wkv.scale": 2,
        }
        quantized_calls: list[torch.Size] = []

        def fake_fp8_weight_quant(x, block_size, scale_fmt, scale_dtype):
            quantized_calls.append(x.shape)
            return x.to(torch.float8_e4m3fn), torch.ones((1, 1), dtype=scale_dtype)

        model = SimpleNamespace(
            config=SimpleNamespace(expert_dtype="fp8"),
            _veomni_fqn_to_index_mapping=mapping,
        )
        monkeypatch.setattr(
            module,
            "export_weights",
            lambda m: iter(
                [
                    ("model.layers.0.self_attn.q_a_proj.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
                    ("model.layers.0.self_attn.kv_proj.weight", torch.ones(8, 8, dtype=torch.bfloat16)),
                ]
            ),
        )
        monkeypatch.setattr(export_quant, "fp8_weight_quant", fake_fp8_weight_quant)

        exported = dict(
            DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS).export_weights(
                model,
                should_quantize=lambda origin: origin == "layers.0.attn.wq_a.weight",
            )
        )

        assert quantized_calls == [torch.Size([4, 4])]
        assert set(exported) == {"layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"}

    def test_dequantizes_block_scaled_fp8_weight(self):
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        expected = weight.float().view(2, 2, 2, 2).mul(scale.view(2, 1, 2, 1)).reshape(4, 4)
        assert torch.equal(_dequantize_scaled_weight(weight, scale), expected)

    def test_dequantizes_packed_fp4_expert_weight(self):
        # Each int8 byte stores two E2M1 values, low nibble first. 0x21 maps
        # to [0.5, 1.0]. DeepSeek applies one E8M0 scale per 32 FP4 values.
        weight = torch.full((1, 32), 0x21, dtype=torch.uint8).view(torch.int8)
        scale = torch.tensor([[2.0, 4.0]])
        expected = torch.tensor([[1.0, 2.0] * 16 + [2.0, 4.0] * 16])

        result = _dequantize_scaled_weight(weight, scale, packed_fp4=True)

        assert result.shape == (1, 64)
        assert torch.equal(result, expected)

    def test_reuses_fp4_lookup_table(self):
        _get_fp4_e2m1_table.cache_clear()
        try:
            first = _get_fp4_e2m1_table(torch.device("cpu"))
            second = _get_fp4_e2m1_table(torch.device("cpu"))
            assert first is second
        finally:
            _get_fp4_e2m1_table.cache_clear()

    def test_dequantizes_conventional_int8_weight_without_unpacking(self):
        weight = torch.arange(16, dtype=torch.int8).reshape(4, 4)
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        expected = weight.float().view(2, 2, 2, 2).mul(scale.view(2, 1, 2, 1)).reshape(4, 4)

        result = _dequantize_scaled_weight(weight, scale)

        assert result.shape == weight.shape
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        ("raw_name", "converted_name"),
        [
            ("embed.weight", "model.embed_tokens.weight"),
            ("model.embed.weight", "model.embed_tokens.weight"),
            ("head.weight", "lm_head.weight"),
            ("model.head.weight", "lm_head.weight"),
            ("norm.weight", "model.norm.weight"),
            ("model.norm.weight", "model.norm.weight"),
            ("hc_head_fn", "model.hc_head.hc_fn"),
            ("model.hc_head_fn", "model.hc_head.hc_fn"),
            ("layers.2.attn_norm.weight", "model.layers.2.input_layernorm.weight"),
            ("layers.2.ffn_norm.weight", "model.layers.2.post_attention_layernorm.weight"),
            ("layers.2.attn.wkv.weight", "model.layers.2.self_attn.kv_proj.weight"),
            ("model.layers.2.attn.wkv.weight", "model.layers.2.self_attn.kv_proj.weight"),
            ("layers.2.attn.wq_a.scale", "model.layers.2.self_attn.q_a_proj.scale"),
            ("layers.2.attn.attn_sink", "model.layers.2.self_attn.sinks"),
            (
                "layers.2.attn.compressor.wgate.weight",
                "model.layers.2.self_attn.compressor.gate_proj.weight",
            ),
            (
                "layers.2.attn.indexer.compressor.wkv.weight",
                "model.layers.2.self_attn.compressor.indexer.kv_proj.weight",
            ),
            (
                "layers.2.attn.indexer.weights_proj.weight",
                "model.layers.2.self_attn.compressor.indexer.scorer.weights_proj.weight",
            ),
            ("layers.2.ffn.gate.bias", "model.layers.2.mlp.gate.e_score_correction_bias"),
            ("layers.2.ffn.gate.tid2eid", "model.layers.2.mlp.gate.tid2eid"),
            (
                "layers.2.ffn.shared_experts.w1.weight",
                "model.layers.2.mlp.shared_experts.gate_proj.weight",
            ),
            (
                "model.layers.2.ffn.shared_experts.w1.weight",
                "model.layers.2.mlp.shared_experts.gate_proj.weight",
            ),
            (
                "layers.2.ffn.experts.7.w3.scale",
                "model.layers.2.mlp.experts.7.w3.scale",
            ),
            ("layers.2.hc_attn_base", "model.layers.2.attn_hc.base"),
            ("layers.2.hc_ffn_scale", "model.layers.2.ffn_hc.scale"),
            ("mtp.0.attn.wkv.weight", "mtp.0.attn.wkv.weight"),
            ("model.mtp.0.attn.wkv.weight", "model.mtp.0.attn.wkv.weight"),
            ("model.custom.weight", "model.custom.weight"),
        ],
    )
    def test_converts_deepseek_inference_checkpoint_names(self, raw_name: str, converted_name: str):
        assert convert_deepseek_v4_checkpoint_key(raw_name) == converted_name

    @pytest.mark.parametrize(
        ("raw_name", "converted_name"),
        [
            ("embed.weight", "embed_tokens.weight"),
            ("model.embed.weight", "embed_tokens.weight"),
            ("norm.weight", "norm.weight"),
            ("model.norm.weight", "norm.weight"),
            ("layers.2.attn.wkv.weight", "layers.2.self_attn.kv_proj.weight"),
            ("model.layers.2.attn.wkv.weight", "layers.2.self_attn.kv_proj.weight"),
            ("layers.2.ffn.experts.7.w3.scale", "layers.2.mlp.experts.7.w3.scale"),
            ("head.weight", "lm_head.weight"),
            ("model.custom.weight", "model.custom.weight"),
        ],
    )
    def test_converts_deepseek_inference_checkpoint_names_for_base_model(self, raw_name: str, converted_name: str):
        assert convert_deepseek_v4_checkpoint_key(raw_name, target_model_prefix="") == converted_name

    def test_raw_inference_fp8_pair_emits_hf_name(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.ones(2, 2)

        assert maybe_convert_checkpoint_tensor("layers.0.attn.wkv.weight", weight, converter) is None
        result = maybe_convert_checkpoint_tensor("layers.0.attn.wkv.scale", scale, converter)

        assert result is not None
        assert result.name == "model.layers.0.self_attn.kv_proj.weight"
        assert torch.equal(result.tensor, weight.float())

    def test_raw_inference_fp8_pair_emits_base_model_name(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS, target_model_prefix="")
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.ones(2, 2)

        assert maybe_convert_checkpoint_tensor("layers.0.attn.wkv.weight", weight, converter) is None
        result = maybe_convert_checkpoint_tensor("layers.0.attn.wkv.scale", scale, converter)

        assert result is not None
        assert result.name == "layers.0.self_attn.kv_proj.weight"
        assert torch.equal(result.tensor, weight.float())

    def test_fp8_weight_scale_pair_emits_dequantized_weight(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight_name = "model.layers.0.attn.wkv.weight"
        scale_name = "model.layers.0.attn.wkv.scale"
        converted_weight_name = "model.layers.0.self_attn.kv_proj.weight"
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])

        assert maybe_convert_checkpoint_tensor(weight_name, weight, converter) is None
        result = maybe_convert_checkpoint_tensor(scale_name, scale, converter)

        assert result is not None
        assert result.name == converted_weight_name
        assert torch.equal(result.tensor, _dequantize_scaled_weight(weight, scale))
        assert converter.finalize() == []

    def test_fp8_scale_can_arrive_before_weight(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight_name = "model.layers.0.attn.wkv.weight"
        scale_name = "model.layers.0.attn.wkv.scale"
        converted_weight_name = "model.layers.0.self_attn.kv_proj.weight"
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])

        assert maybe_convert_checkpoint_tensor(scale_name, scale, converter) is None
        result = maybe_convert_checkpoint_tensor(weight_name, weight, converter)

        assert result is not None
        assert result.name == converted_weight_name
        assert torch.equal(result.tensor, _dequantize_scaled_weight(weight, scale))
        assert converter.finalize() == []

    def test_fp8_weight_scale_inv_suffix_emits_dequantized_weight(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight_name = "model.layers.0.attn.wkv.weight"
        scale_name = "model.layers.0.attn.wkv.weight_scale_inv"
        converted_weight_name = "model.layers.0.self_attn.kv_proj.weight"
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])

        assert maybe_convert_checkpoint_tensor(weight_name, weight, converter) is None
        result = maybe_convert_checkpoint_tensor(scale_name, scale, converter)

        assert result is not None
        assert result.name == converted_weight_name
        assert torch.equal(result.tensor, _dequantize_scaled_weight(weight, scale))
        assert converter.finalize() == []

    def test_scalar_scale_pairs_with_fp8_weight(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight_name = "model.layers.0.attn.wkv.weight"
        scale_name = "model.layers.0.attn.wkv.scale"
        converted_weight_name = "model.layers.0.self_attn.kv_proj.weight"
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        scale = torch.tensor(2.0)

        assert maybe_convert_checkpoint_tensor(weight_name, weight, converter) is None
        result = maybe_convert_checkpoint_tensor(scale_name, scale, converter)

        assert result is not None
        assert result.name == converted_weight_name
        assert torch.equal(result.tensor, weight.float() * scale)
        assert converter.finalize() == []

    def test_legitimate_scale_parameter_passes_through(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        tensor = torch.ones(3)
        result = maybe_convert_checkpoint_tensor("model.layers.0.input_layernorm.scale", tensor, converter)
        assert result is not None
        assert result.name == "model.layers.0.input_layernorm.scale"
        assert result.tensor is tensor

    def test_2d_scale_without_quantized_weight_passes_through_on_finalize(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        tensor = torch.ones(2, 2)
        assert maybe_convert_checkpoint_tensor("model.layers.0.some_module.scale", tensor, converter) is None
        finalized = converter.finalize()
        assert len(finalized) == 1
        assert finalized[0].name == "model.layers.0.some_module.scale"
        assert finalized[0].tensor is tensor

    def test_quantized_weight_without_scale_errors(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        weight_name = "model.layers.0.attn.wkv.weight"
        weight = _to_fp8(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        assert maybe_convert_checkpoint_tensor(weight_name, weight, converter) is None
        with pytest.raises(RuntimeError, match="missing scale"):
            converter.finalize()

    def test_raw_w_keys_convert_to_fused_expert_layout(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        dispatched = {}

        # CI materializes DeepSeek-V4 toy checkpoints with save_original_format=True,
        # which writes raw w1/w2/w3 expert keys. The runtime loader must consume
        # those keys directly instead of letting them pass through as unexpected.
        for proj in ("w1", "w3", "w2"):
            for expert_id in range(NUM_EXPERTS):
                result = maybe_convert_checkpoint_tensor(
                    _make_deepseek_v4_expert_key(0, expert_id, proj),
                    _make_deepseek_v4_expert_tensor(proj, expert_id),
                    converter,
                )
                if result is not None:
                    dispatched[result.name] = result.tensor

        assert converter.finalize() == []
        assert set(dispatched) == {
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        }
        assert dispatched["model.layers.0.mlp.experts.gate_up_proj"].shape == (
            NUM_EXPERTS,
            2 * INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )
        assert dispatched["model.layers.0.mlp.experts.down_proj"].shape == (
            NUM_EXPERTS,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )
        for expert_id in range(NUM_EXPERTS):
            gate_expected = _make_deepseek_v4_expert_tensor("w1", expert_id)
            up_expected = _make_deepseek_v4_expert_tensor("w3", expert_id)
            down_expected = _make_deepseek_v4_expert_tensor("w2", expert_id)
            gate_up = dispatched["model.layers.0.mlp.experts.gate_up_proj"]
            down = dispatched["model.layers.0.mlp.experts.down_proj"]
            assert torch.equal(gate_up[expert_id, :INTERMEDIATE_DIM, :], gate_expected)
            assert torch.equal(gate_up[expert_id, INTERMEDIATE_DIM:, :], up_expected)
            assert torch.equal(down[expert_id], down_expected)

    def test_fp8_expert_weights_are_dequantized_before_fusing(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        dispatched = {}
        scales = {
            "w1": torch.full((1, 1), 2.0),
            "w2": torch.full((1, 1), 3.0),
            "w3": torch.full((1, 1), 4.0),
        }

        for proj in ("w1", "w3", "w2"):
            for expert_id in range(NUM_EXPERTS):
                weight_name = _make_deepseek_v4_expert_key(0, expert_id, proj)
                weight = _to_fp8(_make_deepseek_v4_expert_tensor(proj, expert_id))
                assert maybe_convert_checkpoint_tensor(weight_name, weight, converter) is None
                result = maybe_convert_checkpoint_tensor(
                    weight_name.removesuffix(".weight") + ".scale",
                    scales[proj],
                    converter,
                )
                if result is not None:
                    dispatched[result.name] = result.tensor

        gate_up = dispatched["model.layers.0.mlp.experts.gate_up_proj"]
        down = dispatched["model.layers.0.mlp.experts.down_proj"]
        for expert_id in range(NUM_EXPERTS):
            gate_expected = _to_fp8(_make_deepseek_v4_expert_tensor("w1", expert_id)).float() * scales["w1"]
            up_expected = _to_fp8(_make_deepseek_v4_expert_tensor("w3", expert_id)).float() * scales["w3"]
            down_expected = _to_fp8(_make_deepseek_v4_expert_tensor("w2", expert_id)).float() * scales["w2"]
            assert torch.equal(gate_up[expert_id, :INTERMEDIATE_DIM, :], gate_expected)
            assert torch.equal(gate_up[expert_id, INTERMEDIATE_DIM:, :], up_expected)
            assert torch.equal(down[expert_id], down_expected)
        assert converter.finalize() == []

    @pytest.mark.parametrize("model_prefix", ("", "model."))
    def test_raw_inference_fp4_experts_are_unpacked_before_fusing(self, model_prefix: str):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        dispatched = {}

        for proj in ("w1", "w3", "w2"):
            logical_shape = (HIDDEN_DIM, INTERMEDIATE_DIM) if proj == "w2" else (INTERMEDIATE_DIM, HIDDEN_DIM)
            packed_shape = (logical_shape[0], logical_shape[1] // 2)
            for expert_id in range(NUM_EXPERTS):
                stem = f"{model_prefix}layers.0.ffn.experts.{expert_id}.{proj}"
                packed = torch.full(packed_shape, 0x21, dtype=torch.uint8).view(torch.int8)
                assert maybe_convert_checkpoint_tensor(f"{stem}.weight", packed, converter) is None
                result = maybe_convert_checkpoint_tensor(f"{stem}.scale", torch.tensor(1.0), converter)
                if result is not None:
                    dispatched[result.name] = result.tensor

        assert converter.finalize() == []
        assert dispatched["model.layers.0.mlp.experts.gate_up_proj"].shape == (
            NUM_EXPERTS,
            2 * INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )
        assert dispatched["model.layers.0.mlp.experts.down_proj"].shape == (
            NUM_EXPERTS,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )
        expected_values = torch.tensor([0.5, 1.0] * (HIDDEN_DIM // 2))
        assert torch.equal(dispatched["model.layers.0.mlp.experts.gate_up_proj"][0, 0], expected_values)

    def test_renamed_keys_still_convert(self):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=NUM_EXPERTS)
        emitted = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for expert_id in range(NUM_EXPERTS):
                result = converter.convert(
                    _make_deepseek_v4_expert_key(1, expert_id, proj),
                    _make_deepseek_v4_expert_tensor(proj, expert_id),
                )
                if result is not None:
                    emitted.append(result.name)
        assert sorted(emitted) == [
            "model.layers.1.mlp.experts.down_proj",
            "model.layers.1.mlp.experts.gate_up_proj",
        ]


class TestDeepseekV4ConverterExpertStreaming:
    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("layers.2.ffn.experts.5.w1.weight", ("model.layers.2.mlp.experts.gate_up_proj", 5)),
            ("layers.2.ffn.experts.5.w3.scale", ("model.layers.2.mlp.experts.gate_up_proj", 5)),
            ("layers.2.ffn.experts.5.w2.weight", ("model.layers.2.mlp.experts.down_proj", 5)),
            ("model.layers.2.mlp.experts.1.up_proj.weight", ("model.layers.2.mlp.experts.gate_up_proj", 1)),
            ("layers.2.ffn.shared_experts.w1.weight", None),
            ("layers.2.attn.wq_a.scale", None),
            ("mtp.0.ffn.experts.3.w1.weight", None),
        ],
    )
    def test_fused_expert_target_covers_weights_and_scales(self, name, target):
        converter = DeepseekV4CheckpointTensorConverter(num_experts=8)
        assert converter.fused_expert_target(name) == target
        if target is not None:
            assert checkpoint_converter_fused_expert_target(converter, name) == target

    def test_for_expert_range_stacks_only_its_fp8_experts(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            pytest.skip("torch.float8_e4m3fn is unavailable")
        local = DeepseekV4CheckpointTensorConverter(num_experts=4, target_model_prefix="").for_expert_range(2, 2)
        assert (local.num_experts, local.expert_offset, local.target_model_prefix) == (2, 2, "")

        emitted = {}
        for expert_id in (2, 3):
            for proj in ("w1", "w3", "w2"):
                stem = f"layers.0.ffn.experts.{expert_id}.{proj}"
                weight = _to_fp8(_make_deepseek_v4_expert_tensor(proj, expert_id))
                assert maybe_convert_checkpoint_tensor(f"{stem}.weight", weight, local) is None
                result = maybe_convert_checkpoint_tensor(f"{stem}.scale", torch.tensor([[2.0]]), local)
                if result is not None:
                    emitted[result.name] = result.tensor
        assert local.finalize() == []

        down = emitted["layers.0.mlp.experts.down_proj"]
        assert down.shape == (2, HIDDEN_DIM, INTERMEDIATE_DIM)
        for row, expert_id in enumerate((2, 3)):
            want = _to_fp8(_make_deepseek_v4_expert_tensor("w2", expert_id)).float() * 2.0
            assert torch.equal(down[row], want)
        assert emitted["layers.0.mlp.experts.gate_up_proj"].shape == (2, 2 * INTERMEDIATE_DIM, HIDDEN_DIM)

    def test_a_range_converter_rejects_other_experts(self):
        local = DeepseekV4CheckpointTensorConverter(num_experts=4).for_expert_range(2, 2)
        with pytest.raises(ValueError, match=r"expert 1 is outside this converter's range \[2, 4\)"):
            local.convert("layers.0.ffn.experts.1.w1.weight", torch.zeros(INTERMEDIATE_DIM, HIDDEN_DIM))

    def test_for_expert_range_rejects_an_out_of_bounds_range(self):
        with pytest.raises(ValueError, match="outside this converter's 4 experts"):
            DeepseekV4CheckpointTensorConverter(num_experts=4).for_expert_range(3, 2)


class TestDeepseekV4ConverterFactoryAndMapping:
    def test_factory_creates_converter(self):
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=8),
            named_parameters=lambda: iter([("model.embed_tokens.weight", torch.nn.Parameter(torch.empty(1)))]),
        )
        converter = create_deepseek_v4_checkpoint_tensor_converter(model)
        assert isinstance(converter, DeepseekV4CheckpointTensorConverter)
        assert converter.num_experts == 8
        assert converter.target_model_prefix == "model."

    def test_factory_creates_base_model_converter(self):
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=8),
            named_parameters=lambda: iter([("embed_tokens.weight", torch.nn.Parameter(torch.empty(1)))]),
        )
        converter = create_deepseek_v4_checkpoint_tensor_converter(model)
        assert isinstance(converter, DeepseekV4CheckpointTensorConverter)
        assert converter.num_experts == 8
        assert converter.target_model_prefix == ""

    def test_deepseek_v4_models_do_not_register_fqn_to_index_mapping_converter(self):
        from veomni.models.registry import MODELING_REGISTRY

        causal_lm_cls = MODELING_REGISTRY["deepseek_v4"]("DeepseekV4ForCausalLM")
        base_model_cls = MODELING_REGISTRY["deepseek_v4"]("DeepseekV4Model")
        assert getattr(causal_lm_cls, "_convert_fqn_to_index_mapping", None) is None
        assert getattr(base_model_cls, "_convert_fqn_to_index_mapping", None) is None
        assert get_fqn_to_index_mapping_converter(causal_lm_cls.__new__(causal_lm_cls)) is None

    def test_split_moe_params_splits_gate_up_and_down(self):
        gate_up = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        split_items = dict(split_moe_params("model.layers.0.mlp.experts.gate_up_proj", gate_up, ep_rank=1))
        assert set(split_items) == {
            "model.layers.0.mlp.experts.2.gate_proj.weight",
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            "model.layers.0.mlp.experts.2.up_proj.weight",
            "model.layers.0.mlp.experts.3.up_proj.weight",
        }
        assert torch.equal(split_items["model.layers.0.mlp.experts.2.gate_proj.weight"], gate_up[0, :2])
        assert torch.equal(split_items["model.layers.0.mlp.experts.3.up_proj.weight"], gate_up[1, 2:])


# ---------------------------------------------------------------------------
# Tests for Qwen3VLMoeCheckpointTensorConverter
# ---------------------------------------------------------------------------


VLMOE_NUM_EXPERTS = 4
VLMOE_HIDDEN = 8
VLMOE_INTERMEDIATE = 6  # chosen so hidden != 2*intermediate (8 != 12) — layouts are unambiguous


def _vlmoe_hf_gate_up(layer: int = 0) -> tuple[str, torch.Tensor]:
    key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    tensor = torch.arange(VLMOE_NUM_EXPERTS * VLMOE_HIDDEN * 2 * VLMOE_INTERMEDIATE, dtype=torch.float32).reshape(
        VLMOE_NUM_EXPERTS, VLMOE_HIDDEN, 2 * VLMOE_INTERMEDIATE
    )
    return key, tensor


def _vlmoe_hf_down(layer: int = 0) -> tuple[str, torch.Tensor]:
    key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    tensor = torch.arange(VLMOE_NUM_EXPERTS * VLMOE_INTERMEDIATE * VLMOE_HIDDEN, dtype=torch.float32).reshape(
        VLMOE_NUM_EXPERTS, VLMOE_INTERMEDIATE, VLMOE_HIDDEN
    )
    return key, tensor


class TestQwen3VLMoeConverterCanHandle:
    def setup_method(self):
        self.converter = Qwen3VLMoeCheckpointTensorConverter(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            intermediate_size=VLMOE_INTERMEDIATE,
        )

    def test_matches_fused_keys(self):
        assert self.converter.can_handle("model.language_model.layers.0.mlp.experts.gate_up_proj")
        assert self.converter.can_handle("model.language_model.layers.10.mlp.experts.down_proj")
        # Also matches text-only prefix (for Qwen3VLMoeTextModel standalone).
        assert self.converter.can_handle("model.layers.3.mlp.experts.gate_up_proj")

    def test_rejects_other_keys(self):
        assert not self.converter.can_handle("model.language_model.layers.0.self_attn.q_proj.weight")
        # Per-expert keys (the qwen3_moe HF layout) must NOT match here.
        assert not self.converter.can_handle("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
        assert not self.converter.can_handle("model.embed_tokens.weight")


class TestQwen3VLMoeConverterConvert:
    def setup_method(self):
        self.converter = Qwen3VLMoeCheckpointTensorConverter(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            intermediate_size=VLMOE_INTERMEDIATE,
        )

    def test_hf_gate_up_proj_is_transposed(self):
        key, tensor = _vlmoe_hf_gate_up()
        result = self.converter.convert(key, tensor)
        assert result is not None
        assert result.name == key
        assert result.tensor.shape == (VLMOE_NUM_EXPERTS, 2 * VLMOE_INTERMEDIATE, VLMOE_HIDDEN)
        # transposing back should reproduce the HF tensor.
        assert torch.equal(result.tensor.transpose(1, 2), tensor)
        assert result.tensor.is_contiguous()

    def test_hf_down_proj_is_transposed(self):
        key, tensor = _vlmoe_hf_down()
        result = self.converter.convert(key, tensor)
        assert result is not None
        assert result.tensor.shape == (VLMOE_NUM_EXPERTS, VLMOE_HIDDEN, VLMOE_INTERMEDIATE)
        assert torch.equal(result.tensor.transpose(1, 2), tensor)

    def test_v5_layout_passes_through_unchanged(self):
        # Simulate a checkpoint previously saved by VeOmni in v5-native layout.
        gate_up_v5 = torch.randn(VLMOE_NUM_EXPERTS, 2 * VLMOE_INTERMEDIATE, VLMOE_HIDDEN)
        down_v5 = torch.randn(VLMOE_NUM_EXPERTS, VLMOE_HIDDEN, VLMOE_INTERMEDIATE)

        gate_up_result = self.converter.convert("l.mlp.experts.gate_up_proj", gate_up_v5)
        down_result = self.converter.convert("l.mlp.experts.down_proj", down_v5)

        assert gate_up_result is not None and torch.equal(gate_up_result.tensor, gate_up_v5)
        assert down_result is not None and torch.equal(down_result.tensor, down_v5)

    def test_rejects_non_expert_key(self):
        result = self.converter.convert("model.embed_tokens.weight", torch.randn(4, 4))
        assert result is None

    def test_raises_on_wrong_rank(self):
        with pytest.raises(RuntimeError, match="expected 3-D"):
            self.converter.convert(
                "l.mlp.experts.gate_up_proj",
                torch.randn(VLMOE_NUM_EXPERTS, VLMOE_HIDDEN),  # 2-D
            )

    def test_raises_on_wrong_num_experts(self):
        with pytest.raises(RuntimeError, match="dim-0 == num_experts"):
            self.converter.convert(
                "l.mlp.experts.gate_up_proj",
                torch.randn(VLMOE_NUM_EXPERTS + 1, VLMOE_HIDDEN, 2 * VLMOE_INTERMEDIATE),
            )

    def test_raises_on_unrecognized_middle_dim(self):
        with pytest.raises(RuntimeError, match="unrecognized layout"):
            self.converter.convert(
                "l.mlp.experts.gate_up_proj",
                torch.randn(VLMOE_NUM_EXPERTS, 999, VLMOE_HIDDEN),
            )

    @pytest.mark.parametrize(
        ("projection", "hidden_size", "intermediate_size", "shape"),
        [
            pytest.param("gate_up_proj", 12, 6, (VLMOE_NUM_EXPERTS, 12, 12), id="gate-up"),
            pytest.param("down_proj", 8, 8, (VLMOE_NUM_EXPERTS, 8, 8), id="down"),
        ],
    )
    def test_raises_on_ambiguous_layout(self, projection, hidden_size, intermediate_size, shape):
        converter = Qwen3VLMoeCheckpointTensorConverter(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        tensor = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape)
        original = tensor.clone()
        with pytest.raises(RuntimeError, match="ambiguous layout"):
            maybe_convert_checkpoint_tensor(f"l.mlp.experts.{projection}", tensor, converter)
        assert torch.equal(tensor, original)

    @pytest.mark.parametrize("projection", ("gate_up_proj", "down_proj"))
    @pytest.mark.parametrize("layout", ("hf", "veomni"))
    def test_raises_on_wrong_last_dim(self, projection, layout):
        if projection == "gate_up_proj":
            middle = VLMOE_HIDDEN if layout == "hf" else 2 * VLMOE_INTERMEDIATE
        else:
            middle = VLMOE_INTERMEDIATE if layout == "hf" else VLMOE_HIDDEN
        with pytest.raises(RuntimeError, match="unrecognized layout"):
            self.converter.convert(
                f"l.mlp.experts.{projection}",
                torch.randn(VLMOE_NUM_EXPERTS, middle, 999),
            )


class TestQwen3VLMoeConverterFinalize:
    def test_finalize_is_noop(self):
        converter = Qwen3VLMoeCheckpointTensorConverter(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            intermediate_size=VLMOE_INTERMEDIATE,
        )
        key, tensor = _vlmoe_hf_gate_up()
        converter.convert(key, tensor)
        assert converter.finalize() == []


class TestQwen3VLMoeConverterFactory:
    def test_factory_with_nested_config(self):
        # Top-level Qwen3VLMoeConfig — has nested `text_config`.
        text_config = SimpleNamespace(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            moe_intermediate_size=VLMOE_INTERMEDIATE,
        )
        model = SimpleNamespace(config=SimpleNamespace(text_config=text_config))
        converter = create_qwen3_vl_moe_checkpoint_tensor_converter(model)
        assert isinstance(converter, Qwen3VLMoeCheckpointTensorConverter)
        assert converter.num_experts == VLMOE_NUM_EXPERTS
        assert converter.hidden_size == VLMOE_HIDDEN
        assert converter.intermediate_size == VLMOE_INTERMEDIATE

    def test_factory_with_flat_text_config(self):
        # Qwen3VLMoeTextModel is constructed directly from Qwen3VLMoeTextConfig —
        # the config has `num_experts` etc. at top level (no nested text_config).
        flat = SimpleNamespace(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            moe_intermediate_size=VLMOE_INTERMEDIATE,
        )
        model = SimpleNamespace(config=flat)
        converter = create_qwen3_vl_moe_checkpoint_tensor_converter(model)
        assert converter.num_experts == VLMOE_NUM_EXPERTS
        assert converter.hidden_size == VLMOE_HIDDEN
        assert converter.intermediate_size == VLMOE_INTERMEDIATE


class TestQwen3VLMoeConverterIntegration:
    """End-to-end through `maybe_convert_checkpoint_tensor` using an HF-layout checkpoint."""

    def test_full_layer_conversion(self):
        converter = Qwen3VLMoeCheckpointTensorConverter(
            num_experts=VLMOE_NUM_EXPERTS,
            hidden_size=VLMOE_HIDDEN,
            intermediate_size=VLMOE_INTERMEDIATE,
        )

        non_expert_keys = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.gate.weight",
        ]
        dispatched = {}
        for key in non_expert_keys:
            t = torch.randn(4, 4)
            result = maybe_convert_checkpoint_tensor(key, t, converter)
            assert result is not None and result.name == key
            dispatched[result.name] = result.tensor

        for load_fn in (_vlmoe_hf_gate_up, _vlmoe_hf_down):
            key, t = load_fn()
            result = maybe_convert_checkpoint_tensor(key, t, converter)
            assert result is not None
            dispatched[result.name] = result.tensor

        assert converter.finalize() == []

        # Fused expert tensors are now in v5 modeling layout.
        assert dispatched["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (
            VLMOE_NUM_EXPERTS,
            2 * VLMOE_INTERMEDIATE,
            VLMOE_HIDDEN,
        )
        assert dispatched["model.language_model.layers.0.mlp.experts.down_proj"].shape == (
            VLMOE_NUM_EXPERTS,
            VLMOE_HIDDEN,
            VLMOE_INTERMEDIATE,
        )


# ---------------------------------------------------------------------------
# Tests for Qwen3OmniMoeCheckpointTensorConverter
# ---------------------------------------------------------------------------


OMNIMOE_NUM_EXPERTS = 4
OMNIMOE_HIDDEN = 8
OMNIMOE_INTERMEDIATE = 6  # chosen so hidden != 2*intermediate (8 != 12) — layouts are unambiguous


def _omnimoe_per_expert_key(prefix: str, expert_id: int, proj: str) -> str:
    return f"{prefix}.experts.{expert_id}.{proj}.weight"


def _omnimoe_per_expert_tensors(prefix: str = "thinker.model.layers.0.mlp"):
    """Generate per-expert HF-layout tensors with unique fingerprints per (expert, proj)."""
    tensors = {}
    for e in range(OMNIMOE_NUM_EXPERTS):
        # gate_proj / up_proj: [I, H]
        for proj, base in (("gate_proj", 100.0), ("up_proj", 200.0)):
            tensors[_omnimoe_per_expert_key(prefix, e, proj)] = torch.full(
                (OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN), base + e, dtype=torch.float32
            )
        # down_proj: [H, I]
        tensors[_omnimoe_per_expert_key(prefix, e, "down_proj")] = torch.full(
            (OMNIMOE_HIDDEN, OMNIMOE_INTERMEDIATE), 300.0 + e, dtype=torch.float32
        )
    return tensors


class TestQwen3OmniMoeConverterCanHandle:
    def setup_method(self):
        self.converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)

    def test_matches_thinker_per_expert_keys(self):
        assert self.converter.can_handle("thinker.model.layers.0.mlp.experts.0.gate_proj.weight")
        assert self.converter.can_handle("thinker.model.layers.10.mlp.experts.3.up_proj.weight")
        assert self.converter.can_handle("thinker.model.layers.5.mlp.experts.2.down_proj.weight")

    def test_matches_talker_per_expert_keys(self):
        # Talker tower shares the same expert layout convention.
        assert self.converter.can_handle("talker.model.layers.0.mlp.experts.0.gate_proj.weight")
        assert self.converter.can_handle("talker.model.layers.3.mlp.experts.1.down_proj.weight")

    def test_matches_standalone_text_model_keys(self):
        # When the thinker text submodel is loaded standalone the prefix is just `model.*`.
        assert self.converter.can_handle("model.layers.3.mlp.experts.2.gate_proj.weight")

    def test_rejects_other_keys(self):
        assert not self.converter.can_handle("thinker.model.layers.0.self_attn.q_proj.weight")
        # Fused keys (VeOmni-saved v5 layout) must NOT match — they pass through.
        assert not self.converter.can_handle("thinker.model.layers.0.mlp.experts.gate_up_proj")
        assert not self.converter.can_handle("thinker.model.layers.0.mlp.experts.down_proj")
        assert not self.converter.can_handle("thinker.model.embed_tokens.weight")


class TestQwen3OmniMoeConverterConvert:
    def setup_method(self):
        self.converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)

    def test_buffers_until_all_experts_arrive(self):
        # gate_proj for experts 0..N-2 should return None (still buffering).
        for e in range(OMNIMOE_NUM_EXPERTS - 1):
            key = _omnimoe_per_expert_key("thinker.model.layers.0.mlp", e, "gate_proj")
            result = self.converter.convert(key, torch.randn(OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN))
            assert result is None

    def test_gate_up_merge_after_full_stack(self):
        prefix = "thinker.model.layers.0.mlp"
        tensors = _omnimoe_per_expert_tensors(prefix)

        # Feed all gate_proj — no emit yet (waiting for up_proj).
        for e in range(OMNIMOE_NUM_EXPERTS):
            result = self.converter.convert(
                _omnimoe_per_expert_key(prefix, e, "gate_proj"),
                tensors[_omnimoe_per_expert_key(prefix, e, "gate_proj")],
            )
            assert result is None

        # Feed up_proj — the last one triggers the gate_up merge emission.
        emitted = []
        for e in range(OMNIMOE_NUM_EXPERTS):
            result = self.converter.convert(
                _omnimoe_per_expert_key(prefix, e, "up_proj"),
                tensors[_omnimoe_per_expert_key(prefix, e, "up_proj")],
            )
            if result is not None:
                emitted.append(result)

        assert len(emitted) == 1
        assert emitted[0].name == f"{prefix}.experts.gate_up_proj"
        assert emitted[0].tensor.shape == (OMNIMOE_NUM_EXPERTS, 2 * OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN)

        # Verify the merged tensor: rows 0..I are gate (100+e), rows I..2I are up (200+e).
        merged = emitted[0].tensor
        for e in range(OMNIMOE_NUM_EXPERTS):
            assert torch.allclose(
                merged[e, :OMNIMOE_INTERMEDIATE], torch.full_like(merged[e, :OMNIMOE_INTERMEDIATE], 100.0 + e)
            )
            assert torch.allclose(
                merged[e, OMNIMOE_INTERMEDIATE:], torch.full_like(merged[e, OMNIMOE_INTERMEDIATE:], 200.0 + e)
            )

    def test_down_proj_emits_after_full_stack(self):
        prefix = "thinker.model.layers.0.mlp"
        tensors = _omnimoe_per_expert_tensors(prefix)

        emitted = []
        for e in range(OMNIMOE_NUM_EXPERTS):
            result = self.converter.convert(
                _omnimoe_per_expert_key(prefix, e, "down_proj"),
                tensors[_omnimoe_per_expert_key(prefix, e, "down_proj")],
            )
            if result is not None:
                emitted.append(result)

        assert len(emitted) == 1
        assert emitted[0].name == f"{prefix}.experts.down_proj"
        # v5 layout for down_proj is [E, H, I] which matches stacking of HF [H, I].
        assert emitted[0].tensor.shape == (OMNIMOE_NUM_EXPERTS, OMNIMOE_HIDDEN, OMNIMOE_INTERMEDIATE)

    def test_rejects_non_expert_key(self):
        result = self.converter.convert("thinker.model.embed_tokens.weight", torch.randn(4, 4))
        assert result is None


class TestQwen3OmniMoeConverterFinalize:
    def test_finalize_noop_when_all_flushed(self):
        converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)
        prefix = "thinker.model.layers.0.mlp"
        tensors = _omnimoe_per_expert_tensors(prefix)
        for key, t in tensors.items():
            converter.convert(key, t)
        assert converter.finalize() == []

    def test_finalize_raises_on_incomplete_experts(self):
        converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)
        # Feed only 2 of 4 gate_proj experts.
        for e in range(2):
            converter.convert(
                _omnimoe_per_expert_key("thinker.model.layers.0.mlp", e, "gate_proj"),
                torch.randn(OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN),
            )
        with pytest.raises(RuntimeError, match="incomplete checkpoint"):
            converter.finalize()

    def test_finalize_raises_on_missing_up_after_full_gate_stack(self):
        converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)
        # Complete gate_proj but no up_proj — stacked buffer has dangling 'gate_proj'.
        for e in range(OMNIMOE_NUM_EXPERTS):
            converter.convert(
                _omnimoe_per_expert_key("thinker.model.layers.0.mlp", e, "gate_proj"),
                torch.randn(OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN),
            )
        with pytest.raises(RuntimeError, match="incomplete checkpoint"):
            converter.finalize()


class TestQwen3OmniMoeConverterFactory:
    def test_factory_with_top_level_omni_config(self):
        # Qwen3OmniMoeConfig — has `thinker_config.text_config`.
        text_config = SimpleNamespace(num_experts=OMNIMOE_NUM_EXPERTS)
        thinker_config = SimpleNamespace(text_config=text_config)
        model = SimpleNamespace(config=SimpleNamespace(thinker_config=thinker_config))
        converter = create_qwen3_omni_moe_checkpoint_tensor_converter(model)
        assert isinstance(converter, Qwen3OmniMoeCheckpointTensorConverter)
        assert converter.num_experts == OMNIMOE_NUM_EXPERTS

    def test_factory_with_thinker_config(self):
        # Qwen3OmniMoeThinkerForConditionalGeneration — `config.text_config`.
        text_config = SimpleNamespace(num_experts=OMNIMOE_NUM_EXPERTS)
        model = SimpleNamespace(config=SimpleNamespace(text_config=text_config))
        converter = create_qwen3_omni_moe_checkpoint_tensor_converter(model)
        assert converter.num_experts == OMNIMOE_NUM_EXPERTS

    def test_factory_with_flat_text_config(self):
        # Qwen3OmniMoeThinkerTextModel loaded standalone — flat config.
        flat = SimpleNamespace(num_experts=OMNIMOE_NUM_EXPERTS)
        model = SimpleNamespace(config=flat)
        converter = create_qwen3_omni_moe_checkpoint_tensor_converter(model)
        assert converter.num_experts == OMNIMOE_NUM_EXPERTS


class TestQwen3OmniMoeConverterIntegration:
    """End-to-end through `maybe_convert_checkpoint_tensor` using an HF per-expert checkpoint."""

    def test_full_layer_conversion(self):
        converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)
        prefix = "thinker.model.layers.0.mlp"

        non_expert_keys = [
            "thinker.model.layers.0.self_attn.q_proj.weight",
            f"{prefix}.gate.weight",
        ]
        dispatched = {}
        for key in non_expert_keys:
            t = torch.randn(4, 4)
            result = maybe_convert_checkpoint_tensor(key, t, converter)
            assert result is not None and result.name == key
            dispatched[result.name] = result.tensor

        tensors = _omnimoe_per_expert_tensors(prefix)
        for key, t in tensors.items():
            result = maybe_convert_checkpoint_tensor(key, t, converter)
            if result is not None:
                dispatched[result.name] = result.tensor

        assert converter.finalize() == []

        # Fused expert tensors are now in v5 modeling layout.
        assert dispatched[f"{prefix}.experts.gate_up_proj"].shape == (
            OMNIMOE_NUM_EXPERTS,
            2 * OMNIMOE_INTERMEDIATE,
            OMNIMOE_HIDDEN,
        )
        assert dispatched[f"{prefix}.experts.down_proj"].shape == (
            OMNIMOE_NUM_EXPERTS,
            OMNIMOE_HIDDEN,
            OMNIMOE_INTERMEDIATE,
        )

    def test_veomni_saved_fused_keys_pass_through(self):
        """A VeOmni-saved checkpoint stores fused keys directly; converter should ignore them."""
        converter = Qwen3OmniMoeCheckpointTensorConverter(num_experts=OMNIMOE_NUM_EXPERTS)
        prefix = "thinker.model.layers.0.mlp"
        gate_up = torch.randn(OMNIMOE_NUM_EXPERTS, 2 * OMNIMOE_INTERMEDIATE, OMNIMOE_HIDDEN)
        down = torch.randn(OMNIMOE_NUM_EXPERTS, OMNIMOE_HIDDEN, OMNIMOE_INTERMEDIATE)
        gate_up_res = maybe_convert_checkpoint_tensor(f"{prefix}.experts.gate_up_proj", gate_up, converter)
        down_res = maybe_convert_checkpoint_tensor(f"{prefix}.experts.down_proj", down, converter)
        # Pass-through: same tensor, same name.
        assert gate_up_res is not None and torch.equal(gate_up_res.tensor, gate_up)
        assert down_res is not None and torch.equal(down_res.tensor, down)
        # Nothing was buffered.
        assert converter.finalize() == []
