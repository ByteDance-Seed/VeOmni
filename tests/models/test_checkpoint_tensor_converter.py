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

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from veomni.models.checkpoint_tensor_loading import (
    ConvertedCheckpointTensor,
    get_checkpoint_tensor_converter,
    maybe_convert_checkpoint_tensor,
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


class TestQwen3MoeConverterFactory:
    def test_factory_creates_converter(self):
        model = SimpleNamespace(config=SimpleNamespace(num_experts=8))
        converter = create_qwen3_moe_checkpoint_tensor_converter(model)
        assert isinstance(converter, Qwen3MoeCheckpointTensorConverter)
        assert converter.num_experts == 8


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
