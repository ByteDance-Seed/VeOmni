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
