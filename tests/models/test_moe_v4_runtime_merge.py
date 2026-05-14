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
Tests for transformers v4 on-the-fly MoE expert weight stacking converters.

Only `deepseek_v3` is exercised here: `qwen3_moe` and `qwen3_omni_moe` already
have transformers v5 patchgen support and a v5 runtime converter, so their v4
factories were retired together with the v4 ``__init__`` branches in those
models. The shared `MoEV4StackingConverter` plumbing stays covered by the
generic unit tests further down.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from veomni.models.checkpoint_tensor_loading import maybe_convert_checkpoint_tensor
from veomni.models.transformers._moe_v4_converter import MoEV4StackingConverter
from veomni.models.transformers.deepseek_v3.checkpoint_tensor_converter_v4 import (
    create_deepseek_v3_v4_checkpoint_tensor_converter,
)


NUM_EXPERTS = 4
HIDDEN_DIM = 8
INTERMEDIATE_DIM = 6


def _hf_expert_key(prefix: str, expert: int, proj: str) -> str:
    return f"{prefix}.experts.{expert}.{proj}.weight"


def _hf_expert_tensor(proj: str, expert_id: int) -> torch.Tensor:
    """Deterministic per-expert tensor in HF layout.

    gate_proj/up_proj: [I, H]; down_proj: [H, I]. Filled with `expert_id + offset`
    so we can assert the post-stack tensor has the right value at each index.
    """
    if proj == "down_proj":
        shape = (HIDDEN_DIM, INTERMEDIATE_DIM)
    else:
        shape = (INTERMEDIATE_DIM, HIDDEN_DIM)
    offset = {"gate_proj": 0.0, "up_proj": 0.1, "down_proj": 0.2}[proj]
    return torch.full(shape, float(expert_id) + offset)


# Same regex used by the qwen3_moe / deepseek_v3 / qwen3_omni_moe v4 factories.
_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


# ---------------------------------------------------------------------------
# MoEV4StackingConverter unit tests
# ---------------------------------------------------------------------------


class TestMoEV4StackingConverterCanHandle:
    def setup_method(self):
        self.converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)

    def test_matches_per_expert_keys(self):
        assert self.converter.can_handle("model.layers.0.mlp.experts.0.gate_proj.weight")
        assert self.converter.can_handle("model.layers.7.mlp.experts.31.up_proj.weight")
        assert self.converter.can_handle("model.layers.2.mlp.experts.1.down_proj.weight")

    def test_rejects_non_expert_keys(self):
        # Already-stacked keys (e.g. from a pre-merged checkpoint) MUST pass through unchanged.
        assert not self.converter.can_handle("model.layers.0.mlp.experts.gate_proj")
        assert not self.converter.can_handle("model.layers.0.mlp.experts.up_proj")
        assert not self.converter.can_handle("model.layers.0.mlp.experts.down_proj")
        # Other parameters.
        assert not self.converter.can_handle("model.layers.0.self_attn.q_proj.weight")
        assert not self.converter.can_handle("model.layers.0.mlp.gate.weight")
        assert not self.converter.can_handle("model.embed_tokens.weight")


class TestMoEV4StackingConverterStacks:
    def setup_method(self):
        self.converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)

    def _feed_all(self, prefix: str, proj: str):
        results = []
        for e in range(NUM_EXPERTS):
            key = _hf_expert_key(prefix, e, proj)
            results.append(self.converter.convert(key, _hf_expert_tensor(proj, e)))
        return results

    def test_buffers_until_complete(self):
        # First N-1 experts return None.
        for e in range(NUM_EXPERTS - 1):
            key = _hf_expert_key("model.layers.0.mlp", e, "gate_proj")
            assert self.converter.convert(key, _hf_expert_tensor("gate_proj", e)) is None

    def test_emits_three_separate_keys(self):
        """v4 emits three keys (gate/up/down), unlike v5 which merges gate+up."""
        emitted = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for r in self._feed_all("model.layers.0.mlp", proj):
                if r is not None:
                    emitted[r.name] = r.tensor

        # Three separate stacked tensors — not a single fused gate_up_proj.
        assert "model.layers.0.mlp.experts.gate_proj" in emitted
        assert "model.layers.0.mlp.experts.up_proj" in emitted
        assert "model.layers.0.mlp.experts.down_proj" in emitted
        assert "model.layers.0.mlp.experts.gate_up_proj" not in emitted

        # Shapes match the v4 modeling layout.
        assert emitted["model.layers.0.mlp.experts.gate_proj"].shape == (
            NUM_EXPERTS,
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )
        assert emitted["model.layers.0.mlp.experts.up_proj"].shape == (
            NUM_EXPERTS,
            INTERMEDIATE_DIM,
            HIDDEN_DIM,
        )
        assert emitted["model.layers.0.mlp.experts.down_proj"].shape == (
            NUM_EXPERTS,
            HIDDEN_DIM,
            INTERMEDIATE_DIM,
        )

        # Verify per-expert content survived the stack.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = emitted[f"model.layers.0.mlp.experts.{proj}"]
            for e in range(NUM_EXPERTS):
                assert torch.equal(stacked[e], _hf_expert_tensor(proj, e))

    def test_experts_out_of_order(self):
        """Safetensors shards may yield experts in any order; stacking must still index by expert id."""
        for e in [3, 0, 2, 1]:
            key = _hf_expert_key("model.layers.0.mlp", e, "down_proj")
            self.converter.convert(key, _hf_expert_tensor("down_proj", e))

        # Trigger the emit on the last expert. We re-feed expert 1 with the
        # same tensor, but actually the previous loop already fed all 4, so
        # the last `convert` call returned the stacked result. Fetch it via
        # a fresh independent converter run for clarity:
        converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)
        last = None
        for e in [2, 0, 3, 1]:
            last = converter.convert(
                _hf_expert_key("model.layers.0.mlp", e, "down_proj"),
                _hf_expert_tensor("down_proj", e),
            )
        assert last is not None
        for e in range(NUM_EXPERTS):
            assert torch.equal(last.tensor[e], _hf_expert_tensor("down_proj", e))

    def test_independent_layers(self):
        """Different layers buffer independently."""
        for e in range(NUM_EXPERTS):
            r0 = self.converter.convert(
                _hf_expert_key("model.layers.0.mlp", e, "down_proj"),
                _hf_expert_tensor("down_proj", e),
            )
            r1 = self.converter.convert(
                _hf_expert_key("model.layers.1.mlp", e, "down_proj"),
                _hf_expert_tensor("down_proj", e),
            )
            if e < NUM_EXPERTS - 1:
                assert r0 is None and r1 is None
            else:
                assert r0 is not None and r0.name == "model.layers.0.mlp.experts.down_proj"
                assert r1 is not None and r1.name == "model.layers.1.mlp.experts.down_proj"


class TestMoEV4StackingConverterFinalize:
    def test_noop_when_all_flushed(self):
        converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for e in range(NUM_EXPERTS):
                converter.convert(_hf_expert_key("model.layers.0.mlp", e, proj), _hf_expert_tensor(proj, e))
        assert converter.finalize() == []

    def test_raises_with_helpful_diagnostic_on_incomplete(self):
        converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)
        # Drop expert 3 from gate_proj (simulates a corrupted checkpoint).
        for e in range(NUM_EXPERTS - 1):
            converter.convert(_hf_expert_key("model.layers.0.mlp", e, "gate_proj"), _hf_expert_tensor("gate_proj", e))
        with pytest.raises(RuntimeError, match="incomplete checkpoint detected"):
            converter.finalize()

    def test_finalize_lists_collected_expert_ids(self):
        converter = MoEV4StackingConverter(_PATTERN, num_experts_for=lambda _: NUM_EXPERTS)
        for e in (0, 2):
            converter.convert(_hf_expert_key("model.layers.5.mlp", e, "up_proj"), _hf_expert_tensor("up_proj", e))
        with pytest.raises(RuntimeError) as exc:
            converter.finalize()
        # The diagnostic should name the buffered key and the collected expert ids
        # so users can see which expert is actually missing.
        assert "model.layers.5.mlp.experts.up_proj" in str(exc.value)
        assert "[0, 2]" in str(exc.value)


# ---------------------------------------------------------------------------
# Multi-tower variant (Qwen3-Omni-MoE) — per-prefix expert counts
# ---------------------------------------------------------------------------


class TestMoEV4StackingConverterMultiPrefix:
    def test_per_prefix_expert_counts(self):
        """num_experts_for can vary by prefix — covers thinker/talker towers with different sizes."""
        thinker_n, talker_n = 3, 5

        def num_experts_for(prefix: str) -> int:
            if prefix.startswith("thinker."):
                return thinker_n
            return talker_n

        pattern = re.compile(
            r"^((?:thinker|talker)\.model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
        )
        converter = MoEV4StackingConverter(pattern, num_experts_for=num_experts_for)

        # Feed thinker_n experts under thinker prefix → emits at the thinker_n-th call.
        thinker_emit = None
        for e in range(thinker_n):
            thinker_emit = converter.convert(
                f"thinker.model.layers.0.mlp.experts.{e}.gate_proj.weight",
                torch.full((INTERMEDIATE_DIM, HIDDEN_DIM), float(e)),
            )
        assert thinker_emit is not None
        assert thinker_emit.name == "thinker.model.layers.0.mlp.experts.gate_proj"
        assert thinker_emit.tensor.shape == (thinker_n, INTERMEDIATE_DIM, HIDDEN_DIM)

        # Feed talker_n experts under talker prefix → emits at the talker_n-th call.
        talker_emit = None
        for e in range(talker_n):
            talker_emit = converter.convert(
                f"talker.model.layers.0.mlp.experts.{e}.gate_proj.weight",
                torch.full((INTERMEDIATE_DIM, HIDDEN_DIM), float(e)),
            )
        assert talker_emit is not None
        assert talker_emit.name == "talker.model.layers.0.mlp.experts.gate_proj"
        assert talker_emit.tensor.shape == (talker_n, INTERMEDIATE_DIM, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# deepseek_v3 v4 factory
# ---------------------------------------------------------------------------


class TestDeepseekV3V4Factory:
    def test_factory_uses_n_routed_experts(self):
        # DeepSeek V3 config exposes `n_routed_experts`, NOT `num_experts`.
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=NUM_EXPERTS))
        converter = create_deepseek_v3_v4_checkpoint_tensor_converter(model)
        assert isinstance(converter, MoEV4StackingConverter)

        # Stack one full set; emit happens on the N-th expert.
        last = None
        for e in range(NUM_EXPERTS):
            last = converter.convert(
                _hf_expert_key("model.layers.5.mlp", e, "down_proj"),
                _hf_expert_tensor("down_proj", e),
            )
        assert last is not None
        assert last.name == "model.layers.5.mlp.experts.down_proj"

    def test_dense_layers_pass_through(self):
        """Dense MLP layers (< first_k_dense_replace) have no `experts.*` keys; the
        regex should never match those keys, so they pass through untouched."""
        model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=NUM_EXPERTS))
        converter = create_deepseek_v3_v4_checkpoint_tensor_converter(model)
        # Dense layer: standard MLP weight key, no experts in path.
        dense_key = "model.layers.0.mlp.gate_proj.weight"
        dense_t = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM)
        result = maybe_convert_checkpoint_tensor(dense_key, dense_t, converter)
        assert result is not None
        assert result.name == dense_key
        assert torch.equal(result.tensor, dense_t)
