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

"""DeepSeek-V4 packed-sample isolation: compressors, sparse indices, masks, and cache."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoConfig

from tests.models.compare import eager_ops_config
from veomni.models.configuration_base import as_veomni_config
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_npu as npu_modeling
from veomni.ops import resolve_op
from veomni.ops.kernels.dsa.indexer.deepseek_v4 import cann as cann_indexer
from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates, build_sparse_attention_indices


_TOY_CONFIG = "tests/toy_config/deepseek_v4_toy"


def _toy_config():
    config = AutoConfig.from_pretrained(_TOY_CONFIG)
    as_veomni_config(config, validate=False)._set_ops_implementation(eager_ops_config())
    return config


def _eager_compressors(config):
    """Build both compressor types with zeroed ``position_bias``.

    Compressors allocate ``position_bias`` with ``torch.empty``; the full model
    zeros it in ``_init_weights``. Standalone construction must do the same or
    softmax-gated compression sees allocator garbage / NaNs.
    """
    as_veomni_config(config, validate=False)._set_ops_implementation(eager_ops_config())
    compressors = [cls(config) for cls in (modeling.DeepseekV4HCACompressor, modeling.DeepseekV4CSACompressor)]
    with torch.no_grad():
        for compressor in compressors:
            torch.nn.init.zeros_(compressor.position_bias)
            indexer = getattr(compressor, "indexer", None)
            if indexer is not None:
                torch.nn.init.zeros_(indexer.position_bias)
    return compressors


class _SparseAttentionSpy:
    """Stand-in ``dsa_attention`` op that records what each layer is asked to attend to."""

    def __init__(self) -> None:
        self.attention_masks: list[torch.Tensor | None] = []
        self.candidate_slots: list[torch.Tensor] = []
        self.packed_slices: list[tuple[tuple[int, int], ...] | None] = []
        self.compressed: list[tuple[int, int, CompressedCandidates | None]] = []

    def __call__(self, q, ori_kv, cmp_kv, sinks, position_ids, **kwargs):
        del sinks
        self.attention_masks.append(kwargs["attention_mask"])
        self.packed_slices.append(kwargs["packed_sequence_slices"])
        if cmp_kv is not None:
            self.compressed.append((kwargs["compress_ratio"], cmp_kv.shape[1], kwargs["candidates"]))
        self.candidate_slots.append(
            build_sparse_attention_indices(
                position_ids=position_ids,
                sliding_window=kwargs["sliding_window"],
                kv_len=ori_kv.shape[1],
                compressed_len=0 if cmp_kv is None else cmp_kv.shape[1],
                candidates=kwargs["candidates"],
                query_offset=kwargs["query_offset"],
                attention_mask=kwargs["attention_mask"],
            )
        )
        return torch.zeros_like(q)


def _visible_compressed(candidates: CompressedCandidates, seq_len: int, compressed_len: int) -> torch.Tensor:
    """``[S, C]`` bool: which compressed rows each query of a batch-1 call may read."""
    if candidates.topk_indices is not None:
        indices = candidates.topk_indices[0].long()
        visible = torch.zeros(seq_len, compressed_len + 1, dtype=torch.bool)
        visible.scatter_(-1, torch.where(indices >= 0, indices, compressed_len), True)
        return visible[:, :compressed_len]
    entries = torch.arange(compressed_len)
    starts = candidates.range_starts.reshape(-1, 1)
    ends = candidates.range_ends.reshape(-1, 1)
    return (entries >= starts) & (entries < ends)


def test_deepseek_v4_stateless_forward_does_not_create_decode_cache():
    from transformers.cache_utils import DynamicCache

    config = _toy_config()
    model = modeling.DeepseekV4Model(config)
    seen_caches = []

    for layer in model.layers:

        def passthrough(self, hidden_states, past_key_values=None, **kwargs):
            seen_caches.append(past_key_values)
            return hidden_states

        layer.forward = types.MethodType(passthrough, layer)

    input_ids = torch.arange(8).unsqueeze(0)
    model(input_ids=input_ids, use_cache=False)
    assert seen_caches and all(cache is None for cache in seen_caches)

    seen_caches.clear()
    output = model(input_ids=input_ids, use_cache=True)
    assert seen_caches and all(isinstance(cache, DynamicCache) for cache in seen_caches)
    assert isinstance(output.past_key_values, DynamicCache)


def test_deepseek_v4_packed_compressors_match_independent_sequences():
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    torch.manual_seed(5)
    config = _toy_config()
    segment_lengths = (64, 96)
    total_len = sum(segment_lengths)
    hidden_states = torch.randn(1, total_len, config.hidden_size)
    q_residual = torch.randn(1, total_len, config.q_lora_rank)
    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0)
    sequence_slices = ((0, segment_lengths[0]), (segment_lengths[0], total_len))
    packed_metadata = build_packed_compression_metadata(
        hidden_states, position_ids, sequence_slices, tuple(config.compress_rates.values())
    )

    for compressor in _eager_compressors(config):
        packed_kv, packed_candidates = compressor(
            hidden_states,
            q_residual,
            position_ids,
            None,
            0,
            packed_sequence_slices=sequence_slices,
            packed_compression_metadata=packed_metadata,
        )
        if isinstance(compressor, modeling.DeepseekV4CSACompressor):
            assert packed_candidates.topk_indices is not None
        else:
            assert packed_candidates.topk_indices is None
            assert packed_candidates.range_starts is not None
        packed_visible = _visible_compressed(packed_candidates, total_len, packed_kv.shape[2])

        segment_outputs = []
        kv_offset = 0
        for start, end in sequence_slices:
            segment_kv, segment_candidates = compressor(
                hidden_states[:, start:end],
                q_residual[:, start:end],
                position_ids[:, start:end],
                None,
                0,
            )
            segment_outputs.append(segment_kv)
            kv_end = kv_offset + segment_kv.shape[2]
            segment_visible = _visible_compressed(segment_candidates, end - start, segment_kv.shape[2])
            torch.testing.assert_close(packed_visible[start:end, kv_offset:kv_end], segment_visible)
            assert not packed_visible[start:end, :kv_offset].any()
            assert not packed_visible[start:end, kv_end:].any()
            kv_offset = kv_end

        torch.testing.assert_close(packed_kv, torch.cat(segment_outputs, dim=2))


class _IndexerSpy:
    impl = "eager"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return resolve_op("dsa_indexer", "deepseek_v4", "eager").wrapper(*args, **kwargs)


class _FakeCannOps:
    @staticmethod
    def lightning_indexer_metadata(*args, **kwargs):
        return torch.empty(1, dtype=torch.int32)

    @staticmethod
    def lightning_indexer(q, k, weights, top_k, **kwargs):
        shape = (q.shape[0], 1, top_k)
        return torch.zeros(shape, dtype=torch.int32), torch.zeros(shape)


@pytest.mark.parametrize("module", [modeling, npu_modeling], ids=["gpu", "npu"])
def test_deepseek_v4_packed_indexer_op_receives_segments_cann_accepts(module, monkeypatch):
    """The packed CSA path hands the indexer op segment boundaries whose ranges CANN can mask."""
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    torch.manual_seed(13)
    config = _toy_config()
    segment_lengths = (37, 64, 27)
    total_len = sum(segment_lengths)
    bounds = torch.tensor((0, *segment_lengths)).cumsum(0).tolist()
    sequence_slices = tuple(zip(bounds[:-1], bounds[1:]))
    hidden_states = torch.randn(1, total_len, config.hidden_size)
    q_residual = torch.randn(1, total_len, config.q_lora_rank)
    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0)
    packed_metadata = build_packed_compression_metadata(
        hidden_states, position_ids, sequence_slices, tuple(config.compress_rates.values())
    )
    compressor = module.DeepseekV4CSACompressor(config)
    with torch.no_grad():
        torch.nn.init.zeros_(compressor.position_bias)
        torch.nn.init.zeros_(compressor.indexer.position_bias)
    spy = _IndexerSpy()
    compressor.indexer.veomni_dsa_indexer = spy

    compressor(
        hidden_states,
        q_residual,
        position_ids,
        None,
        0,
        packed_sequence_slices=sequence_slices,
        packed_compression_metadata=packed_metadata,
    )

    ((args, kwargs),) = spy.calls
    assert kwargs["packed_sequence_slices"] == sequence_slices
    assert args[0].shape[:2] == (total_len, 1)
    monkeypatch.setattr(cann_indexer, "_ops", lambda: _FakeCannOps)
    cann_indexer._packed_plan.cache_clear()
    try:
        scores, indices = cann_indexer.wrapper(*args, **kwargs)
    finally:
        cann_indexer._packed_plan.cache_clear()
    assert indices.shape == scores.shape == (1, total_len, min(config.index_topk, args[1].shape[0]))


def test_deepseek_v4_npu_config_collator_preserves_packed_segments_for_model():
    from veomni.data.data_collator import PackingCollator

    npu_config = yaml.safe_load(Path("configs/text/deepseek_v4_npu.yaml").read_text())
    assert npu_config["train"].get("dyn_bsz", True) is True
    assert npu_config["train"]["micro_batch_size"] == 1

    lengths = (5, 3)
    features = [
        {
            "input_ids": torch.arange(length),
            "labels": torch.arange(length),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "position_ids": torch.arange(length),
        }
        for length in lengths
    ]
    batch = PackingCollator()(features)
    seq_len = sum(lengths)
    assert batch["cu_seq_lens_q"].tolist() == [0, lengths[0], seq_len]

    model = npu_modeling.DeepseekV4Model(_toy_config())
    seen_slices = []
    for layer in model.layers:

        def passthrough(self, hidden_states, **kwargs):
            seen_slices.append(kwargs.get("packed_sequence_slices"))
            return hidden_states

        layer.forward = types.MethodType(passthrough, layer)

    model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        cu_seq_lens_q=batch["cu_seq_lens_q"],
        use_cache=False,
    )
    assert seen_slices
    assert all(slices == ((0, lengths[0]), (lengths[0], seq_len)) for slices in seen_slices)


def test_deepseek_v4_sparse_indices_match_attention_mask():
    seq_len, compressed_len, sliding_window = 6, 3, 3
    attention_mask = torch.full((1, 1, seq_len, seq_len + compressed_len), float("-inf"))
    segment_starts = (0, 0, 0, 3, 3, 3)
    compressed_ranges = ((0, 1), (0, 1), (0, 2), (2, 2), (2, 2), (2, 3))
    for query_idx, (segment_start, (compressed_start, compressed_end)) in enumerate(
        zip(segment_starts, compressed_ranges, strict=True)
    ):
        window_start = max(segment_start, query_idx - sliding_window + 1)
        attention_mask[0, 0, query_idx, window_start : query_idx + 1] = 0
        attention_mask[0, 0, query_idx, seq_len + compressed_start : seq_len + compressed_end] = 0

    actual = build_sparse_attention_indices(
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
        sliding_window=sliding_window,
        kv_len=seq_len,
        compressed_len=compressed_len,
        candidates=CompressedCandidates(
            range_starts=torch.tensor([start for start, _ in compressed_ranges], dtype=torch.int32),
            range_ends=torch.tensor([end for _, end in compressed_ranges], dtype=torch.int32),
        ),
    )

    for query_idx in range(seq_len):
        actual_indices = actual[0, query_idx]
        actual_indices = actual_indices[actual_indices >= 0].sort().values
        expected_indices = torch.where(attention_mask[0, 0, query_idx] == 0)[0].to(torch.int32)
        torch.testing.assert_close(actual_indices, expected_indices)


def test_deepseek_v4_mask_free_sparse_indices_match_dense_mask_path():
    """The dense mask the model skips on packed input must not narrow any candidate."""
    from transformers.masking_utils import create_sliding_window_causal_mask

    from veomni.models.transformers.deepseek_v4.packed_utils import (
        build_packed_compression_metadata,
        isolate_packed_causal_mask_,
    )

    torch.manual_seed(7)
    config = _toy_config()
    segment_lengths = (64, 96)
    total_len = sum(segment_lengths)
    sequence_slices = ((0, segment_lengths[0]), (segment_lengths[0], total_len))
    hidden_states = torch.randn(1, total_len, config.hidden_size)
    q_residual = torch.randn(1, total_len, config.q_lora_rank)
    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0)

    # The production oracle: the exact mask DeepseekV4Model.forward builds on the dense path.
    sliding_mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=torch.ones(1, total_len, dtype=torch.long),
        past_key_values=None,
        position_ids=position_ids,
    )
    sliding_mask = isolate_packed_causal_mask_(sliding_mask, sequence_slices)

    packed_metadata = build_packed_compression_metadata(
        hidden_states, position_ids, sequence_slices, tuple(config.compress_rates.values())
    )

    for compressor in _eager_compressors(config):
        compressed_kv, candidates = compressor(
            hidden_states,
            q_residual,
            position_ids,
            None,
            0,
            packed_sequence_slices=sequence_slices,
            packed_compression_metadata=packed_metadata,
        )
        contract = {
            "position_ids": position_ids,
            "sliding_window": config.sliding_window,
            "kv_len": total_len,
            "compressed_len": compressed_kv.shape[2],
            "candidates": candidates,
        }
        expected = build_sparse_attention_indices(**contract, attention_mask=sliding_mask)
        actual = build_sparse_attention_indices(**contract)
        # Width is load-bearing: the TileLang kernel specializes on the last dim,
        # so a narrower "compacted" list would trigger endless recompilation.
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


def _packed_model_inputs(config, segment_lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    total_len = sum(segment_lengths)
    cu_seq_lens = torch.tensor((0, *segment_lengths), dtype=torch.int32).cumsum(0).to(torch.int32)
    return {
        "input_ids": torch.randint(0, config.vocab_size, (1, total_len)),
        "position_ids": torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0),
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
    }


@pytest.mark.parametrize("module", [modeling, npu_modeling], ids=["gpu", "npu"])
def test_deepseek_v4_packed_model_forward_skips_dense_mask(module, monkeypatch):
    """Packed forwards hand the op segments and per-sample candidates, never an O(S^2) mask."""
    torch.manual_seed(11)
    config = _toy_config()
    segment_lengths = (24, 40)
    total_len = sum(segment_lengths)
    model = module.DeepseekV4Model(config).eval()

    def fail_on_dense_mask(*args, **kwargs):
        raise AssertionError("packed forward must not build a dense causal mask")

    monkeypatch.setattr(module, "create_sliding_window_causal_mask", fail_on_dense_mask)

    spy = _SparseAttentionSpy()
    for layer in model.layers:
        layer.self_attn.veomni_dsa_attention = spy

    inputs = _packed_model_inputs(config, segment_lengths)
    with torch.no_grad():
        model(**inputs, use_cache=False)

    assert len(spy.candidate_slots) == config.num_hidden_layers
    assert all(mask is None for mask in spy.attention_masks)
    position_ids = inputs["position_ids"]
    # position_ids restart per sample, so this is each query's own segment start.
    segment_starts = (torch.arange(total_len) - position_ids[0]).to(torch.int32)
    queries = torch.arange(total_len, dtype=torch.int32)
    for topk_indices in spy.candidate_slots:
        sliding = topk_indices[0, :, :]
        is_sliding = (sliding >= 0) & (sliding < total_len)
        assert is_sliding.any(), "no sliding candidate survived"
        within_sample = sliding >= segment_starts[:, None]
        causal = sliding <= queries[:, None]
        assert (within_sample | ~is_sliding).all(), "sliding candidate crossed a packed boundary"
        assert (causal | ~is_sliding).all(), "sliding candidate is not causal"

    assert spy.packed_slices == [((0, 24), (24, total_len))] * config.num_hidden_layers
    assert {ratio for ratio, _, _ in spy.compressed} == set(config.compress_rates.values())
    query_sample = torch.arange(total_len) >= segment_lengths[0]
    for ratio, compressed_len, candidates in spy.compressed:
        assert candidates is not None
        assert (candidates.topk_indices is not None) == (ratio == config.compress_rates["compressed_sparse_attention"])
        visible = _visible_compressed(candidates, total_len, compressed_len)
        entry_sample = torch.arange(compressed_len) >= segment_lengths[0] // ratio
        crossed = visible & (query_sample[:, None] != entry_sample[None, :])
        assert not crossed.any(), f"ratio-{ratio} compressed candidate crossed a packed boundary"


def test_deepseek_v4_mask_free_packed_eager_forward_matches_samples_run_alone(monkeypatch):
    """Without the dense mask, eager attention still keeps every packed sample to itself."""
    torch.manual_seed(12)
    config = _toy_config()
    segment_lengths = (24, 40)
    model = modeling.DeepseekV4Model(config).eval()

    inputs = _packed_model_inputs(config, segment_lengths)
    expected = []
    with torch.no_grad():
        for start, end in zip(inputs["cu_seq_lens_q"][:-1].tolist(), inputs["cu_seq_lens_q"][1:].tolist()):
            alone = model(input_ids=inputs["input_ids"][:, start:end], use_cache=False)
            expected.append(alone.last_hidden_state)

        def fail_on_dense_mask(*args, **kwargs):
            raise AssertionError("packed forward must not build a dense causal mask")

        monkeypatch.setattr(modeling, "create_sliding_window_causal_mask", fail_on_dense_mask)
        packed = model(**inputs, use_cache=False).last_hidden_state

    torch.testing.assert_close(packed, torch.cat(expected, dim=1), atol=1e-5, rtol=1e-5)


def test_deepseek_v4_packed_causal_mask_blocks_previous_samples():
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    from veomni.models.transformers.deepseek_v4.packed_utils import isolate_packed_causal_mask_

    config = _toy_config()
    hidden_states = torch.randn(1, 8, config.hidden_size)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    causal_mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=hidden_states,
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        past_key_values=DynamicCache(config=config),
        position_ids=position_ids,
    )
    assert causal_mask is not None and (causal_mask[0, 0, 4, :4] == 0).all()

    isolate_packed_causal_mask_(causal_mask, ((0, 4), (4, 8)))

    assert (causal_mask[0, 0, 4:, :4] == torch.finfo(causal_mask.dtype).min).all()
    assert causal_mask[0, 0, 4, 4] == 0
