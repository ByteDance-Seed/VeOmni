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

import pytest
import torch
from transformers import AutoConfig

from tests.models.compare import eager_ops_config, ops_config_scope
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_TOY_CONFIG = "tests/toy_config/deepseek_v4_toy"


def _toy_config():
    config = AutoConfig.from_pretrained(_TOY_CONFIG)
    config._attn_implementation = "eager"
    return config


def _eager_compressors(config):
    """Build both compressor types with zeroed ``position_bias``.

    Compressors allocate ``position_bias`` with ``torch.empty``; the full model
    zeros it in ``_init_weights``. Standalone construction must do the same or
    softmax-gated compression sees allocator garbage / NaNs.
    """
    with ops_config_scope(eager_ops_config()):
        compressors = [cls(config) for cls in (modeling.DeepseekV4HCACompressor, modeling.DeepseekV4CSACompressor)]
    with torch.no_grad():
        for compressor in compressors:
            torch.nn.init.zeros_(compressor.position_bias)
            indexer = getattr(compressor, "indexer", None)
            if indexer is not None:
                torch.nn.init.zeros_(indexer.position_bias)
    return compressors


class _TileLangSparseAttentionSpy:
    impl = "tilelang"

    def __init__(self, captured: list[torch.Tensor]):
        self.captured = captured

    def __call__(self, query, key, sinks, topk_indices, **kwargs):
        del key, sinks, kwargs
        self.captured.append(topk_indices)
        return torch.zeros_like(query)


def test_deepseek_v4_stateless_forward_does_not_create_decode_cache():
    from transformers.cache_utils import DynamicCache

    config = _toy_config()
    with ops_config_scope(eager_ops_config()):
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
    from veomni.models.transformers.deepseek_v4.packed_utils import (
        build_packed_compression_metadata,
        build_sparse_attention_indices,
        mask_sparse_attention_indices,
    )

    torch.manual_seed(5)
    config = _toy_config()
    segment_lengths = (64, 96)
    total_len = sum(segment_lengths)
    hidden_states = torch.randn(1, total_len, config.hidden_size)
    q_residual = torch.randn(1, total_len, config.q_lora_rank)
    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0)
    sequence_slices = ((0, segment_lengths[0]), (segment_lengths[0], total_len))
    packed_metadata = build_packed_compression_metadata(
        hidden_states,
        position_ids,
        sequence_slices,
        tuple(config.compress_rates.values()),
        block_bias_rates=(config.compress_rates["heavily_compressed_attention"],),
    )

    for compressor in _eager_compressors(config):
        packed_kv, packed_bias = compressor(
            hidden_states,
            q_residual,
            position_ids,
            None,
            0,
            packed_sequence_slices=sequence_slices,
            packed_compression_metadata=packed_metadata,
        )
        compact_kv, compact_bias, compact_candidates = compressor(
            hidden_states,
            q_residual,
            position_ids,
            None,
            0,
            packed_sequence_slices=sequence_slices,
            packed_compression_metadata=packed_metadata,
            return_topk_indices=True,
        )
        torch.testing.assert_close(compact_kv, packed_kv)
        torch.testing.assert_close(compact_bias, packed_bias)
        if isinstance(compressor, modeling.DeepseekV4CSACompressor):
            compact_indices = compact_candidates.topk_indices
            assert compact_indices is not None
            valid = compact_indices >= 0
            safe_indices = compact_indices.clamp_min(0).unsqueeze(1)
            selected_bias = packed_bias.gather(-1, safe_indices)
            assert (selected_bias[valid.unsqueeze(1)] == 0).all()

            sliding_bias = hidden_states.new_full((1, 1, total_len, total_len), float("-inf"))
            for start, end in sequence_slices:
                for query_idx in range(start, end):
                    window_start = max(start, query_idx - config.sliding_window + 1)
                    sliding_bias[:, :, query_idx, window_start : query_idx + 1] = 0
            full_bias = torch.cat((sliding_bias, packed_bias), dim=-1)
            candidates = build_sparse_attention_indices(
                batch_size=1,
                seq_len=total_len,
                sliding_window=config.sliding_window,
                compressed_len=packed_kv.shape[2],
                compressed_indices=compact_indices,
                device=hidden_states.device,
            )
            filtered_indices = mask_sparse_attention_indices(full_bias, candidates)
            for query_idx in range(total_len):
                actual_indices = filtered_indices[0, query_idx]
                actual_indices = actual_indices[actual_indices >= 0].sort().values
                expected_indices = torch.where(full_bias[0, 0, query_idx] == 0)[0].to(torch.int32)
                torch.testing.assert_close(actual_indices, expected_indices)
        else:
            assert compact_candidates.topk_indices is None
            assert compact_candidates.range_starts is not None

        segment_outputs = []
        segment_biases = []
        for start, end in sequence_slices:
            segment_kv, segment_bias = compressor(
                hidden_states[:, start:end],
                q_residual[:, start:end],
                position_ids[:, start:end],
                None,
                0,
            )
            segment_outputs.append(segment_kv)
            segment_biases.append(segment_bias)

        torch.testing.assert_close(packed_kv, torch.cat(segment_outputs, dim=2))
        kv_offset = 0
        for (start, end), segment_kv, segment_bias in zip(
            sequence_slices, segment_outputs, segment_biases, strict=True
        ):
            kv_end = kv_offset + segment_kv.shape[2]
            torch.testing.assert_close(packed_bias[:, :, start:end, kv_offset:kv_end], segment_bias)
            assert torch.isneginf(packed_bias[:, :, start:end, :kv_offset]).all()
            assert torch.isneginf(packed_bias[:, :, start:end, kv_end:]).all()
            kv_offset = kv_end


def test_deepseek_v4_compact_sparse_indices_match_attention_mask():
    from veomni.models.transformers.deepseek_v4.packed_utils import (
        build_sparse_attention_indices,
        mask_sparse_attention_indices,
    )

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

    candidates = build_sparse_attention_indices(
        batch_size=1,
        seq_len=seq_len,
        sliding_window=sliding_window,
        compressed_len=compressed_len,
        compressed_indices=None,
        device=attention_mask.device,
    )
    actual = mask_sparse_attention_indices(attention_mask, candidates)

    for query_idx in range(seq_len):
        actual_indices = actual[0, query_idx]
        actual_indices = actual_indices[actual_indices >= 0].sort().values
        expected_indices = torch.where(attention_mask[0, 0, query_idx] == 0)[0].to(torch.int32)
        torch.testing.assert_close(actual_indices, expected_indices)


def test_deepseek_v4_mask_free_sparse_indices_match_dense_mask_path():
    """Candidates built from packed metadata must equal what the dense mask allows."""
    from transformers.masking_utils import create_sliding_window_causal_mask

    from veomni.models.transformers.deepseek_v4.packed_utils import (
        build_packed_compression_metadata,
        build_packed_sparse_attention_indices,
        build_sparse_attention_indices,
        isolate_packed_causal_mask_,
        mask_sparse_attention_indices,
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
        hidden_states,
        position_ids,
        sequence_slices,
        tuple(config.compress_rates.values()),
        block_bias_rates=(config.compress_rates["heavily_compressed_attention"],),
    )

    for compressor in _eager_compressors(config):
        packed_kwargs = {
            "packed_sequence_slices": sequence_slices,
            "packed_compression_metadata": packed_metadata,
            "return_topk_indices": True,
        }
        dense_kv, dense_bias, _ = compressor(hidden_states, q_residual, position_ids, None, 0, **packed_kwargs)
        free_kv, free_bias, candidates = compressor(
            hidden_states, q_residual, position_ids, None, 0, build_block_bias=False, **packed_kwargs
        )
        assert free_bias is None
        torch.testing.assert_close(free_kv, dense_kv)

        compressed_len = dense_kv.shape[2]
        expected = mask_sparse_attention_indices(
            torch.cat((sliding_mask, dense_bias), dim=-1),
            build_sparse_attention_indices(
                batch_size=1,
                seq_len=total_len,
                sliding_window=config.sliding_window,
                compressed_len=compressed_len,
                compressed_indices=candidates.topk_indices,
                device=hidden_states.device,
            ),
        )
        actual = build_packed_sparse_attention_indices(
            position_ids=position_ids,
            sliding_window=config.sliding_window,
            compressed_len=compressed_len,
            candidates=candidates,
        )
        # Width is load-bearing: the TileLang kernel specializes on the last dim,
        # so a narrower "compacted" list would trigger endless recompilation.
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="mask-free sparse dispatch requires bf16 CUDA tensors")
def test_deepseek_v4_packed_model_forward_skips_dense_mask(monkeypatch):
    """Packed TileLang forwards must never materialize an O(S^2) mask."""
    torch.manual_seed(11)
    device = torch.device(get_device_type())
    config = _toy_config()
    segment_lengths = (24, 40)
    total_len = sum(segment_lengths)
    with ops_config_scope(eager_ops_config()):
        model = modeling.DeepseekV4Model(config).to(device=device, dtype=torch.bfloat16).eval()

    def fail_on_dense_mask(*args, **kwargs):
        raise AssertionError("packed TileLang forward must not build a dense causal mask")

    monkeypatch.setattr(modeling, "create_sliding_window_causal_mask", fail_on_dense_mask)

    captured: list[torch.Tensor] = []
    for layer in model.layers:
        layer.self_attn.veomni_dsa_attention = _TileLangSparseAttentionSpy(captured)

    position_ids = torch.cat([torch.arange(length) for length in segment_lengths]).unsqueeze(0).to(device)
    cu_seq_lens = torch.tensor([0, segment_lengths[0], total_len], dtype=torch.int32, device=device)
    tilelang_ops = eager_ops_config()
    tilelang_ops.dsa_attention_implementation = "tilelang"
    with ops_config_scope(tilelang_ops), torch.no_grad():
        model(
            input_ids=torch.randint(0, config.vocab_size, (1, total_len), device=device),
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
        )

    assert len(captured) == config.num_hidden_layers
    # position_ids restart per sample, so this is each query's own segment start.
    segment_starts = (torch.arange(total_len, device=device) - position_ids[0]).to(torch.int32)
    queries = torch.arange(total_len, device=device, dtype=torch.int32)
    for topk_indices in captured:
        sliding = topk_indices[0, :, :]
        is_sliding = (sliding >= 0) & (sliding < total_len)
        assert is_sliding.any(), "no sliding candidate survived"
        within_sample = sliding >= segment_starts[:, None]
        causal = sliding <= queries[:, None]
        assert (within_sample | ~is_sliding).all(), "sliding candidate crossed a packed boundary"
        assert (causal | ~is_sliding).all(), "sliding candidate is not causal"


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
