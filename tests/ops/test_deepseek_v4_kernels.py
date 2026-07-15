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

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.deepseek_v4 import linear_bf16_fp32
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()


def test_kernel_package_does_not_import_tilelang_eagerly():
    sys.modules.pop("veomni.ops.kernels.deepseek_v4", None)
    before = "tilelang" in sys.modules

    importlib.import_module("veomni.ops.kernels.deepseek_v4")

    assert ("tilelang" in sys.modules) is before


def test_linear_bf16_fp32_matches_bf16_rounded_fp32_reference():
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)

    actual = linear_bf16_fp32(x, weight)
    expected = x.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().t()

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_linear_bf16_fp32_backward_matches_reference():
    x = torch.randn(7, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(5, 8, dtype=torch.float32, requires_grad=True)
    grad = torch.randn(7, 5)

    linear_bf16_fp32(x, weight).backward(grad)
    expected_x_grad = grad @ weight.detach().to(torch.bfloat16).float()
    expected_weight_grad = grad.t() @ x.detach().to(torch.bfloat16).float()

    torch.testing.assert_close(x.grad, expected_x_grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)


def test_tilelang_wrappers_reject_pre_sm90_before_import(monkeypatch):
    import veomni.ops.kernels.deepseek_v4 as kernels

    monkeypatch.setattr(kernels, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(kernels, "get_gpu_compute_capability", lambda: 89)
    args = (torch.empty(0),) * 4

    with pytest.raises(RuntimeError, match="SM90 or later"):
        kernels.sparse_attn_tilelang(*args)
    with pytest.raises(RuntimeError, match="SM90 or later"):
        kernels.v4_lighting_indexer(*args[:3], compress_ratio=1, topk=1)
    with pytest.raises(RuntimeError, match="SM90 or later"):
        kernels.act_quant(torch.empty(0))


def test_tilelang_wrappers_reject_rocm_before_import(monkeypatch):
    import veomni.ops.kernels.deepseek_v4 as kernels

    monkeypatch.setattr(kernels.torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(kernels, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(kernels, "get_gpu_compute_capability", lambda: 90)
    args = (torch.empty(0),) * 4

    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        kernels.sparse_attn_tilelang(*args)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        kernels.v4_lighting_indexer(*args[:3], compress_ratio=1, topk=1)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        kernels.act_quant(torch.empty(0))


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


def _indexer_reference(q, k, weights, topk_indices):
    logits = torch.einsum("sbhd,tbd->bsht", q.float(), k.float())
    logits = (logits.relu() * weights.permute(1, 0, 2).float().unsqueeze(-1)).sum(dim=2)
    valid = (topk_indices >= 0) & (topk_indices < logits.shape[-1])
    scores = logits.gather(-1, topk_indices.clamp(0, logits.shape[-1] - 1).long())
    return torch.where(valid, scores, float("-inf"))


def _cosine_similarity(actual, expected):
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)


def test_tilelang_indexer_non_power_of_two_topk_forward_backward():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import v4_lighting_indexer

    torch.manual_seed(0)
    seqlen, batch, heads, dim, compress_ratio, topk = 130, 2, 8, 128, 4, 65
    kv_len = seqlen // compress_ratio
    q = torch.randn(seqlen, batch, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(kv_len, batch, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weights = (torch.randn(seqlen, batch, heads, device=DEVICE) * 0.01).requires_grad_()

    indices = torch.full((batch, seqlen, topk), -1, device=DEVICE, dtype=torch.int32)
    for position in range(seqlen):
        valid_count = min((position + 1) // compress_ratio, kv_len, topk)
        if valid_count:
            indices[:, position, :valid_count] = torch.arange(valid_count, device=DEVICE, dtype=torch.int32)
    indices[..., -3] = -1
    indices[..., -2] = -2
    indices[..., -1] = kv_len

    actual, actual_indices = v4_lighting_indexer(q, k, weights, compress_ratio, topk, indices)
    expected = _indexer_reference(q, k, weights, indices)
    valid = (indices >= 0) & (indices < kv_len)
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_indices, indices)

    grad = torch.randn_like(actual).masked_fill(~valid, 0)
    expected_grads = torch.autograd.grad((expected.masked_fill(~valid, 0) * grad).sum(), (q, k, weights))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, weights.grad), expected_grads, strict=True):
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_indexer_packed_forward_backward():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import v4_lighting_indexer

    torch.manual_seed(4)
    segment_len, heads, dim, compress_ratio, topk = 64, 8, 128, 4, 7
    seqlen = 2 * segment_len
    kv_per_segment = segment_len // compress_ratio
    kv_len = 2 * kv_per_segment
    q = torch.randn(seqlen, 1, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(kv_len, 1, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weights = (torch.randn(seqlen, 1, heads, device=DEVICE) * 0.01).requires_grad_()

    local_positions = torch.arange(segment_len, device=DEVICE, dtype=torch.int32).repeat(2)
    cu_seqlen_ks = torch.cat(
        [
            torch.zeros(segment_len, device=DEVICE, dtype=torch.int32),
            torch.full((segment_len,), kv_per_segment, device=DEVICE, dtype=torch.int32),
        ]
    )
    cu_seqlen_ke = cu_seqlen_ks + (local_positions + 1) // compress_ratio

    actual, actual_indices = v4_lighting_indexer(
        q,
        k,
        weights,
        compress_ratio,
        topk,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
    )
    logits = torch.einsum("sbhd,tbd->bsht", q.float(), k.float())
    logits = (logits.relu() * weights.permute(1, 0, 2).float().unsqueeze(-1)).sum(dim=2)
    entries = torch.arange(kv_len, device=DEVICE)
    valid_ranges = (entries >= cu_seqlen_ks[:, None]) & (entries < cu_seqlen_ke[:, None])
    masked_logits = logits.masked_fill(~valid_ranges.unsqueeze(0), float("-inf"))
    expected, expected_indices = masked_logits.topk(topk, dim=-1)
    expected_indices = expected_indices.to(torch.int32).masked_fill(expected == -torch.inf, -1)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_indices, expected_indices)

    valid = actual_indices >= 0
    grad = torch.randn_like(actual).masked_fill(~valid, 0)
    expected_grads = torch.autograd.grad((expected.masked_fill(~valid, 0) * grad).sum(), (q, k, weights))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, weights.grad), expected_grads, strict=True):
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_indexer_rejects_unsupported_head_count():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import v4_lighting_indexer

    q = torch.empty(8, 1, 7, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.empty(2, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.empty(8, 1, 7, device=DEVICE)
    with pytest.raises(ValueError, match="divisible by 8"):
        v4_lighting_indexer(q, k, weights, compress_ratio=4, topk=2)


def _sparse_attention_reference(q, kv, sinks, indices, scale):
    valid = (indices >= 0) & (indices < kv.shape[1])
    safe = indices.clamp(0, kv.shape[1] - 1).long()
    gathered = kv[:, None, :, :].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    max_score = scores.max(dim=-1, keepdim=True).values.clamp_min(-1e30)
    exp_scores = (scores - max_score).exp()
    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, gathered.float())
    denominator = exp_scores.sum(-1) + (sinks.view(1, 1, -1) - max_score.squeeze(-1)).exp()
    return numerator / denominator.unsqueeze(-1)


def test_tilelang_sparse_attention_forward_backward_with_invalid_indices():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang

    torch.manual_seed(1)
    batch, seqlen, heads, dim, kv_len, topk = 1, 32, 8, 512, 48, 65
    q = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(heads, device=DEVICE, requires_grad=True)
    indices = torch.randint(kv_len, (batch, seqlen, topk), device=DEVICE, dtype=torch.int32)
    indices[..., -9:-6] = -1
    indices[..., -6:-3] = -2
    indices[..., -3:] = kv_len
    scale = dim**-0.5

    actual = sparse_attn_tilelang(q, kv, sinks, indices, scale)
    expected = _sparse_attention_reference(q, kv, sinks, indices, scale)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn(batch, heads, seqlen, dim, device=DEVICE, dtype=actual.dtype).transpose(1, 2)
    assert not grad.is_contiguous()
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_deepseek_v4_generated_attention_dispatch_matches_eager():
    _require_tilelang_cuda()
    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    torch.manual_seed(3)
    batch, seqlen, heads, dim = 1, 32, 8, 512
    query = torch.randn(batch, heads, seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    key = torch.randn(batch, 1, seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    causal_mask = torch.full((batch, 1, seqlen, seqlen), float("-inf"), device=DEVICE)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    module = SimpleNamespace(
        num_key_value_groups=heads,
        sinks=torch.randn(heads, device=DEVICE),
        training=False,
        sliding_window=seqlen,
        compressor=None,
    )

    try:
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="eager"))
        expected, _ = modeling.eager_attention_forward(module, query, key, key, causal_mask, dim**-0.5, dropout=0.0)
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="tilelang"))
        actual, weights = modeling.eager_attention_forward(
            module, query, key, key, causal_mask, dim**-0.5, dropout=0.0
        )

        assert weights is None
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

        fp32_output, fp32_weights = modeling.eager_attention_forward(
            module, query.float(), key.float(), key.float(), causal_mask, dim**-0.5, dropout=0.0
        )
        assert fp32_weights is not None
        assert fp32_output.dtype == torch.float32
    finally:
        modeling.veomni_dsa_attention_implementation.bind(SimpleNamespace(dsa_attention_implementation="eager"))


def test_deepseek_v4_generated_indexer_dispatch_and_position_fallback(monkeypatch):
    _require_tilelang_cuda()
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    indexer = modeling.DeepseekV4Indexer(config).to(device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        torch.nn.init.zeros_(indexer.position_bias)
    seq_len = 8
    hidden_states = torch.randn(1, seq_len, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    q_residual = torch.randn(1, seq_len, config.q_lora_rank, device=DEVICE, dtype=torch.bfloat16)
    canonical_positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    calls = []

    def fake_tilelang(q, k, weights, compress_ratio, topk, **kwargs):
        calls.append((q.shape, k.shape, weights.shape, compress_ratio, topk, kwargs))
        indices = torch.zeros(1, q.shape[0], topk, device=DEVICE, dtype=torch.int32)
        return torch.zeros_like(indices, dtype=torch.float32), indices

    monkeypatch.setattr(modeling, "v4_lighting_indexer", fake_tilelang)
    try:
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
        tilelang_indices = indexer(hidden_states, q_residual, canonical_positions, None, 0)
        assert calls
        expected_topk = seq_len // config.compress_rates["compressed_sparse_attention"]
        assert tilelang_indices.shape == (1, seq_len, expected_topk)

        packed_positions = torch.arange(seq_len, device=DEVICE).repeat(2).unsqueeze(0)
        packed_hidden = hidden_states.repeat(1, 2, 1)
        packed_q_residual = q_residual.repeat(1, 2, 1)
        packed_slices = ((0, seq_len), (seq_len, 2 * seq_len))
        packed_metadata = build_packed_compression_metadata(
            packed_hidden,
            packed_positions,
            packed_slices,
            tuple(config.compress_rates.values()),
        )
        packed_indices = indexer(
            packed_hidden,
            packed_q_residual,
            packed_positions,
            None,
            0,
            packed_sequence_slices=packed_slices,
            packed_compression_metadata=packed_metadata,
        )
        assert len(calls) == 2
        packed_kwargs = calls[-1][-1]
        torch.testing.assert_close(
            packed_kwargs["cu_seqlen_ks"],
            torch.tensor([0] * seq_len + [expected_topk] * seq_len, device=DEVICE, dtype=torch.int32),
        )
        assert packed_kwargs["cu_seqlen_ke"][-1] == 2 * expected_topk
        assert packed_indices.shape == (1, 2 * seq_len, 2 * expected_topk)

        eager_indices = indexer(hidden_states, q_residual, canonical_positions + 1, None, 0)
        assert len(calls) == 2
        assert eager_indices.shape == tilelang_indices.shape
    finally:
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="eager"))


def test_deepseek_v4_stateless_forward_does_not_create_decode_cache():
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
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


def test_deepseek_v4_stateless_model_reaches_tilelang_indexer(monkeypatch):
    _require_tilelang_cuda()
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"
    model = modeling.DeepseekV4Model(config).to(device=DEVICE, dtype=torch.bfloat16).eval()
    calls = []

    def fake_tilelang(q, k, weights, compress_ratio, topk, *args, **kwargs):
        calls.append((q.shape, k.shape, weights.shape, compress_ratio, topk))
        indices = torch.zeros(q.shape[1], q.shape[0], topk, device=DEVICE, dtype=torch.int32)
        return torch.zeros_like(indices, dtype=torch.float32), indices

    monkeypatch.setattr(modeling, "v4_lighting_indexer", fake_tilelang)
    try:
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="tilelang"))
        with torch.no_grad():
            model(input_ids=torch.arange(8, device=DEVICE).unsqueeze(0), use_cache=False)
        assert calls
    finally:
        modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="eager"))


def test_deepseek_v4_packed_compressors_match_independent_sequences():
    from transformers import AutoConfig

    from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
    from veomni.models.transformers.deepseek_v4.packed_utils import build_packed_compression_metadata

    torch.manual_seed(5)
    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
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

    modeling.veomni_dsa_indexer_implementation.bind(SimpleNamespace(dsa_indexer_implementation="eager"))
    for compressor_cls in (modeling.DeepseekV4HCACompressor, modeling.DeepseekV4CSACompressor):
        compressor = compressor_cls(config)
        # Compressors allocate ``position_bias`` with ``torch.empty``; the full
        # model zeros it in ``_init_weights``. Standalone construction must do
        # the same or softmax-gated compression sees allocator garbage / NaNs.
        with torch.no_grad():
            torch.nn.init.zeros_(compressor.position_bias)
            indexer = getattr(compressor, "indexer", None)
            if indexer is not None:
                torch.nn.init.zeros_(indexer.position_bias)
        packed_kv, packed_bias = compressor(
            hidden_states,
            q_residual,
            position_ids,
            None,
            0,
            packed_sequence_slices=sequence_slices,
            packed_compression_metadata=packed_metadata,
        )

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

        expected_kv = torch.cat(segment_outputs, dim=2)
        torch.testing.assert_close(packed_kv, expected_kv)
        kv_offset = 0
        for (start, end), segment_kv, segment_bias in zip(
            sequence_slices, segment_outputs, segment_biases, strict=True
        ):
            kv_end = kv_offset + segment_kv.shape[2]
            torch.testing.assert_close(packed_bias[:, :, start:end, kv_offset:kv_end], segment_bias)
            assert torch.isneginf(packed_bias[:, :, start:end, :kv_offset]).all()
            assert torch.isneginf(packed_bias[:, :, start:end, kv_end:]).all()
            kv_offset = kv_end


def test_deepseek_v4_packed_causal_mask_blocks_previous_samples():
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_sliding_window_causal_mask

    from veomni.models.transformers.deepseek_v4.packed_utils import isolate_packed_causal_mask_

    config = AutoConfig.from_pretrained("tests/toy_config/deepseek_v4_toy")
    config._attn_implementation = "eager"
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


def test_tilelang_act_quant_shapes_scales_and_inplace():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4 import act_quant

    torch.manual_seed(2)
    x = torch.randn(3, 256, device=DEVICE, dtype=torch.bfloat16)
    quantized, scales = act_quant(x, block_size=128)

    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 2)
    expected_scales = x.float().view(3, 2, 128).abs().amax(-1).clamp_min(1e-4) / 448.0
    torch.testing.assert_close(scales, expected_scales, rtol=1e-5, atol=1e-7)
    expanded_scales = expected_scales.repeat_interleave(128, dim=-1)
    expected_quantized = (x.float() / expanded_scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(quantized.float(), expected_quantized.float(), rtol=0, atol=0)
    expected_dequantized = (expected_quantized.float() * expanded_scales).to(torch.bfloat16)
    actual_dequantized = (quantized.float() * scales.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
    torch.testing.assert_close(actual_dequantized, expected_dequantized, rtol=0, atol=0)

    inplace = x.clone()
    result = act_quant(inplace, block_size=128, inplace=True)
    assert result.data_ptr() == inplace.data_ptr()
    torch.testing.assert_close(result, expected_dequantized, rtol=0, atol=0)

    x_mx = x.clone()
    x_mx[0].zero_()
    quantized_mx, scales_mx = act_quant(
        x_mx,
        block_size=128,
        scale_fmt="ue8m0",
        scale_dtype=torch.float8_e8m0fnu,
    )
    amax_mx = x_mx.float().view(3, 2, 128).abs().amax(-1).clamp_min(1e-4)
    expected_scales_mx = torch.pow(2.0, torch.ceil(torch.log2(amax_mx / 448.0)))
    torch.testing.assert_close(scales_mx.float(), expected_scales_mx, rtol=0, atol=0)
    expanded_scales_mx = expected_scales_mx.repeat_interleave(128, dim=-1)
    expected_quantized_mx = (x_mx.float() / expanded_scales_mx).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(quantized_mx.float(), expected_quantized_mx.float(), rtol=0, atol=0)
