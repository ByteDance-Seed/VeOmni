# Copyright 2026 ByteDance Ltd. and/or its affiliates
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

import copy
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


DEVICE = get_device_type()


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("Qwen4-Exp TileLang kernels require an NVIDIA CUDA GPU")


def _require_cuda():
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("Qwen4-Exp generated attention requires an NVIDIA CUDA GPU")


def test_kernel_package_does_not_import_tilelang_eagerly():
    sys.modules.pop("veomni.ops.kernels.qwen4_exp", None)
    before = "tilelang" in sys.modules

    importlib.import_module("veomni.ops.kernels.qwen4_exp")

    assert ("tilelang" in sys.modules) is before


def test_qwen4_exp_generated_eager_preserves_non_qsa_fallback():
    _require_cuda()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import eager_attention_forward

    from veomni.models.transformers.qwen4_exp.generated import patched_modeling_qwen4_exp_gpu as modeling

    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(2, 4, 3, 5, device=DEVICE)
    key = torch.randn(2, 2, 3, 5, device=DEVICE)
    value = torch.randn(2, 2, 3, 5, device=DEVICE)
    attention_mask = torch.triu(torch.full((2, 1, 3, 3), float("-inf"), device=DEVICE), diagonal=1)

    try:
        # The TileLang selection applies only to QSA calls. Vision attention
        # reaches this module-level eager fallback without selected_indices.
        _bind_qsa_implementation(modeling, "tilelang")
        expected = eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)
        actual = modeling.eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)
        torch.testing.assert_close(actual, expected)
    finally:
        _bind_qsa_implementation(modeling, "eager")


def _qsa_reference(q, k, v, indices, scale):
    from veomni.models.transformers.qwen4_exp.generated import patched_modeling_qwen4_exp_gpu as modeling

    _bind_qsa_implementation(modeling, "eager")
    output, _ = modeling.eager_attention_forward(
        SimpleNamespace(training=False),
        q.float(),
        k.float(),
        v.float(),
        attention_mask=None,
        scaling=scale,
        dropout=0.0,
        selected_indices=indices,
    )
    return output


def _cosine_similarity(actual, expected):
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)


def _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, kv_heads, seq_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, kv_heads, seq_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.full((batch, seq_len, topk), -1, device=DEVICE, dtype=torch.int32)
    valid_topk = min(seq_len, max(0, topk - 3))
    if valid_topk:
        # QSA selects distinct blocks, whose token ranges do not overlap. Keep
        # the synthetic inputs faithful to that contract: valid indices within
        # each query row are unique, while unused slots remain -1 padding.
        indices[..., :valid_topk] = torch.rand(batch, seq_len, seq_len, device=DEVICE).argsort(dim=-1)[
            ..., :valid_topk
        ]
    return q, k, v, indices


@pytest.mark.parametrize(
    "batch,seq_len,heads,kv_heads,dim,topk",
    [
        (2, 64, 8, 2, 64, 48),  # production-like GQA grouping
        (2, 64, 2, 1, 16, 64),  # toy head dim: zero-pad D up to the 64-wide MMA floor
        (1, 32, 4, 1, 256, 65),  # MQA + non-multiple-of-64 topk padding path
        (1, 33, 6, 6, 128, 32),  # no GQA expansion
        (2, 64, 8, 2, 256, 64),  # production head dim
    ],
)
def test_tilelang_qsa_forward_backward_matches_reference(batch, seq_len, heads, kv_heads, dim, topk):
    _require_tilelang_cuda()
    q, k, v, indices = _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk)
    scale = dim**-0.5
    from veomni.ops.kernels.qwen4_exp import qsa_attn_tilelang

    actual = qsa_attn_tilelang(q, k, v, indices, scale)
    expected = _qsa_reference(q, k, v, indices, scale)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, k, v))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), expected_grads, strict=True):
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        assert actual_grad.shape == expected_grad.shape
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_qsa_default_scale_uses_unpadded_head_dim():
    _require_tilelang_cuda()
    from veomni.ops.kernels.qwen4_exp import qsa_attn_tilelang

    q, k, v, indices = _make_qsa_tensors(1, 64, 2, 1, 16, 32)
    actual = qsa_attn_tilelang(q, k, v, indices)
    expected = _qsa_reference(q, k, v, indices, 16**-0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, k, v))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), expected_grads, strict=True):
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_qsa_causal_packed_indices_match_reference():
    """Indices that respect causality and packed-sample boundaries, as the QSA indexer emits."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.qwen4_exp import qsa_attn_tilelang

    batch, seq_len, heads, kv_heads, dim, topk = 1, 96, 8, 2, 128, 40
    q, k, v, _ = _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk)
    # Two packed segments [0, 40) and [40, 96); selections never cross the
    # boundary and never look ahead.
    boundaries = [(0, 40), (40, 96)]
    indices = torch.full((batch, seq_len, topk), -1, device=DEVICE, dtype=torch.int32)
    for start, end in boundaries:
        for s in range(start, end):
            width = min(s - start + 1, topk)
            pool = torch.arange(s - width + 1, s + 1, device=DEVICE, dtype=torch.int32)
            indices[:, s, :width] = pool
    scale = dim**-0.5

    actual = qsa_attn_tilelang(q, k, v, indices, scale)
    expected = _qsa_reference(q, k, v, indices, scale)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, k, v))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), expected_grads, strict=True):
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_qsa_empty_selection_row_is_zero():
    _require_tilelang_cuda()
    from veomni.ops.kernels.qwen4_exp import qsa_attn_tilelang

    batch, seq_len, heads, kv_heads, dim, topk = 1, 16, 8, 2, 64, 32
    q, k, v, indices = _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk)
    indices[:, 5, :] = -1

    actual = qsa_attn_tilelang(q, k, v, indices, dim**-0.5)
    expected = _qsa_reference(q, k, v, indices, dim**-0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    assert torch.all(actual[:, 5] == 0)

    actual.backward(torch.ones_like(actual))
    assert torch.all(q.grad[:, :, 5] == 0)
    assert torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


def test_tilelang_qsa_rejects_non_bf16_and_bad_shapes():
    _require_tilelang_cuda()
    from veomni.ops.kernels.qwen4_exp import qsa_attn_tilelang

    q, k, v, indices = _make_qsa_tensors(1, 16, 8, 2, 64, 32)
    with pytest.raises(ValueError, match="bfloat16"):
        qsa_attn_tilelang(q.float(), k, v, indices, 0.125)
    with pytest.raises(ValueError, match="power-of-two head dim"):
        qsa_attn_tilelang(
            q[:, :, :, :48].contiguous(),
            k[:, :, :, :48].contiguous(),
            v[:, :, :, :48].contiguous(),
            indices,
            0.125,
        )
    with pytest.raises(ValueError, match="divisible"):
        qsa_attn_tilelang(
            q,
            k[:, :1].repeat(1, 3, 1, 1).contiguous(),
            v[:, :1].repeat(1, 3, 1, 1).contiguous(),
            indices,
            0.125,
        )
    with pytest.raises(ValueError, match="key/value shapes must match"):
        qsa_attn_tilelang(q, k, v[:, :, :-1], indices, 0.125)
    with pytest.raises(ValueError, match="query batch and sequence"):
        qsa_attn_tilelang(q, k, v, indices[:, :-1], 0.125)
    with pytest.raises(TypeError, match="int32 or int64"):
        qsa_attn_tilelang(q, k, v, indices.float(), 0.125)
    invalid_indices = indices.clone()
    invalid_indices[0, 0, 0] = k.shape[2]
    with pytest.raises(ValueError, match="valid global KV token indices"):
        qsa_attn_tilelang(q, k, v, invalid_indices, 0.125)


def _bind_qsa_implementation(modeling, implementation):
    modeling.veomni_qsa_attention_implementation.bind(SimpleNamespace(qsa_attention_implementation=implementation))


def test_qwen4_exp_attention_layer_tilelang_matches_fp32_reference():
    """Full attention-layer integration: indexer -> compact selection -> TileLang kernel.

    Exercises the generated Qwen4ExpTextAttention forward/backward end to end
    (compact selection with packed cu_seq_lens included) and compares against
    the fp32 eager reference, not just the same-precision one.
    """
    _require_tilelang_cuda()
    from transformers import AutoConfig

    from veomni.models.transformers.qwen4_exp.generated import patched_modeling_qwen4_exp_gpu as modeling

    config = AutoConfig.from_pretrained("tests/toy_config/qwen4_exp_toy").text_config
    config.num_attention_heads = 8
    config.num_key_value_heads = 2
    # The toy head_dim (16) is below the TileLang MMA tiling floor; use the
    # smallest tileable dim, keeping the indexer head dim at or above the
    # rotary width as Qwen4-Exp config validation requires.
    config.head_dim = 64
    config.indexer_head_dim = 32
    torch.manual_seed(31)
    reference = modeling.Qwen4ExpTextAttention(config, layer_idx=1).float().to(DEVICE)
    attention = copy.deepcopy(reference).to(dtype=torch.bfloat16)

    seq_len = 64
    packed_boundary = 17
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    position_embeddings = (
        torch.ones(1, seq_len, rotary_dim, device=DEVICE),
        torch.zeros(1, seq_len, rotary_dim, device=DEVICE),
    )
    cu_seq_lens_q = torch.tensor([0, packed_boundary, seq_len], dtype=torch.int32, device=DEVICE)
    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, config.hidden_size, device=DEVICE)

    try:
        _bind_qsa_implementation(modeling, "eager")
        reference_input = hidden.clone().requires_grad_(True)
        expected, _ = reference(
            reference_input,
            position_embeddings=position_embeddings,
            attention_mask=None,
            cu_seq_lens_q=cu_seq_lens_q,
        )
        (expected.sum() / expected.numel()).backward()

        _bind_qsa_implementation(modeling, "tilelang")
        actual_input = hidden.to(torch.bfloat16).requires_grad_(True)
        actual, _ = attention(
            actual_input,
            position_embeddings=tuple(t.to(torch.bfloat16) for t in position_embeddings),
            attention_mask=None,
            cu_seq_lens_q=cu_seq_lens_q,
        )
        (actual.sum() / actual.numel()).backward()

        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
        assert _cosine_similarity(actual_input.grad, reference_input.grad) > 0.95
    finally:
        _bind_qsa_implementation(modeling, "eager")
