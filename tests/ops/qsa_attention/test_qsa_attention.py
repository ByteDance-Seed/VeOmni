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

"""Qwen4-Exp QSA rows: eager reference parity, TileLang and torch_npu kernels, and dispatch guards."""

import copy
import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops import resolve_op
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type


DEVICE = get_device_type()
_VENDOR_PACKAGE = "veomni.ops.kernels.qsa_attention.vendor"


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("Qwen4-Exp TileLang kernels require an NVIDIA CUDA GPU")


def _tilelang_qsa():
    _require_tilelang_cuda()
    return resolve_op("qsa_attention", "standard", "tilelang").wrapper


def _eager_qsa():
    return resolve_op("qsa_attention", "standard", "eager").wrapper


@contextmanager
def _selected_qsa_implementation(implementation):
    """Install one QSA selection for directly constructed test layers."""
    from veomni.ops.config import get_ops_config, set_ops_config

    previous = get_ops_config()
    set_ops_config(SimpleNamespace(qsa_attention_implementation=implementation))
    try:
        yield
    finally:
        set_ops_config(previous)


def test_vendor_package_does_not_import_tilelang_eagerly():
    sys.modules.pop(_VENDOR_PACKAGE, None)
    before = "tilelang" in sys.modules

    importlib.import_module(_VENDOR_PACKAGE)

    assert ("tilelang" in sys.modules) is before


def test_eager_full_causal_selection_matches_hf_eager_attention():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import eager_attention_forward

    torch.manual_seed(0)
    batch, heads, kv_heads, seq_len, dim = 2, 4, 2, 7, 8
    query = torch.randn(batch, heads, seq_len, dim)
    key = torch.randn(batch, kv_heads, seq_len, dim)
    value = torch.randn(batch, kv_heads, seq_len, dim)
    positions = torch.arange(seq_len)
    indices = torch.where(positions[None, :] <= positions[:, None], positions[None, :], -1)
    indices = indices.expand(batch, seq_len, seq_len).to(torch.int32)
    causal_mask = torch.triu(torch.full((batch, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, training=False)

    expected, _ = eager_attention_forward(module, query, key, value, causal_mask, scaling=dim**-0.5)
    actual, _ = _eager_qsa()(query, key, value, indices, scaling=dim**-0.5)

    torch.testing.assert_close(actual, expected)


def test_eager_empty_selection_row_is_zero():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    indices = torch.tensor([[[0, -1], [-1, -1], [1, 2], [3, 0]]], dtype=torch.int64)

    actual, _ = _eager_qsa()(query, key, value, indices, scaling=8**-0.5)

    assert torch.all(actual[:, 1] == 0)
    assert torch.isfinite(actual).all()


def _qsa_reference(q, k, v, indices, scale):
    output, _ = _eager_qsa()(q.float(), k.float(), v.float(), indices, scaling=scale)
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
        (1, 513, 24, 2, 256, 129),  # 12 heads/group and multiple dKV reduction splits
    ],
)
def test_tilelang_qsa_forward_backward_matches_reference(batch, seq_len, heads, kv_heads, dim, topk):
    qsa = _tilelang_qsa()
    q, k, v, indices = _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk)
    scale = dim**-0.5

    actual, weights = qsa(q, k, v, indices, scaling=scale)
    assert weights is None
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
    from veomni.ops.kernels.qsa_attention.vendor.tilelang_qsa import qsa_attn_tilelang

    q, k, v, indices = _make_qsa_tensors(1, 64, 2, 1, 16, 32)
    actual = qsa_attn_tilelang(q, k, v, indices)
    expected = _qsa_reference(q, k, v, indices, 16**-0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, k, v))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), expected_grads, strict=True):
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


@pytest.mark.parametrize("topk", [0, 65, 333])
def test_qsa_reverse_tiles_preserve_exact_selections(topk):
    _require_tilelang_cuda()
    from veomni.ops.kernels.qsa_attention.vendor.tilelang_qsa_dkv import build_reverse_tiles

    batch, seq, kv_len = 2, 37, 301 if topk == 333 else 61
    torch.manual_seed(19)
    indices = torch.full((batch, seq, topk), -1, dtype=torch.int32, device=DEVICE)
    if topk:
        valid_count = kv_len - 7 if topk == 333 else 31
        indices[..., :valid_count] = torch.rand(batch, seq, kv_len, device=DEVICE).argsort(-1)[..., :valid_count]
        indices[:, 0] = -1
        # A single selected token in the final, incomplete KV storage tile.
        indices[:, 1] = -1
        indices[:, 1, -1] = kv_len - 1
    original = indices.clone()
    offsets, queries, masks = (t.cpu() for t in build_reverse_tiles(indices, kv_len))
    actual = torch.zeros(batch, seq, kv_len, dtype=torch.bool)
    num_blocks = (kv_len + 15) // 16
    for b in range(batch):
        for block in range(num_blocks):
            first, last = offsets[b * num_blocks + block : b * num_blocks + block + 2].tolist()
            rows = queries[first:last].tolist()
            assert len(rows) == len(set(rows))
            for row, mask in zip(rows, masks[first:last].tolist(), strict=True):
                assert 0 <= row < seq and mask != 0
                for offset in range(16):
                    if mask & (1 << offset):
                        token = block * 16 + offset
                        assert token < kv_len
                        actual[b, row, token] = True
    reference_indices = indices.cpu().long()
    expected = torch.zeros(batch, seq, kv_len + 1, dtype=torch.bool).scatter_(
        -1, reference_indices.masked_fill(reference_indices < 0, kv_len), True
    )[..., :kv_len]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(indices, original)


def test_tilelang_qsa_unequal_lengths_and_unused_kv_gradients():
    qsa = _tilelang_qsa()

    torch.manual_seed(23)
    q = torch.randn(2, 24, 33, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 79, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.full((2, 33, 17), -1, device=DEVICE, dtype=torch.int64)
    # Cross storage-tile boundaries, leaving complete KV tiles unused.
    indices[0, 1:, :5] = torch.tensor([78, 16, 0, 15, 77], device=DEVICE)
    actual, _ = qsa(q, k, v, indices, scaling=64**-0.5)
    expected = _qsa_reference(q, k, v, indices, 64**-0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(actual)
    borrowed_grad = grad.clone()
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad.float())
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        # Many queries accumulate onto just five tokens. Local cancellation
        # makes a per-element relative tolerance misleading for BF16 sums.
        assert torch.isfinite(actual_grad).all()
        relative_error = (actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
        assert relative_error < 1e-2
        assert torch.count_nonzero(actual_grad[1]) == 0
    assert torch.count_nonzero(actual_grads[1][:, :, 17:77]) == 0
    assert torch.count_nonzero(actual_grads[2][:, :, 17:77]) == 0
    torch.testing.assert_close(grad, borrowed_grad)


def test_tilelang_qsa_causal_packed_indices_match_reference():
    """Indices that respect causality and packed-sample boundaries, as the QSA indexer emits."""
    qsa = _tilelang_qsa()

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

    actual, _ = qsa(q, k, v, indices, scaling=scale)
    expected = _qsa_reference(q, k, v, indices, scale)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, k, v))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), expected_grads, strict=True):
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95


def test_tilelang_qsa_empty_selection_row_is_zero():
    qsa = _tilelang_qsa()

    batch, seq_len, heads, kv_heads, dim, topk = 1, 16, 8, 2, 64, 32
    q, k, v, indices = _make_qsa_tensors(batch, seq_len, heads, kv_heads, dim, topk)
    indices[:, 5, :] = -1

    actual, _ = qsa(q, k, v, indices, scaling=dim**-0.5)
    expected = _qsa_reference(q, k, v, indices, dim**-0.5)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    assert torch.all(actual[:, 5] == 0)

    actual.backward(torch.ones_like(actual))
    assert torch.all(q.grad[:, :, 5] == 0)
    assert torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()


def test_tilelang_qsa_rejects_non_bf16_and_bad_shapes():
    qsa = _tilelang_qsa()

    q, k, v, indices = _make_qsa_tensors(1, 16, 8, 2, 64, 32)
    with pytest.raises(ValueError, match="bfloat16"):
        qsa(q.float(), k, v, indices, scaling=0.125)
    with pytest.raises(ValueError, match="power-of-two head dim"):
        qsa(
            q[:, :, :, :48].contiguous(),
            k[:, :, :, :48].contiguous(),
            v[:, :, :, :48].contiguous(),
            indices,
            scaling=0.125,
        )
    with pytest.raises(ValueError, match="divisible"):
        qsa(
            q,
            k[:, :1].repeat(1, 3, 1, 1).contiguous(),
            v[:, :1].repeat(1, 3, 1, 1).contiguous(),
            indices,
            scaling=0.125,
        )
    with pytest.raises(ValueError, match="key/value shapes must match"):
        qsa(q, k, v[:, :, :-1], indices, scaling=0.125)
    with pytest.raises(ValueError, match="query batch and sequence"):
        qsa(q, k, v, indices[:, :-1], scaling=0.125)
    with pytest.raises(TypeError, match="int32 or int64"):
        qsa(q, k, v, indices.float(), scaling=0.125)
    invalid_indices = indices.clone()
    invalid_indices[0, 0, 0] = k.shape[2]
    with pytest.raises(ValueError, match="valid global KV token indices"):
        qsa(q, k, v, invalid_indices, scaling=0.125)


def _causal_indices(seq_len):
    positions = torch.arange(seq_len, dtype=torch.int32)
    return torch.where(positions[None, :] <= positions[:, None], positions[None, :], -1)[None]


def test_tilelang_row_rejects_an_attention_mask():
    from veomni.ops.kernels.qsa_attention.standard import tilelang as tilelang_row

    q, k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="requires no attention_mask"):
        tilelang_row.wrapper(q, k, v, _causal_indices(4), scaling=0.25, attention_mask=torch.zeros(1, 1, 4, 4))


def test_tilelang_row_dropout_falls_back_to_eager_with_warning(monkeypatch):
    from veomni.ops.kernels.qsa_attention.standard import tilelang as tilelang_row

    warnings = []
    monkeypatch.setattr(tilelang_row.logger, "warning_once", lambda message, *args, **kwargs: warnings.append(message))
    torch.manual_seed(0)
    # fp32 CPU inputs would be rejected by the TileLang kernel, so a matching
    # result also proves that the dropout branch never reached it.
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    indices = _causal_indices(8)

    torch.manual_seed(1)
    actual, _ = tilelang_row.wrapper(q, k, v, indices, scaling=0.25, dropout=0.1)
    torch.manual_seed(1)
    expected, _ = _eager_qsa()(q, k, v, indices, scaling=0.25, dropout=0.1)

    torch.testing.assert_close(actual, expected)
    assert len(warnings) == 1 and "no dropout path" in warnings[0]


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
    with _selected_qsa_implementation("eager"):
        reference = modeling.Qwen4ExpTextAttention(config, layer_idx=1).float().to(DEVICE)
    with _selected_qsa_implementation("tilelang"):
        attention = modeling.Qwen4ExpTextAttention(config, layer_idx=1).to(DEVICE, dtype=torch.bfloat16)
    attention.load_state_dict(copy.deepcopy(reference.state_dict()))
    assert reference.veomni_qsa_attn.impl == "eager"
    assert attention.veomni_qsa_attn.impl == "tilelang"

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

    reference_input = hidden.clone().requires_grad_(True)
    expected, _ = reference(
        reference_input,
        position_embeddings=position_embeddings,
        attention_mask=None,
        cu_seq_lens_q=cu_seq_lens_q,
    )
    (expected.sum() / expected.numel()).backward()

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


def _npu_reference_forward(q, k, v, blocked, scale):
    repeats = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    maximum = scores.masked_fill(blocked, -torch.inf).amax(-1, keepdim=True)
    maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
    weights = torch.exp(scores - maximum).masked_fill(blocked, 0)
    denominator = weights.sum(-1, keepdim=True).clamp_min(1e-30)
    return (weights / denominator) @ v, maximum, denominator


def _npu_reference_backward(q, k, v, grad, blocked, output, maximum, denominator, scale):
    repeats = q.shape[1] // k.shape[1]
    kr, vr = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    probabilities = torch.exp((q @ kr.transpose(-1, -2)) * scale - maximum).masked_fill(blocked, 0) / denominator
    ds = probabilities * ((grad @ vr.transpose(-1, -2)) - (grad * output).sum(-1, keepdim=True))
    dq = (ds @ kr) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = probabilities.transpose(-1, -2) @ grad
    shape = (*k.shape[:2], repeats, *k.shape[2:])
    return dq, dk.reshape(shape).sum(2), dv.reshape(shape).sum(2)


def test_torch_npu_pruned_gqa_output_gradients_and_saved_tensors(monkeypatch):
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_qsa

    torch.manual_seed(71)
    calls = []

    def forward(*args):
        calls.append(args[1].shape[2])
        return _npu_reference_forward(*args)

    monkeypatch.setattr(npu_qsa, "_forward", forward)
    monkeypatch.setattr(npu_qsa, "_backward", _npu_reference_backward)
    q = torch.randn(2, 6, 257, 8, requires_grad=True)
    k = torch.randn(2, 2, 769, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.randint(0, 127, (2, 257, 7))
    indices[:, 128:256] += 512
    indices[:, 0] = -1
    indices[:, -1] = -1
    indices[:, 1, 1] = indices[:, 1, 0]
    indices[:, 2, 1:] = -1
    grad = torch.randn_like(q)
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        result = npu_qsa._CompactFusedAttention.apply(q, k, v, indices, 0.31, 128)
    candidate = torch.autograd.grad(result, (q, k, v), grad)
    assert calls == [128, 128, 1]  # Backward does not replay forward or gather KV.
    assert not any(dtype == torch.bool for dtype, _ in saved)
    allowed = ~npu_qsa._blocked_mask(indices, k.shape[2])
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True, scale=0.31)
    expected = torch.autograd.grad(reference, (q, k, v), grad)
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
    for actual, ref in zip(candidate, expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=3e-5, atol=4e-6)
    assert torch.count_nonzero(result[:, :, 0]) == 0
    assert torch.count_nonzero(result[:, :, -1]) == 0


def test_torch_npu_ranges_cover_every_selected_index_with_partial_last_block():
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_qsa

    indices = torch.tensor([[[300, -1, 300], [511, 420, -1], [768, -1, -1]]])
    ranges = npu_qsa._kv_ranges(indices, 769, 2)
    assert ranges == [(256, 512), (768, 769)]
    for block, (lo, hi) in enumerate(ranges):
        local = indices[:, block * 2 : (block + 1) * 2]
        valid = local[local >= 0]
        assert torch.all((valid >= lo) & (valid < hi))


def test_torch_npu_masked_rows_are_constant_zero_even_with_backend_residue(monkeypatch):
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_qsa

    torch.manual_seed(83)

    def forward(*args):
        output, maximum, denominator = _npu_reference_forward(*args)
        # A fused backend may leave subnormal values on fully masked rows.
        output = output.masked_fill(args[3].all(-1, keepdim=True), 1e-38)
        return output, maximum, denominator

    def backward(*args):
        dq, dk, dv = _npu_reference_backward(*args)
        dq = dq.masked_fill(args[4].all(-1, keepdim=True), 1e-38)
        empty = args[4].all((-1, -2), keepdim=True)
        return dq, dk.masked_fill(empty, 1e-38), dv.masked_fill(empty, 1e-38)

    monkeypatch.setattr(npu_qsa, "_forward", forward)
    monkeypatch.setattr(npu_qsa, "_backward", backward)
    q = torch.randn(2, 4, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 128, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.tensor([[[0, 2], [-1, -1], [-1, -1], [-1, -1]]] * 2)
    output = npu_qsa._CompactFusedAttention.apply(q, k, v, indices, 0.31, 2)
    # Only masked rows contribute to this objective. Every gradient is zero.
    grad = torch.ones_like(output)
    grad[:, :, 0] = 0
    gradients = torch.autograd.grad(output, (q, k, v), grad)
    assert torch.count_nonzero(output[:, :, 1:]) == 0
    for gradient in gradients:
        assert torch.count_nonzero(gradient) == 0


def test_torch_npu_int8_scatter_mask_matches_boolean_reference_exactly():
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_qsa

    # Non-contiguous indices, duplicates, padding and an entirely masked row.
    indices = torch.tensor([[[0, 0, -1, 6], [-1, -1, -1, -1], [5, 2, 5, -1]]])
    indices = indices.repeat(2, 1, 2)[..., ::2]
    reference = torch.ones((*indices.shape[:-1], 8), dtype=torch.bool)
    slots = torch.where(indices >= 0, indices, 7).long()
    reference.scatter_(-1, slots, False)
    expected = reference[..., :7].unsqueeze(1).contiguous()
    actual = npu_qsa._blocked_mask(indices, 7)
    assert actual.dtype == torch.bool
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_torch_npu_row_rejects_an_attention_mask():
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_row

    q, k, v = torch.randn(1, 2, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="requires no attention_mask"):
        npu_row.wrapper(q, k, v, _causal_indices(4), scaling=0.25, attention_mask=torch.zeros(1, 1, 4, 4))


def test_torch_npu_row_dropout_falls_back_to_eager_with_warning(monkeypatch):
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_row

    warnings = []
    monkeypatch.setattr(npu_row.logger, "warning_once", lambda message, *args, **kwargs: warnings.append(message))
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    indices = _causal_indices(8)

    torch.manual_seed(1)
    actual, _ = npu_row.wrapper(q, k, v, indices, scaling=0.25, dropout=0.1)
    torch.manual_seed(1)
    expected, _ = _eager_qsa()(q, k, v, indices, scaling=0.25, dropout=0.1)

    torch.testing.assert_close(actual, expected)
    assert len(warnings) == 1 and "no dropout path" in warnings[0]


def _require_npu():
    pytest.importorskip("torch_npu")
    if not IS_NPU_AVAILABLE:
        pytest.skip("requires an Ascend NPU")


def test_torch_npu_native_int8_mask_and_fused_gqa_backward():
    _require_npu()
    from veomni.ops.kernels.qsa_attention.standard import torch_npu as npu_qsa

    torch.manual_seed(71)
    # Two pruned KV intervals, duplicate indices, padding, one masked row,
    # and a final entirely masked query block exercise native CANN semantics.
    indices = torch.randint(0, 128, (1, 192, 16))
    indices[:, 64:128] += 128
    indices[:, 0] = -1
    indices[:, 128:] = -1
    indices[:, 1, 1] = indices[:, 1, 0]
    indices[:, 2, 1:] = -1
    blocked = torch.ones((1, 192, 257), dtype=torch.bool)
    blocked.scatter_(-1, torch.where(indices >= 0, indices, 256), False)
    blocked = blocked[..., :256].unsqueeze(1).contiguous()
    native_indices = indices.to("npu")
    torch.testing.assert_close(npu_qsa._blocked_mask(native_indices, 256).cpu(), blocked, rtol=0, atol=0)

    shape_q, shape_kv = (1, 4, 192, 64), (1, 2, 256, 64)
    inputs = [torch.randn(shape, dtype=torch.bfloat16) for shape in (shape_q, shape_kv, shape_kv)]
    reference_inputs = [x.float().requires_grad_() for x in inputs]
    native_inputs = [x.to("npu").requires_grad_() for x in inputs]
    grad = torch.randn(shape_q, dtype=torch.bfloat16)
    scale = 64**-0.5
    reference, _, _ = _npu_reference_forward(*reference_inputs, blocked, scale)
    expected_grads = torch.autograd.grad(reference, reference_inputs, grad.float())
    actual = npu_qsa._CompactFusedAttention.apply(*native_inputs, native_indices, scale, 64)
    actual_grads = torch.autograd.grad(actual, native_inputs, grad.to("npu"))
    torch.testing.assert_close(actual.float().cpu(), reference, rtol=0.03, atol=0.02)
    for candidate, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(candidate.float().cpu(), expected, rtol=0.03, atol=0.02)
    assert torch.count_nonzero(actual[:, :, 0]).item() == 0
    assert torch.count_nonzero(actual[:, :, 128:]).item() == 0
    assert torch.count_nonzero(actual_grads[0][:, :, 128:]).item() == 0
