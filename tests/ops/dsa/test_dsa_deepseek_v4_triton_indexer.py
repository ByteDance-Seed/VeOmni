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

"""DeepSeek-V4 Triton indexer (AMD ROCm) against the FP32 reference and the eager row."""

import importlib.util
import sys
from types import ModuleType

import pytest
import torch

from tests.ops.utils import assert_gradient_direction_and_scale
from veomni.ops import resolve_op
from veomni.ops.kernels.dsa.indexer.deepseek_v4 import triton as triton_indexer_adapter
from veomni.utils.device import IS_CUDA_AVAILABLE


_ROCM_AVAILABLE = (
    IS_CUDA_AVAILABLE and torch.version.hip is not None and importlib.util.find_spec("triton") is not None
)
requires_rocm = pytest.mark.skipif(not _ROCM_AVAILABLE, reason="the Triton indexer requires an AMD ROCm GPU")


def _reference_logits(q, k, weights):
    logits = torch.einsum("sbhd,tbd->bsht", q.float(), k.float())
    return (logits.relu() * weights.permute(1, 0, 2).float().unsqueeze(-1)).sum(dim=2)


def _gather(logits, indices):
    valid = (indices >= 0) & (indices < logits.shape[-1])
    scores = logits.gather(-1, indices.clamp(0, logits.shape[-1] - 1).long())
    return torch.where(valid, scores, float("-inf")), valid


def _assert_same_selection(logits, actual_indices, reference_values, reference_indices):
    """Both selections pick the same entries up to swaps among scores tied within FP32 noise."""
    assert torch.equal(actual_indices >= 0, reference_indices >= 0)
    achieved, _ = _gather(logits, actual_indices)
    achieved = achieved.sort(dim=-1, descending=True).values
    torch.testing.assert_close(achieved, reference_values, rtol=1e-5, atol=1e-6)


def _inputs(seqlen, batch, heads, dim, kv_len, seed):
    torch.manual_seed(seed)
    q = torch.randn(seqlen, batch, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(kv_len, batch, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = (torch.randn(seqlen, batch, heads, device="cuda") * 0.01).requires_grad_()
    return q, k, weights


def _assert_grads_match(actual, reference, grad, valid, leaves):
    reference_grads = torch.autograd.grad((reference.masked_fill(~valid, 0) * grad).sum(), leaves)
    actual.backward(grad)
    for leaf, expected in zip(leaves, reference_grads, strict=True):
        assert leaf.grad is not None and leaf.grad.dtype == leaf.dtype
        assert_gradient_direction_and_scale(leaf.grad, expected, min_cosine=0.999, norm_rtol=0.02)


@requires_rocm
def test_triton_indexer_scores_given_indices_and_rejects_invisible_ones():
    from veomni.ops.kernels.dsa.vendor.triton_indexer import triton_lighting_indexer

    seqlen, batch, heads, dim, compress_ratio, topk = 130, 2, 8, 128, 4, 65
    kv_len = seqlen // compress_ratio
    q, k, weights = _inputs(seqlen, batch, heads, dim, kv_len, seed=0)
    indices = torch.full((batch, seqlen, topk), -1, device="cuda", dtype=torch.int32)
    for position in range(seqlen):
        valid_count = min((position + 1) // compress_ratio, kv_len, topk)
        if valid_count:
            indices[:, position, :valid_count] = torch.arange(valid_count, device="cuda", dtype=torch.int32)
    indices[..., -3] = -1
    indices[..., -2] = -2
    indices[..., -1] = kv_len
    # A legal entry the query cannot see yet: scored -inf and given no gradient.
    indices[:, 0, 0] = kv_len - 1

    actual, actual_indices = triton_lighting_indexer(q, k, weights, compress_ratio, topk, indices)
    logits = _reference_logits(q, k, weights)
    causal = torch.arange(kv_len, device="cuda")[None, :] < ((torch.arange(seqlen, device="cuda") + 1) // 4)[:, None]
    reference, valid = _gather(logits.masked_fill(~causal, float("-inf")), indices)
    valid = valid & torch.isfinite(reference)

    assert torch.equal(actual_indices, indices)
    assert torch.equal(torch.isfinite(actual), valid)
    torch.testing.assert_close(actual[valid], reference[valid], rtol=1e-5, atol=1e-6)
    grad = torch.randn_like(actual).masked_fill(~valid, 0)
    _assert_grads_match(actual, reference, grad, valid, (q, k, weights))


@requires_rocm
def test_triton_indexer_streaming_topk_matches_packed_reference(monkeypatch):
    from veomni.ops.kernels.dsa.vendor import triton_indexer

    segment_len, heads, dim, compress_ratio, topk = 64, 8, 128, 4, 7
    seqlen = 2 * segment_len
    kv_per_segment = segment_len // compress_ratio
    q, k, weights = _inputs(seqlen, 1, heads, dim, 2 * kv_per_segment, seed=4)
    local_positions = torch.arange(segment_len, device="cuda", dtype=torch.int32).repeat(2)
    cu_seqlen_ks = torch.cat(
        [
            torch.zeros(segment_len, device="cuda", dtype=torch.int32),
            torch.full((segment_len,), kv_per_segment, device="cuda", dtype=torch.int32),
        ]
    )
    cu_seqlen_ke = cu_seqlen_ks + (local_positions + 1) // compress_ratio
    # Seven pool columns force multiple chunks, including a short final chunk.
    monkeypatch.setattr(triton_indexer, "_SCORES_CHUNK_BYTES", seqlen * torch.float32.itemsize * 7)

    actual, actual_indices = triton_indexer.triton_lighting_indexer(
        q, k, weights, compress_ratio, topk, cu_seqlen_ks=cu_seqlen_ks, cu_seqlen_ke=cu_seqlen_ke
    )
    entries = torch.arange(2 * kv_per_segment, device="cuda")
    visible = (entries >= cu_seqlen_ks[:, None]) & (entries < cu_seqlen_ke[:, None])
    masked = _reference_logits(q, k, weights).masked_fill(~visible.unsqueeze(0), float("-inf"))
    reference_values, reference_indices = masked.topk(topk, dim=-1)
    reference_indices = reference_indices.to(torch.int32).masked_fill(reference_values == -torch.inf, -1)

    _assert_same_selection(masked.detach(), actual_indices, reference_values.detach(), reference_indices)
    valid = actual_indices >= 0
    reference, _ = _gather(masked, actual_indices)
    torch.testing.assert_close(actual[valid], reference[valid], rtol=1e-5, atol=1e-6)
    assert torch.isneginf(actual[~valid]).all()
    grad = torch.randn_like(actual).masked_fill(~valid, 0)
    _assert_grads_match(actual, reference, grad, valid, (q, k, weights))


@requires_rocm
def test_triton_indexer_score_chunk_budget_bounds_long_context_memory():
    from veomni.ops.kernels.dsa.vendor import triton_indexer

    assert triton_indexer._pool_chunk_width(1, 4096, 1024) == 1024
    seq_len, pool = 192000, 48000
    chunk = triton_indexer._pool_chunk_width(1, seq_len, pool)
    assert 0 < chunk < pool
    assert seq_len * chunk * torch.float32.itemsize <= triton_indexer._SCORES_CHUNK_BYTES


@requires_rocm
def test_triton_indexer_row_matches_eager_at_v4_width():
    seqlen, batch, heads, dim, compress_ratio, topk = 512, 1, 64, 128, 4, 64
    kv_len = seqlen // compress_ratio
    q, k, weights = _inputs(seqlen, batch, heads, dim, kv_len, seed=7)
    weights.data.mul_(heads**-0.5 * dim**-0.5 * 100)
    triton_row = resolve_op("dsa_indexer", "deepseek_v4", "triton").wrapper
    eager_row = resolve_op("dsa_indexer", "deepseek_v4", "eager").wrapper
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, weights)]
    expected, expected_indices = eager_row(*leaves, compress_ratio, topk)

    with torch.no_grad():
        _, selected_indices = triton_row(q, k, weights, compress_ratio, topk)
        causal = (
            torch.arange(kv_len, device="cuda")[None, :] < ((torch.arange(seqlen, device="cuda") + 1) // 4)[:, None]
        )
        masked = _reference_logits(q, k, weights).masked_fill(~causal, float("-inf"))
    _assert_same_selection(masked, selected_indices, expected.detach(), expected_indices)

    actual, _ = triton_row(q, k, weights, compress_ratio, topk, expected_indices)
    valid = expected_indices >= 0
    torch.testing.assert_close(actual[valid], expected[valid], rtol=1e-5, atol=1e-6)
    grad = torch.randn_like(actual).masked_fill(~valid, 0)
    expected_grads = torch.autograd.grad((expected.masked_fill(~valid, 0) * grad).sum(), leaves)
    actual.backward(grad)
    for leaf, reference in zip((q, k, weights), expected_grads, strict=True):
        assert_gradient_direction_and_scale(leaf.grad, reference, min_cosine=0.999, norm_rtol=0.02)


@requires_rocm
def test_triton_indexer_handles_an_empty_pool():
    from veomni.ops.kernels.dsa.vendor.triton_indexer import triton_lighting_indexer

    q, k, weights = _inputs(3, 1, 16, 64, 0, seed=9)
    scores, indices = triton_lighting_indexer(q, k, weights, 4, 8)
    assert scores.shape == indices.shape == (1, 3, 0)
    (scores.sum() + q.float().sum() * 0).backward()
    assert torch.equal(q.grad, torch.zeros_like(q))


@requires_rocm
@pytest.mark.parametrize(
    ("heads", "dim", "q_dtype", "w_dtype", "match"),
    (
        (65, 128, torch.bfloat16, torch.float32, "at most 64 heads"),
        (8, 96, torch.bfloat16, torch.float32, "power-of-two head dim"),
        (8, 128, torch.float16, torch.float32, "bfloat16 q/k"),
        (8, 128, torch.bfloat16, torch.bfloat16, "float32 weights"),
    ),
)
def test_triton_indexer_rejects_unsupported_inputs(heads, dim, q_dtype, w_dtype, match):
    from veomni.ops.kernels.dsa.vendor.triton_indexer import triton_lighting_indexer

    q = torch.empty(8, 1, heads, dim, device="cuda", dtype=q_dtype)
    k = torch.empty(2, 1, dim, device="cuda", dtype=q_dtype)
    weights = torch.empty(8, 1, heads, device="cuda", dtype=w_dtype)
    with pytest.raises(ValueError, match=match):
        triton_lighting_indexer(q, k, weights, compress_ratio=4, topk=2)


@requires_rocm
def test_triton_indexer_rejects_mismatched_shapes_devices_and_indices():
    from veomni.ops.kernels.dsa.vendor.triton_indexer import triton_lighting_indexer

    q, k, weights = _inputs(8, 1, 8, 128, 2, seed=13)
    with pytest.raises(ValueError, match=r"q \[S, B, H, D\]"):
        triton_lighting_indexer(q[:, 0], k, weights, 4, 2)
    with pytest.raises(ValueError, match=r"k \[P, B=1, D=128\]"):
        triton_lighting_indexer(q, k.expand(-1, 2, -1), weights, 4, 2)
    with pytest.raises(ValueError, match=r"weights \[S=8, B=1, H=8\]"):
        triton_lighting_indexer(q, k, weights[:, :, :-1], 4, 2)
    with pytest.raises(ValueError, match="one device"):
        triton_lighting_indexer(q, k.cpu(), weights, 4, 2)
    with pytest.raises(ValueError, match="compress_ratio > 0"):
        triton_lighting_indexer(q, k, weights, 0, 2)
    with pytest.raises(ValueError, match="topk >= 0"):
        triton_lighting_indexer(q, k, weights, 4, -1)

    bad_shape = torch.zeros(2, 8, 2, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match=r"topk_indices \[B=1, S=8, K\]"):
        triton_lighting_indexer(q, k, weights, 4, 2, bad_shape)
    bad_dtype = torch.zeros(1, 8, 2, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="int32/int64 topk_indices"):
        triton_lighting_indexer(q, k, weights, 4, 2, bad_dtype)

    ks = torch.zeros(8, device="cuda", dtype=torch.float32)
    ke = torch.ones(8, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="int32/int64 packed ranges"):
        triton_lighting_indexer(q, k, weights, 4, 2, cu_seqlen_ks=ks, cu_seqlen_ke=ke)


@requires_rocm
def test_triton_indexer_backward_masks_indices_past_the_pool():
    from veomni.ops.kernels.dsa.vendor.triton_indexer import triton_lighting_indexer

    seq_len, pool = 4, 2
    q, k, weights = _inputs(seq_len, 1, 8, 128, pool, seed=14)
    indices = torch.full((1, seq_len, 1), pool, device="cuda", dtype=torch.int32)
    ks = torch.zeros(seq_len, device="cuda", dtype=torch.int32)
    # Deliberately extend the visible range past the physical pool. The score is
    # still invalid, and backward must not read or atomically write row ``pool``.
    ke = torch.full((seq_len,), pool + 1, device="cuda", dtype=torch.int32)

    scores, returned_indices = triton_lighting_indexer(
        q,
        k,
        weights,
        compress_ratio=4,
        topk=1,
        topk_indices=indices,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    assert torch.equal(returned_indices, indices)
    assert torch.isneginf(scores).all()
    scores.backward(torch.ones_like(scores))
    for grad in (q.grad, k.grad, weights.grad):
        assert torch.equal(grad, torch.zeros_like(grad))


def test_triton_indexer_adapter_delegates_to_dsa_vendor(monkeypatch):
    calls = []
    vendor = ModuleType("veomni.ops.kernels.dsa.vendor.triton_indexer")
    vendor.triton_lighting_indexer = lambda *args: calls.append(args) or ("scores", "indices")
    monkeypatch.setitem(sys.modules, "veomni.ops.kernels.dsa.vendor.triton_indexer", vendor)

    result = triton_indexer_adapter.wrapper("q", "k", "w", 4, 8, "idx", "ks", "ke", ((0, 1),))
    assert result == ("scores", "indices")
    assert calls == [("q", "k", "w", 4, 8, "idx", "ks", "ke")]
