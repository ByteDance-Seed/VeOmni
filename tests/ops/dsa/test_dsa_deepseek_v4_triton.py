# Copyright 2026 Advanced Micro Devices, Inc. and ByteDance Ltd. and/or its affiliates
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

"""DeepSeek-V4 Triton sparse-attention adapter and ROCm kernel tests."""

from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from tests.ops.tol import ATTN_BF16_GRAD_ATOL, ATTN_GRAD_RTOL
from veomni.ops import resolve_op
from veomni.ops.kernels.dsa.attention.deepseek_v4 import triton as triton_attention
from veomni.ops.kernels.dsa.sparse_indices import CompressedCandidates
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


DEVICE = get_device_type()
_ROCM_AVAILABLE = IS_CUDA_AVAILABLE and torch.version.hip is not None


def _sparse_attention_reference(q: Tensor, kv: Tensor, sinks: Tensor, indices: Tensor, scale: float) -> Tensor:
    """Compute sparse MQA with one sink logit per head in fp32."""
    valid = (indices >= 0) & (indices < kv.shape[1])
    safe = indices.clamp(0, kv.shape[1] - 1).long()
    gathered = kv[:, None].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_logits = sinks.float().view(1, 1, -1, 1).expand(q.shape[0], q.shape[1], -1, -1)
    probabilities = torch.softmax(torch.cat((scores, sink_logits), dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", probabilities, gathered.float())


def _sparse_lse_reference(q: Tensor, kv: Tensor, sinks: Tensor, indices: Tensor, scale: float) -> Tensor:
    """Compute the sink-inclusive sparse MQA log-sum-exp in base-2 units."""
    valid = (indices >= 0) & (indices < kv.shape[1])
    safe = indices.clamp(0, kv.shape[1] - 1).long()
    gathered = kv[:, None].expand(-1, q.shape[1], -1, -1)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, kv.shape[-1]))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_logits = sinks.float().view(1, 1, -1, 1).expand(q.shape[0], q.shape[1], -1, -1)
    return torch.logsumexp(torch.cat((scores, sink_logits), dim=-1), dim=-1) * math.log2(math.e)


def _fake_vendor(monkeypatch, return_value: object) -> Mock:
    module_name = "veomni.ops.kernels.dsa.vendor.triton_sparse_mla"
    module = ModuleType(module_name)
    function = Mock(return_value=return_value)
    module.sparse_attn_triton = function
    monkeypatch.setitem(sys.modules, module_name, module)
    return function


def test_triton_attention_adapter_rejects_unsupported_outputs_before_vendor_import(monkeypatch):
    module_name = "veomni.ops.kernels.dsa.vendor.triton_sparse_mla"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    tensor = torch.empty(0)
    contract = {"sliding_window": 1, "compress_ratio": 1}

    with pytest.raises(ValueError, match="dropout=0"):
        triton_attention.wrapper(tensor, tensor, None, tensor, tensor, **contract, dropout=0.5)
    with pytest.raises(ValueError, match="output_attentions=True"):
        triton_attention.wrapper(tensor, tensor, None, tensor, tensor, **contract, return_attn_weights=True)

    assert module_name not in sys.modules


def test_triton_attention_adapter_delegates_to_vendor(monkeypatch):
    sentinel = object()
    vendor = _fake_vendor(monkeypatch, sentinel)
    q = torch.randn(1, 3, 2, 4)
    ori_kv = torch.randn(1, 3, 4)
    cmp_kv = torch.randn(1, 1, 4)
    sink = torch.randn(2, dtype=torch.bfloat16)
    topk = torch.tensor([[[-1], [-1], [0]]], dtype=torch.int32)

    actual = triton_attention.wrapper(
        q,
        ori_kv,
        cmp_kv,
        sink,
        torch.arange(3).view(1, 3),
        sliding_window=2,
        compress_ratio=4,
        candidates=CompressedCandidates(topk_indices=topk),
        sm_scale=0.25,
        return_lse=True,
    )

    assert actual is sentinel
    (vendor_q, vendor_kv, vendor_sink, vendor_indices, scale, return_lse), _ = vendor.call_args
    torch.testing.assert_close(vendor_q, q)
    torch.testing.assert_close(vendor_kv, torch.cat((ori_kv, cmp_kv), dim=1))
    assert vendor_sink.dtype == torch.float32
    assert vendor_indices.tolist() == [[[-1, 0, -1], [0, 1, -1], [1, 2, 3]]]
    assert (scale, return_lse) == (0.25, True)


@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype", "sink_dtype", "match"),
    [
        (torch.float32, torch.bfloat16, torch.float32, "bfloat16"),
        (torch.bfloat16, torch.float32, torch.float32, "bfloat16"),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, "float32"),
    ],
)
def test_triton_sparse_mla_rejects_unsupported_dtypes(q_dtype, kv_dtype, sink_dtype, match):
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import sparse_attn_triton

    q = torch.empty(1, 2, 4, 16, dtype=q_dtype)
    kv = torch.empty(1, 2, 16, dtype=kv_dtype)
    sink = torch.empty(4, dtype=sink_dtype)

    with pytest.raises(ValueError, match=match):
        sparse_attn_triton(q, kv, sink, torch.zeros(1, 2, 4, dtype=torch.int32))


def test_triton_sparse_mla_rejects_batch_shape_mismatch():
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import sparse_attn_triton

    q = torch.empty(2, 2, 4, 16, dtype=torch.bfloat16)
    kv = torch.empty(1, 2, 16, dtype=torch.bfloat16)
    sinks = torch.empty(4)
    indices = torch.zeros(2, 2, 4, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"needs \[B=2, S=2, K\] indices"):
        sparse_attn_triton(q, kv, sinks, indices)


def _flat_pool_sparse_mla_kernels():
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import TOPK_ALIGN

    def pool_attention(q, kv, indices, attn_sink, kv_lora_rank, scale):
        assert q.shape[-1] == kv.shape[-1] == kv_lora_rank
        assert indices.dtype == torch.int32 and indices.shape[-1] % TOPK_ALIGN == 0
        return _sparse_attention_reference(q[None], kv[None, :, 0], attn_sink, indices[None], scale)[0]

    def forward(q, kv, indices, attn_sink, kv_lora_rank, scale):
        output = pool_attention(q, kv, indices, attn_sink, kv_lora_rank, scale)
        return output.to(q.dtype), q.new_zeros(q.shape[:2], dtype=torch.float32)

    def backward(q, kv, output, grad, indices, lse, attn_sink, kv_lora_rank, scale, small_lds):
        del output, lse
        assert small_lds is True
        with torch.enable_grad():
            leaves = [tensor.detach().float().requires_grad_() for tensor in (q, kv, attn_sink)]
            reference = pool_attention(*leaves[:2], indices, leaves[2], kv_lora_rank, scale)
            dq, dkv, d_sink = torch.autograd.grad(reference, leaves, grad.float())
        return dq.to(q.dtype), dkv.to(kv.dtype), d_sink

    return forward, backward


def test_triton_sparse_mla_adapter_matches_reference_through_flat_pool(monkeypatch):
    from veomni.ops.kernels.dsa.vendor import triton_sparse_mla

    monkeypatch.setattr(triton_sparse_mla, "_load_kernels", _flat_pool_sparse_mla_kernels)
    monkeypatch.setattr(triton_sparse_mla, "_has_small_lds", lambda device: True)

    torch.manual_seed(0)
    batch, seq_len, kv_len, heads, dim, topk = 2, 5, 7, 4, 16, 6
    scale = dim**-0.5
    q = torch.randn(batch, seq_len, heads, dim, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, kv_len, dim, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(heads, requires_grad=True)
    indices = torch.randint(kv_len, (batch, seq_len, topk), dtype=torch.int32)
    indices[..., -1] = -1
    indices[0, :, -2] = kv_len

    actual = triton_sparse_mla.sparse_attn_triton(q, kv, sinks, indices, scale)
    expected = _sparse_attention_reference(q, kv, sinks, indices, scale)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
    actual.backward(grad)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), rtol=2e-2, atol=2e-2)


def test_triton_sparse_mla_rank_chunk_caps_long_sequence_workspace():
    pytest.importorskip("triton")
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla_bwd import (
        _MAX_RANK_CHUNK_WORKSPACE_BYTES,
        _rank_chunk_workspace_bytes,
        _select_dq_launch,
        _select_rank_chunk,
    )

    assert _select_dq_launch(small_lds=True) == (32, 16, 1)
    assert _select_dq_launch(small_lds=False) == (64, 16, None)
    assert _select_rank_chunk(total_tokens=4096, num_heads=64, d_qk=512, topk=640) == 256
    for total_tokens in (32768, 131072):
        chunk = _select_rank_chunk(total_tokens, num_heads=64, d_qk=512, topk=640)
        assert chunk > 0 and chunk & (chunk - 1) == 0
        assert _rank_chunk_workspace_bytes(total_tokens, 64, 512, chunk) <= _MAX_RANK_CHUNK_WORKSPACE_BYTES


@pytest.mark.skipif(not _ROCM_AVAILABLE, reason="DeepSeek V4 Triton sparse attention requires AMD ROCm")
def test_dsa_attention_triton_registry_row_matches_eager():
    torch.manual_seed(3)
    batch, seq_len, heads, dim, kv_len, topk = 1, 32, 8, 512, 48, 64
    q = torch.randn(batch, seq_len, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16)
    sinks = torch.randn(heads, device=DEVICE)
    indices = torch.randint(kv_len, (batch, seq_len, topk), device=DEVICE, dtype=torch.int32)
    indices[..., -6:-3] = -1
    indices[..., -3:] = kv_len
    positions = torch.zeros(batch, seq_len, device=DEVICE, dtype=torch.long)
    candidates = CompressedCandidates(topk_indices=indices)
    eager = resolve_op("dsa_attention", "deepseek_v4", "eager").wrapper
    triton = resolve_op("dsa_attention", "deepseek_v4", "triton").wrapper

    q_e, kv_e, sink_e = (tensor.detach().requires_grad_(True) for tensor in (q, kv, sinks))
    q_t, kv_t, sink_t = (tensor.detach().requires_grad_(True) for tensor in (q, kv, sinks))
    kwargs = {
        "sliding_window": 1,
        "compress_ratio": 1,
        "candidates": candidates,
        "sm_scale": dim**-0.5,
    }
    expected = eager(q_e, kv_e[:, :0], kv_e, sink_e, positions, **kwargs)
    actual = triton(q_t, kv_t[:, :0], kv_t, sink_t, positions, **kwargs)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    grad = torch.randn(batch, heads, seq_len, dim, device=DEVICE, dtype=actual.dtype).transpose(1, 2)
    assert not grad.is_contiguous()
    expected.backward(grad)
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q_t.grad, kv_t.grad, sink_t.grad), (q_e.grad, kv_e.grad, sink_e.grad)):
        torch.testing.assert_close(
            actual_grad.float(),
            expected_grad.float(),
            rtol=ATTN_GRAD_RTOL,
            atol=ATTN_BF16_GRAD_ATOL,
        )


@pytest.mark.skipif(not _ROCM_AVAILABLE, reason="DeepSeek V4 Triton sparse attention requires AMD ROCm")
def test_dsa_attention_triton_broadcasts_singleton_position_batch():
    torch.manual_seed(4)
    batch, seq_len, heads, dim = 2, 16, 8, 512
    q = torch.randn(batch, seq_len, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len, dim, device=DEVICE, dtype=torch.bfloat16)
    sinks = torch.randn(heads, device=DEVICE)
    position_ids = torch.arange(seq_len, device=DEVICE).view(1, seq_len)
    eager = resolve_op("dsa_attention", "deepseek_v4", "eager").wrapper
    triton = resolve_op("dsa_attention", "deepseek_v4", "triton").wrapper

    q_e, kv_e, sink_e = (tensor.detach().requires_grad_(True) for tensor in (q, kv, sinks))
    q_t, kv_t, sink_t = (tensor.detach().requires_grad_(True) for tensor in (q, kv, sinks))
    kwargs = {"sliding_window": 4, "compress_ratio": 1, "sm_scale": dim**-0.5}
    expected = eager(q_e, kv_e, None, sink_e, position_ids, **kwargs)
    actual = triton(q_t, kv_t, None, sink_t, position_ids, **kwargs)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q_t.grad, kv_t.grad, sink_t.grad), (q_e.grad, kv_e.grad, sink_e.grad)):
        torch.testing.assert_close(
            actual_grad.float(),
            expected_grad.float(),
            rtol=ATTN_GRAD_RTOL,
            atol=ATTN_BF16_GRAD_ATOL,
        )


@pytest.mark.skipif(not _ROCM_AVAILABLE, reason="DeepSeek V4 Triton sparse attention requires AMD ROCm")
def test_triton_sparse_attention_all_invalid_row_uses_sink_state():
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import sparse_attn_triton

    batch, seq_len, heads, dim, kv_len, topk = 1, 4, 8, 512, 8, 64
    q = torch.randn(batch, seq_len, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.full((heads,), -1000.0, device=DEVICE, requires_grad=True)
    indices = torch.full((batch, seq_len, topk), -1, device=DEVICE, dtype=torch.int32)

    output, lse = sparse_attn_triton(q, kv, sinks, indices, dim**-0.5, return_lse=True)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    expected_lse = sinks.view(1, 1, -1).expand_as(lse) * math.log2(math.e)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)

    output.backward(torch.randn_like(output))
    for grad in (q.grad, kv.grad, sinks.grad):
        assert grad is not None and torch.isfinite(grad).all()
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)


@pytest.mark.skipif(not _ROCM_AVAILABLE, reason="DeepSeek V4 Triton sparse attention requires AMD ROCm")
def test_triton_sparse_attention_keeps_invalid_prefix_out_of_online_softmax_state():
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import sparse_attn_triton

    batch, seq_len, heads, dim, kv_len, topk = 1, 1, 8, 512, 64, 128
    q = torch.ones(batch, seq_len, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.full((batch, kv_len, dim), -44.0, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.full((heads,), -2000.0, device=DEVICE, requires_grad=True)
    invalid = torch.full((64,), -1, device=DEVICE, dtype=torch.int32)
    valid = torch.arange(kv_len, device=DEVICE, dtype=torch.int32)
    indices = torch.cat((invalid, valid)).view(1, 1, topk)
    scale = dim**-0.5

    actual, lse = sparse_attn_triton(q, kv, sinks, indices, scale, return_lse=True)
    expected = _sparse_attention_reference(q, kv, sinks, indices, scale)
    expected_lse = _sparse_lse_reference(q, kv, sinks, indices, scale)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=5e-3, atol=5e-3)

    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _ROCM_AVAILABLE, reason="DeepSeek V4 Triton sparse attention requires AMD ROCm")
def test_triton_sparse_attention_keeps_mixed_magnitude_dq_across_chunks(monkeypatch):
    from veomni.ops.kernels.dsa.vendor import triton_sparse_mla_bwd
    from veomni.ops.kernels.dsa.vendor.triton_sparse_mla import sparse_attn_triton

    torch.manual_seed(2)
    batch, seq_len, heads, dim, kv_len, topk = 1, 16, 8, 512, 640, 640
    q = torch.zeros(batch, seq_len, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16)
    kv[:, 16:] *= 0.05
    kv.requires_grad_()
    sinks = torch.zeros(heads, device=DEVICE, requires_grad=True)
    indices = torch.arange(topk, device=DEVICE, dtype=torch.int32).view(1, 1, -1).expand(batch, seq_len, -1)
    d_qk = dim + 64
    monkeypatch.setattr(
        triton_sparse_mla_bwd,
        "_MAX_RANK_CHUNK_WORKSPACE_BYTES",
        triton_sparse_mla_bwd._rank_chunk_workspace_bytes(seq_len, heads, d_qk, 16),
    )
    assert triton_sparse_mla_bwd._select_rank_chunk(seq_len, heads, d_qk, topk) == 16

    actual = sparse_attn_triton(q, kv, sinks, indices, dim**-0.5)
    expected = _sparse_attention_reference(q, kv, sinks, indices, dim**-0.5)
    grad = torch.ones_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
    actual.backward(grad)

    for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
        torch.testing.assert_close(actual_grad.float(), expected_grad.float(), rtol=2e-2, atol=2e-2)
