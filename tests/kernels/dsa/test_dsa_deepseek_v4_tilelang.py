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

"""DeepSeek-V4 TileLang vendor-kernel boundary and gradient coverage."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


DEVICE = get_device_type()


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
    from veomni.kernels._kernels.dsa.vendor.tilelang_indexer import v4_lighting_indexer

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
    from veomni.kernels._kernels.dsa.vendor.tilelang_indexer import v4_lighting_indexer

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
    from veomni.kernels._kernels.dsa.vendor.tilelang_indexer import v4_lighting_indexer

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


def test_tilelang_sparse_attention_forward_pipelines_the_gather_without_changing_output():
    """Pipelining the forward's sparse gather must stay legal, real, and exact.

    Legality is load-bearing on the gather reading its row index straight from global
    memory. Indexing a register fragment there instead puts the gather in pipeline
    stage 0 while the fragment's producer lands in a later stage, which is a hard
    TileLang failure -- so a future change of that shape breaks compilation here rather
    than silently losing the overlap. The async-copy check is what distinguishes "still
    pipelined" from "compiles but the pipelining was dropped".

    Pipelining is off by default because it is a loss at the production shape, so this
    only pins that it remains available and exact when asked for.
    """
    _require_tilelang_cuda()
    from veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla_fwd import sparse_mqa_fwd, sparse_mqa_fwd_interface

    torch.manual_seed(4)
    batch, seqlen, heads, dim, kv_len, topk = 1, 64, 8, 512, 128, 640
    q = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16)
    sinks = torch.randn(heads, device=DEVICE)
    indices = torch.randint(kv_len, (batch, seqlen, topk), device=DEVICE, dtype=torch.int32)
    # topk=640 is 10 gather tiles, so the loop has a real steady state; plant sentinels
    # in every tile so a one-tile skew between the mask and the gather cannot hide.
    indices[..., ::64] = -1
    indices[..., 63::64] = kv_len
    scale = dim**-0.5

    pipelined = sparse_mqa_fwd(heads, dim, topk, scale, num_stages=1)
    serial = sparse_mqa_fwd(heads, dim, topk, scale, num_stages=0)
    assert "cp_async_gs" in pipelined.get_kernel_source()
    assert "cp_async_gs" not in serial.get_kernel_source()

    expected_out, expected_lse = serial(q, kv, sinks, indices)
    for out, lse in (
        pipelined(q, kv, sinks, indices),
        sparse_mqa_fwd_interface(q, kv, sinks, indices, scale, num_stages=1),
    ):
        assert torch.equal(out, expected_out)
        assert torch.equal(lse, expected_lse)

    # The depth is bounded by shared memory, so it is derived from the device rather than
    # left to surface as an opaque launch failure. Walk the depth up until the guard
    # refuses, then show the deepest depth it did allow really launches: that measures the
    # bound against the device instead of against a copy of its own formula, and it holds
    # on any opt-in shared-memory limit rather than assuming Hopper's 227 KB.
    wide_heads = 64
    wide_q = torch.randn(batch, seqlen, wide_heads, dim, device=DEVICE, dtype=torch.bfloat16)
    wide_sinks = torch.randn(wide_heads, device=DEVICE)
    deepest = None
    for depth in range(1, 9):
        try:
            candidate = sparse_mqa_fwd(wide_heads, dim, topk, scale, num_stages=depth)
        except AssertionError as exc:
            assert "shared memory per block" in str(exc)
            break
        deepest = candidate
    else:
        pytest.fail("the shared-memory guard never rejected a gather depth")

    assert deepest is not None, "the guard rejected even a single-stage gather"
    deep_out, _ = deepest(wide_q, kv, wide_sinks, indices)
    # Reading the result into a Python bool is the synchronization point: a launch that
    # could not get its shared memory surfaces here rather than staying pending.
    assert torch.isfinite(deep_out).all().item()


def test_tilelang_sparse_attention_forward_defaults_to_no_pipelining():
    """The shipped gather depth is a performance choice that no output can reveal.

    Both depths are bitwise identical, so the numerics test above cannot catch a flipped
    default -- and the depth it would flip to cost 5% of a production step. This lives
    outside that test because it needs no GPU, while the test it would otherwise sit in
    skips below SM90 and therefore never runs on CI's L20 runners.
    """
    pytest.importorskip("tilelang")
    from veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla_fwd import sparse_mqa_fwd, sparse_mqa_fwd_interface

    for entry in (sparse_mqa_fwd, sparse_mqa_fwd_interface):
        # tilelang.jit re-exports the kernel as (*args, **kwargs), hiding its defaults.
        params = inspect.signature(getattr(entry, "func", entry)).parameters
        assert params["num_stages"].default == 0


def test_tilelang_sparse_attention_forward_backward_with_invalid_indices():
    _require_tilelang_cuda()
    from veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla import sparse_attn_tilelang

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
    # dAttnSink is accumulated by an atomic under a replicated T.Parallel loop, so a
    # lost replication guard would scale it by the warp count -- which cosine, being
    # scale-invariant, cannot see.
    torch.testing.assert_close(sinks.grad, expected_grads[2], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "block_H, block_size, expected",
    [
        (16, 32, 128),
        (32, 32, 256),
        (64, 32, 256),
        # block_size is a bwd parameter, so the bound has to follow it: at 16 the
        # column split halves and block_H=32 no longer affords 8 warps.
        (32, 16, 128),
        (64, 16, 256),
        # The one tile where the derived width drops below the historical 128, which
        # was in fact illegal there -- two warps is all this tile can partition.
        (16, 16, 64),
    ],
)
def test_tilelang_bwd_cta_threads_respects_warp_tile_bound(block_H, block_size, expected):
    """``cta_threads`` must stay inside the GEMM warp tile it derives its bound from.

    A width above ``block_size * block_H // 4`` is a TileLang compile-time assertion
    rather than a slow kernel, so this pins both the chosen values and the warp split
    they come from. The production tile is (64, 32); (16, 32) is what every other
    TileLang test in this file exercises, which is why it must stay at 128.
    """
    pytest.importorskip("tilelang")
    from veomni.kernels._kernels.dsa.vendor import tilelang_sparse_mla_bwd

    threads = tilelang_sparse_mla_bwd.cta_threads(block_H, block_size)

    assert threads == expected
    assert threads % 32 == 0 and threads & (threads - 1) == 0
    # Restate TileLang's own partition rather than the closed form above: it splits the
    # columns over at most block_size // 8 warps and gives the rest to the rows, and an
    # MMA row tile below 16 is a hard compile error.
    num_warps = threads // 32
    column_warps = min(num_warps, block_size // 8)
    assert block_H // (num_warps // column_warps) >= 16


def test_tilelang_sparse_attention_backward_matches_reference_on_wide_head_tile():
    """Cover the head tile that gets the widened CTA; the rest of this file does not.

    ``bwd`` derives its CTA width from ``block_H = min(64, max(next_power_of_2(H), 16))``,
    and every other TileLang test here uses 8 heads, i.e. block_H=16, which keeps the
    historical 128 threads. 32 heads is the smallest tile that actually widens to 256,
    so without this case the wide path is never run.
    """
    _require_tilelang_cuda()
    from veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla import sparse_attn_tilelang

    torch.manual_seed(6)
    batch, seqlen, heads, dim, kv_len, topk = 1, 16, 32, 512, 64, 64
    q = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(heads, device=DEVICE, requires_grad=True)
    indices = torch.randint(kv_len, (batch, seqlen, topk), device=DEVICE, dtype=torch.int32)
    indices[..., -1] = -1
    scale = dim**-0.5

    actual = sparse_attn_tilelang(q, kv, sinks, indices, scale)
    expected = _sparse_attention_reference(q, kv, sinks, indices, scale)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        assert _cosine_similarity(actual_grad, expected_grad) > 0.95
    # The CTA width sets the replicate extent of the T.Parallel loop that atomically
    # accumulates dAttnSink, so a lost replication guard would scale it by the warp
    # count while leaving every cosine above intact.
    torch.testing.assert_close(sinks.grad, expected_grads[2], rtol=2e-2, atol=2e-2)


def test_tilelang_sparse_attention_backward_shares_kernel_across_kv_lengths(monkeypatch):
    _require_tilelang_cuda()
    from veomni.kernels._kernels.dsa.vendor import tilelang_sparse_mla_bwd as sparse_mla_bwd
    from veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla import sparse_attn_tilelang

    kv_block = sparse_mla_bwd.KV_BLOCK
    requested_kv_lengths = []
    original_bwd = sparse_mla_bwd.bwd

    def tracking_bwd(B, S, S_kv, *args, **kwargs):
        requested_kv_lengths.append(S_kv)
        return original_bwd(B, S, S_kv, *args, **kwargs)

    monkeypatch.setattr(sparse_mla_bwd, "bwd", tracking_bwd)

    batch, seqlen, heads, dim, topk = 1, 32, 8, 512, 64
    scale = dim**-0.5
    # All three round up to one KV block, so one compiled kernel must serve them all;
    # the last one needs no padding at all.
    for kv_len in (48, 129, kv_block):
        torch.manual_seed(4)
        q = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        kv = torch.randn(batch, kv_len, dim, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        sinks = torch.randn(heads, device=DEVICE, requires_grad=True)
        indices = torch.randint(kv_len, (batch, seqlen, topk), device=DEVICE, dtype=torch.int32)
        # Out of range for the real KV but inside the padded KV: must stay masked out.
        indices[..., -2:] = kv_len

        actual = sparse_attn_tilelang(q, kv, sinks, indices, scale)
        expected = _sparse_attention_reference(q, kv, sinks, indices, scale)
        grad = torch.randn_like(actual)
        expected_grads = torch.autograd.grad((expected * grad.float()).sum(), (q, kv, sinks))
        actual.backward(grad)
        assert kv.grad.shape == kv.shape
        for actual_grad, expected_grad in zip((q.grad, kv.grad, sinks.grad), expected_grads, strict=True):
            assert _cosine_similarity(actual_grad, expected_grad) > 0.95

    # One specialization key for all three lengths, so TileLang compiles once.
    assert set(requested_kv_lengths) == {kv_block}
