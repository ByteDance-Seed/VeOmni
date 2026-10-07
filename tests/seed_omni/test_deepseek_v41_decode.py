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

"""The decode path's attention, held against the kernel it stands in for.

``dense_sparse_attn`` exists so a decode step does not need TileLang or SM90, and
that is only worth having if it is the kernel's equal rather than something
close. The failure it guards against is quiet: a sink dropped from the
denominator, or a mask applied to the scores but not to the values, still yields
finite output of the right shape whose softmax still sums to one. The model would
decode fluent, subtly wrong text.

So the comparison is against ``sparse_attn_tilelang`` itself, with the index
tensor seeded with both kinds of miss the real caller produces -- ``-1`` from a
sliding window that is not yet full, and out-of-range from an indexer that found
fewer visible groups than ``index_topk``.
"""

import math

import pytest
import torch

from veomni.models.seed_omni.modules.deepseek_v41.llm.decode import dense_sparse_attn
from veomni.utils.device import IS_CUDA_AVAILABLE


def _kernel_available() -> bool:
    if not IS_CUDA_AVAILABLE:
        return False
    try:
        from veomni.utils.device import get_gpu_compute_capability

        return get_gpu_compute_capability() >= 90
    except Exception:
        return False


requires_kernel = pytest.mark.skipif(
    not _kernel_available(), reason="the TileLang sparse kernel needs an SM90+ CUDA GPU"
)


def _inputs(batch, seqlen, heads, dim, kv_len, topk, *, seed=0, device="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, dim, device=device, dtype=torch.bfloat16)
    sinks = torch.randn(heads, device=device, dtype=torch.float32)
    idxs = torch.randint(kv_len, (batch, seqlen, topk), device=device, dtype=torch.int32)
    # Both kinds of miss the production caller emits, planted densely enough that
    # a one-tile skew between the mask and the gather cannot hide in a corner.
    idxs[..., ::16] = -1
    idxs[..., 7::16] = kv_len
    return q, kv, sinks, idxs


@requires_kernel
@pytest.mark.parametrize(
    ("seqlen", "kv_len", "topk"),
    [
        (64, 128, 640),  # prefill-shaped
        (8, 256, 640),
    ],
)
def test_dense_matches_tilelang_kernel(seqlen, kv_len, topk):
    from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang

    heads, dim = 8, 512
    q, kv, sinks, idxs = _inputs(1, seqlen, heads, dim, kv_len, topk)
    scale = dim**-0.5

    expected, expected_lse = sparse_attn_tilelang(q, kv, sinks, idxs, scale, return_lse=True)
    actual, actual_lse = dense_sparse_attn(q, kv, sinks, idxs, scale, return_lse=True)

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=6e-3, rtol=6e-3)
    torch.testing.assert_close(actual_lse, expected_lse, atol=6e-3, rtol=6e-3)


@requires_kernel
def test_sink_is_not_silently_dropped():
    """A sink left out of the denominator still produces a valid-looking softmax."""
    from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang

    heads, dim, kv_len, topk = 8, 512, 128, 640
    q, kv, _, idxs = _inputs(1, 16, heads, dim, kv_len, topk)
    scale = dim**-0.5

    # A large sink takes most of the mass; a tiny one takes almost none. If the
    # sink were ignored the two would agree.
    big = torch.full((heads,), 8.0, device="cuda", dtype=torch.float32)
    small = torch.full((heads,), -8.0, device="cuda", dtype=torch.float32)

    for sinks in (big, small):
        expected = sparse_attn_tilelang(q, kv, sinks, idxs, scale)
        actual = dense_sparse_attn(q, kv, sinks, idxs, scale)
        torch.testing.assert_close(actual.float(), expected.float(), atol=6e-3, rtol=6e-3)

    assert not torch.allclose(
        dense_sparse_attn(q, kv, big, idxs, scale).float(),
        dense_sparse_attn(q, kv, small, idxs, scale).float(),
        atol=1e-2,
    )


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="needs CUDA")
def test_decode_shape_runs_without_the_kernel():
    """One query against a cached pool -- the shape every decode step uses."""
    heads, dim, kv_len, topk = 8, 512, 640, 640
    q, kv, sinks, idxs = _inputs(1, 1, heads, dim, kv_len, topk)

    out, lse = dense_sparse_attn(q, kv, sinks, idxs, dim**-0.5, return_lse=True)

    assert out.shape == (1, 1, heads, dim)
    assert lse.shape == (1, 1, heads)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="needs CUDA")
def test_a_fully_masked_query_falls_back_to_the_sink():
    """The first token of a sample sees no compressed group and no window past."""
    heads, dim, kv_len, topk = 4, 512, 32, 64
    q, kv, sinks, _ = _inputs(1, 3, heads, dim, kv_len, topk)
    idxs = torch.full((1, 3, topk), -1, device="cuda", dtype=torch.int32)

    out, lse = dense_sparse_attn(q, kv, sinks, idxs, dim**-0.5, return_lse=True)

    # All mass on the sink, which carries no value: the output is exactly zero
    # and the LSE is the sink alone -- in base two, so log2(exp(sink)).
    assert torch.equal(out, torch.zeros_like(out))
    expected = (sinks / math.log(2.0)).view(1, 1, -1).expand(1, 3, heads).contiguous()
    torch.testing.assert_close(lse, expected)


def test_backward_stays_linear_in_the_pool():
    """DSpark trains through this; its pool is the full gathered row under SP.

    Gathering from a per-query expanded view of ``kv`` gives the same values but a
    backward that allocates ``[B, S, S_kv, D]`` -- what OOMed SP8 at 20k tokens.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    batch, seqlen, heads, dim, kv_len, topk = 2, 24, 4, 16, 40, 6
    q, kv, sinks, idxs = _inputs(batch, seqlen, heads, dim, kv_len, topk, device="cpu")
    q, kv = q.float().requires_grad_(True), kv.float().requires_grad_(True)
    out_grad = torch.randn(batch, seqlen, heads, dim)

    class LargestTensor(TorchDispatchMode):
        numel = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for tensor in out if isinstance(out, (tuple, list)) else [out]:
                if isinstance(tensor, torch.Tensor):
                    self.numel = max(self.numel, tensor.numel())
            return out

    out = dense_sparse_attn(q, kv, sinks, idxs, dim**-0.5)
    tracker = LargestTensor()
    with tracker:
        out.backward(out_grad)
    assert tracker.numel < batch * seqlen * kv_len * dim

    ref_q, ref_kv = q.detach().clone().requires_grad_(True), kv.detach().clone().requires_grad_(True)
    valid = (idxs >= 0) & (idxs < kv_len)
    safe = idxs.clamp(0, kv_len - 1).long()
    gathered = ref_kv[:, None].expand(-1, seqlen, -1, -1).gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, dim))
    scores = torch.einsum("bmhd,bmkd->bmhk", ref_q, gathered) * dim**-0.5
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    logits = torch.cat([scores, sinks.view(1, 1, -1, 1).expand(batch, seqlen, -1, 1)], dim=-1)
    probs = logits.softmax(-1)[..., :-1]
    torch.einsum("bmhk,bmkd->bmhd", probs, gathered).backward(out_grad)

    torch.testing.assert_close(q.grad, ref_q.grad)
    torch.testing.assert_close(kv.grad, ref_kv.grad)


@requires_kernel
def test_lse_is_base_two_like_the_kernel():
    """The kernel exponentiates with exp2; its wrapper never says so.

    A natural-log LSE is wrong by a constant 1.4427 and stays plausible: the only
    consumer renormalizes with it, and a uniformly scaled renormalizer still
    yields a distribution that sums to one. Nothing downstream would complain.
    """
    from veomni.ops.kernels.deepseek_v4 import sparse_attn_tilelang

    heads, dim, kv_len, topk = 8, 512, 128, 640
    q, kv, sinks, idxs = _inputs(1, 16, heads, dim, kv_len, topk)
    scale = dim**-0.5

    _, kernel_lse = sparse_attn_tilelang(q, kv, sinks, idxs, scale, return_lse=True)
    _, dense_lse = dense_sparse_attn(q, kv, sinks, idxs, scale, return_lse=True)

    torch.testing.assert_close(dense_lse, kernel_lse, atol=6e-3, rtol=6e-3)
    # And it really is base two, not an accident of this input.
    natural = dense_lse * math.log(2.0)
    assert not torch.allclose(natural, kernel_lse, atol=1e-2)
