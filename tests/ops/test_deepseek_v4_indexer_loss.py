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

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


def reference_compressed_target(q, kv, attn_sink, topk_idxs, compressed_start, sm_scale):
    """Paper-correct teacher for DeepSeek-V3.2 eq. (4), in fp32.

    Deliberately written without reference to any LSE: the per-head denominator
    comes from one softmax over ``[selected slots ‖ sink]``, so the window and
    sink contributions are structurally present. This is the property Megatron's
    teacher violates (NVIDIA/Megatron-LM#5776) and the reason this reference must
    never be replaced by a call into the implementation.

    Args:
        q:               [B, S, H, D] any float dtype
        kv:              [B, S_kv, D] any float dtype
        attn_sink:       [H]
        topk_idxs:       [B, S, W + C] int, -1 for misses
        compressed_start: W, the index where the compressed slice begins
        sm_scale:        float

    Returns:
        [B, S, C] fp32, L1-normalised over the compressed slice.
    """
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    batch_index = torch.arange(b, device=kv.device).view(b, 1, 1)
    gathered = kv[batch_index, topk_idxs.clamp_min(0)]  # [B, S, W + C, D]
    logits = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    probs = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1)[..., :-1]
    compressed = probs[..., compressed_start:].sum(dim=2)  # head sum -> [B, S, C]
    return compressed / compressed.sum(-1, keepdim=True).clamp_min(1e-20)


def test_reference_target_responds_to_sink_and_window():
    """The two perturbations Megatron's compressed-only teacher is blind to.

    A per-head softmax over the compressed entries alone would make both of
    these no-ops, so this test is what distinguishes a correct teacher from a
    plausible one. Two heads with opposing preferences are required: with one
    head the outer normalisation cancels the denominator entirely.
    """
    torch.manual_seed(0)
    b, s, h, d, w, c = 1, 3, 2, 16, 2, 4
    q = torch.randn(b, s, h, d)
    kv = torch.randn(b, w + c, d)
    topk = torch.arange(w + c).view(1, 1, -1).expand(b, s, -1).contiguous().to(torch.int32)
    sink = torch.zeros(h)
    scale = d**-0.5

    base = reference_compressed_target(q, kv, sink, topk, w, scale)

    bumped_sink = sink.clone()
    bumped_sink[0] = 5.0
    assert not torch.allclose(base, reference_compressed_target(q, kv, bumped_sink, topk, w, scale), atol=1e-4)

    bumped_kv = kv.clone()
    bumped_kv[:, 0] *= 4.0  # a window row, not a compressed row
    assert not torch.allclose(base, reference_compressed_target(q, bumped_kv, sink, topk, w, scale), atol=1e-4)


def test_reference_target_is_normalised():
    torch.manual_seed(1)
    q = torch.randn(2, 5, 4, 16)
    kv = torch.randn(2, 12, 16)
    topk = torch.randint(0, 12, (2, 5, 8), dtype=torch.int32)
    topk[0, 0, 0] = -1
    target = reference_compressed_target(q, kv, torch.zeros(4), topk, 4, 16**-0.5)
    assert torch.allclose(target.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert (target >= 0).all()


def _require_tilelang_cuda():
    pytest.importorskip("tilelang")
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE:
        pytest.skip("DeepSeek V4 TileLang kernels require an NVIDIA CUDA GPU")
    if get_gpu_compute_capability() < 90:
        pytest.skip("DeepSeek V4 TileLang kernels require SM90 or later")


@pytest.mark.parametrize("heads", [8, 16, 64])
def test_target_kernel_matches_reference(heads):
    """``heads=8`` pads to 16 inside the kernel; the padded heads must contribute
    nothing to the head sum. That is not automatic — a padded head has zero Q and
    would contribute ``exp2(0 - lse_pad)`` unless its LSE is padded to +inf."""
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(0)
    b, s, d, w, c = 2, 8, 64, 64, 64
    device = "cuda"
    q = torch.randn(b, s, heads, d, device=device, dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device=device, dtype=torch.bfloat16)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device=device, dtype=torch.int32)
    topk[0, 0, w] = -1  # a compressed miss
    scale = d**-0.5

    _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
    actual = sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)
    actual = actual / actual.sum(-1, keepdim=True).clamp_min(1e-20)

    expected = reference_compressed_target(q, kv, sink, topk, w, scale)
    # A normalised entry is ~1/C ~ 0.015 here, so a tolerance of the order of
    # 1e-2 would exceed the values being compared and accept anything. It is not
    # a hypothetical: summing the wrong axis of the score tile yields a
    # per-head row sum whose normalised form sits 1.5e-2 from the truth, i.e.
    # inside a 2e-2 tolerance. The kernel and this reference consume the same
    # bf16 inputs and both accumulate in fp32, so they agree to 2e-8 absolute
    # and 5e-7 relative as measured on GB200; the bounds below leave three
    # orders of magnitude of headroom for a different GEMM summation order or a
    # TF32 einsum, while still rejecting the wrong-axis sum by a factor of 60.
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-2)


def test_target_kernel_zeroes_invalid_slots():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_fwd import sparse_mqa_fwd_interface
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    torch.manual_seed(2)
    b, s, heads, d, w, c = 1, 4, 16, 64, 64, 64
    q = torch.randn(b, s, heads, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, 256, d, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
    topk = torch.randint(0, 256, (b, s, w + c), device="cuda", dtype=torch.int32)
    topk[:, :, w + 3] = -1
    scale = d**-0.5

    _, lse = sparse_mqa_fwd_interface(q, kv, sink, topk, sm_scale=scale)
    target = sparse_mqa_target_fwd_interface(q, kv, topk[:, :, w:].contiguous(), lse, sm_scale=scale)
    assert (target[:, :, 3] == 0).all()


def test_target_kernel_rejects_more_than_64_heads():
    _require_tilelang_cuda()
    from veomni.ops.kernels.deepseek_v4.tilelang_sparse_mla_target import sparse_mqa_target_fwd_interface

    q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
    topk = torch.zeros(1, 2, 64, device="cuda", dtype=torch.int32)
    lse = torch.zeros(1, 2, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="64"):
        sparse_mqa_target_fwd_interface(q, kv, topk, lse)
