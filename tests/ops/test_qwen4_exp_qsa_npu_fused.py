"""Portable autograd/selection checks; real Ascend parity has a separate gate."""

import torch
import torch.nn.functional as F

from veomni.ops.kernels.qwen4_exp import npu_qsa


def _reference_forward(q, k, v, blocked, scale):
    repeats = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    maximum = scores.masked_fill(blocked, -torch.inf).amax(-1, keepdim=True)
    maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
    weights = torch.exp(scores - maximum).masked_fill(blocked, 0)
    denominator = weights.sum(-1, keepdim=True).clamp_min(1e-30)
    return (weights / denominator) @ v, maximum, denominator


def _reference_backward(q, k, v, grad, blocked, output, maximum, denominator, scale):
    repeats = q.shape[1] // k.shape[1]
    kr, vr = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    probabilities = torch.exp((q @ kr.transpose(-1, -2)) * scale - maximum).masked_fill(blocked, 0) / denominator
    ds = probabilities * ((grad @ vr.transpose(-1, -2)) - (grad * output).sum(-1, keepdim=True))
    dq = (ds @ kr) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = probabilities.transpose(-1, -2) @ grad
    shape = (*k.shape[:2], repeats, *k.shape[2:])
    return dq, dk.reshape(shape).sum(2), dv.reshape(shape).sum(2)


def test_pruned_gqa_output_gradients_and_saved_tensors(monkeypatch):
    torch.manual_seed(71)
    calls = []

    def forward(*args):
        calls.append(args[1].shape[2])
        return _reference_forward(*args)

    monkeypatch.setattr(npu_qsa, "_forward", forward)
    monkeypatch.setattr(npu_qsa, "_backward", _reference_backward)
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
    for actual, ref in zip(candidate, expected):
        torch.testing.assert_close(actual, ref, rtol=3e-5, atol=4e-6)
    assert torch.count_nonzero(result[:, :, 0]) == 0
    assert torch.count_nonzero(result[:, :, -1]) == 0


def test_ranges_cover_every_selected_index_with_partial_last_block():
    indices = torch.tensor([[[300, -1, 300], [511, 420, -1], [768, -1, -1]]])
    ranges = npu_qsa._kv_ranges(indices, 769, 2)
    assert ranges == [(256, 512), (768, 769)]
    for block, (lo, hi) in enumerate(ranges):
        local = indices[:, block * 2 : (block + 1) * 2]
        valid = local[local >= 0]
        assert torch.all((valid >= lo) & (valid < hi))
