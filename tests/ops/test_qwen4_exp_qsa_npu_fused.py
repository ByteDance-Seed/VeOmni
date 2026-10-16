"""Portable autograd/selection checks; real Ascend parity has a separate gate."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


# Keep the portable mask/autograd checks independent of model dependencies.
spec = importlib.util.spec_from_file_location(
    "qsa_kernel_under_test", Path(__file__).parents[2] / "veomni/ops/kernels/qwen4_exp/npu_qsa.py"
)
npu_qsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu_qsa)


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


def test_int8_scatter_mask_matches_boolean_reference_exactly():
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


def _require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")


def test_native_int8_mask_and_fused_gqa_backward():
    _require_npu()
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
    reference, _, _ = _reference_forward(*reference_inputs, blocked, scale)
    expected_grads = torch.autograd.grad(reference, reference_inputs, grad.float())
    actual = npu_qsa._CompactFusedAttention.apply(*native_inputs, native_indices, scale, 64)
    actual_grads = torch.autograd.grad(actual, native_inputs, grad.to("npu"))
    torch.testing.assert_close(actual.float().cpu(), reference, rtol=0.03, atol=0.02)
    for candidate, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(candidate.float().cpu(), expected, rtol=0.03, atol=0.02)
    assert torch.count_nonzero(actual[:, :, 0]).item() == 0
    assert torch.count_nonzero(actual[:, :, 128:]).item() == 0
    assert torch.count_nonzero(actual_grads[0][:, :, 128:]).item() == 0
