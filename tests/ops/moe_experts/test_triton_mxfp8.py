# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 dispatch checks and Blackwell numerical comparisons."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels import moe_experts
from veomni.ops.kernels.moe_experts.shared import mxfp8
from veomni.ops.kernels.moe_experts.shared import mxfp8_group_gemm as mx
from veomni.ops.kernels.moe_experts.standard import triton as grouped
from veomni.ops.kernels.moe_experts.standard import triton_fp8
from veomni.ops.registry import OP_REGISTRY


cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@cpu
def test_registry_and_recipe_constraints(monkeypatch):
    assert "fused_triton_fp8" in OP_REGISTRY.list_registered("moe_experts", "standard")
    assert "fused_triton_fp8" in moe_experts.QUANTIZING_IMPLS
    monkeypatch.setattr(triton_fp8, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    monkeypatch.setattr(mx, "_check_device", lambda *args: None)
    calls = []
    monkeypatch.setattr(triton_fp8, "_experts", lambda x, *args: calls.append(x) or x)
    x = torch.ones(1, 128, dtype=torch.bfloat16)
    empty = x[:0]
    idx = torch.zeros(1, 1, dtype=torch.int64)
    gate = torch.ones(1, 1)
    w1, w2 = x.new_empty(1, 256, 128), x.new_empty(1, 128, 128)
    y = triton_fp8.wrapper(x, gate, idx, empty, empty, w2, w1, num_experts=1)
    torch.testing.assert_close(y, x)
    assert len(calls) == 1
    for kwargs in ({"dtype": "fp4"}, {"activation": "gelu"}):
        with pytest.raises(ValueError, match="FP8 and SiLU"):
            triton_fp8.wrapper(x, gate, idx, empty, empty, w2, w1, num_experts=1, **kwargs)


@pytest.mark.parametrize("ends,tokens", [([3, 2], 2), ([-1, 3], 3), ([1, 2], 3)])
@cpu
def test_rejects_invalid_offsets(ends, tokens):
    with pytest.raises(ValueError, match="nondecreasing"):
        mx._validate_offsets(torch.tensor(ends), tokens, 2, torch.device("cpu"))


@cpu
def test_offsets_with_empty_expert():
    assert mx._validate_offsets(torch.tensor([3, 3, 132, 195]), 195, 4, torch.device("cpu")) == (129, 4)
    with pytest.raises(ValueError, match="integer"):
        mx._validate_offsets(torch.tensor([1.0]), 1, 1, torch.device("cpu"))


@cpu
def test_operand_validation():
    with pytest.raises(ValueError, match="operands"):
        mx.mxfp8_group_gemm(torch.empty(2, 128), torch.empty(128), torch.tensor([2]))
    with pytest.raises(ValueError, match="divisible"):
        mx.mxfp8_group_gemm(
            torch.empty(2, 32, dtype=torch.bfloat16),
            torch.empty(1, 128, 32, dtype=torch.bfloat16),
            torch.tensor([2]),
        )
    with pytest.raises(ValueError, match="CUDA"):
        mx.mxfp8_group_gemm(
            torch.empty(2, 128, dtype=torch.bfloat16),
            torch.empty(1, 128, 128, dtype=torch.bfloat16),
            torch.tensor([2]),
        )


@cpu
def test_rejects_ep_and_split_weights(monkeypatch):
    empty = torch.empty(0)
    args = (empty, empty, empty, empty, empty, empty, empty)
    monkeypatch.setattr(triton_fp8, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    with pytest.raises(ValueError, match="merged"):
        triton_fp8.wrapper(*args, num_experts=1)
    monkeypatch.setattr(triton_fp8, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True))
    with pytest.raises(ValueError, match="EP1"):
        triton_fp8.wrapper(*args, num_experts=1)


def _dequantize(x):
    size = x.shape[-1]
    padded = torch.nn.functional.pad(x.float(), (0, (-size) % 32))
    chunks = padded.unflatten(-1, (-1, 32))
    exponent = (chunks.abs().amax(-1, keepdim=True).clamp_min(2.0**-118) / 448).log2().ceil().clamp(-126, 127)
    scale = exponent.exp2()
    return ((chunks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(-2)[..., :size]


def _require_sm10x():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("MXFP8 numerical tests require an SM10x CUDA GPU")


def test_grouped_gemm_and_wgrad_reference():
    _require_sm10x()
    torch.manual_seed(42)
    counts = [3, 0, 129, 63]
    ends = torch.tensor(counts, device="cuda").cumsum(0)
    x = torch.randn(sum(counts), 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat([_dequantize(inp) @ _dequantize(weight).T for inp, weight in zip(x.split(counts), w)])
    torch.testing.assert_close(mx.mxfp8_group_gemm(x, w, ends), expected.to(x.dtype), rtol=0.01, atol=0.02)
    expected_dgrad = torch.cat(
        [_dequantize(grad) @ _dequantize(weight.T).T for grad, weight in zip(dy.split(counts), w)]
    ).to(x.dtype)
    torch.testing.assert_close(
        mx.mxfp8_group_gemm(dy, w.transpose(1, 2).contiguous(), ends), expected_dgrad, rtol=0.01, atol=0.02
    )
    expected_wgrad = torch.stack(
        [_dequantize(grad.T) @ _dequantize(inp.T).T for grad, inp in zip(dy.split(counts), x.split(counts))]
    ).to(x.dtype)
    actual = mx.mxfp8_group_wgrad(dy, x, ends)
    torch.testing.assert_close(actual, expected_wgrad, rtol=0.01, atol=0.02)
    assert not actual[1].count_nonzero()
    empty = x[:0]
    zeros = ends.new_zeros(4)
    assert mx.mxfp8_group_gemm(empty, w, zeros).shape == (0, 128)
    assert not mx.mxfp8_group_wgrad(dy[:0], empty, zeros).count_nonzero()


@pytest.mark.parametrize("train_weights", [True, False])
def test_mxfp8_forward_backward(train_weights, monkeypatch):
    _require_sm10x()
    monkeypatch.setattr(grouped, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    monkeypatch.setattr(triton_fp8, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    torch.manual_seed(42)
    counts = [3, 0, 129, 63]
    device = dict(device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(4, device="cuda").repeat_interleave(torch.tensor(counts, device="cuda"))[:, None]
    x = torch.randn(sum(counts), 256, **device)
    gate = torch.ones(sum(counts), 1, **device)
    w1 = torch.randn(4, 256, 256, **device) / 16
    w2 = torch.randn(4, 256, 128, **device) / 16
    dout = torch.randn(x.shape, **device)
    outputs, grads = {}, {}
    for use_mxfp8 in (False, True):
        inputs = [t.detach().requires_grad_(i < 2 or train_weights) for i, t in enumerate((x, gate, w1, w2))]
        xx, gg, a, b = inputs
        if use_mxfp8:
            empty = xx.new_empty(0)
            y = triton_fp8.wrapper(xx, gg, indices, empty, empty, b, a, num_experts=4)
        else:
            y = grouped.group_gemm_fused_moe_forward(4, gg, indices, xx, None, None, b, a)
        g = torch.autograd.grad(y, [t for t in inputs if t.requires_grad], dout)
        assert torch.isfinite(y).all() and all(torch.isfinite(t).all() for t in g)
        outputs[use_mxfp8], grads[use_mxfp8] = y, g
    assert not torch.equal(outputs[False], outputs[True])
    assert not torch.equal(grads[False][0], grads[True][0])
    if train_weights:
        assert all(not torch.equal(a, b) for a, b in zip(grads[False][2:], grads[True][2:]))


@pytest.mark.parametrize("train_weights", [True, False])
@pytest.mark.parametrize("train_inputs", [True, False])
@pytest.mark.parametrize("limit", [None, 0.1])
@cpu
def test_composed_mxfp8_gradients_on_cpu(train_weights, train_inputs, limit, monkeypatch):
    """Check actual grouped-linear nodes and routing against an independent BF16 graph.

    Substitute only the hardware GEMMs with BF16 grouped products. Routing,
    activation/clamp, gradient selection and composition remain production code.
    Duplicate routes and an empty expert exercise parameter-gradient accumulation.
    """
    calls = {"gemm": 0, "wgrad": 0}

    def mx_gemm(a, b, ends):
        calls["gemm"] += 1
        counts = torch.diff(ends, prepend=ends.new_zeros(1)).tolist()
        return torch.cat([x @ w.T for x, w in zip(a.split(counts), b)])

    def mx_wgrad(a, b, ends):
        calls["wgrad"] += 1
        counts = torch.diff(ends, prepend=ends.new_zeros(1)).tolist()
        return torch.stack([dy.T @ x for dy, x in zip(a.split(counts), b.split(counts))])

    monkeypatch.setattr(mxfp8, "mxfp8_group_gemm", mx_gemm)
    monkeypatch.setattr(mxfp8, "mxfp8_group_wgrad", mx_wgrad)
    monkeypatch.setattr(mx, "_check_device", lambda *args: None)
    monkeypatch.setattr(triton_fp8, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    torch.manual_seed(42)
    sources = [
        torch.randn(3, 128, dtype=torch.bfloat16),
        torch.rand(3, 2),
        torch.randn(3, 256, 128, dtype=torch.bfloat16) / 16,
        torch.randn(3, 128, 128, dtype=torch.bfloat16) / 16,
    ]
    indices = torch.tensor([[0, 0], [1, 0], [1, 1]])
    flags = [train_inputs, True, train_weights, train_weights]
    inputs = [t.detach().clone().requires_grad_(f) for t, f in zip(sources, flags)]
    x, gate, w1, w2 = inputs
    empty = x.new_empty(0)
    y = triton_fp8.wrapper(x, gate, indices, empty, empty, w2, w1, num_experts=3, swiglu_limit=limit)
    grads = torch.autograd.grad(y, [t for t in inputs if t.requires_grad], torch.ones_like(y))
    assert calls == {
        "gemm": 2 + int(train_inputs or train_weights) + int(train_inputs),
        "wgrad": 2 * int(train_weights),
    }

    refs = [t.detach().clone().requires_grad_(f) for t, f in zip(sources, flags)]
    xx, gg, a, b = refs
    rows = []
    for token in range(3):
        slots = []
        for slot in range(2):
            expert = indices[token, slot]
            projected = xx[token : token + 1] @ a[expert].T
            g, u = projected.chunk(2, dim=-1)
            if limit is not None:
                g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
            slots.append(((torch.nn.functional.silu(g) * u) @ b[expert].T) * gg.to(xx.dtype)[token, slot])
        rows.append(torch.stack(slots).sum(0))
    expected = torch.cat(rows)
    expected_grads = torch.autograd.grad(expected, [t for t in refs if t.requires_grad], torch.ones_like(expected))
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.003)
    for actual, reference in zip(grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.003)
    if train_weights:
        assert not grads[-1][2].count_nonzero()
        assert not grads[-2][2].count_nonzero()
