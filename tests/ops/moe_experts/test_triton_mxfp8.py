# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 dispatch checks and Blackwell numerical comparisons."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops.kernels import moe_experts
from veomni.ops.kernels.moe_experts.shared import dispatch
from veomni.ops.kernels.moe_experts.shared import group_gemm as bf
from veomni.ops.kernels.moe_experts.shared import mxfp8_group_gemm as mx
from veomni.ops.kernels.moe_experts.standard import triton as grouped
from veomni.ops.kernels.moe_experts.standard import triton_fp8
from veomni.ops.registry import OP_REGISTRY


cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@cpu
def test_registry_and_recipe_constraints(monkeypatch):
    assert "fused_triton_fp8" in OP_REGISTRY.list_registered("moe_experts", "standard")
    assert "fused_triton_fp8" in moe_experts.QUANTIZING_IMPLS
    calls = []
    monkeypatch.setattr(grouped, "group_gemm_fused_moe_forward", lambda *args, **kwargs: calls.append(kwargs))
    x = torch.empty(1, 128, dtype=torch.bfloat16)
    empty = x[:0]
    triton_fp8.wrapper(x, x, x, empty, empty, x, x, num_experts=1)
    assert calls[0]["use_mxfp8"] is True
    for kwargs in ({"dtype": "fp4"}, {"activation": "gelu"}):
        with pytest.raises(ValueError, match="FP8 and SiLU"):
            triton_fp8.wrapper(x, x, x, empty, empty, x, x, num_experts=1, **kwargs)


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
    args = (1, None, None, None, None, None, None)
    monkeypatch.setattr(grouped, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=False))
    with pytest.raises(ValueError, match="merged"):
        grouped.group_gemm_fused_moe_forward(*args, use_mxfp8=True)
    monkeypatch.setattr(grouped, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True))
    with pytest.raises(ValueError, match="EP1"):
        grouped.group_gemm_fused_moe_forward(*args, torch.empty(1), use_mxfp8=True)


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
        y = grouped.group_gemm_fused_moe_forward(4, gg, indices, xx, None, None, b, a, use_mxfp8=use_mxfp8)
        g = torch.autograd.grad(y, [t for t in inputs if t.requires_grad], dout)
        assert torch.isfinite(y).all() and all(torch.isfinite(t).all() for t in g)
        outputs[use_mxfp8], grads[use_mxfp8] = y, g
    assert not torch.equal(outputs[False], outputs[True])
    assert not torch.equal(grads[False][0], grads[True][0])
    if train_weights:
        assert all(not torch.equal(a, b) for a, b in zip(grads[False][2:], grads[True][2:]))


@pytest.mark.parametrize("use_mxfp8", [False, True])
@pytest.mark.parametrize("train_weights", [True, False])
@cpu
def test_full_mxfp8_dispatch_on_cpu(use_mxfp8, train_weights, monkeypatch):
    """Exercise the real autograd function with CPU GEMM/scatter stand-ins."""
    calls = {"mx_gemm": 0, "mx_wgrad": 0, "bf_gemm": 0, "bf_wgrad": 0}

    def bf_gemm(a, b, transpose_b=True, **kwargs):
        calls["bf_gemm"] += 1
        return a @ (b[0].T if transpose_b else b[0])

    def bf_wgrad(a, b, c, **kwargs):
        calls["bf_wgrad"] += 1
        c.copy_(a.T @ b)

    def mx_gemm(a, b, ends):
        calls["mx_gemm"] += 1
        return a @ b[0].T

    def mx_wgrad(a, b, ends):
        calls["mx_wgrad"] += 1
        return (a.T @ b).unsqueeze(0)

    monkeypatch.setattr(bf, "group_gemm_same_nk", bf_gemm)
    monkeypatch.setattr(bf, "group_gemm_same_mn", bf_wgrad)
    monkeypatch.setattr(dispatch, "moe_scatter", lambda x, index: x)
    monkeypatch.setattr(dispatch, "moe_gather", lambda x, index: x)
    monkeypatch.setattr(dispatch, "expert_histogram", lambda index, n: torch.tensor([index.numel()]))
    monkeypatch.setattr(bf, "same_nk_tile_schedule", lambda *args: (None, None))
    monkeypatch.setattr(mx, "mxfp8_group_gemm", mx_gemm)
    monkeypatch.setattr(mx, "mxfp8_group_wgrad", mx_wgrad)
    torch.manual_seed(42)
    inputs = [
        torch.randn(3, 128, dtype=torch.bfloat16).requires_grad_(),
        torch.ones(3, 1, dtype=torch.bfloat16).requires_grad_(),
        (torch.randn(1, 256, 128, dtype=torch.bfloat16) / 16).requires_grad_(train_weights),
        (torch.randn(1, 128, 128, dtype=torch.bfloat16) / 16).requires_grad_(train_weights),
    ]
    x, gate, w1, w2 = inputs
    args = (1, gate, torch.zeros(3, 1, dtype=torch.int64), x, w1, w2)
    # The omitted optional argument also verifies the pre-MXFP8 direct-call API.
    y = grouped.MergedFc1TritonFusedMoeExpertFunction.apply(*args, *([None, False, "silu", True] if use_mxfp8 else []))
    y.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in inputs if t.requires_grad)
    assert calls["mx_gemm"] == (4 if use_mxfp8 else 0)
    assert calls["bf_gemm"] == 4 - calls["mx_gemm"]
    assert calls["mx_wgrad"] == (2 if use_mxfp8 and train_weights else 0)
    assert calls["bf_wgrad"] == (2 if not use_mxfp8 and train_weights else 0)
