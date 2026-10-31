# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Parity of fused EP combine against independent multiply/scatter autograd."""

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.ops import OP_REGISTRY, VeomniOp


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_unpermute_registration_and_eager_empty():
    assert {"eager", "triton"} <= set(OP_REGISTRY.list_registered("moe_unpermute", "standard"))
    x = torch.empty(0, 7, requires_grad=True)
    w = torch.empty(0, requires_grad=True)
    result = VeomniOp("moe_unpermute", "standard", "eager")(x, w, torch.empty(0, dtype=torch.int64), num_tokens=3)
    assert result.shape == (3, 7) and result.count_nonzero() == 0
    result.sum().backward()
    assert x.grad is not None and w.grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
@pytest.mark.parametrize(
    "dtype,weight_dtype,rows,hidden",
    [
        pytest.param(torch.bfloat16, torch.float32, 0, 33, id="empty-rank"),
        pytest.param(torch.bfloat16, torch.float32, 159, 129, id="fp32-routing-tail"),
        pytest.param(torch.bfloat16, torch.bfloat16, 257, 2560, id="bf16-routing"),
        pytest.param(torch.float32, torch.float32, 65, 128, id="fp32"),
    ],
)
def test_unpermute_forward_backward(dtype, weight_dtype, rows, hidden):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Requires NVIDIA CUDA.")
    torch.manual_seed(237)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(rows, device="cuda", dtype=weight_dtype, requires_grad=True)
    # Deliberately repeated destinations and unused output rows.
    index = torch.randint(0, 23, (rows,), device="cuda")
    reference_x, reference_w = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    weighted = reference_x * reference_w[:, None]
    expected = torch.zeros(31, hidden, device="cuda", dtype=torch.float32)
    expected.scatter_add_(0, index[:, None].expand_as(weighted), weighted.float())
    expected = expected.to(weighted.dtype)
    actual = VeomniOp("moe_unpermute", "standard", "triton")(x, w, index, num_tokens=31)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    for a, b in [(actual, expected), (x.grad, reference_x.grad), (w.grad, reference_w.grad)]:
        assert a.dtype == b.dtype and a.shape == b.shape and torch.isfinite(a).all()
        diff = (a.double() - b.double()).square().sum() / (
            a.double().square().sum() + b.double().square().sum()
        ).clamp_min(1e-30)
        assert diff < (1e-10 if a.dtype == torch.float32 else 1e-6), diff.item()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=1))
def test_unpermute_strided_inputs_and_router_only_gradients():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("Requires NVIDIA CUDA.")
    torch.manual_seed(237)
    x = torch.randn(65, 19, device="cuda", dtype=torch.bfloat16).T
    w = torch.randn(38, device="cuda", dtype=torch.float32)[::2].detach().requires_grad_()
    index = torch.randint(0, 7, (38,), device="cuda", dtype=torch.int32)[::2]
    reference_x, reference_w = x.clone(), w.detach().clone().requires_grad_()
    eager = VeomniOp("moe_unpermute", "standard", "eager")
    fused = VeomniOp("moe_unpermute", "standard", "triton")
    actual = fused(x, w, index, num_tokens=7)
    expected = eager(reference_x, reference_w, index.long(), num_tokens=7)
    grad = torch.randn(65, 7, device="cuda").T
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert x.grad is None
    torch.testing.assert_close(w.grad, reference_w.grad, atol=2e-5, rtol=1e-5)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ep_unpermute_keeps_deterministic_dispatch(monkeypatch):
    import veomni.ops as ops
    import veomni.utils.device as device
    from veomni.distributed.moe.moe_utils import unpermute

    selected = []
    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(device, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        ops, "VeomniOp", lambda op, variant, impl: lambda *args, **kwargs: (selected.append(impl), args[0])[1]
    )
    x, weights, mapping, mask = torch.empty(2, 3), torch.ones(2, 1), torch.arange(2), torch.ones(1, 2)
    for deterministic, expected in [(False, "triton"), (True, "eager")]:
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda enabled=deterministic: enabled)
        unpermute(x, weights, x.shape, mapping, mask)
        assert selected[-1] == expected


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_ep_unpermute_preserves_token_dtype_with_fp32_routing_weights():
    from veomni.distributed.moe.moe_utils import unpermute

    torch.manual_seed(53)
    tokens = torch.randn(6, 7, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)
    mapping = torch.tensor([0, 1, 2, 0, 1, 2])
    mask = torch.ones(2, 3, dtype=torch.bool)
    reference_tokens = tokens.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    weighted = reference_tokens * reference_weights.T.flatten()[:, None]
    reference = torch.zeros(3, 7, dtype=torch.float32)
    reference.scatter_add_(0, mapping[:, None].expand_as(tokens), weighted)
    reference = reference.to(tokens.dtype)
    actual = unpermute(tokens, weights, reference.shape, mapping, mask)
    grad = torch.randn_like(reference)
    actual.backward(grad)
    reference.backward(grad)
    assert actual.dtype == tokens.dtype
    for value, expected in (
        (actual, reference),
        (tokens.grad, reference_tokens.grad),
        (weights.grad, reference_weights.grad),
    ):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
