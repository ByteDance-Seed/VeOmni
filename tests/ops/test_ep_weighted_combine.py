"""Preserve ordered BF16 combine outputs and both gradients exactly."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.parallel.test_moe_ep_load_balance_planner import ep_load_balance


# Keep the real compact helper import without triggering the package's eager
# accelerator imports; restore every module introduced by this CPU seam.
_MOE_PACKAGE_NAME = "veomni.distributed.moe"
_MOE_UTILS_NAME = f"{_MOE_PACKAGE_NAME}.moe_utils"
_saved_modules = {name: sys.modules.get(name) for name in (_MOE_PACKAGE_NAME, _MOE_UTILS_NAME)}
_moe_package = types.ModuleType(_MOE_PACKAGE_NAME)
_moe_package.__path__ = [str(Path(ep_load_balance.__file__).parent)]
sys.modules[_MOE_PACKAGE_NAME] = _moe_package
try:
    spec = importlib.util.spec_from_file_location(
        f"{_MOE_PACKAGE_NAME}._weighted_combine_test",
        Path(ep_load_balance.__file__).with_name("weighted_combine.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    for _name, _saved in _saved_modules.items():
        if _saved is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _saved
weighted_combine = module.weighted_combine
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def deterministic():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def reference(value, weights, order, physical):
    rows = value.index_select(0, physical)
    ordered_weights = weights.flatten().index_select(0, order)
    weighted = rows * ordered_weights[:, None]
    indices = (order // weights.shape[1])[:, None].expand_as(rows)
    output = torch.zeros((weights.shape[0], value.shape[1]), dtype=torch.float32, device=value.device)
    return output.scatter_add_(0, indices, weighted.float()).to(weighted.dtype)


def check_case(device, dtype, tokens, hidden, topk, grad_flags=(True, True), strided=False, zero=False):
    torch.manual_seed(203)
    value = torch.randn(tokens * topk, hidden * (2 if strided else 1), dtype=dtype, device=device)
    weights = torch.randn(tokens, topk * (2 if strided else 1), dtype=dtype, device=device)
    if strided:
        value, weights = value[:, ::2], weights[:, ::2]
    if zero:
        value.zero_()
        weights.zero_()
        value[::2].neg_()
        weights[::2].neg_()
    value.requires_grad_(grad_flags[0])
    weights.requires_grad_(grad_flags[1])
    order = torch.randperm(tokens * topk, device=device)
    physical = torch.randperm(tokens * topk, device=device)
    if strided:
        order = torch.stack((order, order), dim=1).flatten()[::2]
        physical = torch.stack((physical, physical), dim=1).flatten()[::2]
    actual = weighted_combine(value, weights, (tokens, hidden), order, physical)
    expected = reference(value, weights, order, physical)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    inputs = [tensor for tensor in (value, weights) if tensor.requires_grad]
    if inputs:
        # Exercise a strided upstream gradient even for contiguous fast inputs.
        gradient = torch.randn(tokens, hidden * 2, device=device, dtype=actual.dtype)[:, ::2]
        old = torch.autograd.grad(expected, inputs, gradient)
        new = torch.autograd.grad(actual, inputs, gradient)
        for left, right in zip(old, new):
            assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("topk", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("shape", [(0, 16), (3, 0), (1, 1), (7, 257), (7, 256), (7, 272), (33, 2048)])
def test_exact_output_and_both_gradients(device, topk, shape):
    check_case(device, torch.bfloat16, *shape, topk)


@CUDA
@pytest.mark.parametrize("flags", [(True, False), (False, True), (False, False)])
def test_frozen_inputs(flags):
    check_case("cuda", torch.bfloat16, 11, 528, 8, grad_flags=flags)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_fallbacks(device, dtype):
    check_case(device, dtype, 9, 31, 3)
    check_case(device, dtype, 9, 31, 8, strided=True)


@CUDA
def test_signed_zero():
    check_case("cuda", torch.bfloat16, 7, 16, 8, zero=True)


@CUDA
def test_flat_grid_beyond_65535_tokens():
    check_case("cuda", torch.bfloat16, 65537, 16, 1)


@CUDA
@pytest.mark.parametrize("hidden", [1, 17, 257, 2049])
def test_unaligned_width_does_not_use_fused_backward(monkeypatch, hidden):
    def fail(*_args):
        pytest.fail("Unaligned reductions must keep the original row ordering.")

    monkeypatch.setattr(module._WeightedCombine, "apply", fail)
    check_case("cuda", torch.bfloat16, 7, hidden, 8)


@CUDA
@pytest.mark.parametrize("hidden", [16, 256, 272, 2048])
def test_aligned_cancellation_and_repeatability(hidden):
    torch.manual_seed(710)
    value = torch.randn(56, hidden, device="cuda", dtype=torch.bfloat16)
    # Mix very large opposing terms with small values in the same row. These
    # are intentionally sensitive to changes in the weight-gradient sum tree.
    value[:, 0::4] = 2**25
    value[:, 1::4] = 1
    value[:, 2::4] = -(2**25)
    value.requires_grad_()
    weights = torch.randn(7, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    order, physical = torch.randperm(56, device="cuda"), torch.randperm(56, device="cuda")
    expected = reference(value, weights, order, physical)
    old = torch.autograd.grad(expected, (value, weights), torch.ones_like(expected))
    for _ in range(3):
        actual = weighted_combine(value, weights, (7, hidden), order, physical)
        assert type(actual.grad_fn).__name__ == "_WeightedCombineBackward"
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        new = torch.autograd.grad(actual, (value, weights), torch.ones_like(actual))
        for left, right in zip(old, new):
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@CUDA
def test_misaligned_row_cancellation_regression(monkeypatch):
    value = torch.zeros(2, 257, device="cuda", dtype=torch.bfloat16)
    value[1, 0], value[1, 7], value[1, 8] = 2**30, -(2**30), 1
    value.requires_grad_()
    weights = torch.zeros(1, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    order = torch.tensor([0, 1], device="cuda")
    physical = torch.tensor([1, 0], device="cuda")

    def fail(*_args):
        pytest.fail("Moving this row before sum can change its cancellation order.")

    monkeypatch.setattr(module._WeightedCombine, "apply", fail)
    expected = reference(value, weights, order, physical)
    actual = weighted_combine(value, weights, (1, 257), order, physical)
    for left, right in zip(
        torch.autograd.grad(expected, (value, weights), torch.ones_like(expected)),
        torch.autograd.grad(actual, (value, weights), torch.ones_like(actual)),
    ):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@CUDA
def test_contiguous_but_unaligned_storage_fallback(monkeypatch):
    value = torch.randn(8 * 16 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(8, 16)
    value.requires_grad_()
    weights = torch.randn(1, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    order, physical = torch.randperm(8, device="cuda"), torch.randperm(8, device="cuda")

    def fail(*_args):
        pytest.fail("Unaligned base pointers must use the fallback.")

    monkeypatch.setattr(module._WeightedCombine, "apply", fail)
    actual = weighted_combine(value, weights, (1, 16), order, physical)
    expected = reference(value, weights, order, physical)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("error", ["value_shape", "output_shape", "map_dtype", "map_length", "map_rank"])
def test_invalid_metadata(error):
    value = torch.randn(8, 3)
    weights = torch.randn(4, 2)
    order, physical = torch.arange(8), torch.arange(8)
    shape = (4, 3)
    if error == "value_shape":
        value = value[:7]
    elif error == "output_shape":
        shape = (4, 4)
    elif error == "map_dtype":
        order = order.int()
    elif error == "map_length":
        physical = physical[:7]
    else:
        order = order.reshape(4, 2)
    with pytest.raises(ValueError):
        weighted_combine(value, weights, shape, order, physical)


@CUDA
def test_mixed_dtype_uses_reference():
    value = torch.randn(16, 31, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    order, physical = torch.randperm(16, device="cuda"), torch.randperm(16, device="cuda")
    actual = weighted_combine(value, weights, (2, 31), order, physical)
    expected = reference(value, weights, order, physical)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.equal(actual, expected)
    for old, new in zip(
        torch.autograd.grad(expected.sum(), (value, weights)), torch.autograd.grad(actual.sum(), (value, weights))
    ):
        assert torch.equal(old, new)
