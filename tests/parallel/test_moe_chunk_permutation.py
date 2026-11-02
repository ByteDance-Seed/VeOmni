"""Exact copy/gradient semantics for fused EP chunk layouts."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tests.parallel.test_moe_ep_load_balance_planner import ep_load_balance


# Avoid the optional eager MoE kernel imports during CPU-only collection.
spec = importlib.util.spec_from_file_location(
    "veomni.distributed.moe._chunk_permutation_test",
    Path(ep_load_balance.__file__).with_name("chunk_permutation.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ChunkPermutation = module.ChunkPermutation


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sizes", [[], [0, 0], [1], [0, 2, 3, 0, 1], [31, 0, 129, 3, 7]])
def test_output_gradient_inverse_and_noncontiguous_input(device, dtype, sizes):
    order = list(reversed(range(len(sizes))))
    rows = sum(sizes)
    value = torch.randn(rows, 18, device=device, dtype=dtype)[:, ::2].detach().requires_grad_()
    mapping = ChunkPermutation(sizes, order, device)
    chunks = value.split(sizes)
    expected = torch.cat([chunks[index] for index in order]) if sizes else value.clone()
    actual = mapping.apply(value)
    assert torch.equal(actual, expected)
    assert torch.equal(mapping.apply(actual, inverse=True), value)
    gradient = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad(expected, value, gradient)[0]
    actual_gradient = torch.autograd.grad(actual, value, gradient)[0]
    assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    "sizes,order", [([-1], [0]), ([1, 2], [0, 0]), ([1], [1]), ([1], []), ([1.0], [0]), ([True], [0])]
)
def test_invalid_metadata_rejected(sizes, order):
    with pytest.raises((TypeError, ValueError)):
        ChunkPermutation(sizes, order, "cpu")


def test_invalid_tensor_metadata_and_input_shape():
    with pytest.raises(ValueError):
        ChunkPermutation(torch.ones(2, 2, dtype=torch.long), [0, 1], "cpu")
    mapping = ChunkPermutation([2, 3], [1, 0], "cpu")
    with pytest.raises(ValueError):
        mapping.apply(torch.zeros(4, 3))
    with pytest.raises(ValueError):
        mapping.apply(torch.zeros(5, 2, 2))


def test_copy_preserves_signed_zero_nan_and_infinity_bits():
    value = torch.tensor([[0.0, -0.0], [float("nan"), float("inf")], [-float("inf"), 1.0]])
    mapping = ChunkPermutation([1, 2], [1, 0], "cpu")
    actual = mapping.apply(value)
    expected = torch.cat((value[1:], value[:1]))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_first_and_second_order_cpu_gradient_checks():
    mapping = ChunkPermutation([1, 0, 2], [2, 1, 0], "cpu")
    value = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(mapping.apply, (value,))
    assert torch.autograd.gradgradcheck(mapping.apply, (value,))
