"""Aliases must not reorder logical forward or input-gradient accumulation."""

import pytest
import torch

from tests.parallel.test_moe_compact_dispatch import utils


@pytest.mark.parametrize("count", [0, 1, 11])
@pytest.mark.parametrize("topk", [2, 3, 8])
@pytest.mark.parametrize(
    "dtype,weight_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
    ],
)
def test_alias_gathers_preserve_logical_combine_and_gradients(count, topk, dtype, weight_dtype):
    torch.manual_seed(207)
    logical = (
        torch.stack([torch.randperm(16)[:topk] for _ in range(count)])
        if count
        else torch.empty(0, topk, dtype=torch.long)
    )
    physical = (logical * 7 + 3) % 17
    x = torch.randn(count, 5, dtype=dtype).requires_grad_()
    routing = torch.randn(count, topk, dtype=weight_dtype).requires_grad_()
    coefficients = torch.randn(count * topk, 5, dtype=dtype)
    expected_physical, physical_order = utils.compact_permute(x, physical)
    logical_tokens, expected_logical_order = utils.compact_permute(x, logical)
    actual, (logical_order, restore) = utils.alias_permute(x, physical, logical)
    assert torch.equal(actual, expected_physical)
    assert torch.equal(logical_order, expected_logical_order)
    dx_reference = torch.autograd.grad((logical_tokens * coefficients[logical_order]).sum(), x, retain_graph=True)[0]
    dx_actual = torch.autograd.grad((actual * coefficients[physical_order]).sum(), x, retain_graph=True)[0]
    assert torch.equal(dx_actual, dx_reference)
    expected = utils.compact_unpermute(coefficients[logical_order], routing, x.shape, logical_order)
    restored = coefficients[physical_order].index_select(0, restore)
    output = utils.compact_unpermute(restored, routing, x.shape, logical_order)
    assert torch.equal(output, expected)
    grad = torch.randn_like(output)
    assert torch.equal(
        torch.autograd.grad(output, routing, grad, retain_graph=True)[0],
        torch.autograd.grad(expected, routing, grad)[0],
    )


def test_alias_order_cancellation_regression():
    logical, physical = torch.tensor([[0, 1, 2]]), torch.tensor([[5, 1, 3]])
    x = torch.zeros(1, 1, dtype=torch.bfloat16, requires_grad=True)
    coefficients = torch.tensor([[1.0], [100139008.0], [-100139008.0]], dtype=torch.bfloat16)
    baseline, baseline_order = utils.compact_permute(x, logical)
    old, physical_order = utils.compact_permute(x, physical)
    fixed, (logical_order, restore) = utils.alias_permute(x, physical, logical)
    reference_dx = torch.autograd.grad((baseline * coefficients[baseline_order]).sum(), x, retain_graph=True)[0]
    old_dx = torch.autograd.grad((old * coefficients[physical_order]).sum(), x, retain_graph=True)[0]
    fixed_dx = torch.autograd.grad((fixed * coefficients[physical_order]).sum(), x)[0]
    assert reference_dx.item() == fixed_dx.item() == 0
    assert old_dx.item() == 1
    weights = torch.ones(1, 3, dtype=torch.bfloat16)
    old_output = utils.compact_unpermute(coefficients[physical_order], weights, x.shape, physical_order)
    fixed_output = utils.compact_unpermute(coefficients[physical_order][restore], weights, x.shape, logical_order)
    assert old_output.item() == 1 and fixed_output.item() == 0


def test_alias_permute_rejects_invalid_shapes_and_empty_topk():
    with pytest.raises(ValueError, match="shape"):
        utils.alias_permute(
            torch.zeros(1, 4), torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="at least one"):
        utils.alias_permute(
            torch.zeros(1, 4), torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, dtype=torch.long)
        )
