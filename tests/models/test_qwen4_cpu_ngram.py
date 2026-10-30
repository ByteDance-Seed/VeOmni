"""Local CPU lookup precision and validation; no distributed placement claim."""

import runpy
from pathlib import Path

import pytest
import torch


lookup_class = runpy.run_path(str(Path(__file__).parents[2] / "veomni/models/transformers/qwen4_exp/cpu_ngram.py"))[
    "FrozenCpuNgramLookup"
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lookup_order_duplicates_empty_and_immutable_copy(dtype):
    weights = [torch.arange(15).reshape(5, 3).to(dtype), torch.arange(21).reshape(7, 3).to(dtype) + 30]
    lookup = lookup_class(weights)
    ids = torch.tensor([1, 0, 1, 1])
    rows = torch.tensor([6, 1, 2, 6])
    expected = torch.stack([weights[s][r] for s, r in zip(ids, rows)])
    weights[0].zero_()
    torch.testing.assert_close(lookup(ids, rows), expected, rtol=0, atol=0)
    assert not lookup(ids, rows).requires_grad
    assert lookup(ids[:0], rows[:0]).shape == (0, 3)


def test_lookup_rejects_trainable_and_invalid_requests():
    with pytest.raises(ValueError, match="Freeze"):
        lookup_class([torch.ones(3, 2, requires_grad=True)])
    lookup = lookup_class([torch.ones(3, 2)])
    with pytest.raises(ValueError, match="shard ID"):
        lookup(torch.tensor([1]), torch.tensor([0]))
    with pytest.raises(ValueError, match="row ID"):
        lookup(torch.tensor([0]), torch.tensor([3]))
