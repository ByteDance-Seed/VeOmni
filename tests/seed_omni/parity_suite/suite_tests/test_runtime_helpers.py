"""Tests for shared parity-suite runtime helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.core.runtime import (
    resolve_torch_dtype,
    sample_grad,
    sample_named_param,
    sample_tensor,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, torch.float32),
        (torch.float16, torch.float16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_resolve_torch_dtype_accepts_known_aliases(raw: object, expected: torch.dtype) -> None:
    assert resolve_torch_dtype(raw) is expected


def test_resolve_torch_dtype_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="Unsupported reference dtype"):
        resolve_torch_dtype("fp8")


def test_sample_tensor_slices_without_rows() -> None:
    tensor = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    sampled = sample_tensor(tensor)
    assert torch.equal(sampled, tensor[:4, :4])


def test_sample_tensor_indexes_rows() -> None:
    tensor = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    rows = torch.tensor([0, 2], dtype=torch.long)
    sampled = sample_tensor(tensor, rows=rows)
    assert torch.equal(sampled, tensor[rows])


def test_sample_grad_delegates_to_sample_tensor() -> None:
    param = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    param.grad = torch.arange(16, dtype=torch.float32).reshape(4, 4) * 2
    assert torch.equal(sample_grad(param), sample_tensor(param.grad))


def test_sample_named_param_reads_parameters() -> None:
    module = nn.Linear(4, 2, bias=False)
    rows = torch.tensor([0], dtype=torch.long)
    assert torch.equal(sample_named_param(module, "weight", rows=rows), sample_tensor(module.weight, rows=rows))
