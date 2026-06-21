"""Tests for shared parity-suite runtime helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.seed_omni.parity_suite.core.runtime import (
    resolve_torch_dtype,
    run_capture_context,
    run_worker_context,
    sample_grad,
    sample_named_param,
    sample_tensor,
)


def test_resolve_torch_dtype_accepts_known_aliases() -> None:
    cases = [
        (None, torch.float32),
        (torch.float16, torch.float16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ]

    for raw, expected in cases:
        assert resolve_torch_dtype(raw) is expected


def test_resolve_torch_dtype_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="Unsupported reference dtype"):
        resolve_torch_dtype("fp8")


def test_run_capture_context_defaults_capture_budget() -> None:
    with run_capture_context({}) as options:
        assert options.max_tensor_numel == 1_000_000


def test_run_capture_context_reads_capture_budget_from_run_options() -> None:
    with run_capture_context({"max_tensor_numel": 12}) as options:
        assert options.max_tensor_numel == 12


def test_run_worker_context_defaults_debug_log_off() -> None:
    with run_worker_context({}) as options:
        assert options.debug_log is False
        assert options.env()["VEOMNI_PARITY_WORKER_DEBUG_LOG"] == "false"


def test_run_worker_context_reads_debug_log_from_run_options() -> None:
    with run_worker_context({"debug_log": True}) as options:
        assert options.debug_log is True
        assert options.env()["VEOMNI_PARITY_WORKER_DEBUG_LOG"] == "true"


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
