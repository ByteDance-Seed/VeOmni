"""Tests for shared parity-suite reference loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.seed_omni.parity_suite.reference.model import empty_init_context, load_safetensors_weights


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)


def test_empty_init_context_places_parameters_on_meta_device() -> None:
    with empty_init_context():
        model = _TinyModel()

    assert model.linear.weight.device.type == "meta"


def test_load_safetensors_weights_missing_file_raises(tmp_path: Path) -> None:
    model = _TinyModel()
    missing = tmp_path / "missing.safetensors"

    with pytest.raises(FileNotFoundError, match="Reference weights not found"):
        load_safetensors_weights(
            model,
            missing,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_load_safetensors_weights_filters_prefixes_and_casts(tmp_path: Path) -> None:
    model = _TinyModel()
    weights_path = tmp_path / "weights.safetensors"
    save_file(
        {
            "linear.weight": torch.ones(2, 2, dtype=torch.float32),
            "ignored.bias": torch.zeros(2, dtype=torch.float32),
        },
        str(weights_path),
    )

    load_safetensors_weights(
        model,
        weights_path,
        include_prefixes=("linear.",),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    assert model.linear.weight.dtype == torch.float16
    assert torch.allclose(model.linear.weight, torch.ones(2, 2, dtype=torch.float16))


def test_load_safetensors_weights_reports_missing_prefixed_keys(tmp_path: Path) -> None:
    model = _TinyModel()
    weights_path = tmp_path / "weights.safetensors"
    save_file({"other.weight": torch.ones(2, 2)}, str(weights_path))

    with pytest.raises(RuntimeError, match="Missing reference weight keys"):
        load_safetensors_weights(
            model,
            weights_path,
            include_prefixes=("linear.",),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
