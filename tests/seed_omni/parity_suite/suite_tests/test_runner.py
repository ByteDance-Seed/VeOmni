"""Tests for parity runner helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.seed_omni.parity_suite.core import apply_v2_selector, load_probe_catalog


def test_unique_consecutive_v2_selector_collapses_adjacent_equal_tensors(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
infer.step:
  v2:
    node: toy.generate
    field: hidden
    selector: unique_consecutive
  ref:
    field: hidden
  tol: hidden
""",
        encoding="utf-8",
    )
    [probe] = load_probe_catalog(probes_path).probes

    selected = apply_v2_selector(
        [
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
        ],
        probe,
    )

    assert len(selected) == 3
    assert torch.equal(selected[0], torch.tensor([1.0]))
    assert torch.equal(selected[1], torch.tensor([2.0]))
    assert torch.equal(selected[2], torch.tensor([1.0]))
