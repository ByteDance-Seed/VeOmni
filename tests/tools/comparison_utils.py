"""Tensor comparison utilities for VeOmni tests.

Provides a configurable ``TensorComparator`` class, convenience assertion
functions, and a pretty-printed comparison table for diagnosing mismatches.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from rich.console import Console
from rich.table import Table


@dataclass
class TensorComparator:
    """Configurable tensor comparator with named tolerance presets.

    Parameters
    ----------
    rtol : float
        Relative tolerance for ``torch.testing.assert_close``.
    atol : float
        Absolute tolerance for ``torch.testing.assert_close``.
    """

    rtol: float = 1e-2
    atol: float = 1e-2

    def compare(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        """Assert that *actual* and *expected* are close within tolerances."""
        torch.testing.assert_close(actual, expected, rtol=self.rtol, atol=self.atol)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    """Convenience wrapper around ``torch.testing.assert_close``."""
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert bitwise equality between two tensors."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def compare_metrics(
    outputs: Dict[str, Dict[str, Any]],
    *,
    rtol: float = 0.01,
    atol: float = 0.01,
    keys: Optional[Sequence[str]] = None,
) -> None:
    """Compare metrics across multiple runs.

    Parameters
    ----------
    outputs : dict
        ``{run_name: {metric_name: value}}`` mapping.
    rtol, atol : float
        Tolerances for ``torch.testing.assert_close``.
    keys : sequence of str, optional
        If provided, only compare these metric keys. Otherwise compare all.

    Raises
    ------
    AssertionError
        If any metric differs beyond tolerance.
    """
    base_name = next(iter(outputs))
    base = outputs[base_name]
    check_keys = keys or list(base.keys())

    for run_name, run_output in outputs.items():
        if run_name == base_name:
            continue
        for key in check_keys:
            base_val = base[key]
            run_val = run_output[key]
            if not isinstance(base_val, torch.Tensor):
                base_val = torch.tensor(base_val)
            if not isinstance(run_val, torch.Tensor):
                run_val = torch.tensor(run_val)
            try:
                torch.testing.assert_close(run_val, base_val, rtol=rtol, atol=atol)
            except AssertionError as err:
                print_comparison_table(outputs, key)
                raise AssertionError(f"Metric '{key}' mismatch: {base_name} vs {run_name}") from err


def print_comparison_table(
    outputs: Dict[str, Any],
    metric_key: str,
    title: str = "",
) -> None:
    """Pretty-print a comparison table for a single metric across runs."""
    console = Console()
    table = Table(title=f"Comparison: {title} {metric_key}")
    table.add_column("Run", style="cyan")
    table.add_column(metric_key.upper(), style="bold green", justify="right")

    for name, output in outputs.items():
        val = output.get(metric_key, "N/A")
        if isinstance(val, (list, tuple)):
            val_str = ", ".join(f"{v:.8f}" for v in val)
        elif hasattr(val, "item"):
            val_str = f"{val.item():.8f}"
        elif isinstance(val, float):
            val_str = f"{val:.8f}"
        else:
            val_str = str(val)
        table.add_row(str(name), val_str)

    console.print(table)
