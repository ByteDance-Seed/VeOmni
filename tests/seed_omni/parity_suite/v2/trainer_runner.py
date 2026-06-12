"""Trainer-level helper interface for parity adapters."""

from __future__ import annotations

from typing import Any


def run_forward_backward_step(trainer: Any, micro_batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    return trainer.forward_backward_step(micro_batch)
