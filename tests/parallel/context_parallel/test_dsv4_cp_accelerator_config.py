"""TrainingArguments admits context parallelism, but only without Ulysses.

Complements ``test_dsv4_cp_parallel_state.py``: that file covers the
``ParallelState`` gate, this one covers the independent gate in
``TrainingArguments._validate_accelerator`` that a config-file / CLI run must
pass through before a ``ParallelState`` is ever constructed.
"""

from __future__ import annotations

import pytest

from veomni.arguments.arguments_types import AcceleratorConfig, TrainingArguments


def test_accelerator_context_parallel_alone_is_accepted(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = TrainingArguments(accelerator=AcceleratorConfig(cp_size=2, ulysses_size=1))
    assert args.accelerator.cp_size == 2
    # dp_size = world_size // (pp * ulysses * cp * tp) = 2 // (1*1*2*1) = 1
    assert args.accelerator.dp_size == 1


def test_accelerator_hybrid_context_and_ulysses_is_rejected(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(AssertionError, match="ulysses_size"):
        TrainingArguments(accelerator=AcceleratorConfig(cp_size=2, ulysses_size=2))
