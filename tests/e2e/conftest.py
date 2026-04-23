"""Shared fixtures for `tests/e2e/`.

Each `test_*_parallel_align` parametrizes over `DummyDataset(seq_len=2048,
dataset_type=...)`. The session-scoped fixtures below wrap that so the
dataset files are only materialized once per pytest invocation.

Fixture names are kept stable so the previously-monolithic
`test_e2e_parallel.py` (and its future per-modality splits) can reference
them without extra plumbing.
"""

from __future__ import annotations

import pytest

from ..tools import DummyDataset


_SEQ_LEN = 2048


def _make(dataset_type: str):
    dummy = DummyDataset(seq_len=_SEQ_LEN, dataset_type=dataset_type)
    return dummy


@pytest.fixture(scope="session")
def dummy_text_dataset():
    dummy = _make("text")
    yield dummy.save_path
    del dummy


@pytest.fixture(scope="session")
def dummy_qwen2vl_dataset():
    dummy = _make("qwen2vl")
    yield dummy.save_path
    del dummy


@pytest.fixture(scope="session")
def dummy_qwen3vl_dataset():
    dummy = _make("qwen3vl")
    yield dummy.save_path
    del dummy


@pytest.fixture(scope="session")
def dummy_qwen2omni_dataset():
    dummy = _make("qwen2omni")
    yield dummy.save_path
    del dummy


@pytest.fixture(scope="session")
def dummy_qwen3omni_dataset():
    dummy = _make("qwen3omni")
    yield dummy.save_path
    del dummy


@pytest.fixture(scope="session")
def dummy_wan_t2v_dataset():
    dummy = _make("wan_t2v")
    yield dummy.save_path
    del dummy
