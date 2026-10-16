"""Shared fixtures for the model tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def restore_modeling_backend():
    """Undo ``MODELING_BACKEND`` writes so test order cannot change behaviour.

    ``utils.set_environ_param`` flips the variable with a raw ``os.environ``
    assignment mid-test (it has to outlive the model build inside the same test),
    which otherwise persists for the whole pytest process. Every later test module
    then reads the last case's backend — and under ``hf`` the veomni-only config
    normalization (``flex_attention`` -> ``veomni_flex_attention_with_sp``) stops
    happening, so unrelated suites fail only when run after this directory.
    """
    previous = os.environ.get("MODELING_BACKEND")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODELING_BACKEND", None)
        else:
            os.environ["MODELING_BACKEND"] = previous
