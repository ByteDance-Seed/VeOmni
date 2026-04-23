"""Top-level pytest configuration for VeOmni tests.

This module is intentionally small: it only centralizes cross-test plumbing
that multiple test packages need, so each test file can focus on its own
logic.

Responsibilities:

1. Expose the CI NFS paths (`CI_HF_MODELS_DIR`, `CI_SAMPLES_DIR`,
   `CI_DATASET_DIR`) as pytest fixtures so tests don't re-read `os.environ`
   ad-hoc. These are mounted read-only into the CI container by the
   workflow definitions.

2. Enforce device gates declared via `pytest.mark.gpu_only` /
   `pytest.mark.npu_only`. The active device is read from
   ``VEOMNI_TEST_DEVICE`` (set by CI), or auto-detected from the runtime
   when unset (so local invocations still work without extra wiring).

3. Enforce the ``v5_only`` marker so tests that require transformers v5
   get skipped cleanly under the transformers-stable group without each
   file needing its own ``pytest.mark.skipif(...)``.

Dataset fixtures specific to `tests/e2e/` live in `tests/e2e/conftest.py`
and are not imported here.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# CI path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ci_models_dir() -> str:
    """HuggingFace model cache mounted at /mnt/veomni_ci/models in CI.

    Falls back to "." so unit tests that don't actually need real weights
    still collect cleanly in a local checkout.
    """
    return os.environ.get("CI_HF_MODELS_DIR", ".")


@pytest.fixture(scope="session")
def ci_samples_dir() -> str:
    """Multimodal sample assets (videos/audio) used by a few data tests."""
    return os.environ.get("CI_SAMPLES_DIR", ".")


@pytest.fixture(scope="session")
def ci_dataset_dir() -> str:
    """Pre-downloaded HF datasets used by e2e training tests."""
    return os.environ.get("CI_DATASET_DIR", ".")


# ---------------------------------------------------------------------------
# Marker enforcement
# ---------------------------------------------------------------------------


def _active_device() -> str:
    """Return "gpu" or "npu" depending on the runtime.

    Priority:
      1. ``VEOMNI_TEST_DEVICE`` env var (set explicitly by CI workflows).
      2. Runtime detection via ``veomni.utils.device.IS_NPU_AVAILABLE``.
      3. Default to "gpu" (safest when nothing is available, since most
         tests explicitly check CUDA themselves).
    """
    explicit = os.environ.get("VEOMNI_TEST_DEVICE", "").strip().lower()
    if explicit in {"gpu", "npu"}:
        return explicit
    try:
        from veomni.utils.device import IS_NPU_AVAILABLE
    except Exception:
        return "gpu"
    return "npu" if IS_NPU_AVAILABLE else "gpu"


def _transformers_is_v5() -> bool:
    try:
        from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to
    except Exception:
        return False
    return is_transformers_version_greater_or_equal_to("5.0.0")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    device = _active_device()
    skip_gpu_only = pytest.mark.skip(reason=f"gpu_only test skipped on device={device}")
    skip_npu_only = pytest.mark.skip(reason=f"npu_only test skipped on device={device}")
    skip_v5_only = pytest.mark.skip(reason="v5_only test skipped under transformers < 5.0.0")

    is_v5 = _transformers_is_v5()

    for item in items:
        if "gpu_only" in item.keywords and device != "gpu":
            item.add_marker(skip_gpu_only)
        if "npu_only" in item.keywords and device != "npu":
            item.add_marker(skip_npu_only)
        if "v5_only" in item.keywords and not is_v5:
            item.add_marker(skip_v5_only)
