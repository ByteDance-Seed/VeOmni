"""Shared CPU isolation fixtures for model and composed-module tests."""

import pytest

from scripts.ci.accelerator.pytest_plugin import resolve_plan
from veomni.utils.device import get_device_type


@pytest.fixture(autouse=True)
def cpu_model_attention_kernel_guard(monkeypatch, request):
    """Catch accidental optional attention preloading even when packages are installed."""
    device = get_device_type()
    # Local MLU tests do not use the CUDA/NPU CI planner or its capability schema.
    if device == "mlu":
        return
    plan = resolve_plan(request.node)
    if device != "cpu" and any(spec.device != "cpu" for spec in plan.launches):
        return

    def forbid_loading(*args, **kwargs):
        raise AssertionError(
            "CPU model construction must select eager/SDPA before initialization; "
            "optional FlashAttention preloading is forbidden."
        )

    monkeypatch.setattr("transformers.modeling_utils.lazy_import_flash_attention", forbid_loading)
    monkeypatch.setattr("veomni.ops.install._load_veomni_local_flash_kernel", forbid_loading)
