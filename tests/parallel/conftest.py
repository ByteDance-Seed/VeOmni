"""Stub heavy dependencies to allow CPU-only testing of vision_dp utilities.

The veomni import chain pulls in datasets, flash_attn, CUDA ops, etc.
We stub everything except the specific module under test (vision_dp.py)
so that pytest can collect and run the tests on a CPU-only machine.
"""

import importlib
import sys
import types


def _ensure_stub(name, **attrs):
    """Create a stub module with __path__ if it doesn't already exist."""
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        mod.__path__ = [name.replace(".", "/")]
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── Stub veomni top-level (prevent __init__.py from importing ops/data) ──
_ensure_stub("veomni")

_ensure_stub("veomni.ops")
_ensure_stub("veomni.data")
_ensure_stub("veomni.data.constants", IGNORE_INDEX=-100)

# ── Stub utils ──
_ensure_stub("veomni.utils")


class _FakeLogger:
    def __getattr__(self, name):
        return lambda *a, **kw: None


_ensure_stub("veomni.utils.logging", get_logger=lambda name=None: _FakeLogger())
_ensure_stub(
    "veomni.utils.device",
    get_device_type=lambda: "cpu",
    get_device_id=lambda: "cpu",
    IS_NPU_AVAILABLE=False,
    IS_CUDA_AVAILABLE=False,
)
_ensure_stub("veomni.utils.import_utils", is_torch_version_greater_than=lambda v: True)

# ── Stub distributed ──
_ensure_stub("veomni.distributed")


class _FakeParallelState:
    sp_enabled = False
    sp_size = 1
    sp_rank = 0
    sp_group = None


_ensure_stub(
    "veomni.distributed.parallel_state",
    get_parallel_state=lambda: _FakeParallelState(),
    ParallelState=_FakeParallelState,
)

# ── Stub the sequence_parallel __init__ and its heavy sub-modules ──
# We need to prevent the real __init__.py from running (it imports
# async_ulysses, comm, data, loss, ulysses, utils which have heavy deps).
# So we register the package stub FIRST, then load vision_dp.py directly.
_sp_pkg = _ensure_stub("veomni.distributed.sequence_parallel")

for _sub in [
    "veomni.distributed.sequence_parallel.async_ulysses",
    "veomni.distributed.sequence_parallel.comm",
    "veomni.distributed.sequence_parallel.data",
    "veomni.distributed.sequence_parallel.loss",
    "veomni.distributed.sequence_parallel.ulysses",
    "veomni.distributed.sequence_parallel.utils",
]:
    _ensure_stub(_sub)

# Now load vision_dp.py for real (it only depends on torch, dist, parallel_state)
_vision_dp_spec = importlib.util.spec_from_file_location(
    "veomni.distributed.sequence_parallel.vision_dp",
    "veomni/distributed/sequence_parallel/vision_dp.py",
)
_vision_dp_mod = importlib.util.module_from_spec(_vision_dp_spec)
sys.modules["veomni.distributed.sequence_parallel.vision_dp"] = _vision_dp_mod
_vision_dp_spec.loader.exec_module(_vision_dp_mod)

# Attach it to the parent package
_sp_pkg.vision_dp = _vision_dp_mod
