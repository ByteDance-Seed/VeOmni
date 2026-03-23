"""
Copy and modified from https://github.com/thinking-machines-lab/batch_invariant_ops
"""

import contextlib

from ...utils.device import IS_CUDA_AVAILABLE


_batch_invariant_MODE = False
_batch_invariant_LIB = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    import torch

    from .batch_invariant_ops import (
        _log_softmax_batch_invariant,
        addmm_batch_invariant,
        mean_batch_invariant,
        mm_batch_invariant,
    )

    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    dispatch_key = getattr(torch.accelerator.current_accelerator(), "type", "cpu").upper()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)
    if enabled and IS_CUDA_AVAILABLE:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data


__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]
