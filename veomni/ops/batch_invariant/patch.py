"""Lifecycle management for the process-wide batch-invariant ATen patch."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from ...utils.device import IS_CUDA_AVAILABLE


_batch_invariant_lib: torch.library.Library | None = None


def is_batch_invariant_mode_enabled() -> bool:
    """Return whether the batch-invariant ATen implementations are installed."""
    return _batch_invariant_lib is not None


def _batch_invariant_implementations():
    """Return the ATen names and deterministic CUDA implementations to install."""
    # Keep Triton out of the import path until the patch is actually enabled.
    from .triton import (
        _log_softmax_batch_invariant,
        addmm_batch_invariant,
        mean_batch_invariant,
        mm_batch_invariant,
    )

    return (
        ("aten::mm", mm_batch_invariant),
        ("aten::addmm", addmm_batch_invariant),
        ("aten::_log_softmax", _log_softmax_batch_invariant),
        ("aten::mean.dim", mean_batch_invariant),
    )


def enable_batch_invariant_mode() -> None:
    """Install batch-invariant implementations for the current accelerator."""
    global _batch_invariant_lib

    if _batch_invariant_lib is not None:
        return

    dispatch_key = getattr(torch.accelerator.current_accelerator(), "type", "cpu").upper()
    library = torch.library.Library("aten", "IMPL")
    try:
        for op_name, implementation in _batch_invariant_implementations():
            library.impl(op_name, implementation, dispatch_key)
    except BaseException:
        library._destroy()
        raise

    _batch_invariant_lib = library


def disable_batch_invariant_mode() -> None:
    """Remove the installed batch-invariant implementations, if any."""
    global _batch_invariant_lib

    library, _batch_invariant_lib = _batch_invariant_lib, None
    if library is not None:
        library._destroy()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True) -> Iterator[None]:
    """Temporarily set batch-invariant mode and restore its previous state.

    The mode is active only when CUDA is available. Restoration is
    exception-safe and supports nested enabled/disabled scopes.
    """
    restore_enabled = is_batch_invariant_mode_enabled()
    target_enabled = enabled and IS_CUDA_AVAILABLE

    if target_enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()

    try:
        yield
    finally:
        if restore_enabled:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()


__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]
