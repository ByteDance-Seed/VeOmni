"""Reference-side online capture helpers for the parity suite."""

from .capture import (
    ExtractorTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    ReferenceCaptureResult,
    ReferenceDriver,
    capture_reference_taps,
)
from .hooks import HookTap, capture_hook_taps, resolve_submodule
from .model import ParityReferenceModel, reference_options
from .tensors import DEFAULT_MAX_CAPTURE_TENSOR_NUMEL, materialize_reference_value


__all__ = [
    "DEFAULT_MAX_CAPTURE_TENSOR_NUMEL",
    "ExtractorTap",
    "HookTap",
    "ParityReferenceModel",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "ReferenceCaptureResult",
    "ReferenceDriver",
    "capture_hook_taps",
    "capture_reference_taps",
    "materialize_reference_value",
    "reference_options",
    "resolve_submodule",
]
