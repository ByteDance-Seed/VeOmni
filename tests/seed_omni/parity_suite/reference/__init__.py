"""Reference-side online capture helpers for the parity suite."""

from .capture import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    ExtractorTap,
    HookTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    ReferenceCaptureResult,
    ReferenceDriver,
    build_reference_capture_plan,
    capture_hook_taps,
    capture_reference_taps,
    materialize_reference_value,
    resolve_submodule,
)
from .contract import (
    ReferenceRunOutput,
    canonical_from_reference_output,
    is_reference_run_output,
    make_reference_run_output,
)
from .model import (
    ParityReferenceModel,
    empty_init_context,
    load_safetensors_weights,
    load_transformers_reference_model,
    reference_options,
)


__all__ = [
    "DEFAULT_MAX_CAPTURE_TENSOR_NUMEL",
    "ExtractorTap",
    "HookTap",
    "ParityReferenceModel",
    "empty_init_context",
    "load_safetensors_weights",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "ReferenceCaptureResult",
    "ReferenceDriver",
    "ReferenceRunOutput",
    "build_reference_capture_plan",
    "canonical_from_reference_output",
    "capture_hook_taps",
    "capture_reference_taps",
    "is_reference_run_output",
    "load_transformers_reference_model",
    "make_reference_run_output",
    "materialize_reference_value",
    "reference_options",
    "resolve_submodule",
]
