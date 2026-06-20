"""Reference-side capture helpers for the parity suite."""

from .observation import (
    MethodPatchObservationCapture,
    NullReferenceObservationCapture,
    ReferenceObservationCapture,
    ReferenceObservationSatisfied,
    ReferenceStopPolicy,
    reference_observation_stop_policy,
)
from .runtime import (
    DEFAULT_MAX_CAPTURE_TENSOR_NUMEL,
    _empty_cache,
    capture_hook_taps,
    materialize_reference_value,
    resolve_submodule,
)
from .spec import (
    ExtractorTap,
    FieldTap,
    HookTap,
    ReferenceCaptureContext,
    ReferenceCapturePlan,
    build_reference_capture_plan,
)


__all__ = [
    "DEFAULT_MAX_CAPTURE_TENSOR_NUMEL",
    "ExtractorTap",
    "FieldTap",
    "HookTap",
    "MethodPatchObservationCapture",
    "NullReferenceObservationCapture",
    "ReferenceObservationSatisfied",
    "ReferenceCaptureContext",
    "ReferenceCapturePlan",
    "ReferenceObservationCapture",
    "ReferenceStopPolicy",
    "_empty_cache",
    "build_reference_capture_plan",
    "capture_hook_taps",
    "materialize_reference_value",
    "reference_observation_stop_policy",
    "resolve_submodule",
]
