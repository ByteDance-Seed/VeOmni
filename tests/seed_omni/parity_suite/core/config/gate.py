"""Runtime gates for SeedOmni V2 parity cases."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .spec import PARITY_ENABLE_ENV


if TYPE_CHECKING:
    from .discovery import ParityCase


# Public gate evaluation -------------------------------------------------------


def case_skip_reason(case: ParityCase) -> str | None:
    """Return the reason a parity case should be skipped, if any."""

    reason = _static_skip_reason(case)
    if reason:
        return reason
    return _cuda_skip_reason(case)


# Internal gate checks ---------------------------------------------------------


def _static_skip_reason(case: ParityCase) -> str | None:
    gate = case.effective_gate
    if gate.requires_parity_env and os.environ.get(PARITY_ENABLE_ENV) != "1":
        return f"Set {PARITY_ENABLE_ENV}=1 to run {case.node_id}."
    if (
        gate.requires_reference_checkpoint
        and case.model.reference.hf_model is not None
        and case.model.reference.hf_model.checkpoint is not None
        and not case.model.reference.hf_model.checkpoint.exists()
    ):
        return f"Reference checkpoint does not exist: {case.model.reference.hf_model.checkpoint}"
    if gate.requires_v2_model and case.v2_model.model_root is not None and not case.v2_model.model_root.exists():
        return f"V2 model root does not exist: {case.v2_model.model_root}"
    return None


def _cuda_skip_reason(case: ParityCase) -> str | None:
    if not case.requires_cuda:
        return None

    import torch

    if not torch.cuda.is_available():
        return f"{case.node_id} requires CUDA."
    detected_cuda_devices = torch.cuda.device_count()
    if case.min_cuda_devices and detected_cuda_devices < case.min_cuda_devices:
        return f"{case.node_id} requires {case.min_cuda_devices} CUDA devices; detected {detected_cuda_devices}."
    return None
