"""Base driver contract for model-specific parity execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityCase, configure_torch_determinism
from tests.seed_omni.parity_suite.driver.observations import TrainObservationMixin
from tests.seed_omni.parity_suite.driver.reference import ReferenceMixin
from tests.seed_omni.parity_suite.driver.requests import RequestDispatchMixin
from tests.seed_omni.parity_suite.driver.runtime import DriverRuntimeMixin
from tests.seed_omni.parity_suite.driver.v2_loading import V2LoadingMixin


# Driver composition -----------------------------------------------------------


class ParityDriver(
    ReferenceMixin,
    V2LoadingMixin,
    RequestDispatchMixin,
    DriverRuntimeMixin,
    TrainObservationMixin,
):
    """Model-specific execution contract used by the shared parity runner.

    Reference handlers used with ``ref_tap: {output: ...}`` should return
    ``{"canonical": ..., "reference": ...}``. Shared tier runners pass that
    canonical payload to ``v2_request_kwargs()`` for both inference and training.
    """

    # Shared generation defaults -------------------------------------------------

    generation_defaults: Mapping[str, Any] = {
        "max_new_tokens": 1,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    # Lifecycle and runtime policy ----------------------------------------------

    def __init__(self, case: ParityCase) -> None:
        self.case = case

    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def configure_determinism(self, seed: int) -> None:
        configure_torch_determinism(seed)
