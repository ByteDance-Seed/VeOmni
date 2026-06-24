"""Driver runtime extension hooks."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch

from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext
from veomni.models.seed_omni.modeling_omni import OmniModel


class DriverRuntimeMixin:
    """Optional hooks used by shared V2 tier runners."""

    def runtime_sdpa_kernel_modules(self) -> tuple[Any, ...]:
        """Return modules whose local ``sdpa_kernel`` import should follow run runtime options.

        Some model files import ``sdpa_kernel`` as a module global. The suite
        patches those globals only while a deterministic-SDPA run is active. A
        driver may return both reference and V2 modules here; patching a module
        that is not active in the current phase is harmless.
        """

        return ()

    def v2_parameter_samples(
        self,
        ctx: V2RunContext,
        model: OmniModel,
        sample_context: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del ctx, model, sample_context
        return {}

    def v2_parameter_sample_context(
        self,
        ctx: V2RunContext,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del ctx
        return batch

    def v2_execution_context(
        self,
        ctx: V2RunContext,
        *,
        model: Any | None = None,
        batch: Mapping[str, Any] | None = None,
    ):
        del ctx, model, batch
        return nullcontext()
