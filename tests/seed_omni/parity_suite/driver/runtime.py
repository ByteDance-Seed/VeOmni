"""Driver runtime extension hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

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

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del model, batch
        return {}
