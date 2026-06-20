"""V2 tier runner dispatch wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import ParityReport
from veomni.models.seed_omni.modeling_omni import OmniModel


class TierRunnerMixin:
    """Dispatch parity execution to shared graph, module, and framework runners."""

    # Graph tier ------------------------------------------------------------------

    def run_v2_infer_graph_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.graph import run_v2_infer_graph

        return run_v2_infer_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_graph_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 training graph through the shared graph dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.graph import run_v2_train_graph

        return run_v2_train_graph(self, reference_output, whitelist, device=device, dtype=dtype)

    # Module tier -----------------------------------------------------------------

    def run_v2_infer_module_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run a V2 inference graph through the shared module dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.module import run_v2_infer_module

        return run_v2_infer_module(self, reference_output, whitelist, device=device, dtype=dtype)

    # Framework tier --------------------------------------------------------------

    def run_v2_infer_framework_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run an inference framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.framework import run_v2_infer_framework

        return run_v2_infer_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    def run_v2_train_framework_recipe(
        self,
        reference_output: Any,
        whitelist: Mapping[tuple[str, str], frozenset[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any] | ParityReport:
        """Run a training framework-tier case through the shared framework dispatcher."""

        from tests.seed_omni.parity_suite.v2.tier_runners.framework import run_v2_train_framework

        return run_v2_train_framework(self, reference_output, whitelist, device=device, dtype=dtype)

    # Driver extension hooks ------------------------------------------------------

    def sample_v2_framework_parameters(
        self,
        model: OmniModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        del model, batch
        return {}
