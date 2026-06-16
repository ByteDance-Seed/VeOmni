"""V2 model and module loading hooks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import ParityCase
from tests.seed_omni.parity_suite.v2.model import (
    graph_active_module_names,
    load_graph_active_omni_config,
    load_graph_active_omni_modules,
)
from veomni.models.seed_omni.modeling_omni import OmniModel


class V2LoadingMixin:
    """Load graph-active V2 models and modules for parity execution."""

    case: ParityCase

    def load_v2_model(self, *, device: torch.device, dtype: torch.dtype) -> OmniModel:
        """Load the V2 model under test."""

        module_names = self.v2_module_names()
        config = load_graph_active_omni_config(self.case, module_names)
        modules = self.load_v2_modules(config.module_names, device=device, dtype=dtype)
        return OmniModel(config, modules).eval()

    def v2_module_names(self) -> frozenset[str]:
        """Return the complete module set referenced by the selected V2 graph."""

        return graph_active_module_names(self.case)

    def load_v2_modules(
        self,
        module_names: tuple[str, ...] | list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, nn.Module]:
        """Load graph-active V2 modules.

        Drivers with non-standard checkpoint layouts can override this hook
        while keeping the shared graph-driven config behavior.
        """

        return load_graph_active_omni_modules(self.case, module_names, device=device, dtype=dtype)
