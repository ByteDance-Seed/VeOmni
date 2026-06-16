"""Reference model loading and recipe execution hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.core import ParityCase, ParityReport
from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.model import load_transformers_reference_model, reference_options


class ReferenceMixin:
    """Reference oracle inputs, loading, and recipe execution."""

    case: ParityCase

    def reference_inputs(self) -> Mapping[str, Any]:
        return self.case.recipe.stimulus

    def generation_kwargs(self, model_or_config: Any, reference_output: Any) -> dict[str, Any]:
        del reference_output
        config = getattr(model_or_config, "config", model_or_config)
        kwargs = dict(getattr(config, "generation_kwargs", None) or {})
        for key, default in self.generation_defaults.items():
            kwargs[key] = default
        kwargs.update(self.case.recipe.stimulus)
        return kwargs

    def reference_model_load_kwargs(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        """Return model-specific kwargs for ``AutoModel.from_pretrained``."""

        del device, dtype
        return {}

    def load_reference_model(self, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Load the independent reference oracle."""

        return load_transformers_reference_model(
            module=self.case.model.reference.module,
            checkpoint=self.case.model.reference.checkpoint,
            **self.reference_model_load_kwargs(device=device, dtype=dtype),
        )

    def run_reference_recipe(
        self,
        ref_model: nn.Module,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> Any:
        """Run the reference recipe and return driver-owned canonical output."""

        reference = self.case.recipe.reference
        kind = reference.get("kind")
        if kind is not None:
            kind = str(kind)
        options = reference.get("options", {}) or {}
        with reference_options(ref_model, options):
            run_reference_kind = getattr(ref_model, "run_reference_kind", None)
            if run_reference_kind is not None:
                return run_reference_kind(kind, inputs, context)
            if kind is None:
                return ref_model(**inputs)
        raise NotImplementedError(f"{type(ref_model).__name__} does not implement reference kind {kind!r}.")

    def run_reference_only_recipe(self) -> ParityReport:
        """Run a reference-only recipe."""

        raise NotImplementedError(f"{type(self).__name__} does not implement reference-only execution.")
