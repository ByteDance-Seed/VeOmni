"""Reference model loading and recipe execution hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.seed_omni.parity_suite.core import ParityCase, ParityReport
from tests.seed_omni.parity_suite.core.stimulus import conversation_stimulus_to_batched_specs
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext
from tests.seed_omni.parity_suite.reference.contract import ReferenceOracle
from tests.seed_omni.parity_suite.reference.oracles.factory import build_reference_oracle


class ReferenceMixin:
    """Reference oracle inputs and selection."""

    case: ParityCase

    # Reference inputs and generation --------------------------------------------

    def reference_inputs(self) -> Mapping[str, Any]:
        stimulus = self.case.recipe.stimulus
        batched_conversation = conversation_stimulus_to_batched_specs(stimulus)
        if batched_conversation is None:
            return stimulus
        inputs = dict(stimulus)
        inputs.pop("batched_conversation_list", None)
        inputs["conversation_list"] = batched_conversation
        return inputs

    def v2_generation_kwargs(self, ctx: V2RunContext, model_or_config: Any) -> dict[str, Any]:
        del ctx
        config = getattr(model_or_config, "config", model_or_config)
        kwargs = dict(getattr(config, "generation_kwargs", None) or {})
        for key, default in self.generation_defaults.items():
            kwargs[key] = default
        stimulus_kwargs = self.case.recipe.stimulus.get("generation_kwargs", {})
        if stimulus_kwargs:
            if not isinstance(stimulus_kwargs, Mapping):
                raise TypeError("stimulus.generation_kwargs must be a mapping.")
            kwargs.update(stimulus_kwargs)
        return kwargs

    # Reference oracle ------------------------------------------------------------

    def reference_oracle(self) -> ReferenceOracle:
        """Return the configured independent reference oracle."""

        return build_reference_oracle(self.case, self)

    def run_reference_only_recipe(self) -> ParityReport:
        """Run a reference-only recipe."""

        raise NotImplementedError(f"{type(self).__name__} does not implement reference-only execution.")
