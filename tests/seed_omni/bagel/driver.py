from __future__ import annotations

from typing import Any

import torch

from tests.seed_omni.parity_suite.core import effective_reference_kind
from tests.seed_omni.parity_suite.driver import ParityDriver


class BagelParityDriver(ParityDriver):
    def v2_request_kwargs(self, reference_output: Any, *, device: torch.device) -> dict[str, Any]:
        if (
            self.case.recipe.reference.get("oracle") == "hf_module.flow_connector_decode"
            and effective_reference_kind(self.case) == "infer_gen"
        ):
            canonical = {} if reference_output is None else dict(reference_output.canonical)
            try:
                hidden_states = canonical["denoise_hidden"][-1]
            except (KeyError, IndexError) as exc:
                raise KeyError(
                    "BAGEL flow_connector_decode infer_gen module parity requires reference denoise_hidden "
                    "as the single-module decode_velocity input."
                ) from exc
            if hidden_states.ndim == 2 and hidden_states.shape[0] >= 3:
                hidden_states = hidden_states[1:-1]
            return {"hidden_states": hidden_states.to(device=device)}
        return super().v2_request_kwargs(reference_output, device=device)


def create_driver(case) -> BagelParityDriver:
    return BagelParityDriver(case)
