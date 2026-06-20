"""Small oracle fixtures used by parity-suite oracle tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from tests.seed_omni.parity_suite.reference.capture import ReferenceCaptureContext
from tests.seed_omni.parity_suite.reference.contract import ReferenceRunResult


EVENTS: list[tuple[str, Any]] = []


class TinyModuleReference(nn.Module):
    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.seed = seed
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(float(seed))

    def run_reference_encode(
        self,
        inputs: Mapping[str, Any],
        context: ReferenceCaptureContext,
    ) -> ReferenceRunResult:
        EVENTS.append(("subject", "encode", context.ref_model is self))
        output = self.linear(inputs["x"])
        context.record_extra("extra", output + 1)
        return ReferenceRunResult(
            canonical={"x": inputs["x"]},
            observations={"subject_hidden": [output]},
            raw_output={"kind": "encode"},
        )


def reset_events() -> None:
    EVENTS.clear()


def load_tiny_module_reference(
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    options: Mapping[str, Any] | None = None,
) -> TinyModuleReference:
    EVENTS.append(("loader", seed, device.type, dtype, dict(options or {})))
    return TinyModuleReference(seed=seed).to(device=device, dtype=dtype)


def run_tiny_module_reference(
    ref_model: TinyModuleReference,
    inputs: Mapping[str, Any],
    context: ReferenceCaptureContext,
    *,
    kind: str | None,
    options: Mapping[str, Any],
) -> ReferenceRunResult:
    EVENTS.append(("runner", kind, dict(options), context.ref_model is ref_model))
    output = ref_model.linear(inputs["x"])
    context.record_extra("extra", output + 1)
    return ReferenceRunResult(
        canonical={"x": inputs["x"]},
        observations={"hidden": [output], "scale": [options["scale"]]},
        raw_output={"kind": kind},
    )


def tiny_extractor(context: ReferenceCaptureContext) -> torch.Tensor:
    return context.extras["extra"]
