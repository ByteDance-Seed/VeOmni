"""Thin V2 graph runner used by model adapters."""

from __future__ import annotations

from typing import Any

from veomni.models.seed_omni.modeling_omni import OmniModel


def run_training_graph(model: OmniModel, batch: dict[str, Any], *, trace: list[str] | None = None) -> dict[str, Any]:
    return model(trace=trace, **batch)


def run_generation_graph(
    model: OmniModel,
    request: dict[str, Any],
    *,
    generation_kwargs: dict[str, Any] | None = None,
    trace: list[str] | None = None,
) -> dict[str, Any]:
    return model.generate(request, generation_kwargs=generation_kwargs, trace=trace)
