"""Default module-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import RunCaptureOptions, autocast_for_dtype
from tests.seed_omni.parity_suite.v2.generation_runtime import (
    ModuleRuntime,
    cpu_origin_randn_for_reference_parity,
    materialize_for_device,
    run_generation_fsm,
)
from tests.seed_omni.parity_suite.v2.model import load_graph_active_omni_config
from veomni.models.seed_omni.generation_graph import GenerationGraph


def run_v2_infer_module(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> dict[str, Any]:
    """Run module-tier inference through eager modules and generation-graph automation."""

    config = load_graph_active_omni_config(driver.case, driver.v2_module_names())
    modules = driver.load_v2_modules(config.module_names, device=device, dtype=dtype)
    for module in modules.values():
        module.eval()
    driver.configure_determinism(driver.case.model.seed)
    model = ModuleRuntime(
        config=config,
        modules=modules,
        device=device,
        generation_graph=GenerationGraph(config.generation_graph) if config.has_generation_graph() else None,
    )
    request = driver.v2_request_kwargs(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model, reference_output)
    with torch.no_grad(), autocast_for_dtype(device, dtype), cpu_origin_randn_for_reference_parity():
        return run_generation_fsm(
            model,
            materialize_for_device(dict(request), device),
            whitelist,
            generation_kwargs=materialize_for_device(generation_kwargs, device),
            policy=driver.v2_infer_module_policy(reference_output, whitelist, capture_options=capture_options),
        )


__all__ = [
    "run_v2_infer_module",
]
