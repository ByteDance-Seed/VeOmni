"""Default graph-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import autocast_for_dtype, zero_module_grads
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer


def run_v2_infer_graph(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run a V2 inference graph through ``OmniModel.generate``."""

    model = driver.load_v2_model(device=device, dtype=dtype)
    request = driver.v2_request_kwargs(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model, reference_output)
    trace: list[str] = []
    with torch.no_grad(), arm_generation_observer(whitelist) as observations:
        ctx = model.generate(request, trace=trace, generation_kwargs=dict(generation_kwargs))
    return {"observations": dict(observations), "ctx": ctx, "trace": trace}


def run_v2_train_graph(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run a V2 training graph through ``OmniModel.forward``."""

    return run_v2_train_graph_batch(
        driver,
        driver.v2_request_kwargs(reference_output, device=device),
        whitelist,
        device=device,
        dtype=dtype,
    )


def run_v2_train_graph_batch(
    driver: Any,
    batch_kwargs: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = driver.load_v2_model(device=device, dtype=dtype).train()
    zero_module_grads(model.modules_dict.values())
    batch = dict(batch_kwargs)
    with torch.enable_grad(), autocast_for_dtype(device, dtype):
        forward_result = model(**batch)
        loss = forward_result["loss"]
        if loss is None:
            raise RuntimeError(f"{type(driver).__name__} V2 train graph produced no loss.")
    loss.backward()
    observations = driver.collect_v2_train_observations(model, forward_result, whitelist, batch=batch)
    return {"observations": observations, "ctx": forward_result, "trace": ["train:graph"]}


__all__ = ["run_v2_infer_graph", "run_v2_train_graph", "run_v2_train_graph_batch"]
