"""Default graph-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from tests.seed_omni.parity_suite.core import RunCaptureOptions, autocast_for_dtype, zero_module_grads
from tests.seed_omni.parity_suite.driver.v2_run import V2RunContext, canonical_from_reference_output
from tests.seed_omni.parity_suite.v2.cpu_preprocess import apply_training_cpu_preprocessors
from tests.seed_omni.parity_suite.v2.generation_runtime import cpu_origin_randn_for_reference_parity
from tests.seed_omni.parity_suite.v2.observation import arm_generation_observer


def run_v2_infer_graph(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> dict[str, Any]:
    """Run a V2 inference graph through ``OmniModel.generate``."""

    run_ctx = _v2_run_context(
        driver,
        reference_output,
        whitelist,
        device=device,
        dtype=dtype,
        capture_options=capture_options,
    )
    model = driver.load_v2_model(device=device, dtype=dtype)
    driver.configure_determinism(driver.case.model.seed)
    request = driver.build_v2_request(run_ctx)
    generation_kwargs = driver.v2_generation_kwargs(run_ctx, model)
    trace: list[str] = []
    with (
        torch.no_grad(),
        autocast_for_dtype(device, dtype),
        cpu_origin_randn_for_reference_parity(),
        driver.v2_execution_context(run_ctx, model=model, batch=request),
        arm_generation_observer(whitelist, max_tensor_numel=capture_options.max_tensor_numel) as observations,
    ):
        ctx = model.generate(request, trace=trace, generation_kwargs=dict(generation_kwargs))
    return {"observations": dict(observations), "ctx": ctx, "trace": trace}


def run_v2_train_graph(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> dict[str, Any]:
    """Run a V2 training graph through ``OmniModel.forward``."""

    run_ctx = _v2_run_context(
        driver,
        reference_output,
        whitelist,
        device=device,
        dtype=dtype,
        capture_options=capture_options,
    )
    return _run_v2_train_graph_batch(
        driver,
        driver.build_v2_request(run_ctx),
        whitelist,
        run_ctx=run_ctx,
        device=device,
        dtype=dtype,
    )


def _run_v2_train_graph_batch(
    driver: Any,
    batch_kwargs: Mapping[str, Any],
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    run_ctx: V2RunContext,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    del whitelist
    model = driver.load_v2_model(device=device, dtype=dtype).train()
    driver.configure_determinism(driver.case.model.seed)
    zero_module_grads(model.modules_dict.values())
    batch = dict(batch_kwargs)
    apply_training_cpu_preprocessors(model, batch)
    with (
        torch.enable_grad(),
        autocast_for_dtype(device, dtype),
        driver.v2_execution_context(run_ctx, model=model, batch=batch),
    ):
        forward_result = model(**batch)
        loss = forward_result["loss"]
        if loss is None:
            raise RuntimeError(f"{type(driver).__name__} V2 train graph produced no loss.")
    loss.backward()
    observations = driver.collect_v2_observations(run_ctx, model=model, result=forward_result, batch=batch)
    return {"observations": observations, "ctx": forward_result, "trace": ["train:graph"]}


def _v2_run_context(
    driver: Any,
    reference_output: Any,
    whitelist: Mapping[tuple[str, str], frozenset[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    capture_options: RunCaptureOptions,
) -> V2RunContext:
    return V2RunContext(
        case=driver.case,
        tier=driver.case.tier,
        domain=driver.case.graph.domain,
        reference_output=reference_output,
        canonical=canonical_from_reference_output(reference_output),
        whitelist=whitelist,
        device=device,
        dtype=dtype,
        capture_options=capture_options,
    )


__all__ = ["run_v2_infer_graph", "run_v2_train_graph"]
