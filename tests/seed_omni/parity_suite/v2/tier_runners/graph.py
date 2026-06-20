"""Default graph-tier execution for SeedOmni V2 parity."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
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
    driver.configure_determinism(driver.case.model.seed)
    request = driver.v2_request_kwargs(reference_output, device=device)
    generation_kwargs = driver.generation_kwargs(model, reference_output)
    max_tensor_numel = int(driver.case.run.options.get("max_tensor_numel", 1_000_000))
    trace: list[str] = []
    with (
        torch.no_grad(),
        autocast_for_dtype(device, dtype),
        _cpu_origin_randn_for_reference_parity(),
        arm_generation_observer(whitelist, max_tensor_numel=max_tensor_numel) as observations,
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
) -> dict[str, Any]:
    """Run a V2 training graph through ``OmniModel.forward``."""

    return _run_v2_train_graph_batch(
        driver,
        driver.v2_request_kwargs(reference_output, device=device),
        whitelist,
        device=device,
        dtype=dtype,
    )


def _run_v2_train_graph_batch(
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
    observations = driver.collect_v2_train_graph_observations(model, forward_result, whitelist, batch=batch)
    return {"observations": observations, "ctx": forward_result, "trace": ["train:graph"]}


@contextmanager
def _cpu_origin_randn_for_reference_parity():
    """Match official references that sample generation noise on CPU first."""

    original_randn = torch.randn
    original_randn_like = torch.randn_like

    def randn_cpu_origin(*args: Any, **kwargs: Any) -> torch.Tensor:
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        out = kwargs.get("out")
        if out is not None:
            return original_randn(*args, **kwargs)
        tensor = original_randn(*args, **kwargs)
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    def randn_like_cpu_origin(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        device = kwargs.pop("device", input_tensor.device)
        dtype = kwargs.pop("dtype", input_tensor.dtype)
        out = kwargs.get("out")
        if out is not None:
            return original_randn_like(input_tensor, *args, **kwargs)
        cpu_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        tensor = original_randn_like(input_tensor.detach().cpu().to(dtype=cpu_dtype), *args, **kwargs)
        return tensor.to(device=device, dtype=dtype)

    torch.randn = randn_cpu_origin
    torch.randn_like = randn_like_cpu_origin
    try:
        yield
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like


__all__ = ["run_v2_infer_graph", "run_v2_train_graph"]
