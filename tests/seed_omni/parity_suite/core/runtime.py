"""Generic runtime helpers for parity tests."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from veomni.utils.helper import enable_full_determinism


# Determinism ------------------------------------------------------------------


def configure_torch_determinism(seed: int, *, strict: bool = False) -> None:
    """Configure RNG and deterministic knobs for parity execution."""

    enable_full_determinism(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.set_float32_matmul_precision("highest")


@dataclass(frozen=True)
class RunCaptureOptions:
    max_tensor_numel: int = 1_000_000


@dataclass(frozen=True)
class RunWorkerOptions:
    debug_log: bool = False

    def env(self) -> dict[str, str]:
        return {
            "VEOMNI_PARITY_WORKER_DEBUG_LOG": str(self.debug_log).lower(),
        }


@contextmanager
def run_runtime_context(
    run_options: Mapping[str, Any] | None,
    *,
    sdpa_kernel_modules: Iterable[Any] = (),
) -> Iterator[None]:
    """Apply opt-in runtime settings for one parity run."""

    options = _runtime_options(run_options)
    if not options["deterministic_sdpa"]:
        yield
        return
    with _deterministic_sdpa_context(sdpa_kernel_modules=sdpa_kernel_modules):
        yield


@contextmanager
def run_capture_context(run_options: Mapping[str, Any] | None) -> Iterator[RunCaptureOptions]:
    """Resolve observation/capture settings for one parity run."""

    yield _capture_options(run_options)


@contextmanager
def run_worker_context(run_options: Mapping[str, Any] | None) -> Iterator[RunWorkerOptions]:
    """Resolve subprocess-worker settings for one parity run."""

    yield _worker_options(run_options)


def _runtime_options(run_options: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = run_options or {}
    if not isinstance(raw, Mapping):
        raise TypeError("run.options must be a mapping.")
    return {
        "deterministic_sdpa": bool(raw.get("deterministic_sdpa", False)),
    }


def _capture_options(run_options: Mapping[str, Any] | None) -> RunCaptureOptions:
    raw = run_options or {}
    if not isinstance(raw, Mapping):
        raise TypeError("run.options must be a mapping.")
    return RunCaptureOptions(max_tensor_numel=int(raw.get("max_tensor_numel", 1_000_000)))


def _worker_options(run_options: Mapping[str, Any] | None) -> RunWorkerOptions:
    raw = run_options or {}
    if not isinstance(raw, Mapping):
        raise TypeError("run.options must be a mapping.")
    return RunWorkerOptions(debug_log=bool(raw.get("debug_log", False)))


@contextmanager
def _deterministic_sdpa_context(*, sdpa_kernel_modules: Iterable[Any]) -> Iterator[None]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def math_sdpa_kernel(*args: Any, **kwargs: Any):
        del args, kwargs
        return sdpa_kernel(backends=[SDPBackend.MATH])

    modules = tuple(sdpa_kernel_modules)
    originals = tuple((module, module.sdpa_kernel) for module in modules if hasattr(module, "sdpa_kernel"))
    cuda_state: tuple[bool, bool, bool] | None = None
    if torch.cuda.is_available():
        cuda_state = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        for module, _original in originals:
            module.sdpa_kernel = math_sdpa_kernel
        yield
    finally:
        for module, original in originals:
            module.sdpa_kernel = original
        if cuda_state is not None:
            flash, mem_efficient, math = cuda_state
            torch.backends.cuda.enable_flash_sdp(flash)
            torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient)
            torch.backends.cuda.enable_math_sdp(math)


# Device and dtype helpers -----------------------------------------------------


def to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested value to ``device``."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def to_cpu(value: Any) -> Any:
    """Recursively detach tensors and materialize them on CPU."""

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def resolve_torch_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported reference dtype: {dtype!r}")


def autocast_for_dtype(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", enabled=True, dtype=dtype)


# Loss helpers -----------------------------------------------------------------


def sum_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    if not losses:
        return None
    values = iter(losses.values())
    total = next(values)
    for value in values:
        total = total + value
    return total


# Tensor and gradient sampling -------------------------------------------------


def sample_tensor(tensor: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    if rows is not None:
        return tensor.detach().cpu()[rows.detach().cpu().to(dtype=torch.long)]
    if tensor.dim() >= 2:
        return tensor.detach().cpu()[:4, :4]
    return tensor.detach().cpu()[:16]


def sample_grad(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    grad = param.grad
    if grad is None:
        raise RuntimeError(f"Expected gradient for {tuple(param.shape)}, got None.")
    return sample_tensor(grad, rows=rows)


def sample_named_grad(module: nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    return sample_grad(dict(module.named_parameters())[name], rows=rows)


def sample_named_param(module: nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    return sample_tensor(dict(module.named_parameters())[name], rows=rows)


def zero_module_grads(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


# Test-side patches ------------------------------------------------------------


@contextmanager
def patched_randn_like(fixed_noise: torch.Tensor) -> Iterator[None]:
    orig = torch.randn_like

    def fake_randn_like(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return fixed_noise.to(device=input_tensor.device, dtype=input_tensor.dtype)

    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.randn_like = orig


__all__ = [
    "RunCaptureOptions",
    "RunWorkerOptions",
    "autocast_for_dtype",
    "configure_torch_determinism",
    "patched_randn_like",
    "resolve_torch_dtype",
    "run_capture_context",
    "run_runtime_context",
    "run_worker_context",
    "sample_grad",
    "sample_named_grad",
    "sample_named_param",
    "sample_tensor",
    "sum_losses",
    "to_cpu",
    "to_device",
    "zero_module_grads",
]
