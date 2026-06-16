"""Generic runtime helpers for parity tests."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch
from torch import nn


def configure_torch_determinism(seed: int, *, strict: bool = False) -> None:
    """Configure PyTorch RNG and deterministic knobs for parity execution."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.set_float32_matmul_precision("highest")


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


def sum_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    if not losses:
        return None
    values = iter(losses.values())
    total = next(values)
    for value in values:
        total = total + value
    return total


def sample_grad(param: torch.nn.Parameter, rows: torch.Tensor | None = None) -> torch.Tensor:
    grad = param.grad
    if grad is None:
        raise RuntimeError(f"Expected gradient for {tuple(param.shape)}, got None.")
    if rows is not None:
        return grad.detach().cpu()[rows.detach().cpu().to(dtype=torch.long)]
    if grad.dim() >= 2:
        return grad.detach().cpu()[:4, :4]
    return grad.detach().cpu()[:16]


def sample_named_grad(module: nn.Module, name: str, rows: torch.Tensor | None = None) -> torch.Tensor:
    return sample_grad(dict(module.named_parameters())[name], rows=rows)


def zero_module_grads(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


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
    "autocast_for_dtype",
    "configure_torch_determinism",
    "patched_randn_like",
    "resolve_torch_dtype",
    "sample_grad",
    "sample_named_grad",
    "sum_losses",
    "to_cpu",
    "to_device",
    "zero_module_grads",
]
