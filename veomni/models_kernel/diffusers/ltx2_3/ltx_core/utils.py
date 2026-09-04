"""Adapted from https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/utils.py"""

from pathlib import Path
from typing import Any

import torch

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import resolve_kernel_impl


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.

    Always calls the interned ``VeomniKernel`` handle. Missing weight uses
    the ``unweighted`` variant.
    """
    impl = resolve_kernel_impl("rms_norm_implementation")
    if weight is None:
        return VeomniKernel("rms_norm", "unweighted", impl)(x, eps=eps)
    return VeomniKernel("rms_norm", "standard", impl)(x, weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


def find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.
    If root_path is already a file matching the pattern, return it directly.
    """
    p = Path(root_path)
    if p.is_file() and p.match(pattern):
        return p
    matches = list(p.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]
