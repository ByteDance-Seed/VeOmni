"""Common parity metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_a": list(actual.shape),
            "shape_b": list(expected.shape),
            "shape_match": False,
            "passes": False,
        }

    a_float = actual.detach().float()
    b_float = expected.detach().float()
    diff = (a_float - b_float).abs()
    actual_norm = float(a_float.norm().item())
    expected_norm = float(b_float.norm().item())
    cosine = 1.0
    if actual.numel() > 0 and actual_norm > 0 and expected_norm > 0:
        cosine = float(F.cosine_similarity(a_float.reshape(1, -1), b_float.reshape(1, -1), dim=-1).item())
    return {
        "shape": list(actual.shape),
        "dtype_a": str(actual.dtype),
        "dtype_b": str(expected.dtype),
        "shape_match": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
        "relative_l2": float((a_float - b_float).norm().item() / max(expected_norm, 1e-12)),
        "cosine_similarity": cosine,
    }


def tensor_passes(metrics: dict[str, Any], tolerance: dict[str, float]) -> bool:
    if not metrics.get("shape_match"):
        return False
    near_zero_norm = tolerance.get("near_zero_norm", 0.0)
    if metrics["actual_norm"] <= near_zero_norm and metrics["expected_norm"] <= near_zero_norm:
        return bool(
            metrics["max_abs_diff"] <= tolerance.get("max_abs_diff", 0.0)
            and metrics["mean_abs_diff"] <= tolerance.get("mean_abs_diff", 0.0)
        )
    return bool(
        (
            metrics["max_abs_diff"] <= tolerance.get("max_abs_diff", 0.0)
            and metrics["mean_abs_diff"] <= tolerance.get("mean_abs_diff", 0.0)
            and metrics["cosine_similarity"] >= tolerance.get("cosine_similarity_min", -1.0)
        )
        or (
            metrics["relative_l2"] <= tolerance.get("relative_l2_max", 0.0)
            and metrics["cosine_similarity"] >= tolerance.get("cosine_similarity_min", -1.0)
        )
    )


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor, tolerance: dict[str, float]) -> dict[str, Any]:
    metrics = tensor_metrics(actual, expected)
    metrics["passes"] = tensor_passes(metrics, tolerance)
    return metrics
