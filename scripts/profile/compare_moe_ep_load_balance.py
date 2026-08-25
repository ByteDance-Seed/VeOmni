#!/usr/bin/env python3
"""Compare matched MoE EP load-balance training records.

The reporter deliberately uses only the Python standard library so saved
training artifacts can be checked without importing VeOmni or an accelerator
runtime. It never infers absent performance or memory measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_CURVES = ("loss", "grad_norm")
STEP_PERFORMANCE_CURVES = (
    "step_time_s",
    "step_tokens",
    "tokens_per_second",
    "p2p_bytes",
    "p2p_wait_time_s",
)
PEAK_MEMORY_FIELDS = ("peak_accelerator_memory_bytes", "peak_host_memory_bytes")
CRITICAL_METADATA_FIELDS = frozenset(
    {
        "model",
        "model_name",
        "model_config",
        "model_revision",
        "checkpoint",
        "checkpoint_id",
        "dataset",
        "dataset_revision",
        "seed",
        "dtype",
        "backend",
        "world_size",
        "fsdp_mode",
        "sp_size",
        "ep_size",
        "global_batch_size",
        "micro_batch_size",
        "max_seq_len",
        "gradient_checkpointing",
        "optimizer",
        "scheduler",
    }
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LOSS_PATTERN = re.compile(rf"(?<![\w/])(?:total_)?loss\s*[:=]\s*({_NUMBER})(?![\w.])", re.IGNORECASE)
_GRAD_PATTERN = re.compile(
    rf"(?<![\w/])(?:grad_norm|gradient_norm)\s*[:=]\s*({_NUMBER})(?![\w.])",
    re.IGNORECASE,
)


def _validate_schema_version(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ValueError(f"{field} must be integer {SCHEMA_VERSION}.")


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} contains a bool; expected a finite numeric value.")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must contain only numeric values.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must contain only finite values.")
    return result


def _numeric_curve(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a numeric array.")
    if not value:
        raise ValueError(f"{field} must not be empty.")
    return [_finite_number(item, field=field) for item in value]


def _validate_nonnegative(values: list[float], *, field: str, positive: bool = False) -> None:
    invalid = any(value <= 0.0 for value in values) if positive else any(value < 0.0 for value in values)
    if invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must contain only {qualifier} values.")


def normalize_run(payload: Any, *, source: str = "input") -> dict[str, Any]:
    """Normalize a canonical envelope or flat log dictionary."""
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: top-level input must be a JSON object.")

    if "schema_version" in payload:
        _validate_schema_version(payload.get("schema_version"), field=f"{source}: schema_version")
        metadata = payload.get("metadata")
        metrics = payload.get("metrics")
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}: metadata must be an object.")
        if not isinstance(metrics, dict):
            raise ValueError(f"{source}: metrics must be an object.")
    else:
        metadata = {}
        metrics = payload

    normalized_metrics: dict[str, Any] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source}: metric names must be non-empty strings.")
        field = f"{source}: metric {name!r}"
        if name in REQUIRED_CURVES or name in STEP_PERFORMANCE_CURVES:
            curve = _numeric_curve(value, field=field)
            if name == "step_time_s":
                _validate_nonnegative(curve, field=field, positive=True)
            elif name in {"step_tokens", "tokens_per_second", "p2p_bytes", "p2p_wait_time_s"}:
                _validate_nonnegative(curve, field=field)
            normalized_metrics[name] = curve
        elif name in PEAK_MEMORY_FIELDS:
            if isinstance(value, (list, tuple)):
                curve = _numeric_curve(value, field=field)
                _validate_nonnegative(curve, field=field)
                normalized_metrics[name] = curve
            else:
                number = _finite_number(value, field=field)
                _validate_nonnegative([number], field=field)
                normalized_metrics[name] = number
        elif isinstance(value, (list, tuple)):
            normalized_metrics[name] = _numeric_curve(value, field=field)
        else:
            normalized_metrics[name] = _finite_number(value, field=field)

    for required in REQUIRED_CURVES:
        if required not in normalized_metrics:
            raise ValueError(f"{source}: required metric {required!r} is absent.")
    if len(normalized_metrics["loss"]) != len(normalized_metrics["grad_norm"]):
        raise ValueError(f"{source}: loss and grad_norm must have equal lengths.")
    step_count = len(normalized_metrics["loss"])
    for name in STEP_PERFORMANCE_CURVES:
        if name in normalized_metrics and len(normalized_metrics[name]) != step_count:
            raise ValueError(
                f"{source}: {name} must have the same number of steps as loss and grad_norm ({step_count})."
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "metrics": normalized_metrics,
    }


def _jsonl_payload(lines: list[str], *, source: str) -> dict[str, Any] | None:
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: each JSONL step must be an object.")
        rows.append((line_number, row))
    if not rows:
        return None

    metadata: dict[str, Any] = {}
    metadata_seen = False
    expected_metric_names: set[str] | None = None
    collected: dict[str, list[Any]] = {}
    for line_number, row in rows:
        if "metadata" in row:
            row_metadata = row["metadata"]
            if not isinstance(row_metadata, dict):
                raise ValueError(f"{source}:{line_number}: metadata must be an object.")
            if metadata_seen and row_metadata != metadata:
                raise ValueError(f"{source}:{line_number}: metadata changes between JSONL steps.")
            metadata = dict(row_metadata)
            metadata_seen = True

        if "schema_version" in row:
            _validate_schema_version(row["schema_version"], field=f"{source}:{line_number}: schema_version")
        if "metrics" in row:
            step_metrics = row["metrics"]
            if not isinstance(step_metrics, dict):
                raise ValueError(f"{source}:{line_number}: metrics must be an object.")
        else:
            step_metrics = {key: value for key, value in row.items() if key not in {"metadata", "schema_version"}}
        metric_names = set(step_metrics)
        if expected_metric_names is None:
            expected_metric_names = metric_names
            collected = {name: [] for name in sorted(metric_names)}
        elif metric_names != expected_metric_names:
            missing = sorted(expected_metric_names - metric_names)
            added = sorted(metric_names - expected_metric_names)
            if missing:
                raise ValueError(f"{source}:{line_number}: {missing[0]} is missing from JSONL step.")
            raise ValueError(f"{source}:{line_number}: unexpected JSONL metric {added[0]!r}.")
        for name in sorted(metric_names):
            collected[name].append(step_metrics[name])

    return {"schema_version": SCHEMA_VERSION, "metadata": metadata, "metrics": collected}


def _plain_log_payload(lines: list[str]) -> dict[str, Any]:
    loss = []
    grad_norm = []
    for line in lines:
        loss_match = _LOSS_PATTERN.search(line)
        grad_match = _GRAD_PATTERN.search(line)
        if loss_match is not None:
            loss.append(float(loss_match.group(1)))
        if grad_match is not None:
            grad_norm.append(float(grad_match.group(1)))
    metrics = {}
    if loss:
        metrics["loss"] = loss
    if grad_norm:
        metrics["grad_norm"] = grad_norm
    return {"schema_version": SCHEMA_VERSION, "metadata": {}, "metrics": metrics}


def load_run(path: str | Path) -> dict[str, Any]:
    """Load canonical JSON, flat JSON, per-step JSONL, or a plain training log."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{input_path}: input is empty.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        payload = _jsonl_payload(lines, source=str(input_path))
        if payload is None:
            payload = _plain_log_payload(lines)
    return normalize_run(payload, source=str(input_path))


def _validate_comparison_number(value: Any, *, name: str, positive: bool = False) -> float:
    result = _finite_number(value, field=name)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    if not positive and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _pearson(baseline: list[float], candidate: list[float]) -> tuple[float | None, str | None]:
    if len(baseline) < 2:
        return None, "fewer than two points"
    baseline_mean = sum(baseline) / len(baseline)
    candidate_mean = sum(candidate) / len(candidate)
    baseline_delta = [value - baseline_mean for value in baseline]
    candidate_delta = [value - candidate_mean for value in candidate]
    baseline_ss = sum(value * value for value in baseline_delta)
    candidate_ss = sum(value * value for value in candidate_delta)
    if baseline_ss == 0.0:
        return None, "baseline has zero variance"
    if candidate_ss == 0.0:
        return None, "candidate has zero variance"
    covariance = sum(left * right for left, right in zip(baseline_delta, candidate_delta, strict=True))
    return covariance / math.sqrt(baseline_ss * candidate_ss), None


def _precision_metric(
    baseline: list[float],
    candidate: list[float],
    *,
    rtol: float,
    atol: float,
    relative_error_threshold: float,
    epsilon: float,
) -> dict[str, Any]:
    absolute_errors = [abs(right - left) for left, right in zip(baseline, candidate, strict=True)]
    relative_errors = [error / max(abs(left), epsilon) for left, error in zip(baseline, absolute_errors, strict=True)]
    close_hits = [error <= atol + rtol * abs(left) for left, error in zip(baseline, absolute_errors, strict=True)]
    relative_hits = [error <= relative_error_threshold for error in relative_errors]
    correlation, correlation_reason = _pearson(baseline, candidate)
    return {
        "points": len(baseline),
        "max_absolute_error": max(absolute_errors),
        "max_relative_error": max(relative_errors),
        "repository_close_hit_rate": sum(close_hits) / len(close_hits),
        "threshold_relative_hit_rate": sum(relative_hits) / len(relative_hits),
        "pearson_correlation": correlation,
        "correlation_unavailable_reason": correlation_reason,
    }


def _type_sensitive_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_type_sensitive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _type_sensitive_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _metadata_validation(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_metadata = baseline["metadata"]
    candidate_metadata = candidate["metadata"]
    compared = sorted(CRITICAL_METADATA_FIELDS & baseline_metadata.keys() & candidate_metadata.keys())
    mismatches = [
        name for name in compared if not _type_sensitive_equal(baseline_metadata[name], candidate_metadata[name])
    ]
    if mismatches:
        details = ", ".join(
            f"{name} ({baseline_metadata[name]!r} != {candidate_metadata[name]!r})" for name in mismatches
        )
        raise ValueError(f"critical metadata mismatch: {details}.")
    return {"status": "pass", "compared_fields": compared}


def _missing_metric_reason(metric: str, baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> str:
    baseline_missing = metric not in baseline_metrics
    candidate_missing = metric not in candidate_metrics
    if baseline_missing and candidate_missing:
        return f"{metric} missing on both baseline and candidate"
    if baseline_missing:
        return f"{metric} missing on baseline"
    return f"{metric} missing on candidate"


def _unavailable_timing(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "baseline": None,
        "candidate": None,
        "steady_delta_s": None,
        "steady_speedup": None,
    }


def _unavailable_throughput(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "baseline_steady_tokens_per_second": None,
        "candidate_steady_tokens_per_second": None,
        "steady_delta_tokens_per_second": None,
        "steady_speedup": None,
    }


def _unavailable_memory(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "baseline_peak_bytes": None,
        "candidate_peak_bytes": None,
        "delta_bytes": None,
        "baseline_over_candidate_ratio": None,
    }


def _unavailable_p2p_bytes(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "baseline": None,
        "candidate": None,
        "steady_total_delta_bytes": None,
        "steady_total_ratio": None,
        "steady_mean_delta_bytes": None,
        "steady_mean_ratio": None,
    }


def _matched_curve(
    name: str,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    *,
    warmup_steps: int,
) -> tuple[list[float], list[float]] | None:
    if name not in baseline_metrics or name not in candidate_metrics:
        return None
    baseline = baseline_metrics[name]
    candidate = candidate_metrics[name]
    if len(baseline) != len(candidate):
        raise ValueError(f"baseline and candidate {name} curves must have equal lengths.")
    if warmup_steps >= len(baseline):
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must leave at least one steady {name} point out of {len(baseline)}."
        )
    return baseline, candidate


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0.0 else None


def _timing_report(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], *, warmup_steps: int
) -> dict[str, Any]:
    curves = _matched_curve("step_time_s", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    if curves is None:
        return _unavailable_timing(_missing_metric_reason("step_time_s", baseline_metrics, candidate_metrics))
    baseline, candidate = curves
    baseline_steady = sum(baseline[warmup_steps:]) / len(baseline[warmup_steps:])
    candidate_steady = sum(candidate[warmup_steps:]) / len(candidate[warmup_steps:])
    baseline_warmup = sum(baseline[:warmup_steps]) / warmup_steps if warmup_steps else None
    candidate_warmup = sum(candidate[:warmup_steps]) / warmup_steps if warmup_steps else None
    return {
        "status": "available",
        "reason": None,
        "baseline": {"warmup_mean_s": baseline_warmup, "steady_mean_s": baseline_steady},
        "candidate": {"warmup_mean_s": candidate_warmup, "steady_mean_s": candidate_steady},
        "steady_delta_s": candidate_steady - baseline_steady,
        "steady_speedup": _ratio(baseline_steady, candidate_steady),
    }


def _explicit_throughput_report(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], *, warmup_steps: int
) -> dict[str, Any]:
    curves = _matched_curve("tokens_per_second", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    if curves is None:
        return _unavailable_throughput(
            _missing_metric_reason("tokens_per_second", baseline_metrics, candidate_metrics)
        )
    baseline, candidate = curves
    baseline_steady = sum(baseline[warmup_steps:]) / len(baseline[warmup_steps:])
    candidate_steady = sum(candidate[warmup_steps:]) / len(candidate[warmup_steps:])
    return {
        "status": "available",
        "reason": None,
        "baseline_steady_tokens_per_second": baseline_steady,
        "candidate_steady_tokens_per_second": candidate_steady,
        "steady_delta_tokens_per_second": candidate_steady - baseline_steady,
        "steady_speedup": _ratio(candidate_steady, baseline_steady),
    }


def _p2p_bytes_report(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], *, warmup_steps: int
) -> dict[str, Any]:
    curves = _matched_curve("p2p_bytes", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    if curves is None:
        return _unavailable_p2p_bytes(_missing_metric_reason("p2p_bytes", baseline_metrics, candidate_metrics))
    baseline, candidate = curves

    def summarize(values: list[float]) -> dict[str, float | None]:
        warmup = values[:warmup_steps]
        steady = values[warmup_steps:]
        return {
            "warmup_total_bytes": sum(warmup) if warmup else None,
            "warmup_mean_bytes": sum(warmup) / len(warmup) if warmup else None,
            "steady_total_bytes": sum(steady),
            "steady_mean_bytes": sum(steady) / len(steady),
        }

    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    baseline_total = baseline_summary["steady_total_bytes"]
    candidate_total = candidate_summary["steady_total_bytes"]
    baseline_mean = baseline_summary["steady_mean_bytes"]
    candidate_mean = candidate_summary["steady_mean_bytes"]
    assert baseline_total is not None and candidate_total is not None
    assert baseline_mean is not None and candidate_mean is not None
    return {
        "status": "available",
        "reason": None,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "steady_total_delta_bytes": candidate_total - baseline_total,
        "steady_total_ratio": _ratio(candidate_total, baseline_total),
        "steady_mean_delta_bytes": candidate_mean - baseline_mean,
        "steady_mean_ratio": _ratio(candidate_mean, baseline_mean),
    }


def _p2p_wait_time_report(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], *, warmup_steps: int
) -> dict[str, Any]:
    curves = _matched_curve("p2p_wait_time_s", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    if curves is None:
        return _unavailable_timing(_missing_metric_reason("p2p_wait_time_s", baseline_metrics, candidate_metrics))
    baseline, candidate = curves
    baseline_steady = sum(baseline[warmup_steps:]) / len(baseline[warmup_steps:])
    candidate_steady = sum(candidate[warmup_steps:]) / len(candidate[warmup_steps:])
    baseline_warmup = sum(baseline[:warmup_steps]) / warmup_steps if warmup_steps else None
    candidate_warmup = sum(candidate[:warmup_steps]) / warmup_steps if warmup_steps else None
    return {
        "status": "available",
        "reason": None,
        "baseline": {"warmup_mean_s": baseline_warmup, "steady_mean_s": baseline_steady},
        "candidate": {"warmup_mean_s": candidate_warmup, "steady_mean_s": candidate_steady},
        "steady_delta_s": candidate_steady - baseline_steady,
        "steady_speedup": _ratio(baseline_steady, candidate_steady),
    }


def _derived_missing_reason(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> str:
    required = ("step_time_s", "step_tokens")
    if all(name not in baseline_metrics and name not in candidate_metrics for name in required):
        return "step_time_s and step_tokens missing on both baseline and candidate"
    reasons = []
    for name in required:
        if name not in baseline_metrics or name not in candidate_metrics:
            reasons.append(_missing_metric_reason(name, baseline_metrics, candidate_metrics))
    return "; ".join(reasons)


def _derived_throughput_report(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any], *, warmup_steps: int
) -> dict[str, Any]:
    if any(name not in baseline_metrics or name not in candidate_metrics for name in ("step_time_s", "step_tokens")):
        return _unavailable_throughput(_derived_missing_reason(baseline_metrics, candidate_metrics))
    time_curves = _matched_curve("step_time_s", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    token_curves = _matched_curve("step_tokens", baseline_metrics, candidate_metrics, warmup_steps=warmup_steps)
    assert time_curves is not None and token_curves is not None
    baseline_time, candidate_time = time_curves
    baseline_tokens, candidate_tokens = token_curves
    for side, times, tokens in (
        ("baseline", baseline_time, baseline_tokens),
        ("candidate", candidate_time, candidate_tokens),
    ):
        if len(times) != len(tokens):
            raise ValueError(f"{side} step_time_s and step_tokens curves must have equal lengths.")
    baseline_steady = sum(baseline_tokens[warmup_steps:]) / sum(baseline_time[warmup_steps:])
    candidate_steady = sum(candidate_tokens[warmup_steps:]) / sum(candidate_time[warmup_steps:])
    return {
        "status": "available",
        "reason": None,
        "baseline_steady_tokens_per_second": baseline_steady,
        "candidate_steady_tokens_per_second": candidate_steady,
        "steady_delta_tokens_per_second": candidate_steady - baseline_steady,
        "steady_speedup": _ratio(candidate_steady, baseline_steady),
    }


def _peak_memory_report(
    name: str, baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, Any]:
    if name not in baseline_metrics or name not in candidate_metrics:
        return _unavailable_memory(_missing_metric_reason(name, baseline_metrics, candidate_metrics))

    def peak(value: float | list[float]) -> float:
        return max(value) if isinstance(value, list) else value

    baseline_peak = peak(baseline_metrics[name])
    candidate_peak = peak(candidate_metrics[name])
    return {
        "status": "available",
        "reason": None,
        "baseline_peak_bytes": baseline_peak,
        "candidate_peak_bytes": candidate_peak,
        "delta_bytes": candidate_peak - baseline_peak,
        "baseline_over_candidate_ratio": _ratio(baseline_peak, candidate_peak),
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    warmup_steps: int,
    rtol: float,
    atol: float,
    relative_error_threshold: float,
    epsilon: float,
) -> dict[str, Any]:
    """Validate and compare two already-loaded or canonical run dictionaries."""
    baseline = normalize_run(baseline, source="baseline")
    candidate = normalize_run(candidate, source="candidate")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer.")
    rtol = _validate_comparison_number(rtol, name="rtol")
    atol = _validate_comparison_number(atol, name="atol")
    relative_error_threshold = _validate_comparison_number(relative_error_threshold, name="relative_error_threshold")
    epsilon = _validate_comparison_number(epsilon, name="epsilon", positive=True)

    metadata_validation = _metadata_validation(baseline, candidate)
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    precision_metrics = {}
    for name in REQUIRED_CURVES:
        left = baseline_metrics[name]
        right = candidate_metrics[name]
        if len(left) != len(right):
            raise ValueError(f"baseline and candidate {name} curves must have equal lengths.")
        precision_metrics[name] = _precision_metric(
            left,
            right,
            rtol=rtol,
            atol=atol,
            relative_error_threshold=relative_error_threshold,
            epsilon=epsilon,
        )
    passed = all(precision_metrics[name]["repository_close_hit_rate"] == 1.0 for name in REQUIRED_CURVES)
    precision = {"status": "pass" if passed else "fail", "passed": passed, **precision_metrics}
    performance = {
        "warmup_steps": warmup_steps,
        "timing": _timing_report(baseline_metrics, candidate_metrics, warmup_steps=warmup_steps),
        "explicit_tokens_per_second": _explicit_throughput_report(
            baseline_metrics, candidate_metrics, warmup_steps=warmup_steps
        ),
        "derived_tokens_per_second": _derived_throughput_report(
            baseline_metrics, candidate_metrics, warmup_steps=warmup_steps
        ),
        "p2p_bytes": _p2p_bytes_report(baseline_metrics, candidate_metrics, warmup_steps=warmup_steps),
        "p2p_wait_time_s": _p2p_wait_time_report(baseline_metrics, candidate_metrics, warmup_steps=warmup_steps),
        "peak_accelerator_memory_bytes": _peak_memory_report(
            "peak_accelerator_memory_bytes", baseline_metrics, candidate_metrics
        ),
        "peak_host_memory_bytes": _peak_memory_report("peak_host_memory_bytes", baseline_metrics, candidate_metrics),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": {
            "rtol": rtol,
            "atol": atol,
            "relative_error_threshold": relative_error_threshold,
            "epsilon": epsilon,
            "warmup_steps": warmup_steps,
        },
        "metadata_validation": metadata_validation,
        "precision": precision,
        "performance": performance,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a comparison report in a deterministic human-readable order."""
    lines = [
        "# MoE EP Load-Balance Comparison",
        "",
        f"Precision: {report['precision']['status'].upper()}",
        "",
        "## Precision",
        "",
        "| Metric | Max abs error | Max relative error | Repository-close hit rate | Relative-threshold hit rate | Pearson correlation | Correlation note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in REQUIRED_CURVES:
        metric = report["precision"][name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_number(metric["max_absolute_error"]),
                    _format_number(metric["max_relative_error"]),
                    _format_number(metric["repository_close_hit_rate"]),
                    _format_number(metric["threshold_relative_hit_rate"]),
                    _format_number(metric["pearson_correlation"]),
                    metric["correlation_unavailable_reason"] or "—",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Performance", ""])
    timing = report["performance"]["timing"]
    lines.append(f"- Timing: {timing['status']}" + (f" — {timing['reason']}" if timing["reason"] else ""))
    if timing["status"] == "available":
        lines.extend(
            [
                f"  - Baseline warmup mean (s): {_format_number(timing['baseline']['warmup_mean_s'])}",
                f"  - Candidate warmup mean (s): {_format_number(timing['candidate']['warmup_mean_s'])}",
                f"  - Baseline steady mean (s): {_format_number(timing['baseline']['steady_mean_s'])}",
                f"  - Candidate steady mean (s): {_format_number(timing['candidate']['steady_mean_s'])}",
                f"  - Candidate - baseline (s): {_format_number(timing['steady_delta_s'])}",
                f"  - Speedup (baseline/candidate): {_format_number(timing['steady_speedup'])}",
            ]
        )
    for section_name, label in (
        ("explicit_tokens_per_second", "Explicit tokens/s"),
        ("derived_tokens_per_second", "Derived tokens/s"),
    ):
        section = report["performance"][section_name]
        lines.append(f"- {label}: {section['status']}" + (f" — {section['reason']}" if section["reason"] else ""))
        if section["status"] == "available":
            lines.extend(
                [
                    f"  - Baseline steady tokens/s: {_format_number(section['baseline_steady_tokens_per_second'])}",
                    f"  - Candidate steady tokens/s: {_format_number(section['candidate_steady_tokens_per_second'])}",
                    f"  - Candidate - baseline: {_format_number(section['steady_delta_tokens_per_second'])}",
                    f"  - Speedup (candidate/baseline): {_format_number(section['steady_speedup'])}",
                ]
            )
    p2p_bytes = report["performance"]["p2p_bytes"]
    lines.append(f"- P2P bytes: {p2p_bytes['status']}" + (f" — {p2p_bytes['reason']}" if p2p_bytes["reason"] else ""))
    if p2p_bytes["status"] == "available":
        lines.extend(
            [
                f"  - Baseline warmup total bytes: {_format_number(p2p_bytes['baseline']['warmup_total_bytes'])}",
                f"  - Candidate warmup total bytes: {_format_number(p2p_bytes['candidate']['warmup_total_bytes'])}",
                f"  - Baseline warmup mean bytes: {_format_number(p2p_bytes['baseline']['warmup_mean_bytes'])}",
                f"  - Candidate warmup mean bytes: {_format_number(p2p_bytes['candidate']['warmup_mean_bytes'])}",
                f"  - Baseline steady total bytes: {_format_number(p2p_bytes['baseline']['steady_total_bytes'])}",
                f"  - Candidate steady total bytes: {_format_number(p2p_bytes['candidate']['steady_total_bytes'])}",
                f"  - Baseline steady mean bytes: {_format_number(p2p_bytes['baseline']['steady_mean_bytes'])}",
                f"  - Candidate steady mean bytes: {_format_number(p2p_bytes['candidate']['steady_mean_bytes'])}",
                f"  - Steady total candidate - baseline bytes: {_format_number(p2p_bytes['steady_total_delta_bytes'])}",
                f"  - Steady total ratio (candidate/baseline): {_format_number(p2p_bytes['steady_total_ratio'])}",
                f"  - Steady mean candidate - baseline bytes: {_format_number(p2p_bytes['steady_mean_delta_bytes'])}",
                f"  - Steady mean ratio (candidate/baseline): {_format_number(p2p_bytes['steady_mean_ratio'])}",
            ]
        )
    p2p_wait = report["performance"]["p2p_wait_time_s"]
    lines.append(f"- P2P wait time: {p2p_wait['status']}" + (f" — {p2p_wait['reason']}" if p2p_wait["reason"] else ""))
    if p2p_wait["status"] == "available":
        lines.extend(
            [
                f"  - Baseline warmup mean (s): {_format_number(p2p_wait['baseline']['warmup_mean_s'])}",
                f"  - Candidate warmup mean (s): {_format_number(p2p_wait['candidate']['warmup_mean_s'])}",
                f"  - Baseline steady mean (s): {_format_number(p2p_wait['baseline']['steady_mean_s'])}",
                f"  - Candidate steady mean (s): {_format_number(p2p_wait['candidate']['steady_mean_s'])}",
                f"  - Candidate - baseline (s): {_format_number(p2p_wait['steady_delta_s'])}",
                f"  - Speedup (baseline/candidate): {_format_number(p2p_wait['steady_speedup'])}",
            ]
        )
    for section_name, label in (
        ("peak_accelerator_memory_bytes", "Peak accelerator memory"),
        ("peak_host_memory_bytes", "Peak host memory"),
    ):
        section = report["performance"][section_name]
        lines.append(f"- {label}: {section['status']}" + (f" — {section['reason']}" if section["reason"] else ""))
        if section["status"] == "available":
            lines.extend(
                [
                    f"  - Baseline peak bytes: {_format_number(section['baseline_peak_bytes'])}",
                    f"  - Candidate peak bytes: {_format_number(section['candidate_peak_bytes'])}",
                    f"  - Candidate - baseline bytes: {_format_number(section['delta_bytes'])}",
                    f"  - Baseline/candidate ratio: {_format_number(section['baseline_over_candidate_ratio'])}",
                ]
            )
    return "\n".join(lines) + "\n"


def _json_text(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--relative-error-threshold", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        baseline = load_run(args.baseline)
        candidate = load_run(args.candidate)
        report = compare_runs(
            baseline,
            candidate,
            warmup_steps=args.warmup_steps,
            rtol=args.rtol,
            atol=args.atol,
            relative_error_threshold=args.relative_error_threshold,
            epsilon=args.epsilon,
        )
        json_text = _json_text(report)
        markdown_text = render_markdown(report)
        if args.json_out is not None:
            _write_text(args.json_out, json_text)
        if args.markdown_out is not None:
            _write_text(args.markdown_out, markdown_text)
        if args.json_out is None and args.markdown_out is None:
            sys.stdout.write(json_text)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if report["precision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
