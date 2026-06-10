"""Compare BAGEL text-only official fixture reruns for self-consistency."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import assert_text_fixture_schema  # noqa: E402


EXACT_TENSOR_PATHS = (
    ("prepared", "prompt", "packed_text_ids"),
    ("prepared", "prompt", "packed_text_position_ids"),
    ("prepared", "prompt", "text_token_lens"),
    ("prepared", "prompt", "packed_text_indexes"),
    ("prepared", "prompt", "packed_key_value_indexes"),
    ("prepared", "prompt", "key_values_lens"),
    ("prepared", "start", "packed_start_tokens"),
    ("prepared", "start", "packed_query_position_ids"),
    ("prepared", "start", "key_values_lens"),
    ("prepared", "start", "packed_key_value_indexes"),
    ("prepared", "packed_query_indexes"),
    ("prepared", "packed_key_value_indexes_for_step"),
    ("prepared", "query_lens"),
    ("one_step", "greedy_token"),
)

EXACT_VALUE_PATHS = (
    ("metadata", "case_id"),
    ("metadata", "dtype"),
    ("metadata", "seed"),
    ("raw_input", "prompt"),
    ("raw_input", "do_sample"),
    ("raw_input", "temperature"),
    ("raw_input", "max_new_tokens"),
    ("tokenizer", "new_token_ids"),
    ("tokenizer", "encoded_prompt_ids"),
    ("prepared", "kv_lens_after_prompt"),
    ("prepared", "ropes_after_prompt"),
)

FLOAT_TENSOR_PATHS = (
    ("one_step", "hidden_state"),
    ("one_step", "logits"),
)


def _get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


def _path_name(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}

    a_float = a.detach().float()
    b_float = b.detach().float()
    diff = (a_float - b_float).abs()
    if a.numel() == 0:
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(a_float.reshape(1, -1), b_float.reshape(1, -1), dim=-1).item())
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape_match": True,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": cosine,
    }


def _compare_exact_values(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    return {_path_name(path): _get(left, path) == _get(right, path) for path in EXACT_VALUE_PATHS}


def _compare_exact_tensors(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for path in EXACT_TENSOR_PATHS:
        a = _get(left, path)
        b = _get(right, path)
        results[_path_name(path)] = torch.equal(a, b)
    return results


def _compare_float_tensors(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_path_name(path): _tensor_metrics(_get(left, path), _get(right, path)) for path in FLOAT_TENSOR_PATHS}


def _compare_cache(left_cache: dict[str, Any], right_cache: dict[str, Any]) -> dict[str, Any]:
    if left_cache["num_layers"] != right_cache["num_layers"]:
        return {
            "num_layers_match": False,
            "left_num_layers": left_cache["num_layers"],
            "right_num_layers": right_cache["num_layers"],
        }

    layer_metrics: dict[str, Any] = {}
    aggregate = {
        "max_abs_diff": 0.0,
        "mean_abs_diff_max": 0.0,
        "cosine_similarity_min": 1.0,
        "all_shapes_match": True,
    }

    for layer_idx in range(left_cache["num_layers"]):
        layer_result: dict[str, Any] = {}
        for kind in ("key", "value"):
            a = left_cache[kind][layer_idx]
            b = right_cache[kind][layer_idx]
            name = f"{kind}_{layer_idx}"
            if a is None or b is None:
                layer_result[name] = {"both_none": a is None and b is None}
                aggregate["all_shapes_match"] = aggregate["all_shapes_match"] and a is None and b is None
                continue
            metrics = _tensor_metrics(a, b)
            layer_result[name] = metrics
            aggregate["all_shapes_match"] = aggregate["all_shapes_match"] and bool(metrics.get("shape_match"))
            if metrics.get("shape_match"):
                aggregate["max_abs_diff"] = max(aggregate["max_abs_diff"], metrics["max_abs_diff"])
                aggregate["mean_abs_diff_max"] = max(aggregate["mean_abs_diff_max"], metrics["mean_abs_diff"])
                aggregate["cosine_similarity_min"] = min(
                    aggregate["cosine_similarity_min"],
                    metrics["cosine_similarity"],
                )
        layer_metrics[str(layer_idx)] = layer_result

    return {
        "num_layers_match": True,
        "aggregate": aggregate,
        "layers": layer_metrics,
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    assert_text_fixture_schema(left)
    assert_text_fixture_schema(right)
    exact_values = _compare_exact_values(left, right)
    exact_tensors = _compare_exact_tensors(left, right)
    float_tensors = _compare_float_tensors(left, right)
    cache_after_prefill = _compare_cache(left["cache_after_prefill"], right["cache_after_prefill"])
    cache_after_step = _compare_cache(left["one_step"]["cache_after_step"], right["one_step"]["cache_after_step"])

    return {
        "left": left["metadata"],
        "right": right["metadata"],
        "exact_values": exact_values,
        "exact_tensors": exact_tensors,
        "all_exact_checks_pass": all(exact_values.values()) and all(exact_tensors.values()),
        "float_tensors": float_tensors,
        "cache_after_prefill": cache_after_prefill,
        "cache_after_step": cache_after_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    report = compare(left, right)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
