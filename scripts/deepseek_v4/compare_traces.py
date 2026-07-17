"""Compare official and VeOmni DeepSeek-V4 trace files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("official", type=Path)
    parser.add_argument("veomni", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def topk_set_match(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    if left.shape != right.shape:
        return {"shape_match": 0, "left_numel": left.numel(), "right_numel": right.numel()}
    ordered_per_token = (left == right).all(dim=-1)
    left = left.long().sort(dim=-1).values
    right = right.long().sort(dim=-1).values
    per_token = (left == right).all(dim=-1)
    insertion = torch.searchsorted(right.contiguous(), left.contiguous(), side="left")
    found = insertion < right.shape[-1]
    candidates = right.gather(-1, insertion.clamp_max(right.shape[-1] - 1))
    overlap = (found & (candidates == left)).float().mean(dim=-1)
    mismatching_tokens = (~per_token).flatten().nonzero(as_tuple=False).flatten().tolist()
    return {
        "shape_match": 1,
        "matching_tokens": int(per_token.sum()),
        "ordered_matching_tokens": int(ordered_per_token.sum()),
        "total_tokens": per_token.numel(),
        "match_rate": float(per_token.float().mean()),
        "mean_topk_overlap": float(overlap.mean()),
        "min_topk_overlap": float(overlap.min()),
        "p01_topk_overlap": float(overlap.quantile(0.01)),
        "mismatching_tokens": mismatching_tokens,
    }


def compare_topk_group(official: dict, veomni: dict) -> dict[str, object]:
    result: dict[str, object] = {}
    for layer_id in sorted(set(official) | set(veomni)):
        if layer_id not in official or layer_id not in veomni:
            result[str(layer_id)] = {"missing": "official" if layer_id not in official else "veomni"}
        else:
            result[str(layer_id)] = topk_set_match(official[layer_id], veomni[layer_id])
    return result


def main() -> None:
    args = parse_args()
    official = torch.load(args.official, map_location="cpu", weights_only=True)
    veomni = torch.load(args.veomni, map_location="cpu", weights_only=True)
    if not torch.equal(official["input_ids"], veomni["input_ids"]):
        raise ValueError("Trace input IDs differ")

    left_logprobs = official["logprobs"].float()
    right_logprobs = veomni["logprobs"].float()
    logprob_shape_match = left_logprobs.shape == right_logprobs.shape
    logprob_diff = None if not logprob_shape_match else (left_logprobs - right_logprobs).abs()
    report = {
        "tokens": int(official["input_ids"].numel()),
        "logprobs": {
            "shape_match": logprob_shape_match,
            "official_shape": list(left_logprobs.shape),
            "veomni_shape": list(right_logprobs.shape),
            "max_abs_diff": None if logprob_diff is None else float(logprob_diff.max()),
            "mean_abs_diff": None if logprob_diff is None else float(logprob_diff.mean()),
            "p99_abs_diff": None if logprob_diff is None else float(logprob_diff.quantile(0.99)),
        },
        "moe_topk": compare_topk_group(official["moe_topk"], veomni["moe_topk"]),
        "indexer_topk": compare_topk_group(official["indexer_topk"], veomni["indexer_topk"]),
        "attention_topk": compare_topk_group(official["attention_topk"], veomni["attention_topk"]),
    }

    hidden_report = {}
    for layer_id in sorted(set(official["hidden"]) & set(veomni["hidden"])):
        layer_report = {}
        for name in ("mean", "rms", "sample"):
            diff = (official["hidden"][layer_id][name].float() - veomni["hidden"][layer_id][name].float()).abs()
            layer_report[name] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
        hidden_report[str(layer_id)] = layer_report
    report["hidden"] = hidden_report

    terminal_report = {}
    official_terminal = official.get("terminal", {})
    veomni_terminal = veomni.get("terminal", {})
    for name in sorted(set(official_terminal) | set(veomni_terminal)):
        if name not in official_terminal or name not in veomni_terminal:
            terminal_report[name] = {"missing": "official" if name not in official_terminal else "veomni"}
            continue
        left = official_terminal[name].float()
        right = veomni_terminal[name].float()
        if left.shape != right.shape:
            terminal_report[name] = {
                "shape_match": False,
                "official_shape": list(left.shape),
                "veomni_shape": list(right.shape),
            }
            continue
        diff = (left - right).abs()
        terminal_report[name] = {
            "shape_match": True,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "rms_diff": float(diff.square().mean().sqrt()),
        }
    report["terminal"] = terminal_report

    detail_report = {}
    official_details = official.get("details", {})
    veomni_details = veomni.get("details", {})
    for layer_id in sorted(set(official_details) | set(veomni_details)):
        if layer_id not in official_details or layer_id not in veomni_details:
            detail_report[str(layer_id)] = {"missing": "official" if layer_id not in official_details else "veomni"}
            continue
        layer_report = {}
        official_layer = official_details[layer_id]
        veomni_layer = veomni_details[layer_id]
        for name in sorted(set(official_layer) | set(veomni_layer)):
            if name not in official_layer or name not in veomni_layer:
                layer_report[name] = {"missing": "official" if name not in official_layer else "veomni"}
                continue
            left = official_layer[name].float()
            right = veomni_layer[name].float()
            if left.shape != right.shape:
                layer_report[name] = {
                    "shape_match": False,
                    "official_shape": list(left.shape),
                    "veomni_shape": list(right.shape),
                }
                continue
            diff = (left - right).abs()
            layer_report[name] = {
                "shape_match": True,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "rms_diff": float(diff.square().mean().sqrt()),
            }
        detail_report[str(layer_id)] = layer_report
    report["details"] = detail_report

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
