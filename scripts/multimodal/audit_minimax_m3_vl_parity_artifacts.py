#!/usr/bin/env python3
"""Audit MiniMax M3 VL parity artifacts.

This script checks that the JSON artifacts produced by the MiniMax M3 VL
precision parity gates contain enough evidence to support their claims. By
default it audits the evidence currently expected in this PR. Pass
--require-full-checkpoint-forward with one or more --full-forward-json paths on
target machines after the full 869 GB public checkpoint has been loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS_DIR = Path("docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity")
REQUIRED_TOY_CHECKS = {
    "input.input_ids",
    "input.attention_mask",
    "input.position_ids",
    "input.multimodal_metadata_contract",
    "forward.loss",
    "forward.logits",
    "forward.image_hidden_states",
    "forward.video_hidden_states",
}
REQUIRED_CHECKPOINT_FORWARD_CHECKS = {
    "input.input_ids",
    "input.attention_mask",
    "input.position_ids",
    "input.image_grid_thw",
    "input.video_grid_thw",
    "input.pixel_values",
    "input.pixel_values_videos",
    "forward.logits",
    "forward.image_hidden_states",
    "forward.video_hidden_states",
    "forward.last_token_topk_ids",
    "generate.greedy_ids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument(
        "--full-forward-json",
        type=Path,
        action="append",
        default=[],
        help="Full public-checkpoint forward parity artifact. May be repeated.",
    )
    parser.add_argument(
        "--require-full-checkpoint-forward",
        action="store_true",
        help="Fail unless at least one --full-forward-json proves full public-checkpoint forward parity.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def check_runtime_for_npu(data: dict[str, Any], issues: list[str], prefix: str) -> None:
    runtime = data.get("runtime") or {}
    add_issue(issues, runtime.get("torch_npu_version") is not None, f"{prefix}: missing runtime.torch_npu_version")
    add_issue(issues, runtime.get("torch_npu_available") is True, f"{prefix}: torch_npu is not available")
    device_count = runtime.get("torch_npu_device_count")
    add_issue(issues, isinstance(device_count, int) and device_count >= 1, f"{prefix}: invalid torch_npu_device_count")
    ascend_env = runtime.get("ascend_env") or {}
    add_issue(
        issues,
        ascend_env.get("ASCEND_RT_VISIBLE_DEVICES") is not None or ascend_env.get("ASCEND_VISIBLE_DEVICES") is not None,
        f"{prefix}: missing visible Ascend device env",
    )


def check_toy_precision(path: Path, *, expect_reference: str, expect_candidate: str | None) -> dict[str, Any]:
    data = load_json(path)
    issues: list[str] = []
    prefix = path.name

    add_issue(issues, data.get("passed") is True, f"{prefix}: passed is not true")
    add_issue(issues, data.get("failed") == [], f"{prefix}: failed is not empty")
    add_issue(issues, isinstance(data.get("num_checks"), int) and data["num_checks"] >= 38, f"{prefix}: expected >=38 checks")
    add_issue(issues, data.get("reference_device") == expect_reference, f"{prefix}: unexpected reference_device")
    if expect_candidate is not None:
        add_issue(issues, data.get("candidate_device") == expect_candidate, f"{prefix}: unexpected candidate_device")
    state_load = data.get("state_dict_load") or {}
    add_issue(issues, state_load.get("strict") is True, f"{prefix}: state_dict load is not strict")
    add_issue(issues, state_load.get("missing_keys") == [], f"{prefix}: missing state_dict keys")
    add_issue(issues, state_load.get("unexpected_keys") == [], f"{prefix}: unexpected state_dict keys")

    check_names = {item.get("name") for item in data.get("checks", [])}
    missing_checks = sorted(REQUIRED_TOY_CHECKS - check_names)
    add_issue(issues, not missing_checks, f"{prefix}: missing checks {missing_checks}")
    add_issue(issues, (data.get("optimizer") or {}).get("name") == "AdamW", f"{prefix}: missing AdamW optimizer evidence")
    if "npu" in str(data.get("candidate_device")):
        check_runtime_for_npu(data, issues, prefix)

    return {"path": str(path), "passed": not issues, "issues": issues}


def check_payload_sample(path: Path) -> dict[str, Any]:
    data = load_json(path)
    issues: list[str] = []
    prefix = path.name
    payload = data.get("payload") or {}
    metadata = data.get("metadata_comparison") or {}
    sampled = data.get("sampled_state_load") or {}

    add_issue(issues, data.get("passed") is True, f"{prefix}: passed is not true")
    add_issue(issues, data.get("mode") == "payload", f"{prefix}: mode is not payload")
    add_issue(issues, data.get("full_checkpoint_load_executed") is False, f"{prefix}: payload sample must not claim full load")
    add_issue(issues, payload.get("public_keys_read", 0) > 0, f"{prefix}: no public keys read")
    add_issue(issues, payload.get("payload_bytes_read", 0) > 0, f"{prefix}: no tensor payload bytes read")
    add_issue(issues, payload.get("converter_finalize_error") is None, f"{prefix}: converter finalize failed")
    add_issue(issues, metadata.get("missing_model_key_count") == 0, f"{prefix}: missing model keys")
    add_issue(issues, metadata.get("shape_mismatch_count") == 0, f"{prefix}: shape mismatches")
    add_issue(issues, sampled.get("passed") is True, f"{prefix}: sampled state load failed")
    add_issue(issues, sampled.get("loaded_tensor_count", 0) > 0, f"{prefix}: no sampled tensors loaded")
    add_issue(issues, sampled.get("value_mismatch_count") == 0, f"{prefix}: sampled value mismatches")

    return {"path": str(path), "passed": not issues, "issues": issues}


def check_checkpoint_forward(path: Path, *, require_public_full: bool) -> dict[str, Any]:
    data = load_json(path)
    issues: list[str] = []
    prefix = path.name
    forward = data.get("forward") or {}
    payload = data.get("payload") or {}
    metadata = data.get("metadata_comparison") or {}
    state_load = forward.get("state_dict_load") or {}

    add_issue(issues, data.get("passed") is True, f"{prefix}: passed is not true")
    add_issue(issues, data.get("mode") == "forward", f"{prefix}: mode is not forward")
    add_issue(issues, data.get("full_checkpoint_load_executed") is True, f"{prefix}: full checkpoint forward not executed")
    add_issue(issues, data.get("failed") == [], f"{prefix}: failed is not empty")
    add_issue(issues, data.get("num_checks") == forward.get("num_checks"), f"{prefix}: num_checks mismatch")
    add_issue(issues, forward.get("failed") == [], f"{prefix}: forward.failed is not empty")
    add_issue(issues, payload.get("streaming_model_load") is True, f"{prefix}: streaming model load not recorded")
    add_issue(issues, payload.get("converter_finalize_error") is None, f"{prefix}: converter finalize failed")
    add_issue(issues, metadata.get("missing_model_key_count") == 0, f"{prefix}: unexpected converted model keys")
    add_issue(issues, metadata.get("missing_state_key_count") == 0, f"{prefix}: missing state keys")
    add_issue(issues, metadata.get("shape_mismatch_count") == 0, f"{prefix}: shape mismatches")
    add_issue(issues, state_load.get("strict") is True, f"{prefix}: state load is not strict")
    add_issue(issues, state_load.get("missing_key_count") == 0, f"{prefix}: missing state_dict key count")
    add_issue(issues, state_load.get("unexpected_key_count") == 0, f"{prefix}: unexpected state_dict key count")

    check_names = {item.get("name") for item in forward.get("checks", [])}
    missing_checks = sorted(REQUIRED_CHECKPOINT_FORWARD_CHECKS - check_names)
    add_issue(issues, not missing_checks, f"{prefix}: missing checks {missing_checks}")
    tolerances = data.get("tolerances") or {}
    add_issue(issues, "forward" in tolerances and "input" in tolerances, f"{prefix}: missing tolerance evidence")
    if "npu" in str(data.get("candidate_device")):
        check_runtime_for_npu(data, issues, prefix)

    if require_public_full:
        add_issue(issues, data.get("selected_shard_count", 0) >= 59, f"{prefix}: did not read all public shards")
        add_issue(issues, payload.get("public_keys_read", 0) >= 20000, f"{prefix}: did not read full public weight map")
        add_issue(issues, payload.get("payload_bytes_read", 0) > 800_000_000_000, f"{prefix}: payload bytes below full checkpoint")
        checkpoint_dir = str(data.get("checkpoint_dir") or "")
        add_issue(issues, "toy" not in checkpoint_dir.lower(), f"{prefix}: checkpoint_dir looks like a toy checkpoint")

    return {"path": str(path), "passed": not issues, "issues": issues}


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    results = {
        "toy_precision": [
            check_toy_precision(artifacts_dir / "toy_hf_veomni_parity.json", expect_reference="cpu", expect_candidate="cpu"),
            check_toy_precision(artifacts_dir / "toy_hf_veomni_parity_npu.json", expect_reference="npu:0", expect_candidate="npu:0"),
            check_toy_precision(
                artifacts_dir / "toy_hf_cpu_veomni_npu_parity.json",
                expect_reference="cpu",
                expect_candidate="npu:0",
            ),
        ],
        "checkpoint_payload_sample": [
            check_payload_sample(artifacts_dir / "real_checkpoint_payload_remote_sample.json"),
        ],
        "checkpoint_forward_smoke": [
            check_checkpoint_forward(artifacts_dir / "toy_checkpoint_forward_parity.json", require_public_full=False),
            check_checkpoint_forward(artifacts_dir / "toy_checkpoint_cpu_npu_forward_parity.json", require_public_full=False),
        ],
        "full_checkpoint_forward": [
            check_checkpoint_forward(path, require_public_full=True) for path in args.full_forward_json
        ],
    }
    full_forward_passed = any(item["passed"] for item in results["full_checkpoint_forward"])
    if args.require_full_checkpoint_forward and not full_forward_passed:
        results["full_checkpoint_forward"].append(
            {
                "path": None,
                "passed": False,
                "issues": ["no full public-checkpoint forward artifact passed"],
            }
        )

    all_items = [item for group in results.values() for item in group]
    report = {
        "passed": all(item["passed"] for item in all_items),
        "require_full_checkpoint_forward": args.require_full_checkpoint_forward,
        "full_checkpoint_forward_passed": full_forward_passed,
        "results": results,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
