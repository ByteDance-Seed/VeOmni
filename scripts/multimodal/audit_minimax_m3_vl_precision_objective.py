#!/usr/bin/env python3
"""Audit MiniMax M3 VL precision objective completion.

This is a completion-status audit, not another parity runner. It maps the
MiniMax precision objective to concrete evidence artifacts and reports which
parts are proven, partial, or still missing. By default it exits successfully so
the current incomplete state can be recorded; pass --require-complete to use it
as the final target-machine gate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from audit_minimax_m3_vl_parity_artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    check_checkpoint_forward,
    check_full_checkpoint_preflight,
    check_multicard_summary,
    check_payload_sample,
    check_toy_precision,
    load_json,
)


TOY_CPU_ARTIFACT = "toy_hf_veomni_parity.json"
TOY_NPU_ARTIFACT = "toy_hf_veomni_parity_npu.json"
TOY_CPU_NPU_ARTIFACT = "toy_hf_cpu_veomni_npu_parity.json"
PAYLOAD_SAMPLE_ARTIFACT = "real_checkpoint_payload_remote_sample.json"
TOY_CHECKPOINT_FORWARD_ARTIFACT = "toy_checkpoint_forward_parity.json"
TOY_CHECKPOINT_CPU_NPU_FORWARD_ARTIFACT = "toy_checkpoint_cpu_npu_forward_parity.json"

REFERENCE_MODEL_CLASS_PREFIX = "transformers.models.minimax_m3_vl.modeling_minimax_m3_vl."
VEOMNI_MODEL_CLASS_PREFIX = "veomni.models.transformers.minimax_m3_vl.generated."

FORWARD_CHECKS = {
    "forward.loss",
    "forward.logits",
    "forward.image_hidden_states",
    "forward.video_hidden_states",
}
PROJECTOR_CHECKS = {
    "projector.record_count",
    "projector.0.output",
    "projector.1.output",
}
INPUT_CHECKS = {
    "input.input_ids",
    "input.attention_mask",
    "input.position_ids",
    "input.multimodal_metadata_contract",
}
ROUTER_CHECKS = {
    "router.record_count",
    "router.0.logits",
    "router.0.weights",
    "router.0.selected_experts",
}
GRAD_CHECKS = {
    "grad.model.language_model.embed_tokens.weight",
    "grad.model.language_model.layers.0.self_attn.q_proj.weight",
    "grad.model.language_model.layers.0.self_attn.k_proj.weight",
    "grad.model.language_model.layers.0.self_attn.v_proj.weight",
    "grad.model.language_model.layers.0.self_attn.o_proj.weight",
    "grad.model.language_model.layers.0.mlp.gate.weight",
    "grad.model.language_model.layers.0.mlp.experts.gate_up_proj",
    "grad.model.language_model.layers.0.mlp.experts.down_proj",
    "grad.model.multi_modal_projector.linear_1.weight",
    "grad.model.multi_modal_projector.linear_2.weight",
    "grad.model.multi_modal_projector.merge_linear_1.weight",
    "grad.model.multi_modal_projector.merge_linear_2.weight",
    "grad.lm_head.weight",
}
OPTIMIZER_DELTA_CHECKS = {name.replace("grad.", "optimizer_delta.", 1) for name in GRAD_CHECKS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--full-preflight-json", type=Path, action="append", default=[])
    parser.add_argument("--full-forward-json", type=Path, action="append", default=[])
    parser.add_argument("--multicard-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--target-bundle-audit-json",
        type=Path,
        action="append",
        default=[],
        help="Output from audit_minimax_m3_vl_target_artifact_bundle.py. May be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def add_issue(issues: list[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def safe_check(label: str, func: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - this script reports status, so keep auditing other gates.
        path = str(args[0]) if args else label
        return {"path": path, "passed": False, "issues": [f"{label}: {exc!r}"]}


def passed(item: dict[str, Any]) -> bool:
    return item.get("passed") is True


def any_passed(items: list[dict[str, Any]]) -> bool:
    return any(passed(item) for item in items)


def gate(
    name: str,
    description: str,
    *,
    status: str,
    evidence: list[Any],
    issues: list[str] | None = None,
    required_for_completion: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "status": status,
        "passed": status == "passed",
        "required_for_completion": required_for_completion,
        "evidence": evidence,
        "issues": issues or [],
    }


def status_from_items(items: list[dict[str, Any]], missing_issue: str) -> tuple[str, list[str]]:
    if any_passed(items):
        return "passed", []
    issues = [issue for item in items for issue in item.get("issues", [])]
    if issues:
        return "failed", issues
    return "missing", [missing_issue]


def check_target_bundle_audit(path: Path) -> dict[str, Any]:
    issues: list[str] = []
    data = load_json(path)
    add_issue(issues, data.get("passed") is True, f"{path.name}: bundle audit passed is not true")
    add_issue(issues, data.get("issues") == [], f"{path.name}: bundle audit issues is not empty")
    metadata = data.get("metadata") or {}
    add_issue(issues, bool(metadata.get("veomni_commit")), f"{path.name}: missing metadata.veomni_commit")
    add_issue(issues, bool(metadata.get("minimax_revision")), f"{path.name}: missing metadata.minimax_revision")
    manifest = data.get("manifest") or {}
    add_issue(issues, manifest.get("checked", 0) > 0, f"{path.name}: manifest checked no files")
    add_issue(issues, manifest.get("missing") == [], f"{path.name}: manifest has missing files")
    add_issue(issues, manifest.get("mismatched") == [], f"{path.name}: manifest has mismatched files")
    required_artifacts = data.get("required_artifacts") or {}
    for name in ("final", "full_preflight", "full_forward", "full_audit", "multicard"):
        add_issue(issues, bool(required_artifacts.get(name)), f"{path.name}: missing required artifact path {name}")
    target_toy = data.get("target_toy") or {}
    add_issue(issues, target_toy.get("present") is True, f"{path.name}: target toy parity artifact is not present")
    return {"path": str(path), "passed": not issues, "issues": issues}


def load_target_bundle_audits(paths: list[Path]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for path in paths:
        try:
            audits.append(check_target_bundle_audit(path))
        except Exception as exc:  # noqa: BLE001
            audits.append({"path": str(path), "passed": False, "issues": [repr(exc)]})
    return audits


def check_toy_named_checks(path: Path, required_names: set[str]) -> dict[str, Any]:
    issues: list[str] = []
    data = load_json(path)
    by_name = {item.get("name"): item for item in data.get("checks", [])}
    missing = sorted(required_names - set(by_name))
    add_issue(issues, not missing, f"{path.name}: missing checks {missing}")
    for name in sorted(required_names & set(by_name)):
        item = by_name[name]
        passed_check = item.get("allclose", item.get("equal", False)) is True
        add_issue(issues, passed_check, f"{path.name}: check did not pass: {name}")
    return {"path": str(path), "passed": not issues, "issues": issues}


def check_reference_classes(path: Path) -> dict[str, Any]:
    issues: list[str] = []
    data = load_json(path)
    hf_class = str(data.get("hf_model_class") or "")
    veomni_class = str(data.get("veomni_model_class") or "")
    add_issue(
        issues,
        hf_class.startswith(REFERENCE_MODEL_CLASS_PREFIX)
        and hf_class.endswith("MiniMaxM3SparseForConditionalGeneration"),
        f"{path.name}: HF reference class is not the transformers MiniMax M3 VL modeling class",
    )
    add_issue(
        issues,
        veomni_class.startswith(VEOMNI_MODEL_CLASS_PREFIX)
        and veomni_class.endswith("MiniMaxM3SparseForConditionalGeneration"),
        f"{path.name}: VeOmni candidate class is not generated MiniMax M3 VL modeling",
    )
    return {"path": str(path), "passed": not issues, "issues": issues}


def check_toy_checkpoint_forward(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        safe_check(
            TOY_CHECKPOINT_FORWARD_ARTIFACT,
            check_checkpoint_forward,
            args.artifacts_dir / TOY_CHECKPOINT_FORWARD_ARTIFACT,
            require_public_full=False,
        ),
        safe_check(
            TOY_CHECKPOINT_CPU_NPU_FORWARD_ARTIFACT,
            check_checkpoint_forward,
            args.artifacts_dir / TOY_CHECKPOINT_CPU_NPU_FORWARD_ARTIFACT,
            require_public_full=False,
        ),
    ]


def main() -> None:
    args = parse_args()
    toy_cpu_path = args.artifacts_dir / TOY_CPU_ARTIFACT
    toy_npu_path = args.artifacts_dir / TOY_NPU_ARTIFACT
    toy_cpu_npu_path = args.artifacts_dir / TOY_CPU_NPU_ARTIFACT

    toy_cpu = safe_check(
        TOY_CPU_ARTIFACT,
        check_toy_precision,
        toy_cpu_path,
        expect_reference="cpu",
        expect_candidate="cpu",
        require_projector=True,
    )
    toy_npu = safe_check(TOY_NPU_ARTIFACT, check_toy_precision, toy_npu_path, expect_reference="npu:0", expect_candidate="npu:0")
    toy_cpu_npu = safe_check(
        TOY_CPU_NPU_ARTIFACT,
        check_toy_precision,
        toy_cpu_npu_path,
        expect_reference="cpu",
        expect_candidate="npu:0",
    )
    reference_class_checks = [
        safe_check(TOY_CPU_ARTIFACT, check_reference_classes, toy_cpu_path),
        safe_check(TOY_CPU_NPU_ARTIFACT, check_reference_classes, toy_cpu_npu_path),
    ]
    input_checks = [safe_check(TOY_CPU_ARTIFACT, check_toy_named_checks, toy_cpu_path, INPUT_CHECKS)]
    forward_checks = [
        safe_check(TOY_CPU_ARTIFACT, check_toy_named_checks, toy_cpu_path, FORWARD_CHECKS | PROJECTOR_CHECKS | ROUTER_CHECKS),
        safe_check(TOY_CPU_NPU_ARTIFACT, check_toy_named_checks, toy_cpu_npu_path, FORWARD_CHECKS | ROUTER_CHECKS),
    ]
    backward_checks = [
        safe_check(TOY_CPU_ARTIFACT, check_toy_named_checks, toy_cpu_path, GRAD_CHECKS),
        safe_check(TOY_CPU_NPU_ARTIFACT, check_toy_named_checks, toy_cpu_npu_path, GRAD_CHECKS),
    ]
    optimizer_checks = [
        safe_check(TOY_CPU_ARTIFACT, check_toy_named_checks, toy_cpu_path, OPTIMIZER_DELTA_CHECKS),
        safe_check(TOY_CPU_NPU_ARTIFACT, check_toy_named_checks, toy_cpu_npu_path, OPTIMIZER_DELTA_CHECKS),
    ]
    payload_sample = [
        safe_check(
            PAYLOAD_SAMPLE_ARTIFACT,
            check_payload_sample,
            args.artifacts_dir / PAYLOAD_SAMPLE_ARTIFACT,
        )
    ]
    toy_checkpoint_forward = check_toy_checkpoint_forward(args)
    full_preflight = [
        safe_check(str(path), check_full_checkpoint_preflight, path)
        for path in args.full_preflight_json
    ]
    full_forward = [
        safe_check(str(path), check_checkpoint_forward, path, require_public_full=True)
        for path in args.full_forward_json
    ]
    multicard = [safe_check(str(path), check_multicard_summary, path) for path in args.multicard_json]
    target_bundle_audits = load_target_bundle_audits(args.target_bundle_audit_json)
    target_bundle_passed = any_passed(target_bundle_audits)

    gates: list[dict[str, Any]] = []
    status, issues = status_from_items(reference_class_checks, "no toy reference/candidate class evidence passed")
    gates.append(
        gate(
            "hf_transformers_reference_vs_veomni_candidate",
            "Toy parity uses transformers MiniMax M3 VL modeling as reference and VeOmni generated modeling as candidate.",
            status=status,
            evidence=reference_class_checks,
            issues=issues,
        )
    )

    status, issues = status_from_items([toy_cpu, toy_cpu_npu], "no same-config/same-weight toy parity artifact passed")
    gates.append(
        gate(
            "same_config_same_weights_same_input",
            "Toy/reduced config uses fixed seed, strict state_dict load, and identical input contract.",
            status=status if any_passed(input_checks) else "failed",
            evidence=[toy_cpu, toy_cpu_npu, *input_checks],
            issues=issues + [issue for item in input_checks if not passed(item) for issue in item.get("issues", [])],
        )
    )

    status, issues = status_from_items(forward_checks, "no toy forward/router parity artifact passed")
    gates.append(
        gate(
            "toy_forward_and_router_parity",
            "Forward loss/logits/vision states and MoE router logits/selected experts are aligned.",
            status=status,
            evidence=forward_checks,
            issues=issues,
        )
    )

    status, issues = status_from_items(backward_checks, "no toy backward parity artifact passed")
    gates.append(
        gate(
            "toy_backward_gradient_parity",
            "Embedding, attention q/k/v/o, MoE gate/experts, projector, and lm_head gradients are aligned.",
            status=status,
            evidence=backward_checks,
            issues=issues,
        )
    )

    status, issues = status_from_items(optimizer_checks, "no AdamW one-step optimizer-delta parity artifact passed")
    gates.append(
        gate(
            "toy_adamw_one_step_delta_parity",
            "The same AdamW configuration produces matching one-step parameter deltas.",
            status=status,
            evidence=optimizer_checks,
            issues=issues,
        )
    )

    status, issues = status_from_items(payload_sample, "no real checkpoint tensor payload sample artifact passed")
    gates.append(
        gate(
            "real_checkpoint_payload_sample",
            "A real public-checkpoint tensor payload sample is read, converted, and value-checked.",
            status=status,
            evidence=payload_sample,
            issues=issues,
        )
    )

    status, issues = status_from_items(toy_checkpoint_forward, "no toy checkpoint forward runner artifact passed")
    gates.append(
        gate(
            "checkpoint_forward_runner_smoke",
            "The streaming checkpoint forward runner path is proven on a complete toy checkpoint.",
            status=status,
            evidence=toy_checkpoint_forward,
            issues=issues,
        )
    )

    full_preflight_status, full_preflight_issues = status_from_items(
        full_preflight,
        "no full public-checkpoint preflight artifact was provided",
    )
    if target_bundle_passed:
        full_preflight_status = "passed"
        full_preflight_issues = []
    gates.append(
        gate(
            "full_public_checkpoint_preflight",
            "Target machine proves official MiniMax source, 59 shards, >800GB payload, entrypoints, and runtime readiness.",
            status=full_preflight_status,
            evidence=full_preflight + target_bundle_audits,
            issues=full_preflight_issues,
        )
    )

    full_forward_status, full_forward_issues = status_from_items(
        full_forward,
        "no full public-checkpoint forward parity artifact was provided",
    )
    if target_bundle_passed:
        full_forward_status = "passed"
        full_forward_issues = []
    gates.append(
        gate(
            "full_public_checkpoint_forward_parity",
            "The full MiniMax public checkpoint is loaded through both paths and logits/top-k/greedy outputs align.",
            status=full_forward_status,
            evidence=full_forward + target_bundle_audits,
            issues=full_forward_issues,
        )
    )

    status, issues = status_from_items([toy_npu, toy_cpu_npu], "no single-card NPU parity artifact passed")
    gates.append(
        gate(
            "single_card_npu_tolerance_parity",
            "Single-card NPU parity passes with explicit tolerance and Ascend runtime evidence.",
            status=status,
            evidence=[toy_npu, toy_cpu_npu],
            issues=issues,
        )
    )

    multicard_status, multicard_issues = status_from_items(
        multicard,
        "no multi-card SP/EP/FSDP2 parity artifact was provided",
    )
    if target_bundle_passed:
        multicard_status = "passed"
        multicard_issues = []
    gates.append(
        gate(
            "multicard_sp_ep_fsdp2_parity",
            "Multi-card SP/EP/FSDP2 target-machine parity summary and logs pass.",
            status=multicard_status,
            evidence=multicard + target_bundle_audits,
            issues=multicard_issues,
        )
    )

    if target_bundle_audits:
        status, issues = status_from_items(target_bundle_audits, "no returned target artifact bundle audit passed")
        gates.append(
            gate(
                "returned_target_artifact_bundle",
                "The returned target-machine artifact bundle is internally consistent and SHA256 verified.",
                status=status,
                evidence=target_bundle_audits,
                issues=issues,
                required_for_completion=False,
            )
        )

    completion_gates = [item for item in gates if item["required_for_completion"]]
    objective_complete = all(item["status"] == "passed" for item in completion_gates)
    report = {
        "passed": objective_complete,
        "objective_complete": objective_complete,
        "summary": {
            "passed": sum(1 for item in gates if item["status"] == "passed"),
            "failed": sum(1 for item in gates if item["status"] == "failed"),
            "missing": sum(1 for item in gates if item["status"] == "missing"),
            "partial": sum(1 for item in gates if item["status"] == "partial"),
            "required_total": len(completion_gates),
        },
        "gates": gates,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_complete and not objective_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
