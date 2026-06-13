"""Tests for mapping schema and resolver."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.seed_omni.parity_suite.core import MappingSpec, NodeSpec, load_mapping_spec, resolve_mapping
from tests.seed_omni.parity_suite.reference import ReferenceCaptureContext


def test_mapping_yaml_parses_hook_and_extractor_taps(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref_tap: model.norm
      tol: hidden
    text.greedy_token:
      v2_field: greedy
      ref_tap: {extractor: tests.seed_omni.parity_suite.test_mapping_resolver:fake_extractor}
      tol: exact
""",
        encoding="utf-8",
    )

    spec = load_mapping_spec(mapping_path)

    assert len(spec.probes) == 2
    assert spec.probes[0].ref_tap.kind == "hook"
    assert spec.probes[1].ref_tap.kind == "extractor"


def test_resolver_builds_reference_plan_and_v2_whitelist(tmp_path: Path) -> None:
    mapping = MappingSpec(
        probes=load_mapping_spec(_write_mapping(tmp_path)).for_probe_names(["text.hidden", "text.greedy_token"])
    )
    nodes = (
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="text_ar"),
    )

    resolved = resolve_mapping(mappings=mapping.probes, nodes=nodes)

    assert [tap.name for tap in resolved.reference_plan.hook_taps] == ["text.hidden"]
    assert [tap.name for tap in resolved.reference_plan.extractor_taps] == ["text.greedy_token"]
    assert resolved.reference_plan.extractor_taps[0].extractor(_fake_context()).item() == 3
    assert resolved.v2_whitelist == {
        ("prompt", "toy.generate"): frozenset({"hidden", "greedy"}),
        ("text_ar", "toy.generate"): frozenset({"hidden", "greedy"}),
    }


def _write_mapping(tmp_path: Path) -> Path:
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref_tap: model.norm
      tol: hidden
    text.greedy_token:
      v2_field: greedy
      ref_tap: {extractor: tests.seed_omni.parity_suite.test_mapping_resolver:fake_extractor}
      tol: exact
""",
        encoding="utf-8",
    )
    return mapping_path


def fake_extractor(context: ReferenceCaptureContext) -> torch.Tensor:
    del context
    return torch.tensor(3)


def _fake_context() -> ReferenceCaptureContext:
    return ReferenceCaptureContext(ref_model=None, inputs={}, hook_taps={})
