"""Tests for probe catalog schema and resolver."""

from __future__ import annotations

from pathlib import Path

import torch

from tests.seed_omni.parity_suite.core import NodeSpec, ProbeCatalog, load_probe_catalog, resolve_probes
from tests.seed_omni.parity_suite.reference import ReferenceCaptureContext, build_reference_capture_plan


def test_probes_yaml_parses_hook_and_extractor_taps(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref_tap: model.norm
      tol: hidden
    text.greedy_token:
      v2_field: greedy
      ref_tap: {extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor}
      tol: exact
""",
        encoding="utf-8",
    )

    catalog = load_probe_catalog(probes_path)

    assert len(catalog.probes) == 2
    assert catalog.probes[0].ref_tap.kind == "hook"
    assert catalog.probes[1].ref_tap.kind == "extractor"


def test_probes_yaml_parses_output_tap(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref_tap:
        output: reference.hidden
      tol: hidden
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes
    resolved = resolve_probes(
        probes=(probe,),
        nodes=(NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),),
    )
    reference_plan = build_reference_capture_plan(resolved.ref_taps)

    assert probe.ref_tap.kind == "output"
    context = ReferenceCaptureContext(ref_model=None, inputs={}, hook_taps={})
    context.output = {"reference": {"hidden": torch.tensor([5])}}
    assert reference_plan.extractor_taps[0].extractor(context).item() == 5


def test_resolver_builds_reference_plan_and_v2_whitelist(tmp_path: Path) -> None:
    catalog = ProbeCatalog(
        probes=load_probe_catalog(_write_probes(tmp_path)).for_probe_names(["text.hidden", "text.greedy_token"])
    )
    nodes = (
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="text_ar"),
    )

    resolved = resolve_probes(probes=catalog.probes, nodes=nodes)
    reference_plan = build_reference_capture_plan(resolved.ref_taps)

    assert [tap.name for tap in reference_plan.hook_taps] == ["text.hidden"]
    assert [tap.name for tap in reference_plan.extractor_taps] == ["text.greedy_token"]
    assert reference_plan.extractor_taps[0].extractor(_fake_context()).item() == 3
    assert resolved.v2_whitelist == {
        ("prompt", "toy.generate"): frozenset({"hidden", "greedy"}),
        ("text_ar", "toy.generate"): frozenset({"hidden", "greedy"}),
    }


def test_resolver_honors_optional_state_scope(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.generate:
    image.hidden:
      state: image_flow
      v2_field: hidden
      ref_tap: {extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor}
      tol: hidden
""",
        encoding="utf-8",
    )
    nodes = (
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="image_flow"),
    )

    resolved = resolve_probes(probes=load_probe_catalog(probes_path).probes, nodes=nodes)

    assert resolved.v2_whitelist == {("image_flow", "toy.generate"): frozenset({"hidden"})}


def test_probes_yaml_parses_v2_gradient_spec(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.forward:
    train.grad_weight:
      v2_field: grad_weight
      v2_grad:
        module: toy_module
        parameter: linear.weight
        rows_from: packed.labels
      ref_tap: {extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor}
      tol: gradient
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes

    assert probe.v2_grad is not None
    assert probe.v2_grad.module == "toy_module"
    assert probe.v2_grad.parameter == "linear.weight"
    assert probe.v2_grad.rows_from == "packed.labels"


def _write_probes(tmp_path: Path) -> Path:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref_tap: model.norm
      tol: hidden
    text.greedy_token:
      v2_field: greedy
      ref_tap: {extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor}
      tol: exact
""",
        encoding="utf-8",
    )
    return probes_path


def fake_extractor(context: ReferenceCaptureContext) -> torch.Tensor:
    del context
    return torch.tensor(3)


def _fake_context() -> ReferenceCaptureContext:
    return ReferenceCaptureContext(ref_model=None, inputs={}, hook_taps={})
