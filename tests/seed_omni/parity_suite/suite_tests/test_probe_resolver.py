"""Tests for probe catalog schema and resolver."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.seed_omni.parity_suite.core import NodeSpec, ProbeCatalog, load_probe_catalog, resolve_probes
from tests.seed_omni.parity_suite.reference.capture.spec import ReferenceCaptureContext, build_reference_capture_plan


def test_probes_yaml_parses_hook_and_extractor_taps(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
text.hidden:
  v2:
    node: toy.generate
    field: hidden
  ref:
    hook: model.norm
  tol: hidden
text.greedy_token:
  v2:
    node: toy.generate
    field: greedy
  ref:
    extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor
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
text.hidden:
  v2:
    node: toy.generate
    field: hidden
  ref:
    field: hidden
  tol: hidden
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes
    resolved = resolve_probes(
        probes=(probe,),
        nodes=(NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),),
    )

    assert probe.ref_tap.kind == "field"
    assert probe.ref_tap.target == "hidden"
    assert probe.ref_tap.field == "hidden"
    assert build_reference_capture_plan(resolved.ref_taps).extractor_taps == ()


def test_resolver_builds_reference_plan_and_v2_whitelist(tmp_path: Path) -> None:
    catalog = ProbeCatalog(
        probes=load_probe_catalog(_write_probes(tmp_path)).for_probe_names(["text.hidden", "text.greedy_token"])
    )
    nodes = (NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),)

    resolved = resolve_probes(probes=catalog.probes, nodes=nodes)
    reference_plan = build_reference_capture_plan(resolved.ref_taps)

    assert [tap.name for tap in reference_plan.hook_taps] == ["text.hidden"]
    assert [tap.name for tap in reference_plan.extractor_taps] == ["text.greedy_token"]
    assert reference_plan.extractor_taps[0].extractor(_fake_context()).item() == 3
    assert resolved.v2_whitelist == {
        ("prompt", "toy.generate"): frozenset({"hidden", "greedy"}),
    }


def test_resolver_honors_optional_state_scope(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
image.hidden:
  v2:
    node: toy.generate
    state: image_flow
    field: hidden
  ref:
    extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor
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
train.grad_weight:
  v2:
    node: toy.forward
    field: grad_weight
    grad:
      module: toy_module
      parameter: linear.weight
  ref:
    field: grad_weight
  tol: gradient
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes

    assert probe.v2_grad is not None
    assert probe.v2_grad.module == "toy_module"
    assert probe.v2_grad.parameter == "linear.weight"
    assert probe.ref_tap.kind == "field"
    assert probe.ref_tap.target == "grad_weight"
    assert probe.ref_tap.field == "grad_weight"


def test_probes_yaml_parses_v2_selector(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
infer.step:
  v2:
    node: toy.generate
    field: hidden
    selector: unique_consecutive
  ref:
    field: hidden
  tol: hidden
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes

    assert probe.v2_selector == "unique_consecutive"


def test_probes_yaml_rejects_unknown_v2_selector(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
infer.step:
  v2:
    node: toy.generate
    field: hidden
    selector: middle
  ref:
    field: hidden
  tol: hidden
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported v2.selector"):
        load_probe_catalog(probes_path)


def test_probes_yaml_infers_grad_module_from_v2_node(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
train.grad_weight:
  v2:
    node: toy_module.forward
    field: grad_weight
    grad:
      parameter: linear.weight
  ref:
    field: grad_weight
  tol: gradient
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes

    assert probe.v2_grad is not None
    assert probe.v2_grad.module == "toy_module"
    assert probe.v2_grad.parameter == "linear.weight"


def test_probes_yaml_ignores_rows_from_in_v2_grad(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
train.grad_weight:
  v2:
    node: toy_module.forward
    field: grad_weight
    grad:
      module: toy_module
      parameter: linear.weight
      rows_from: packed.labels
  ref:
    field: grad_weight
  tol: gradient
""",
        encoding="utf-8",
    )

    # Gradient row selection moved to ParityDriver.gradient_rows. A stale
    # rows_from is ignored rather than rejected.
    catalog = load_probe_catalog(probes_path)
    (probe,) = catalog.probes
    assert probe.v2_grad is not None
    assert probe.v2_grad.module == "toy_module"
    assert probe.v2_grad.parameter == "linear.weight"
    assert not hasattr(probe.v2_grad, "rows_from")


def test_probes_yaml_resolves_loss_field_alias(tmp_path: Path) -> None:
    from tests.seed_omni.parity_suite.v2.observation import LOSS_FIELD

    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
train.ce_loss:
  v2:
    node: toy.decode
    field: loss
  ref:
    field: train_ce_loss
  tol: loss
""",
        encoding="utf-8",
    )

    [probe] = load_probe_catalog(probes_path).probes

    assert probe.v2_field == LOSS_FIELD


def test_resolver_requires_state_for_multi_state_nodes(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
text.hidden:
  v2:
    node: toy.generate
    field: hidden
  ref:
    field: hidden
  tol: hidden
""",
        encoding="utf-8",
    )
    probe = load_probe_catalog(probes_path).probes[0]
    nodes = (
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="prompt"),
        NodeSpec(name="toy.generate", module="toy", method="generate", graph="infer", state="text_ar"),
    )

    with pytest.raises(ValueError, match="appears in multiple states"):
        resolve_probes(probes=(probe,), nodes=nodes)


def test_load_probe_catalog_requires_probe_keyed_schema(tmp_path: Path) -> None:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
nodes:
  toy.generate:
    text.hidden:
      v2_field: hidden
      ref: hidden
      tol: hidden
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be probe-keyed"):
        load_probe_catalog(probes_path)


def _write_probes(tmp_path: Path) -> Path:
    probes_path = tmp_path / "probes.yaml"
    probes_path.write_text(
        """
text.hidden:
  v2:
    node: toy.generate
    field: hidden
  ref:
    hook: model.norm
  tol: hidden
text.greedy_token:
  v2:
    node: toy.generate
    field: greedy
  ref:
    extractor: tests.seed_omni.parity_suite.suite_tests.test_probe_resolver:fake_extractor
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
