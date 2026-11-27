"""The training graph YAMLs we ship have to agree with the modules beside them.

A training DAG names modules and methods as opaque strings, resolved only when
a run builds the real model — on N GPUs against a checkpoint neither CI nor a
developer's laptop has. So the failure modes these configs actually exhibit
(a node whose module is not declared, a method that does not exist, a producer
ordered after its consumer) all surface minutes into a distributed job.

Every check below is a pure function of the shipped YAML plus the module
registry, which is importable without weights.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import GradientCheckpointingConfig
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.base import is_end
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.modules import OMNI_ACCELERATED_MODEL_REGISTRY


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "seed_omni")
GRAPH_FILES = sorted(glob.glob(os.path.join(CONFIG_ROOT, "**", "graph_train*.yaml"), recursive=True))


def _load(path: str) -> object:
    with open(path) as handle:
        return yaml.safe_load(handle)


def _relative(path: str) -> str:
    return os.path.relpath(path, CONFIG_ROOT)


def _scenarios(path: str) -> dict[str, list]:
    """Every training DAG in a shipped graph file, keyed by scenario.

    Two shapes ship under this glob: a launcher ``train_graph:`` entry is one
    bare edge list, while a convert-time file — the source for a checkpoint's
    ``training_graph.yaml`` sidecar — holds a family's scenarios keyed by
    ``train_type``. Read through the config's own loader rather than deciding
    that here, because a second copy of the shape rule is what went stale when
    the convert-time shape changed: this file kept handing the whole mapping to
    ``TrainingGraph`` as if it were an edge list.
    """
    return OmniConfig._read_graph_file(path, list)


# Parametrized per scenario, not per file, so a family that ships several
# train_types has all of them checked and a failure names the one that broke.
GRAPHS = [(path, name, edges) for path in GRAPH_FILES for name, edges in _scenarios(path).items()]
GRAPH_IDS = [f"{_relative(path)}::{name}" for path, name, _ in GRAPHS]


def _module_lists_beside(graph_path: str) -> list[tuple[str, dict]]:
    """Every ``modules_*.yaml`` in the same scenario directory.

    A directory ships one list per launch variant — ``infer/`` carries an eager
    and an FSDP2 file, ``train/`` a plain and a packed one — and the graph has to
    agree with all of them, since each is a way that same graph gets run. So
    every list is checked rather than one being singled out.

    A graph shipped with no list at all is not something this test can check, and
    says so by skipping rather than by passing.
    """
    candidates = sorted(glob.glob(os.path.join(os.path.dirname(graph_path), "modules_*.yaml")))
    if not candidates:
        pytest.skip(f"{_relative(graph_path)}: no modules_*.yaml beside it")
    return [(path, _load(path) or {}) for path in candidates]


def _launcher_beside(graph_path: str) -> dict:
    """The scenario's ``base.yaml`` — where per-module overrides are merged onto."""
    path = os.path.join(os.path.dirname(graph_path), "base.yaml")
    if not os.path.exists(path):
        pytest.skip(f"{_relative(graph_path)}: no base.yaml beside it to read defaults from")
    return _load(path) or {}


def _gradient_checkpointing_enabled(launcher: dict, overrides: dict) -> bool:
    """The effective flag for one module: its own override, else the launcher's.

    Reading the launcher matters more than the override does — a module block
    that says nothing about gradient checkpointing still gets it, and that is
    how a module that cannot support it ends up asked to.

    The fallback is the dataclass default rather than ``False``, and that is the
    whole check: it is ``enable=True``, so the config that ends up asking a module
    it cannot support is precisely the one that mentions nothing. Hardcoding
    ``False`` here made this test pass on every such config.
    """
    for source in (overrides, launcher.get("model", {})):
        section = source.get("accelerator", {}).get("gradient_checkpointing", {})
        if "enable" in section:
            return bool(section["enable"])
    return GradientCheckpointingConfig().enable


def test_there_are_shipped_graphs_to_check() -> None:
    """Guard the glob: a directory rename would make every test below vacuous."""
    assert GRAPHS


@pytest.mark.parametrize(("path", "scenario", "edges"), GRAPHS, ids=GRAPH_IDS)
def test_a_shipped_training_graph_is_a_dag(path: str, scenario: str, edges: list) -> None:
    """Loads, and its edges topologically sort — no cycle, no orphan node."""
    del scenario
    graph = TrainingGraph(edges)

    assert graph.execution_order
    assert graph.sources


@pytest.mark.parametrize(("path", "scenario", "edges"), GRAPHS, ids=GRAPH_IDS)
def test_every_node_names_a_registered_module_and_a_real_method(path: str, scenario: str, edges: list) -> None:
    """The strings resolve. Caught here, they would otherwise fail on N GPUs.

    Against the *accelerated* class, which is what a training run resolves a
    node to. Several graph methods (``pack_encode``, and every training hook)
    exist only there — the native class is the eager/inference shape.
    """
    del scenario
    graph = TrainingGraph(edges)
    known = set(OMNI_ACCELERATED_MODEL_REGISTRY.valid_keys())

    for name in graph.execution_order:
        if is_end(name):
            continue
        module_type = graph.module_of(name)
        assert module_type in known, f"{_relative(path)}: node {name!r} names no registered module"
        method = graph.method_of(name)
        assert hasattr(OMNI_ACCELERATED_MODEL_REGISTRY[module_type](), method), (
            f"{_relative(path)}: {module_type} has no method {method!r}"
        )


@pytest.mark.parametrize(("path", "scenario", "edges"), GRAPHS, ids=GRAPH_IDS)
def test_every_node_is_declared_in_the_module_list_beside_it(path: str, scenario: str, edges: list) -> None:
    """A node whose module is undeclared is never built, so the graph cannot run.

    The two files are edited apart — adding a node is one diff, declaring its
    module another — and the omission survives until a run resolves the graph.
    """
    del scenario
    graph = TrainingGraph(edges)
    referenced = {graph.module_of(name) for name in graph.execution_order if not is_end(name)}

    for list_path, modules in _module_lists_beside(path):
        assert referenced <= set(modules), (
            f"{_relative(path)}: graph uses {sorted(referenced - set(modules))}, which "
            f"{_relative(list_path)} does not declare"
        )


@pytest.mark.parametrize(("path", "scenario", "edges"), GRAPHS, ids=GRAPH_IDS)
def test_no_module_is_asked_to_gradient_checkpoint_when_it_cannot(path: str, scenario: str, edges: list) -> None:
    """``gradient_checkpointing_enable`` raises on a module that does not support it.

    It raises during the build, before step 0, and on the module it was called
    on rather than on a submodule — so a class that never declared
    ``supports_gradient_checkpointing`` aborts the run even though every layer
    inside it would have honoured the flag. The trap is the *inherited* value:
    a module block that mentions nothing still gets the launcher's default.
    """
    del scenario
    graph = TrainingGraph(edges)
    launcher = _launcher_beside(path)

    for list_path, modules in _module_lists_beside(path):
        for name in graph.execution_order:
            if is_end(name):
                continue
            module_type = graph.module_of(name)
            overrides = modules.get(module_type) or {}
            if not _gradient_checkpointing_enabled(launcher, overrides):
                continue
            cls = OMNI_ACCELERATED_MODEL_REGISTRY[module_type]()
            assert cls.supports_gradient_checkpointing, (
                f"{_relative(list_path)}: {module_type} is asked to gradient-checkpoint but {cls.__name__}"
                f".supports_gradient_checkpointing is False — the build would raise. Either set "
                f"accelerator.gradient_checkpointing.enable: false for it, or declare support on the class."
            )
