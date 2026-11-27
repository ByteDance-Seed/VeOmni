"""The generation graph YAMLs we ship have to be drivable, not just parseable.

A graph can load cleanly and still never terminate. Raising a module signal ends
the body it was raised in (``GenerationGraph.iter_nodes`` returns as soon as
``FSM_SIGNAL_KEY`` appears in the context), so a state whose body chains two
signalling modules never reaches the second one — and if the state does not
transition on the first module's signal it re-enters and re-runs it until the
step budget is gone, producing nothing.

Which module raises which signal is only knowable at runtime, so there is no
generic structural check for this. What follows is the cheap generic part —
every shipped graph loads and its transitions name real states — plus a named
guard for the one graph where the trap has already been sprung.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph


CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "seed_omni")
# Two naming conventions are in use: ``graph_infer*.yaml`` (the flat form) and
# ``infer/<recipe>/graph.yaml`` (a graph beside the module list it is run with).
# Both must be collected — a pattern that only matched the flat form silently
# stopped covering the recipes that moved.
GRAPH_FILES = sorted(
    glob.glob(os.path.join(CONFIG_ROOT, "**", "graph_infer*.yaml"), recursive=True)
    + glob.glob(os.path.join(CONFIG_ROOT, "**", "infer", "*", "graph.yaml"), recursive=True)
)


def _spec(path: str) -> dict:
    """The single FSM spec in a launcher graph file.

    Only valid for the launcher shape; convert-time files hold several. Every
    named guard below points at a launcher file.
    """
    with open(path) as handle:
        return yaml.safe_load(handle)


def _specs(path: str) -> dict[str, dict]:
    """Every FSM in a shipped graph file, keyed by scenario.

    Two shapes ship under this glob. A launcher ``infer_graph:`` entry is one
    bare FSM (``initial`` / ``states``) chosen by ``infer_type``. A convert-time
    file — the source for a checkpoint's ``generation_graph.yaml`` sidecar —
    holds all of a family's scenarios keyed by infer_type. Both end up driving
    the same :class:`GenerationGraph`, so both are checked here rather than
    excluding one by filename.

    Read through the config's own loader rather than deciding the shape here.
    The copy that used to live in this file recognised the convert-time shape by
    a ``generation_graphs:`` wrapping key, and when that key went away the file
    read as one bare FSM with no ``initial`` — so the checks below raised
    ``KeyError`` instead of reporting what was wrong with the graph.
    """
    return OmniConfig._read_graph_file(path, dict)


def _relative(path: str) -> str:
    return os.path.relpath(path, CONFIG_ROOT)


GRAPHS = [(path, name, spec) for path in GRAPH_FILES for name, spec in _specs(path).items()]
GRAPH_IDS = [f"{_relative(path)}::{name}" for path, name, _ in GRAPHS]


def _signals_handled_by(state: dict) -> list[str]:
    return [
        transition["condition"]["key"]
        for transition in state.get("transitions", [])
        if transition.get("condition", {}).get("type") == "module_signal"
    ]


def test_there_are_shipped_graphs_to_check() -> None:
    """Guard the glob: a directory rename would make every test below vacuous."""
    assert GRAPHS


@pytest.mark.parametrize(("path", "scenario", "spec"), GRAPHS, ids=GRAPH_IDS)
def test_shipped_graph_loads(path: str, scenario: str, spec: dict) -> None:
    del path, scenario
    graph = GenerationGraph(spec)

    assert graph.current_state_name == spec["initial"]


@pytest.mark.parametrize(("path", "scenario", "spec"), GRAPHS, ids=GRAPH_IDS)
def test_every_transition_names_a_real_state(path: str, scenario: str, spec: dict) -> None:
    del scenario
    states = spec["states"]
    known = set(states) | {"done"}

    assert spec["initial"] in states, f"{_relative(path)}: initial state is not defined"
    for name, state in states.items():
        for transition in state.get("transitions", []):
            target = transition["next_state"]
            assert target in known, f"{_relative(path)}: state '{name}' transitions to unknown state '{target}'"


def test_the_qwen3omni_audio_graph_separates_the_talker_from_the_codec() -> None:
    """The regression itself, named.

    The talker raises ``codes_done`` and the codec raises ``audio_done``. Sharing
    one body, the talker's signal ends the iteration before the codec runs, and
    because the state only transitioned on ``audio_done`` it re-entered and
    regenerated codes until the budget ran out — a run that reported a text
    reply and no audio.
    """
    path = os.path.join(CONFIG_ROOT, "Qwen", "qwen3omni_30b_a3b", "infer", "audio_output", "graph.yaml")
    states = _spec(path)["states"]

    def state_running(module: str) -> str:
        owners = [name for name, s in states.items() if any(edge["from"] == module for edge in s["body"])]
        assert len(owners) == 1, f"expected exactly one state to run {module}, got {owners}"
        return owners[0]

    talker_state = state_running("qwen3omni_talker")
    codec_state = state_running("qwen3omni_codec")

    assert talker_state != codec_state, "the talker's signal would end the body before the codec ran"
    assert "codes_done" in _signals_handled_by(states[talker_state])
    assert "audio_done" in _signals_handled_by(states[codec_state])


def test_the_qwen3omni_thinker_only_graph_never_reaches_the_speech_modules() -> None:
    """The thinker-only scenario shares the thinker states, so it is easy to over-copy.

    Its module list is a sibling file, not a shared base, so the graph and the
    modules it is paired with have to agree on their own — this reads the graph
    side.
    """
    path = os.path.join(CONFIG_ROOT, "Qwen", "qwen3omni_30b_a3b", "infer", "thinker_only", "graph.yaml")
    states = _spec(path)["states"]
    modules = {edge["from"] for state in states.values() for edge in state["body"]}

    assert "qwen3omni_talker" not in modules
    assert "qwen3omni_codec" not in modules


RECIPE_GRAPH_FILES = sorted(glob.glob(os.path.join(CONFIG_ROOT, "**", "infer", "*", "graph.yaml"), recursive=True))


def test_every_infer_recipe_pairs_its_graph_with_a_matching_module_list() -> None:
    """Each ``infer/<recipe>/`` directory is a graph plus the modules it needs.

    The two are separate files because `model_config.modules` takes one path and
    applies it whole, with no overlay — so nothing structurally forces them to
    agree, and a recipe that declares a module its graph never runs (or omits one
    it does) only fails at load time, on a machine with the checkpoint.

    "Agrees" is both directions: a graph module must be declared, and a declared
    module must be one the graph enters. The exception is a module that is only
    there to be *built* — there is none today, so any would be a deliberate choice
    worth writing down here.
    """
    assert RECIPE_GRAPH_FILES, "no infer/<recipe>/graph.yaml found; the glob is wrong"

    for path in RECIPE_GRAPH_FILES:
        recipe_dir = os.path.dirname(path)
        modules_path = os.path.join(recipe_dir, "modules.yaml")
        assert os.path.isfile(modules_path), f"{path} has no sibling modules.yaml"

        spec = _spec(path)
        used = {edge["from"].split(".")[0] for state in spec["states"].values() for edge in state["body"]}
        with open(modules_path) as handle:
            declared = set(yaml.safe_load(handle) or {})

        assert not used - declared, (
            f"{recipe_dir}: graph runs modules that are not declared: {sorted(used - declared)}"
        )
        assert not declared - used, f"{recipe_dir}: declares modules the graph never enters: {sorted(declared - used)}"
