"""Unit tests for the SeedOmni graph layer (flat edge-list training subset)."""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from veomni.models.seed_omni import EdgeDef, NodeDef
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.graphs.base import END
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph
from veomni.models.seed_omni.mixins.base_mixin import BaseMixin
from veomni.models.seed_omni.mixins.inference_module_mixin import InferenceModuleMixin
from veomni.models.seed_omni.mixins.training_module_mixin import TrainingModuleMixin
from veomni.models.seed_omni.modeling_omni import OmniModel


def test_from_endpoint_default_method():
    n = NodeDef.from_endpoint("module_A", default_method="forward")
    assert n.module == "module_A" and n.method == "forward"
    assert n.name == "module_A.forward"


def test_from_endpoint_dotted_form():
    n = NodeDef.from_endpoint("module_A.encode", default_method="forward")
    assert n.module == "module_A" and n.method == "encode"
    assert n.name == "module_A.encode"


def test_from_endpoint_generate_default():
    n = NodeDef.from_endpoint("module_A", default_method="generate")
    assert n.module == "module_A" and n.method == "generate"


def test_from_endpoint_rejects_reserved_end():
    with pytest.raises(ValueError, match=f"'{END}' is the virtual sink"):
        NodeDef.from_endpoint(END, default_method="forward")


def test_from_endpoint_rejects_empty():
    with pytest.raises(ValueError, match="non-empty 'module"):
        NodeDef.from_endpoint("   ", default_method="forward")


def test_parse_edge():
    e = EdgeDef.parse({"from": "module_A", "to": "module_B"}, default_method="forward")
    assert e.from_ == "module_A.forward" and e.to == "module_B.forward"
    assert e.from_node.module == "module_A" and e.to_node.module == "module_B"
    assert not e.is_sink()


def test_parse_edge_to_end_is_sink():
    e = EdgeDef.parse({"from": "module_A.decode", "to": "end"}, default_method="forward")
    assert e.is_sink() and e.to == END and e.to_node is None
    assert e.from_ == "module_A.decode"


def test_parse_edge_rejects_from_end():
    with pytest.raises(ValueError, match="`from: end` is forbidden"):
        EdgeDef.parse({"from": "end", "to": "module_B"}, default_method="forward")


def test_parse_edge_rejects_node_fields():
    with pytest.raises(ValueError, match="must not contain node fields"):
        EdgeDef.parse({"from": "a", "to": "b", "module": "x"}, default_method="forward")


def test_parse_edge_rejects_missing_endpoints():
    with pytest.raises(ValueError, match="must declare both"):
        EdgeDef.parse({"from": "a"}, default_method="forward")


def _multi_method_edges() -> list[dict]:
    """``module_B`` appears under two methods, with an explicit ``to: end`` sink."""
    return [
        {"from": "module_A", "to": "module_C"},
        {"from": "module_B.encode", "to": "module_C"},
        {"from": "module_C", "to": "module_B.decode"},
        {"from": "module_B.decode", "to": "end"},
    ]


def _fan_in_edges() -> list[dict]:
    """Two sources → ``module_C``, simple DAG with end-sink."""
    return [
        {"from": "module_A", "to": "module_C"},
        {"from": "module_B", "to": "module_C"},
        {"from": "module_C", "to": "end"},
    ]


def test_missing_edges_raises():
    with pytest.raises(ValueError, match="non-empty `training_graph`"):
        TrainingGraph([])


def test_duplicate_edge_raises():
    with pytest.raises(ValueError, match="Duplicate edge"):
        TrainingGraph(
            [
                {"from": "module_A", "to": "module_B"},
                {"from": "module_A", "to": "module_B"},
            ]
        )


def test_single_node_with_only_end_edge():
    """``[{from: module_A, to: end}]`` derives exactly one real node."""
    g = TrainingGraph([{"from": "module_A", "to": "end"}])
    assert g.execution_order == ["module_A.forward"]
    assert g.sources == ["module_A.forward"] and g.sinks == ["module_A.forward"]


def test_fan_in_topological_order():
    g = TrainingGraph(_fan_in_edges())
    assert g.execution_order[-1] == "module_C.forward"
    assert set(g.execution_order[:-1]) == {"module_A.forward", "module_B.forward"}


def test_multi_method_topological_order():
    """``module_B`` appears as TWO nodes; topo must place them on either side of module_C."""
    g = TrainingGraph(_multi_method_edges())
    order = g.execution_order
    assert order.index("module_B.decode") > order.index("module_C.forward")
    assert order.index("module_C.forward") > order.index("module_B.encode")
    assert order.index("module_C.forward") > order.index("module_A.forward")


def test_cycle_in_active_set_raises():
    with pytest.raises(ValueError, match="Circular dependency"):
        TrainingGraph(
            [
                {"from": "module_A", "to": "module_B"},
                {"from": "module_B", "to": "module_A"},
            ]
        )


def test_sources_and_sinks_fan_in():
    g = TrainingGraph(_fan_in_edges())
    assert set(g.sources) == {"module_A.forward", "module_B.forward"}
    # module_C's only outgoing edge targets `end`, so it's a sink.
    assert g.sinks == ["module_C.forward"]


def test_sources_and_sinks_multi_method():
    g = TrainingGraph(_multi_method_edges())
    assert set(g.sources) == {"module_A.forward", "module_B.encode"}
    # module_B.decode is the only sink (its only outgoing edge goes to `end`).
    assert g.sinks == ["module_B.decode"]


def test_module_and_method_lookup():
    g = TrainingGraph(_multi_method_edges())
    assert g.module_of("module_B.encode") == "module_B"
    assert g.method_of("module_B.encode") == "encode"
    assert g.module_of("module_B.decode") == "module_B"
    assert g.method_of("module_B.decode") == "decode"
    assert g.method_of("module_C.forward") == "forward"


def test_module_lookup_raises_for_unknown():
    g = TrainingGraph(_multi_method_edges())
    with pytest.raises(KeyError):
        g.module_of("not_a_node")


class _FakeOmniModule(nn.Module, TrainingModuleMixin, BaseMixin, InferenceModuleMixin):
    """Minimal stand-in for an OmniModule: callable (→ forward) + pre/post hooks.

    ``__call__`` delegates to ``self.forward`` so the non-``forward`` alias trick
    (``raw.forward = encode``) works exactly as on a real ``nn.Module``.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def pre_forward(self, method, **kwargs):
        return kwargs

    def post_forward(self, method, **outputs):
        return outputs

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        trace = list(kwargs.get("trace", []))
        trace.append(f"{self.name}.forward")
        return {"trace": trace}

    def encode(self, **kwargs):
        trace = list(kwargs.get("trace", []))
        trace.append(f"{self.name}.encode")
        return {"trace": trace}

    def generate(self, **kwargs):
        trace = list(kwargs.get("trace", []))
        trace.append(f"{self.name}.generate")
        return {"trace": trace}


def _fake_modules(g: TrainingGraph) -> dict:
    return {name: _FakeOmniModule(name) for name in {g.module_of(n) for n in g.execution_order}}


def test_cursor_lifecycle():
    g = TrainingGraph(_fan_in_edges())
    assert not g.is_done()
    assert g.current_node_name == g.execution_order[0]
    # Walk the cursor manually.
    seen = []
    while not g.is_done():
        seen.append(g.current_node_name)
        g.maybe_transition()
    assert seen == g.execution_order
    assert g.is_done()
    with pytest.raises(RuntimeError, match="cursor past the last node"):
        _ = g.current_node_name
    g.reset()
    assert not g.is_done() and g.current_node_name == g.execution_order[0]


def _minimal_generation_graph(module: str = "module_C") -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(module: str = "module_C") -> dict:
    return {"infer_gen": _minimal_generation_graph(module)}


def test_omni_model_forward_runs_fake_module_chain():
    """Eager training graph walks ``fake_module_a → fake_module_b`` (no conversation)."""
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.configuration import FakeModuleAConfig
    from veomni.models.seed_omni.modules.fake_model.fake_module_a.modeling import FakeModuleA
    from veomni.models.seed_omni.modules.fake_model.fake_module_b.configuration import FakeModuleBConfig
    from veomni.models.seed_omni.modules.fake_model.fake_module_b.modeling import FakeModuleB

    hidden_size = 8
    edges = [{"from": "fake_module_a", "to": "fake_module_b"}, {"from": "fake_module_b", "to": "end"}]
    a = FakeModuleA(FakeModuleAConfig(hidden_size=hidden_size))
    b = FakeModuleB(FakeModuleBConfig(hidden_size=hidden_size))
    config = OmniConfig(
        modules={"fake_module_a": {"subfolder": "fake_module_a"}, "fake_module_b": {"subfolder": "fake_module_b"}},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(module="fake_module_a"),
    )
    model = OmniModel(config, {"fake_module_a": a, "fake_module_b": b})

    hidden = torch.ones(2, hidden_size)
    batch: dict = {"hidden": hidden}
    out = model(batch)

    # Each Linear is ones-initialized, so a row of ones becomes 8, then 64.
    assert torch.allclose(batch["hidden"], torch.full((2, hidden_size), 64.0))
    assert out == {"loss": None, "losses": {}}


class _ArtefactModule(nn.Module):
    """Generation stand-in that emits one artefact per ``generate`` call."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def generate(self, **kwargs):
        return {"generated": {"type": "text", "value": self.name}}


def test_generate_keeps_every_artefact_a_body_pass_emits():
    """``ctx`` is one shared dict: a later node overwrites an earlier artefact.

    Draining once per body pass therefore kept only whatever the last node of
    the pass wrote. Both nodes of an ``a -> b -> end`` body emit here.
    """
    config = OmniConfig(
        modules={"module_A": {"subfolder": "module_A"}, "module_B": {"subfolder": "module_B"}},
        training_graph=[{"from": "module_A", "to": "module_B"}, {"from": "module_B", "to": "end"}],
        generation_graphs={
            "infer_gen": {
                "initial": "run",
                "states": {
                    "run": {
                        "body": [{"from": "module_A", "to": "module_B"}, {"from": "module_B", "to": "end"}],
                        "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                    }
                },
            }
        },
    )
    model = OmniModel(config, {"module_A": _ArtefactModule("module_A"), "module_B": _ArtefactModule("module_B")})

    model.reset()
    generated = model.generate({})

    assert [item["value"] for item in generated] == ["module_A", "module_B"]


def test_omni_model_without_a_training_graph_generates_but_refuses_to_train():
    """An inference-only checkpoint (``training_graph: []``) must still load."""
    config = OmniConfig(
        modules={"module_A": {"subfolder": "module_A"}},
        training_graph=[],
        generation_graphs=_minimal_generation_graphs(module="module_A"),
    )

    model = OmniModel(config, {"module_A": _ArtefactModule("module_A")})

    assert model.training_graph is None
    assert [item["value"] for item in model.generate({})] == ["module_A"]
    with pytest.raises(ValueError, match="no training graph"):
        model({})


def test_modeling_omni_imports_no_veomni_runtime_package():
    """``modeling_omni`` must stay liftable into another framework.

    Everything runtime-specific about running a node (wrapper unwrap,
    ParallelState scoping) reaches ``OmniModel.forward``
    through its ``node_runner`` argument, so the modeling needs no import from
    VeOmni's accelerator / distributed / trainer layers — not even a lazy one
    inside a function body.
    """
    import ast
    import pathlib

    from veomni.models.seed_omni import modeling_omni

    forbidden = {"accelerator", "distributed", "trainer"}

    def _veomni_paths(stmt: ast.stmt) -> list[list[str]]:
        """Segments of each imported first-party path, ``veomni.`` prefix stripped."""
        if isinstance(stmt, ast.Import):
            return [a.name.split(".")[1:] for a in stmt.names if a.name.startswith("veomni.")]
        if isinstance(stmt, ast.ImportFrom):
            base = (stmt.module or "").split(".")
            if stmt.level == 0:  # absolute: only veomni is first-party
                if base[:1] != ["veomni"]:
                    return []
                base = base[1:]
            return [[*base, a.name] for a in stmt.names]
        return []

    tree = ast.parse(pathlib.Path(modeling_omni.__file__).read_text(encoding="utf-8"))
    offenders = [
        ".".join(path) for stmt in ast.walk(tree) for path in _veomni_paths(stmt) if forbidden.intersection(path)
    ]

    assert not offenders, f"modeling_omni must not import VeOmni runtime code: {sorted(set(offenders))}"


# Generation FSM tests below drive the graph only: it selects nodes, it never calls one.


def _two_state_graph() -> GenerationGraph:
    """``s1`` watches one signal and falls back to ``default``; ``s2`` has a two-node body."""
    return GenerationGraph(
        {
            "initial": "s1",
            "states": {
                "s1": {
                    "body": [{"from": "a", "to": "end"}],
                    "transitions": [
                        {"condition": {"type": "module_signal", "key": "watched"}, "next_state": "s2"},
                        {"condition": {"type": "default"}, "next_state": "s2"},
                    ],
                },
                "s2": {
                    "body": [{"from": "c", "to": "d"}, {"from": "d", "to": "end"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                },
            },
        }
    )


def test_body_runs_in_declared_order():
    g = _two_state_graph()
    ctx: dict = {}
    assert [n.name for n in g.iter_nodes(ctx)] == ["a.generate"]
    g.maybe_transition(ctx)
    assert [n.name for n in g.iter_nodes(ctx)] == ["c.generate", "d.generate"]


def test_signal_mid_body_stops_the_rest_of_the_body():
    """A node saying 'done' ends the body it was running in, not just its own edge."""
    g = _two_state_graph()
    ctx: dict = {}
    g.maybe_transition(ctx)  # leave s1 for s2 on default

    ran = []
    for node in g.iter_nodes(ctx):
        ran.append(node.name)
        ctx["module_signal"] = "watched"  # as if the node wrote it
    assert ran == ["c.generate"]


def test_matched_signal_fires_its_transition_and_clears_the_signal():
    g = _two_state_graph()
    ctx = {"module_signal": "watched"}
    fired = g.maybe_transition(ctx)
    assert (fired.to_state, fired.condition) == ("s2", "module_signal(watched)")
    assert "module_signal" not in ctx


def test_unmatched_signal_is_cleared_by_the_default_transition():
    """A signal is one-shot per body, whichever transition consumes the state.

    A module may emit a signal the current state does not name — it leaves on
    ``default`` instead. Were the key left behind, ``iter_nodes`` would see it
    as "stop" after the first node of every later body, for the rest of the run.
    """
    g = _two_state_graph()
    ctx = {"module_signal": "not_watched_here"}

    fired = g.maybe_transition(ctx)
    assert (fired.to_state, fired.condition) == ("s2", "default")
    assert "module_signal" not in ctx
    assert [n.name for n in g.iter_nodes(ctx)] == ["c.generate", "d.generate"]


def test_a_signal_no_transition_consumes_does_not_wedge_the_next_pass():
    """A loop state must keep running its whole body after an unwatched signal.

    A state that stays put until one named signal fires — the shape of every AR
    decode loop — has no ``default`` to fall through, so ``maybe_transition``
    matches nothing and pops nothing. If the body pass did not clear the signal
    itself, the next pass would stop after its first node, and every pass after
    that, burning steps up to ``max_new_tokens`` while the graph made no
    progress and raised nothing.
    """
    g = GenerationGraph(
        {
            "initial": "loop",
            "states": {
                "loop": {
                    "body": [{"from": "c", "to": "d"}, {"from": "d", "to": "end"}],
                    "transitions": [
                        {"condition": {"type": "module_signal", "key": "watched"}, "next_state": "done"},
                    ],
                },
            },
        }
    )
    ctx: dict = {}

    # Pass 1: a node emits a signal this state does not watch.
    ran = [node.name for node in g.iter_nodes(ctx)]
    ctx["module_signal"] = "not_watched_here"
    assert ran == ["c.generate", "d.generate"]

    assert g.maybe_transition(ctx) is None  # nothing matches, so nothing pops
    assert g.current_state_name == "loop"

    # Pass 2 must be a full body, not a single node.
    assert [node.name for node in g.iter_nodes(ctx)] == ["c.generate", "d.generate"]


def test_a_feedback_edge_does_not_gate_its_destination():
    """``to: X`` *after* X's own turn as a source is feedback, not an input.

    It updates ctx for the next iteration, so X must still run on first sight
    rather than waiting on an edge that only exists to feed the round after it.
    """
    g = GenerationGraph(
        {
            "initial": "s1",
            "states": {
                "s1": {
                    "body": [{"from": "d", "to": "end"}, {"from": "c", "to": "d"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    )
    assert [n.name for n in g.iter_nodes({})] == ["d.generate", "c.generate"]


def test_to_mermaid_multi_method_contains_node_labels_and_end_sink():
    g = TrainingGraph(_multi_method_edges())
    out = g.to_mermaid(title="Multi-method training")

    # Frontmatter, ELK renderer hint, then LR flowchart.
    assert out.startswith("---\ntitle: Multi-method training\n---\n")
    assert "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%" in out
    assert "flowchart LR" in out

    # Node ids sanitise dots → underscores; labels keep the canonical name.
    assert re.search(r'\bmodule_A_forward\["<i>module_A\.forward</i>"\]', out)
    assert re.search(r'\bmodule_B_encode\["<i>module_B\.encode</i>"\]', out)
    assert re.search(r'\bmodule_C_forward\["<i>module_C\.forward</i>"\]', out)
    assert re.search(r'\bmodule_B_decode\["<i>module_B\.decode</i>"\]', out)

    assert "module_A_forward -->" in out and "module_C_forward" in out
    assert "module_B_encode -->" in out
    assert "module_C_forward -->" in out and "module_B_decode" in out

    # `end` rendered as the dashed terminal.
    assert "end_sink" in out and "module_B_decode --> end_sink" in out

    assert ":::source" in out and ":::sink" in out

    # Per-rank invisible subgraphs (col0 = sources, col1 = middle, col2 = sinks).
    assert "subgraph col0" in out and "subgraph col1" in out and "subgraph col2" in out
    assert "style col0 fill:transparent,stroke:none" in out

    assert "data -.-> module_A_forward" in out
    assert "data -.-> module_B_encode" in out

    # Single-loss protocol — no `losses` collector node.
    assert "losses" not in out


def test_to_mermaid_always_draws_data_pseudo_node():
    g = TrainingGraph(_multi_method_edges())
    out = g.to_mermaid()
    assert "data[(data)]" in out
    assert "data -.-> module_A_forward" in out
    assert "losses" not in out
    assert "end_sink" in out


def test_generation_graph_mermaid_stacks_state_body_nodes():
    g = GenerationGraph(
        {
            "initial": "prompt",
            "states": {
                "prompt": {
                    "body": [{"from": "encoder.encode", "to": "decoder.decode"}],
                    "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
                }
            },
        }
    )

    out = g.to_mermaid(title="Compact FSM")

    assert "flowchart LR" in out
    assert "subgraph state_prompt [prompt]\n        direction TB" in out
    assert "prompt__encoder_encode --> prompt__decoder_decode" in out


class _DdpStyleWrapper(nn.Module):
    """Minimal DDP-shaped wrapper: hooks live on ``.module``."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.module = inner

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_named_omni_modules_yields_modules_as_attached():
    """Bare :class:`OmniModel` yields sub-modules exactly as stored (no unwrap)."""
    edges = _fan_in_edges()
    g = TrainingGraph(edges)
    raw_modules = _fake_modules(g)
    wrapped_modules = {name: _DdpStyleWrapper(mod) for name, mod in raw_modules.items()}
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in raw_modules},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, wrapped_modules)

    resolved = dict(model.named_omni_modules())
    assert set(resolved) == set(wrapped_modules)
    for name, mod in resolved.items():
        assert mod is wrapped_modules[name]


def test_named_omni_modules_yields_all_graph_participants():
    class _PlainModule(nn.Module):
        def forward(self, x):
            return x

    plain = _PlainModule()
    config = OmniConfig(
        modules={"plain": {"subfolder": "plain"}},
        training_graph=[{"from": "plain", "to": "end"}],
        generation_graphs=_minimal_generation_graphs("plain"),
    )
    model = OmniModel(config, {"plain": plain})
    assert list(model.named_omni_modules()) == [("plain", plain)]
