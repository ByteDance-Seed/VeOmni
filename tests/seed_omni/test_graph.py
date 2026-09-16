"""Unit tests for the SeedOmni V2 graph layer (flat edge-list training subset)."""

from __future__ import annotations

import re

import pytest
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
from veomni.models.seed_omni.utils import graph_profiler
from veomni.models.seed_omni.utils.graph_profiler import GraphProfiler


def test_from_endpoint_default_method():
    n = NodeDef.from_endpoint("ar_llm", default_method="forward")
    assert n.module == "ar_llm" and n.method == "forward"
    assert n.name == "ar_llm.forward"


def test_from_endpoint_dotted_form():
    n = NodeDef.from_endpoint("vq_decoder.encode", default_method="forward")
    assert n.module == "vq_decoder" and n.method == "encode"
    assert n.name == "vq_decoder.encode"


def test_from_endpoint_generate_default():
    n = NodeDef.from_endpoint("ar_llm", default_method="generate")
    assert n.module == "ar_llm" and n.method == "generate"


def test_from_endpoint_rejects_reserved_end():
    with pytest.raises(ValueError, match=f"'{END}' is the virtual sink"):
        NodeDef.from_endpoint(END, default_method="forward")


def test_from_endpoint_rejects_empty():
    with pytest.raises(ValueError, match="non-empty 'module"):
        NodeDef.from_endpoint("   ", default_method="forward")


def test_parse_edge():
    e = EdgeDef.parse({"from": "vision_encoder", "to": "run_ar"}, default_method="forward")
    assert e.from_ == "vision_encoder.forward" and e.to == "run_ar.forward"
    assert e.from_node.module == "vision_encoder" and e.to_node.module == "run_ar"
    assert not e.is_sink()


def test_parse_edge_to_end_is_sink():
    e = EdgeDef.parse({"from": "tok_decode", "to": "end"}, default_method="forward")
    assert e.is_sink() and e.to == END and e.to_node is None
    assert e.from_ == "tok_decode.forward"


def test_parse_edge_rejects_from_end():
    with pytest.raises(ValueError, match="`from: end` is forbidden"):
        EdgeDef.parse({"from": "end", "to": "run_ar"}, default_method="forward")


def test_parse_edge_rejects_node_fields():
    with pytest.raises(ValueError, match="must not contain node fields"):
        EdgeDef.parse({"from": "a", "to": "b", "module": "x"}, default_method="forward")


def test_parse_edge_rejects_missing_endpoints():
    with pytest.raises(ValueError, match="must declare both"):
        EdgeDef.parse({"from": "a"}, default_method="forward")


def _janus_joint_edges() -> list[dict]:
    """Janus joint training edges: vq_decoder appears under TWO methods.

    Adds an explicit ``to: end`` sink for the leaf so every node is visible to
    the active subset purely from the edge list.
    """
    return [
        {"from": "vision_encoder", "to": "run_ar"},
        {"from": "vq_decoder.encode", "to": "run_ar"},
        {"from": "run_ar", "to": "vq_decoder.gen_loss"},
        {"from": "vq_decoder.gen_loss", "to": "end"},
    ]


def _understanding_only_edges() -> list[dict]:
    """Two encoders → ar_llm, simple DAG with end-sink."""
    return [
        {"from": "vision_encoder", "to": "run_ar"},
        {"from": "vq_decoder", "to": "run_ar"},
        {"from": "run_ar", "to": "end"},
    ]


def test_missing_edges_raises():
    with pytest.raises(ValueError, match="non-empty `training_graph`"):
        TrainingGraph([])


def test_duplicate_edge_raises():
    with pytest.raises(ValueError, match="Duplicate edge"):
        TrainingGraph(
            [
                {"from": "vision_encoder", "to": "run_ar"},
                {"from": "vision_encoder", "to": "run_ar"},
            ]
        )


def test_single_node_with_only_end_edge():
    """``[{from: ar_llm, to: end}]`` derives exactly one real node."""
    g = TrainingGraph([{"from": "ar_llm", "to": "end"}])
    assert g.execution_order == ["ar_llm.forward"]
    assert g.sources == ["ar_llm.forward"] and g.sinks == ["ar_llm.forward"]


def test_understanding_only_topological_order():
    g = TrainingGraph(_understanding_only_edges())
    assert g.execution_order[-1] == "run_ar.forward"
    assert set(g.execution_order[:-1]) == {"vision_encoder.forward", "vq_decoder.forward"}


def test_janus_joint_topological_order():
    """vq_decoder appears as TWO nodes; topo must place them on either side of run_ar."""
    g = TrainingGraph(_janus_joint_edges())
    order = g.execution_order
    assert order.index("vq_decoder.gen_loss") > order.index("run_ar.forward")
    assert order.index("run_ar.forward") > order.index("vq_decoder.encode")
    assert order.index("run_ar.forward") > order.index("vision_encoder.forward")


def test_cycle_in_active_set_raises():
    with pytest.raises(ValueError, match="Circular dependency"):
        TrainingGraph(
            [
                {"from": "vq_decoder", "to": "ar_llm"},
                {"from": "ar_llm", "to": "vq_decoder"},
            ]
        )


def test_sources_and_sinks_understanding_only():
    g = TrainingGraph(_understanding_only_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.forward"}
    # run_ar's only outgoing edge targets `end`, so it's a sink.
    assert g.sinks == ["run_ar.forward"]


def test_sources_and_sinks_janus_joint():
    g = TrainingGraph(_janus_joint_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.encode"}
    # vq_decoder.gen_loss is the only sink (its only outgoing edge goes to `end`).
    assert g.sinks == ["vq_decoder.gen_loss"]


def test_module_and_method_lookup():
    g = TrainingGraph(_janus_joint_edges())
    assert g.module_of("vq_decoder.encode") == "vq_decoder"
    assert g.method_of("vq_decoder.encode") == "encode"
    assert g.module_of("vq_decoder.gen_loss") == "vq_decoder"
    assert g.method_of("vq_decoder.gen_loss") == "gen_loss"
    assert g.method_of("run_ar.forward") == "forward"


def test_module_lookup_raises_for_unknown():
    g = TrainingGraph(_janus_joint_edges())
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
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.forward")
        return {"conversation_list": cl}

    def encode(self, **kwargs):
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.encode")
        return {"conversation_list": cl}

    def generate(self, **kwargs):
        cl = list(kwargs.get("conversation_list", []))
        cl.append(f"{self.name}.generate")
        return {"conversation_list": cl}


def _fake_modules(g: TrainingGraph) -> dict:
    return {name: _FakeOmniModule(name) for name in {g.module_of(n) for n in g.execution_order}}


def test_cursor_lifecycle():
    g = TrainingGraph(_understanding_only_edges())
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


def _minimal_generation_graph(module: str = "run_ar") -> dict:
    return {
        "initial": "run",
        "states": {
            "run": {
                "body": [{"from": module, "to": "end"}],
                "transitions": [{"condition": {"type": "default"}, "next_state": "done"}],
            }
        },
    }


def _minimal_generation_graphs(module: str = "run_ar") -> dict:
    return {"infer_gen": _minimal_generation_graph(module)}


def test_omni_model_forward_runs_graph_without_a_runner():
    """Without an injected runner the graph runs on the modeling's own eager path."""
    edges = _understanding_only_edges()
    g = TrainingGraph(edges)
    modules = _fake_modules(g)
    config = OmniConfig(
        modules={name: {"subfolder": name} for name in {g.module_of(n) for n in g.execution_order}},
        training_graph=edges,
        generation_graphs=_minimal_generation_graphs(),
    )
    model = OmniModel(config, modules)

    batch: dict = {"conversation_list": []}
    out = model(batch)

    assert batch["conversation_list"] == [f"{g.module_of(n)}.forward" for n in g.execution_order]
    assert out == {"loss": None, "losses": {}}


def test_modeling_omni_imports_no_veomni_runtime_package():
    """``modeling_omni`` must stay liftable into another framework.

    Everything runtime-specific about running a node (wrapper unwrap,
    ParallelState scoping, metering, profiling) reaches ``OmniModel.forward``
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


def test_graph_profiler_can_append_request_peak_memory(monkeypatch):
    class _FakeDevice:
        def __init__(self):
            self.reset_calls = 0

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def max_memory_allocated(self):
            return 2 * 1024**3

        def max_memory_reserved(self):
            return 3 * 1024**3

    device = _FakeDevice()
    monkeypatch.setattr(graph_profiler, "get_torch_device", lambda: device)

    profiler = GraphProfiler(enable_memory=True)
    with profiler.node("forward:run_ar.forward"):
        pass

    assert device.reset_calls == 1
    assert profiler.save_records() == ["forward:run_ar.forward | peak_allocated_gb=2.000 | peak_reserved_gb=3.000"]


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


def test_to_mermaid_janus_joint_contains_node_labels_and_end_sink():
    g = TrainingGraph(_janus_joint_edges())
    out = g.to_mermaid(title="Janus Joint Training")

    # Frontmatter, ELK renderer hint, then LR flowchart.
    assert out.startswith("---\ntitle: Janus Joint Training\n---\n")
    assert "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%" in out
    assert "flowchart LR" in out

    # Node ids sanitise dots → underscores; labels keep the canonical name.
    assert re.search(r'\bvision_encoder_forward\["<i>vision_encoder\.forward</i>"\]', out)
    assert re.search(r'\bvq_decoder_encode\["<i>vq_decoder\.encode</i>"\]', out)
    assert re.search(r'\brun_ar_forward\["<i>run_ar\.forward</i>"\]', out)
    assert re.search(r'\bvq_decoder_gen_loss\["<i>vq_decoder\.gen_loss</i>"\]', out)

    assert "vision_encoder_forward -->" in out and "run_ar_forward" in out
    assert "vq_decoder_encode -->" in out
    assert "run_ar_forward -->" in out and "vq_decoder_gen_loss" in out

    # `end` rendered as the dashed terminal.
    assert "end_sink" in out and "vq_decoder_gen_loss --> end_sink" in out

    assert ":::source" in out and ":::sink" in out

    # Per-rank invisible subgraphs (col0 = sources, col1 = middle, col2 = sinks).
    assert "subgraph col0" in out and "subgraph col1" in out and "subgraph col2" in out
    assert "style col0 fill:transparent,stroke:none" in out

    assert "data -.-> vision_encoder_forward" in out
    assert "data -.-> vq_decoder_encode" in out

    # Single-loss protocol — no `losses` collector node.
    assert "losses" not in out


def test_to_mermaid_always_draws_data_pseudo_node():
    g = TrainingGraph(_janus_joint_edges())
    out = g.to_mermaid()
    assert "data[(data)]" in out
    assert "data -.-> vision_encoder_forward" in out
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
    edges = _understanding_only_edges()
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
