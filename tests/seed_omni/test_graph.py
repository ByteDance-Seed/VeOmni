"""Unit tests for the SeedOmni V2 graph layer (flat edge-list training subset)."""

from __future__ import annotations

import re

import pytest

from veomni.models.seed_omni import EdgeDef, NodeDef
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph
from veomni.models.seed_omni.graphs.graph import END
from veomni.models.seed_omni.graphs.training_graph import TrainingGraph


# ── NodeDef parsing ───────────────────────────────────────────────────────────


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


# ── EdgeDef parsing ───────────────────────────────────────────────────────────


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


# ── Fixtures ──────────────────────────────────────────────────────────────────


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


# ── Validation ────────────────────────────────────────────────────────────────


def test_missing_edges_raises():
    with pytest.raises(ValueError, match="non-empty `edges`"):
        TrainingGraph(edges=[])


def test_duplicate_edge_raises():
    with pytest.raises(ValueError, match="Duplicate edge"):
        TrainingGraph(
            edges=[
                {"from": "vision_encoder", "to": "run_ar"},
                {"from": "vision_encoder", "to": "run_ar"},
            ]
        )


def test_single_node_with_only_end_edge():
    """``[{from: ar_llm, to: end}]`` derives exactly one real node."""
    g = TrainingGraph(edges=[{"from": "ar_llm", "to": "end"}])
    assert g.execution_order == ["ar_llm.forward"]
    assert g.sources == ["ar_llm.forward"] and g.sinks == ["ar_llm.forward"]


# ── Topological order ─────────────────────────────────────────────────────────


def test_understanding_only_topological_order():
    g = TrainingGraph(edges=_understanding_only_edges())
    assert g.execution_order[-1] == "run_ar.forward"
    assert set(g.execution_order[:-1]) == {"vision_encoder.forward", "vq_decoder.forward"}


def test_janus_joint_topological_order():
    """vq_decoder appears as TWO nodes; topo must place them on either side of run_ar."""
    g = TrainingGraph(edges=_janus_joint_edges())
    order = g.execution_order
    assert order.index("vq_decoder.gen_loss") > order.index("run_ar.forward")
    assert order.index("run_ar.forward") > order.index("vq_decoder.encode")
    assert order.index("run_ar.forward") > order.index("vision_encoder.forward")


def test_cycle_in_active_set_raises():
    with pytest.raises(ValueError, match="Circular dependency"):
        TrainingGraph(
            edges=[
                {"from": "vq_decoder", "to": "ar_llm"},
                {"from": "ar_llm", "to": "vq_decoder"},
            ]
        )


# ── Sources / sinks ───────────────────────────────────────────────────────────


def test_sources_and_sinks_understanding_only():
    g = TrainingGraph(edges=_understanding_only_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.forward"}
    # run_ar's only outgoing edge targets `end`, so it's a sink.
    assert g.sinks == ["run_ar.forward"]


def test_sources_and_sinks_janus_joint():
    g = TrainingGraph(edges=_janus_joint_edges())
    assert set(g.sources) == {"vision_encoder.forward", "vq_decoder.encode"}
    # vq_decoder.gen_loss is the only sink (its only outgoing edge goes to `end`).
    assert g.sinks == ["vq_decoder.gen_loss"]


# ── module / method accessors ────────────────────────────────────────────────


def test_module_and_method_lookup():
    g = TrainingGraph(edges=_janus_joint_edges())
    assert g.module_of("vq_decoder.encode") == "vq_decoder"
    assert g.method_of("vq_decoder.encode") == "encode"
    assert g.module_of("vq_decoder.gen_loss") == "vq_decoder"
    assert g.method_of("vq_decoder.gen_loss") == "gen_loss"
    assert g.method_of("run_ar.forward") == "forward"


def test_module_lookup_raises_for_unknown():
    g = TrainingGraph(edges=_janus_joint_edges())
    with pytest.raises(KeyError):
        g.module_of("not_a_node")


# ── Execution lifecycle (cursor + step + maybe_transition) ────────────────────


class _FakeOmniModule:
    """Minimal stand-in for an OmniModule: callable (→ forward) + pre/post hooks.

    ``__call__`` delegates to ``self.forward`` so the non-``forward`` alias trick
    (``raw.forward = encode``) works exactly as on a real ``nn.Module``.
    """

    def __init__(self, name: str):
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


def _fake_modules(g: TrainingGraph) -> dict:
    return {name: _FakeOmniModule(name) for name in {g.module_of(n) for n in g.execution_order}}


def test_cursor_lifecycle():
    g = TrainingGraph(edges=_understanding_only_edges())
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


def test_step_loop_flows_carrier_in_topological_order():
    """Driving step + maybe_transition mirrors OmniModel.forward; carrier accretes per node."""
    g = TrainingGraph(edges=_understanding_only_edges())
    modules = _fake_modules(g)
    batch = {"conversation_list": []}
    trace: list[str] = []
    g.reset()
    while not g.is_done():
        batch = g.step(modules, batch, trace=trace)
        g.maybe_transition(trace=trace)
    # run_ar runs last; both encoders precede it.
    assert batch["conversation_list"][-1] == "run_ar.forward"
    assert set(batch["conversation_list"]) == {"vision_encoder.forward", "vq_decoder.forward", "run_ar.forward"}
    assert [t for t in trace if t.startswith("forward:")] == [f"forward:{n}" for n in g.execution_order]


def test_step_dispatches_non_forward_method_via_wrapper():
    """A dotted ``module.encode`` node must run the module's ``encode`` (alias trick)."""
    g = TrainingGraph(edges=[{"from": "vq_decoder.encode", "to": "end"}])
    modules = _fake_modules(g)
    batch = g.step(modules, {"conversation_list": []})
    assert batch["conversation_list"] == ["vq_decoder.encode"]
    # forward restored after the aliased call.
    assert modules["vq_decoder"].forward.__name__ == "forward"


def test_step_unwraps_ddp_style_wrapper():
    """A wrapper without ``pre_forward`` is unwrapped via ``.module`` (DDP)."""

    class _DDPWrap:
        def __init__(self, inner):
            self.module = inner

        def __call__(self, **kwargs):
            return self.module.forward(**kwargs)

    g = TrainingGraph(edges=[{"from": "run_ar", "to": "end"}])
    inner = _FakeOmniModule("run_ar")
    batch = g.step({"run_ar": _DDPWrap(inner)}, {"conversation_list": []})
    assert batch["conversation_list"] == ["run_ar.forward"]


def test_step_applies_module_scope():
    from contextlib import contextmanager

    scoped: list[str] = []

    @contextmanager
    def scope_fn(name: str):
        scoped.append(name)
        yield

    g = TrainingGraph(edges=[{"from": "run_ar", "to": "end"}])
    g.step(_fake_modules(g), {"conversation_list": []}, scope_fn=scope_fn)
    assert scoped == ["run_ar"]


def test_step_merges_loss_into_batch():
    class _LossModule(_FakeOmniModule):
        def forward(self, **kwargs):
            out = super().forward(**kwargs)
            out["_loss"] = 1.5
            return out

    g = TrainingGraph(edges=[{"from": "run_ar", "to": "end"}])
    batch = g.step({"run_ar": _LossModule("run_ar")}, {"conversation_list": []})
    assert batch["_loss"] == 1.5


def test_step_raises_for_missing_module():
    g = TrainingGraph(edges=[{"from": "run_ar", "to": "end"}])
    with pytest.raises(KeyError, match="missing from modules dict"):
        g.step({}, {"conversation_list": []})


# ── Mermaid visualisation ────────────────────────────────────────────────────


def test_to_mermaid_janus_joint_contains_node_labels_and_end_sink():
    g = TrainingGraph(edges=_janus_joint_edges())
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
    g = TrainingGraph(edges=_janus_joint_edges())
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
