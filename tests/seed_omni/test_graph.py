"""Unit tests for the SeedOmni V2 graph layer (flat edge-list training subset)."""

from __future__ import annotations

import re

import pytest

from veomni.models.seed_omni import EdgeDef, NodeDef
from veomni.models.seed_omni.generation_graph import GenerationGraph
from veomni.models.seed_omni.graph import END
from veomni.models.seed_omni.training_graph import TrainingGraph


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


# ── collect_inputs ────────────────────────────────────────────────────────────


def test_collect_inputs_returns_raw_batch_copy():
    """Topology-only edges: every node sees the same shared batch dict."""
    g = TrainingGraph(edges=_janus_joint_edges())
    raw_batch = {"conversation_list": {"hidden_states": "HID"}, "input_ids": "X"}
    upstream = {"run_ar.forward": {"hidden_states": "SHOULD_NOT_ROUTE"}}
    kwargs = g.collect_inputs("vq_decoder.gen_loss", upstream, raw_batch)
    assert kwargs is not raw_batch
    assert kwargs == raw_batch
    assert kwargs["conversation_list"]["hidden_states"] == "HID"


def test_collect_inputs_ignores_upstream_outputs():
    """Upstream node output dicts are not merged into kwargs — carrier holds cross-node state."""
    g = TrainingGraph(edges=_understanding_only_edges())
    upstream = {
        "vision_encoder.forward": {"image_embeds": "VIS"},
        "vq_decoder.forward": {"gen_embeds": "GEN"},
    }
    kwargs = g.collect_inputs("run_ar.forward", upstream, {"input_ids": "X"})
    assert kwargs == {"input_ids": "X"}
    assert "und_image_embeds" not in kwargs


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


# ── input_ids sequence helpers (HF generate alignment) ───────────────────────


def test_append_input_ids_grows_sequence():
    import torch

    from veomni.models.seed_omni.graph import append_input_ids, is_step_input_ids, scalar_token_id

    prompt = torch.tensor([[100, 101]], dtype=torch.long)
    step = torch.tensor([[42]], dtype=torch.long)
    full = append_input_ids(prompt, step)
    assert full.tolist() == [[100, 101, 42]]
    assert is_step_input_ids(step)
    assert not is_step_input_ids(prompt)
    assert scalar_token_id(full) == 42
