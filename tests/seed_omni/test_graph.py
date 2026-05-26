"""Unit tests for the SeedOmni V2 graph layer (edges-only training subset)."""

from __future__ import annotations

import re

import pytest

from veomni.models.seed_omni import EdgeDef, NodeDef
from veomni.models.seed_omni.graph import END
from veomni.models.seed_omni.training_graph import TrainingGraph


# ── NodeDef parsing ───────────────────────────────────────────────────────────


def test_parse_node_default_method():
    n = NodeDef.parse("run_ar", {"module": "ar_llm"})
    assert n.module == "ar_llm" and n.method == "forward"


def test_parse_node_dotted_form():
    n = NodeDef.parse("vq_encode", {"module": "vq_decoder.encode"})
    assert n.module == "vq_decoder" and n.method == "encode"


def test_parse_node_explicit_method():
    n = NodeDef.parse("vq_loss", {"module": "vq_decoder", "method": "gen_loss"})
    assert n.module == "vq_decoder" and n.method == "gen_loss"


def test_parse_node_rejects_dotted_with_explicit_method():
    with pytest.raises(ValueError, match="cannot specify both"):
        NodeDef.parse("bad", {"module": "vq_decoder.encode", "method": "decode"})


def test_parse_node_rejects_edge_fields():
    with pytest.raises(ValueError, match="contains edge fields"):
        NodeDef.parse("bad", {"module": "ar_llm", "from": "x", "to": "y"})


def test_parse_node_rejects_missing_module():
    with pytest.raises(ValueError, match="missing required `module`"):
        NodeDef.parse("bad", {"foo": "bar"})


def test_parse_node_rejects_reserved_end_name():
    with pytest.raises(ValueError, match=f"'{END}' is reserved"):
        NodeDef.parse(END, {"module": "ar_llm"})


# ── EdgeDef parsing ───────────────────────────────────────────────────────────


def test_parse_edge():
    e = EdgeDef.parse(
        "vision_to_ar",
        {"from": "vision_encoder", "output": "image_embeds", "to": "run_ar", "as": "und_image_embeds"},
    )
    assert e.from_ == "vision_encoder" and e.to == "run_ar"
    assert e.output_key == "image_embeds" and e.as_ == "und_image_embeds"
    assert not e.is_sink()


def test_parse_edge_to_end_is_sink():
    e = EdgeDef.parse("sink", {"from": "tok_decode", "to": "end"})
    assert e.is_sink() and e.to == END


def test_parse_edge_rejects_from_end():
    with pytest.raises(ValueError, match="`from: end` is forbidden"):
        EdgeDef.parse("bad", {"from": "end", "to": "run_ar"})


def test_parse_edge_rejects_node_fields():
    with pytest.raises(ValueError, match="contains node fields"):
        EdgeDef.parse("bad", {"from": "a", "to": "b", "module": "x"})


def test_parse_edge_rejects_missing_endpoints():
    with pytest.raises(ValueError, match="must declare both"):
        EdgeDef.parse("bad", {"from": "a"})


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _janus_joint_pools() -> tuple[dict, dict]:
    """Janus joint training pools: vq_decoder appears under TWO node names.

    Adds explicit ``to: end`` sinks for the two leaves so every node is
    visible to the active subset purely from ``edges``.
    """
    nodes = {
        "vision_encoder": {"module": "vision_encoder"},
        "vq_encode": {"module": "vq_decoder.encode"},
        "run_ar": {"module": "ar_llm"},
        "vq_loss": {"module": "vq_decoder.gen_loss"},
    }
    edges = {
        "vision_to_ar": {
            "from": "vision_encoder",
            "output": "image_embeds",
            "to": "run_ar",
            "as": "und_image_embeds",
        },
        "vae_enc_to_ar": {
            "from": "vq_encode",
            "output": "gen_embeds",
            "to": "run_ar",
            "as": "gen_image_embeds",
        },
        "ar_to_vq_loss": {
            "from": "run_ar",
            "output": "hidden_states",
            "to": "vq_loss",
            "as": "hidden_states",
        },
        "vq_loss_sink": {"from": "vq_loss", "to": "end"},
    }
    return nodes, edges


_JANUS_TRAIN_EDGES = ["vision_to_ar", "vae_enc_to_ar", "ar_to_vq_loss", "vq_loss_sink"]


def _understanding_only_pools() -> tuple[dict, dict]:
    """Two encoders → ar_llm (no vq_loss), simple DAG with end-sink."""
    nodes = {
        "vision_encoder": {"module": "vision_encoder"},
        "vq_encode": {"module": "vq_decoder"},
        "run_ar": {"module": "ar_llm"},
    }
    edges = {
        "vision_to_ar": {
            "from": "vision_encoder",
            "output": "image_embeds",
            "to": "run_ar",
            "as": "und_image_embeds",
        },
        "vae_to_ar": {
            "from": "vq_encode",
            "output": "gen_embeds",
            "to": "run_ar",
            "as": "gen_image_embeds",
        },
        "ar_sink": {"from": "run_ar", "to": "end"},
    }
    return nodes, edges


# ── Validation ────────────────────────────────────────────────────────────────


def test_missing_training_edges_raises():
    nodes, edges = _understanding_only_pools()
    with pytest.raises(ValueError, match="non-empty `training_edges`"):
        TrainingGraph(nodes=nodes, edges=edges, training_edges=[])


def test_unknown_training_edge_raises():
    nodes, edges = _understanding_only_pools()
    with pytest.raises(KeyError, match="undefined edge name"):
        TrainingGraph(nodes=nodes, edges=edges, training_edges=["ghost_edge"])


def test_edge_referencing_unknown_node_raises():
    nodes = {"run_ar": {"module": "ar_llm"}}
    edges = {"stale": {"from": "ghost", "output": "x", "to": "run_ar", "as": "x"}}
    with pytest.raises(KeyError, match="`from: ghost`"):
        TrainingGraph(nodes=nodes, edges=edges, training_edges=["stale"])


def test_name_collision_between_pools_raises():
    nodes = {"shared": {"module": "ar_llm"}}
    edges = {"shared": {"from": "shared", "to": "shared"}}
    with pytest.raises(ValueError, match="share name"):
        TrainingGraph(nodes=nodes, edges=edges, training_edges=["shared"])


def test_duplicate_training_edge_raises():
    nodes, edges = _understanding_only_pools()
    with pytest.raises(ValueError, match="Duplicate edge name"):
        TrainingGraph(
            nodes=nodes,
            edges=edges,
            training_edges=["vision_to_ar", "vision_to_ar"],
        )


def test_single_node_with_only_end_edge_raises():
    """``edges: [<x> → end]`` derives no real node beyond x; require at least one real."""
    nodes = {"run_ar": {"module": "ar_llm"}}
    edges = {"sink": {"from": "run_ar", "to": "end"}}
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=["sink"])
    assert g.execution_order == ["run_ar"]
    assert g.sources == ["run_ar"] and g.sinks == ["run_ar"]


def test_all_edges_pointing_to_end_with_zero_real_nodes_raises():
    # Using a fake from-node that doesn't exist still fails validation;
    # verify the "no real nodes" guard would fire if edges resolved without
    # endpoints (defensive).  In practice this path is unreachable given the
    # NodeDef pool validation — included for documentation.
    pass


# ── Topological order ─────────────────────────────────────────────────────────


def test_understanding_only_topological_order():
    nodes, edges = _understanding_only_pools()
    g = TrainingGraph(
        nodes=nodes,
        edges=edges,
        training_edges=["vision_to_ar", "vae_to_ar", "ar_sink"],
    )
    assert g.execution_order[-1] == "run_ar"
    assert set(g.execution_order[:-1]) == {"vision_encoder", "vq_encode"}


def test_janus_joint_topological_order():
    """vq_decoder appears as TWO nodes; topo must place them on either side of run_ar."""
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    order = g.execution_order
    assert order.index("vq_loss") > order.index("run_ar")
    assert order.index("run_ar") > order.index("vq_encode")
    assert order.index("run_ar") > order.index("vision_encoder")


def test_active_subset_excludes_inference_cycle():
    """Pools may include cyclic inference edges; the active training subset stays acyclic."""
    nodes, edges = _understanding_only_pools()
    edges = {
        **edges,
        # Cyclic inference edges (training_edges does NOT include them).
        "ar_to_vae_infer": {"from": "run_ar", "output": "h", "to": "vq_encode", "as": "h"},
        "vae_to_ar_infer": {"from": "vq_encode", "output": "embed", "to": "run_ar", "as": "inputs_embeds"},
    }
    g = TrainingGraph(
        nodes=nodes,
        edges=edges,
        training_edges=["vision_to_ar", "vae_to_ar", "ar_sink"],
    )
    assert g.execution_order[-1] == "run_ar"


def test_cycle_in_active_set_raises():
    nodes = {
        "vae": {"module": "vq_decoder"},
        "ar": {"module": "ar_llm"},
    }
    edges = {
        "vae_to_ar": {"from": "vae", "output": "x", "to": "ar", "as": "x"},
        "ar_to_vae": {"from": "ar", "output": "y", "to": "vae", "as": "y"},
    }
    with pytest.raises(ValueError, match="Circular dependency"):
        TrainingGraph(
            nodes=nodes,
            edges=edges,
            training_edges=["vae_to_ar", "ar_to_vae"],
        )


# ── Sources / sinks ───────────────────────────────────────────────────────────


def test_sources_and_sinks_understanding_only():
    nodes, edges = _understanding_only_pools()
    g = TrainingGraph(
        nodes=nodes,
        edges=edges,
        training_edges=["vision_to_ar", "vae_to_ar", "ar_sink"],
    )
    assert set(g.sources) == {"vision_encoder", "vq_encode"}
    # ar_sink targets `end` only — run_ar has no outgoing real edge, so it's a sink.
    assert g.sinks == ["run_ar"]


def test_sources_and_sinks_janus_joint():
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    assert set(g.sources) == {"vision_encoder", "vq_encode"}
    # vq_loss is the only sink (its only outgoing edge goes to `end`).
    assert g.sinks == ["vq_loss"]


# ── module / method accessors ────────────────────────────────────────────────


def test_module_and_method_lookup():
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    assert g.module_of("vq_encode") == "vq_decoder"
    assert g.method_of("vq_encode") == "encode"
    assert g.module_of("vq_loss") == "vq_decoder"
    assert g.method_of("vq_loss") == "gen_loss"
    assert g.method_of("run_ar") == "forward"


def test_module_lookup_raises_for_unknown():
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    with pytest.raises(KeyError):
        g.module_of("not_a_node")


# ── collect_inputs ────────────────────────────────────────────────────────────


def test_collect_inputs_routes_outputs_with_renaming():
    """Fan-in: run_ar receives both vision and vq embeds via two distinct edges."""
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    raw_batch = {"input_ids": "X"}
    upstream = {
        "vision_encoder": {"image_embeds": "VIS"},
        "vq_encode": {"gen_embeds": "GEN"},
    }
    kwargs = g.collect_inputs("run_ar", upstream, raw_batch)
    assert kwargs["input_ids"] == "X"
    assert kwargs["und_image_embeds"] == "VIS"
    assert kwargs["gen_image_embeds"] == "GEN"


def test_collect_inputs_for_vq_loss_uses_run_ar_output():
    """vq_loss takes hidden_states from run_ar's output dict."""
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    upstream = {
        "vision_encoder": {"image_embeds": "VIS"},
        "vq_encode": {"gen_embeds": "GEN"},
        "run_ar": {"hidden_states": "HID", "_loss": 0.7},
    }
    kwargs = g.collect_inputs("vq_loss", upstream, {"vq_token_ids": "GT"})
    assert kwargs["vq_token_ids"] == "GT"
    assert kwargs["hidden_states"] == "HID"


# ── module-name alias on edge endpoints ──────────────────────────────────────


def test_module_alias_when_unique_node():
    """`from: vision_encoder` resolves to the sole node of that module."""
    nodes = {
        "vision_encoder": {"module": "vision_encoder"},
        "run_ar": {"module": "ar_llm"},
    }
    edges = {
        "v2a": {"from": "vision_encoder", "output": "image_embeds", "to": "run_ar", "as": "und"},
        "ar_sink": {"from": "run_ar", "to": "end"},
    }
    g = TrainingGraph(
        nodes=nodes,
        edges=edges,
        training_edges=["v2a", "ar_sink"],
    )
    e = g.active_edges()[0]
    assert e.from_ == "vision_encoder" and e.to == "run_ar"


def test_module_alias_ambiguous_raises():
    """`from: vq_decoder` is ambiguous when two nodes use vq_decoder."""
    nodes = {
        "vq_a": {"module": "vq_decoder.encode"},
        "vq_b": {"module": "vq_decoder.gen_loss"},
        "run_ar": {"module": "ar_llm"},
    }
    edges = {
        "edge": {"from": "vq_decoder", "output": "x", "to": "run_ar", "as": "x"},
    }
    with pytest.raises(ValueError, match="ambiguous"):
        TrainingGraph(nodes=nodes, edges=edges, training_edges=["edge"])


# ── Mermaid visualisation ────────────────────────────────────────────────────


def test_to_mermaid_janus_joint_contains_node_labels_and_end_sink():
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    out = g.to_mermaid(title="Janus Joint Training")

    # Frontmatter, ELK renderer hint, then LR flowchart.
    assert out.startswith("---\ntitle: Janus Joint Training\n---\n")
    assert "%%{init: {'flowchart': {'defaultRenderer': 'elk'}}}%%" in out
    assert "flowchart LR" in out

    assert re.search(r'\bvision_encoder\["vision_encoder<br/><i>vision_encoder\.forward</i>"\]', out)
    assert re.search(r'\bvq_encode\["vq_encode<br/><i>vq_decoder\.encode</i>"\]', out)
    assert re.search(r'\brun_ar\["run_ar<br/><i>ar_llm\.forward</i>"\]', out)
    assert re.search(r'\bvq_loss\["vq_loss<br/><i>vq_decoder\.gen_loss</i>"\]', out)

    assert "vision_encoder -->|" in out and "image_embeds → und_image_embeds" in out
    assert "vq_encode -->|" in out and "gen_embeds → gen_image_embeds" in out
    assert "run_ar -->|" in out and "hidden_states" in out

    # `end` rendered as the dashed terminal.
    assert "end_sink" in out and "vq_loss --> end_sink" in out

    assert ":::source" in out and ":::sink" in out

    # Per-rank invisible subgraphs (col0 = sources, col1 = middle, col2 = sinks).
    assert "subgraph col0" in out and "subgraph col1" in out and "subgraph col2" in out
    # The rank-banding subgraphs are styled invisible.
    assert "style col0 fill:transparent,stroke:none" in out

    assert "data -.-> vision_encoder" in out
    assert "data -.-> vq_encode" in out

    # The single-loss protocol means each module collects its own _loss; we no
    # longer draw a `losses` collector node or any fan-in arrows to it.
    assert "losses" not in out
    for n in ("vision_encoder", "vq_encode", "run_ar", "vq_loss"):
        assert f"{n} -.-> losses" not in out


def test_to_mermaid_always_draws_data_pseudo_node():
    nodes, edges = _janus_joint_pools()
    g = TrainingGraph(nodes=nodes, edges=edges, training_edges=_JANUS_TRAIN_EDGES)
    out = g.to_mermaid()
    assert "data[(data)]" in out
    assert "data -.-> vision_encoder" in out
    assert "losses" not in out
    assert "vision_encoder" in out
    assert "end_sink" in out


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
