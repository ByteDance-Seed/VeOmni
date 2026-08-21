from __future__ import annotations

import pytest
import yaml

from tests.seed_omni.bagel.helpers import bagel_cfg_dir, load_omni_config
from veomni.models.seed_omni.graphs.generation_graph import GenerationGraph


def test_bagel_train_yaml_loads_with_v2_module_names():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / "graph_infer_gen.yaml",
    )

    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert isinstance(cfg.training_graph, list) and cfg.training_graph
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "bagel_text_encoder.encode" in endpoints
    assert "bagel_siglip_navit" in endpoints
    assert "bagel_vae.encode" in endpoints
    assert "bagel_flow_connector.embed_latent" in endpoints
    assert "bagel_text_encoder.decode" in endpoints
    assert "bagel_flow_connector.decode_velocity" in endpoints
    assert "end" in endpoints
    assert cfg.module_ops_implementation("bagel_qwen2_mot")["attn_implementation"] == "veomni_flex_attention_with_sp"


def test_bagel_train_graph_fan_in_execution_order():
    from veomni.models.seed_omni.graphs.training_graph import TrainingGraph

    graph = TrainingGraph(yaml.safe_load((bagel_cfg_dir() / "graph_train.yaml").read_text()))
    order = graph.execution_order
    assert order.index("bagel_qwen2_mot.forward") > order.index("bagel_text_encoder.encode")
    assert order.index("bagel_qwen2_mot.forward") > order.index("bagel_siglip_navit.forward")
    assert order.index("bagel_qwen2_mot.forward") > order.index("bagel_flow_connector.embed_latent")
    assert order.index("bagel_flow_connector.embed_latent") > order.index("bagel_vae.encode")
    assert set(graph.sources) == {
        "bagel_text_encoder.encode",
        "bagel_siglip_navit.forward",
        "bagel_vae.encode",
    }


def test_bagel_infer_gen_graph_uses_siglip_context_without_vae_context():
    data = yaml.safe_load((bagel_cfg_dir() / "graph_infer_gen.yaml").read_text())
    prompt_body = data["states"]["prompt_encode"]["body"]

    assert {"from": "bagel_text_encoder", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_siglip_navit", "to": "bagel_qwen2_mot"} in prompt_body
    assert all(edge["from"] != "bagel_vae.encode_context" for edge in prompt_body)
    assert all(edge["from"] != "bagel_flow_connector.embed_context_latents" for edge in prompt_body)


def test_bagel_edit_prompt_graph_exposes_independent_prompt_producers() -> None:
    graph_config = yaml.safe_load((bagel_cfg_dir() / "graph_infer_edit.yaml").read_text())
    prompt_body = graph_config["states"]["prompt_encode"]["body"]

    assert {"from": "bagel_text_encoder", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_siglip_navit", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_vae.encode_context", "to": "bagel_flow_connector.embed_context_latents"} in prompt_body
    assert {"from": "bagel_flow_connector.embed_context_latents", "to": "bagel_qwen2_mot"} in prompt_body
    assert all(edge["to"] != "bagel_text_encoder.encode_image_markers" for edge in prompt_body)

    forbidden_edges = {
        ("bagel_vae.encode_context", "bagel_siglip_navit"),
        ("bagel_siglip_navit", "bagel_vae.encode_context"),
        ("bagel_siglip_navit", "bagel_flow_connector.embed_context_latents"),
    }
    assert not forbidden_edges.intersection({(edge["from"], edge["to"]) for edge in prompt_body})

    graph = GenerationGraph(graph_config)
    sequence = graph.state_node_sequence("prompt_encode")
    assert "bagel_vae.encode_context" in sequence
    assert "bagel_siglip_navit.generate" in sequence


@pytest.mark.parametrize(
    "infer_graph",
    ["graph_infer_und.yaml", "graph_infer_gen.yaml", "graph_infer_edit.yaml"],
)
def test_bagel_train_plus_infer_merges_generation_graph(infer_graph: str):
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_graph_path=bagel_cfg_dir() / infer_graph,
    )
    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert cfg.generation_graph is not None
    assert cfg.module_ops_implementation("bagel_qwen2_mot")["attn_implementation"] == "veomni_flex_attention_with_sp"
    assert cfg.generation_graph["initial"] == "prompt_encode"
    assert "done" not in cfg.generation_graph["states"]
    assert any(
        t.get("next_state") == "done"
        for state in cfg.generation_graph["states"].values()
        for t in state.get("transitions", [])
    ), f"{infer_graph} has no transition to `done`."
    for state_name, state in cfg.generation_graph["states"].items():
        for e in state.get("body", []):
            assert isinstance(e, dict) and "from" in e and "to" in e, (
                f"state '{state_name}' body item must be a `{{from, to}}` dict: {e!r}"
            )
