from __future__ import annotations

import pytest
import yaml

from tests.seed_omni.bagel.contracts.helpers import bagel_cfg_dir, load_omni_config


def test_bagel_train_yaml_loads_with_v2_module_names():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
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


def test_bagel_train_graph_fan_in_execution_order():
    from veomni.models.seed_omni.graphs.training_graph import TrainingGraph

    graph = TrainingGraph(_bagel_train_edges())
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


@pytest.mark.parametrize(
    "infer_graph",
    ["graph_infer_und.yaml", "graph_infer_gen.yaml", "graph_infer_edit.yaml"],
)
def test_bagel_train_plus_infer_merges_generation_graph(infer_graph: str):
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train.yaml",
        infer_modules=bagel_cfg_dir() / "modules_infer_eager.yaml",
        infer_graph_path=bagel_cfg_dir() / infer_graph,
    )
    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert cfg.has_generation_graph()
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


@pytest.mark.parametrize(
    "infer_graph",
    ["graph_infer_und.yaml", "graph_infer_gen.yaml", "graph_infer_edit.yaml"],
)
def test_bagel_infer_graph_yaml_is_graph_only(infer_graph: str):
    data = yaml.safe_load((bagel_cfg_dir() / infer_graph).read_text())

    assert "generation_graph" in data
    assert "generation_kwargs" not in data


def _bagel_train_edges() -> list[dict]:
    data = yaml.safe_load((bagel_cfg_dir() / "graph_train.yaml").read_text())
    return data["training_graph"]
