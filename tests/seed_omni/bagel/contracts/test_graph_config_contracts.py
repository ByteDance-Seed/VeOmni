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
    assert cfg.modules["bagel_qwen2_mot"]["model"]["ops_implementation"]["attn_implementation"] == "flex_attention"
    assert (
        cfg.module_config("bagel_qwen2_mot").model.ops_implementation.attn_implementation
        == "veomni_flex_attention_with_sp"
    )


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


def test_bagel_infer_gen_graph_uses_siglip_context_without_vae_context():
    data = yaml.safe_load((bagel_cfg_dir() / "graph_infer_gen.yaml").read_text())
    prompt_body = data["generation_graph"]["states"]["prompt_encode"]["body"]

    assert {"from": "bagel_text_encoder", "to": "bagel_qwen2_mot"} in prompt_body
    assert {"from": "bagel_siglip_navit", "to": "bagel_qwen2_mot"} in prompt_body
    assert all(edge["from"] != "bagel_vae.encode_context" for edge in prompt_body)
    assert all(edge["from"] != "bagel_flow_connector.embed_context_latents" for edge in prompt_body)


def test_bagel_offline_cache_yaml_loads_encode_only_vae():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train_offline_cache.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train_offline_cache.yaml",
    )

    assert set(cfg.modules) == {"bagel_vae"}
    assert cfg.module_config("bagel_vae").model.model_config == {"support_cache": True}
    assert cfg.training_graph == [{"from": "bagel_vae.offline_encode", "to": "end"}]


def test_bagel_train_with_cache_yaml_loads_process_only_vae():
    cfg = load_omni_config(
        modules_path=bagel_cfg_dir() / "modules_train_with_cache.yaml",
        train_graph_path=bagel_cfg_dir() / "graph_train_with_cache.yaml",
    )

    assert set(cfg.modules) == {
        "bagel_text_encoder",
        "bagel_siglip_navit",
        "bagel_qwen2_mot",
        "bagel_flow_connector",
        "bagel_vae",
    }
    assert cfg.module_config("bagel_vae").model.model_config == {"support_cache": True}
    endpoints = {e["from"] for e in cfg.training_graph} | {e["to"] for e in cfg.training_graph}
    assert "bagel_vae.online_process" in endpoints
    assert "bagel_vae.encode" not in endpoints
    assert cfg.modules["bagel_qwen2_mot"]["model"]["ops_implementation"]["attn_implementation"] == "flex_attention"


def test_bagel_offline_cache_full_entry_yamls_point_to_cache_graphs():
    offline_cache = yaml.safe_load((bagel_cfg_dir() / "offline_cache.yaml").read_text())
    train_with_cache = yaml.safe_load((bagel_cfg_dir() / "train_with_cache.yaml").read_text())

    assert offline_cache["train"]["train_type"] == "offline_cache"
    assert offline_cache["train"]["offline_cache_dir"] == "outputs/bagel_vae_cached_dataset"
    assert offline_cache["data"]["data_type"] == "seedomni"
    assert offline_cache["model"]["modules"].endswith("modules_train_offline_cache.yaml")
    assert offline_cache["model"]["train_graph"].endswith("graph_train_offline_cache.yaml")

    assert train_with_cache["train"]["train_type"] == "train_with_cache"
    assert train_with_cache["data"]["data_type"] == "seedomni_cached"
    assert train_with_cache["data"]["train_path"] == offline_cache["train"]["offline_cache_dir"]
    assert train_with_cache["model"]["modules"].endswith("modules_train_with_cache.yaml")
    assert train_with_cache["model"]["train_graph"].endswith("graph_train_with_cache.yaml")


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
    assert cfg.modules["bagel_qwen2_mot"]["model"]["ops_implementation"]["attn_implementation"] == "flex_attention"
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
