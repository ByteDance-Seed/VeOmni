import pytest
from transformers import PretrainedConfig

from veomni.models.loader import get_model_class, get_model_config, get_model_processor
from veomni.models.transformers.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from veomni.utils.helper import get_cache_dir
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


local_test_cases = [
    pytest.param("./tests/toy_config/qwen2vl_toy", True, False, ["config", "model", "processor"], ["model"]),
    pytest.param("./tests/toy_config/janus_siglip_toy", False, True, [], ["config", "model", "processor"]),
]


@pytest.mark.parametrize(
    "config_path, is_hf_model, load_processor, hf_registered, veomni_registered", local_test_cases
)
def test_local_model_registry(monkeypatch, config_path, is_hf_model, load_processor, hf_registered, veomni_registered):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    if is_hf_model:
        save_path = get_cache_dir(config_path)
        hf_config = get_model_config(config_path)
        assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
        hf_config.save_pretrained(save_path)
        hf_model_class = get_model_class(hf_config)
        assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
        if load_processor:
            hf_processor = get_model_processor(config_path)
            assert hf_processor.__class__.__module__.startswith(
                "transformers." if "processor" in hf_registered else "veomni."
            )
            hf_processor.save_pretrained(save_path)

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    save_path = get_cache_dir(config_path)
    veomni_config = get_model_config(config_path)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    if load_processor:
        veomni_processor = get_model_processor(config_path)
        assert veomni_processor.__class__.__module__.startswith(
            "veomni." if "processor" in veomni_registered else "transformers."
        )
        veomni_processor.save_pretrained(save_path)


def test_minimax_m3_vl_config_registry_and_modeling_gate(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy")

    assert config.__class__.__module__.startswith("veomni.")
    assert config.model_type == "minimax_m3_vl"
    assert config.text_config.model_type == "minimax_m3_vl_text"
    assert config.vision_config.model_type == "minimax_m3_vl_vision"
    assert config.image_token_index == 200025
    assert config.video_token_index == 200026
    assert config.text_config.hidden_act == "silu"
    assert config.text_config.layer_types == ["minimax_m3_sparse", "full_attention"]
    assert config.text_config.mlp_layer_types == ["sparse", "dense"]
    assert config.text_config.index_block_size == 16
    assert config.text_config.rope_parameters == {"rope_theta": 5000000.0, "rope_type": "default"}
    assert config.vision_config.rope_parameters == {"rope_theta": 10000.0, "rope_type": "default"}

    save_path = get_cache_dir("./tests/toy_config/minimax_m3_vl_toy")
    config.save_pretrained(save_path)
    roundtrip_config = get_model_config(save_path)
    assert roundtrip_config.text_config.layer_types == config.text_config.layer_types
    assert roundtrip_config.text_config.mlp_layer_types == config.text_config.mlp_layer_types

    if is_transformers_version_greater_or_equal_to("5.12.0"):
        model_class = get_model_class(config)
        assert model_class.__module__.startswith("veomni.models.transformers.minimax_m3_vl.generated.")
        assert hasattr(model_class, "get_parallel_plan")
        assert hasattr(model_class, "get_position_id_func")
    else:
        with pytest.raises(RuntimeError, match="transformers>=5.12.0"):
            get_model_class(config)


def test_minimax_m3_vl_config_accepts_pretrained_nested_configs():
    text_config = PretrainedConfig(
        model_type="minimax_m3_vl_text",
        hidden_size=128,
        num_hidden_layers=2,
        sparse_attention_config={"sparse_attention_freq": [True, False]},
        moe_layer_freq=[True, False],
    )
    vision_config = PretrainedConfig(
        model_type="minimax_m3_vl_vision",
        hidden_size=64,
        image_size=32,
        patch_size=16,
        spatial_merge_size=2,
    )

    config = MiniMaxM3VLConfig(text_config=text_config, vision_config=vision_config)

    assert config.text_config.model_type == "minimax_m3_vl_text"
    assert config.text_config.hidden_size == 128
    assert config.text_config.layer_types == ["minimax_m3_sparse", "full_attention"]
    assert config.text_config.mlp_layer_types == ["sparse", "dense"]
    assert config.vision_config.model_type == "minimax_m3_vl_vision"
    assert config.vision_config.hidden_size == 64
    assert config.vision_config.image_size == 32


remote_test_cases = [
    pytest.param("Qwen/Qwen2-VL-2B-Instruct", ["config", "model", "processor"], ["model"]),
    pytest.param(
        "deepseek-community/Janus-Pro-1B", ["config", "model", "processor"], ["config", "model", "processor"]
    ),
]


@pytest.mark.xfail(reason="Remote path test may get too many requests error.")
@pytest.mark.parametrize("config_path, hf_registered, veomni_registered", remote_test_cases)
def test_remote_model_registry(monkeypatch, config_path, hf_registered, veomni_registered):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    save_path = get_cache_dir(config_path)
    hf_config = get_model_config(config_path)
    assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
    hf_config.save_pretrained(save_path)
    hf_model_class = get_model_class(hf_config)
    assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
    hf_processor = get_model_processor(config_path)
    assert hf_processor.__class__.__module__.startswith("transformers." if "processor" in hf_registered else "veomni.")
    hf_processor.save_pretrained(save_path)

    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    veomni_config = get_model_config(config_path)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    veomni_processor = get_model_processor(config_path)
    assert veomni_processor.__class__.__module__.startswith(
        "veomni." if "processor" in veomni_registered else "transformers."
    )
    veomni_processor.save_pretrained(save_path)


if __name__ == "__main__":
    test_remote_model_registry("deepseek-community/Janus-Pro-1B")
