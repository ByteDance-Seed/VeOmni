from types import SimpleNamespace

import pytest
import torch

from veomni.models.loader import MODEL_PROCESSOR_REGISTRY, get_model_class, get_model_config, get_model_processor
from veomni.utils.helper import get_cache_dir


@pytest.mark.parametrize("toy_name", ["qwen3vl", "qwen3vlmoe", "qwen3_5", "qwen3_5_moe"])
def test_generated_multimodal_children(monkeypatch, toy_name):
    """AutoModel inside a generated parent must not bypass the patched towers."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = get_model_config(f"./tests/toy_config/{toy_name}_toy")
    model_class = get_model_class(config)
    with torch.device("meta"):
        model = model_class._from_config(config, attn_implementation="eager")
    for tower in (model.model.visual, model.model.language_model):
        assert type(tower).__module__ == model_class.__module__


local_test_cases = [
    pytest.param("./tests/toy_config/qwen2vl_toy", True, False, ["config", "model", "processor"], ["model"]),
    pytest.param("./tests/toy_config/movqgan_toy", False, True, [], ["config", "model", "processor"]),
    pytest.param("./tests/toy_config/gpt_oss_toy", True, False, ["config", "model"], ["model"]),
]


# Public MiniMax M3 checkpoints ship their own processing_minimax.py, so which
# class name AutoProcessor resolves depends on the checkpoint's auto_map. Both
# must route to VeOmni's processor.
@pytest.mark.parametrize("processor_class_name", ["MiniMaxM3VLProcessor", "MiniMaxVLProcessor"])
def test_minimax_m3_vl_processor_registry_resolves_to_veomni_class(processor_class_name):
    from veomni.models.transformers.minimax_m3_vl.processing_minimax_m3_vl import MiniMaxM3VLProcessor

    assert MODEL_PROCESSOR_REGISTRY[processor_class_name]() is MiniMaxM3VLProcessor


def test_minimax_m3_vl_processor_builds_from_native_subprocessors(monkeypatch):
    # The checkpoint's bundled processor is unusable, so from_pretrained must
    # build the native tokenizer / image / video processors by concrete class
    # with trust_remote_code=False, and take the chat template from the tokenizer.
    import transformers

    from veomni.models.transformers.minimax_m3_vl import processing_minimax_m3_vl as mod

    calls = {}

    def _record(name, result):
        def _factory(path, **kwargs):
            calls[name] = kwargs
            return result

        return _factory

    fake_tokenizer = SimpleNamespace(chat_template="{{ tokenizer_template }}")
    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", staticmethod(_record("tokenizer", fake_tokenizer)))
    monkeypatch.setattr(
        transformers.MiniMaxM3VLImageProcessor, "from_pretrained", staticmethod(_record("image", "<image>"))
    )
    monkeypatch.setattr(
        transformers.MiniMaxM3VLVideoProcessor, "from_pretrained", staticmethod(_record("video", "<video>"))
    )
    constructed = {}
    monkeypatch.setattr(mod.MiniMaxM3VLProcessor, "__init__", lambda self, **kwargs: constructed.update(kwargs))

    mod.MiniMaxM3VLProcessor.from_pretrained("some/path", trust_remote_code=True, max_pixels=123)

    assert all(kwargs.get("trust_remote_code") is False for kwargs in calls.values())
    assert calls["image"].get("max_pixels") == 123
    assert constructed == {
        "image_processor": "<image>",
        "tokenizer": fake_tokenizer,
        "video_processor": "<video>",
        "chat_template": "{{ tokenizer_template }}",
    }


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


remote_test_cases = [
    pytest.param("Qwen/Qwen2-VL-2B-Instruct", ["config", "model", "processor"], ["model"]),
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
