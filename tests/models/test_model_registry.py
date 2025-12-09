import pytest

from veomni.models.loader import get_model_class, get_model_config, get_model_processor
from veomni.utils.helper import get_cache_dir


local_test_cases = [
    pytest.param("./tests/models/toy_config/qwen2vl_toy", True, False, ["config", "model", "processor"], ["model"]),
    pytest.param("./tests/models/toy_config/janus_siglip_toy", False, True, [], ["config", "model", "processor"]),
]


@pytest.mark.parametrize(
    "config_path, is_hf_model, load_processor, hf_registered, veomni_registered", local_test_cases
)
def test_local_model_registry(config_path, is_hf_model, load_processor, hf_registered, veomni_registered):
    if is_hf_model:
        save_path = get_cache_dir(config_path)
        hf_config = get_model_config(config_path, force_use_huggingface=True)
        assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
        hf_config.save_pretrained(save_path)
        hf_model_class = get_model_class(hf_config, force_use_huggingface=True)
        assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
        if load_processor:
            hf_processor = get_model_processor(config_path, force_use_huggingface=True)
            assert hf_processor.__class__.__module__.startswith(
                "transformers." if "processor" in hf_registered else "veomni."
            )
            hf_processor.save_pretrained(save_path)

    save_path = get_cache_dir(config_path)
    veomni_config = get_model_config(config_path, force_use_huggingface=False)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config, force_use_huggingface=False)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    if load_processor:
        veomni_processor = get_model_processor(config_path, force_use_huggingface=False)
        assert veomni_processor.__class__.__module__.startswith(
            "veomni." if "processor" in veomni_registered else "transformers."
        )
        veomni_processor.save_pretrained(save_path)


remote_test_cases = [
    pytest.param("Qwen/Qwen2-VL-2B-Instruct", ["config", "model", "processor"], ["model"]),
    pytest.param(
        "deepseek-community/Janus-Pro-1B", ["config", "model", "processor"], ["config", "model", "processor"]
    ),
]


@pytest.mark.parametrize("config_path, hf_registered, veomni_registered", remote_test_cases)
def test_remote_model_registry(config_path, hf_registered, veomni_registered):
    save_path = get_cache_dir(config_path)
    hf_config = get_model_config(config_path, force_use_huggingface=True)
    assert hf_config.__class__.__module__.startswith("transformers." if "config" in hf_registered else "veomni.")
    hf_config.save_pretrained(save_path)
    hf_model_class = get_model_class(hf_config, force_use_huggingface=True)
    assert hf_model_class.__module__.startswith("transformers." if "model" in hf_registered else "veomni.")
    hf_processor = get_model_processor(config_path, force_use_huggingface=True)
    assert hf_processor.__class__.__module__.startswith("transformers." if "processor" in hf_registered else "veomni.")
    hf_processor.save_pretrained(save_path)

    veomni_config = get_model_config(config_path, force_use_huggingface=False)
    assert veomni_config.__class__.__module__.startswith(
        "veomni." if "config" in veomni_registered else "transformers."
    )
    veomni_config.save_pretrained(save_path)
    veomni_model_class = get_model_class(veomni_config, force_use_huggingface=False)
    assert veomni_model_class.__module__.startswith("veomni." if "model" in veomni_registered else "transformers.")
    veomni_processor = get_model_processor(config_path, force_use_huggingface=False)
    assert veomni_processor.__class__.__module__.startswith(
        "veomni." if "processor" in veomni_registered else "transformers."
    )
    veomni_processor.save_pretrained(save_path)


if __name__ == "__main__":
    test_remote_model_registry("deepseek-community/Janus-Pro-1B")
