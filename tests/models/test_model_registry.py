import pytest

from veomni.models import loader
from veomni.models.loader import get_model_class, get_model_config, get_model_processor
from veomni.utils.helper import get_cache_dir


local_test_cases = [
    pytest.param("./tests/toy_config/qwen2vl_toy", True, False, ["config", "model", "processor"], ["model"]),
    pytest.param("./tests/toy_config/janus_siglip_toy", False, True, [], ["config", "model", "processor"]),
    pytest.param("./tests/toy_config/gpt_oss_toy", True, False, ["config", "model"], ["model"]),
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


def test_degraded_autoprocessor_does_not_shadow_registered_processor(monkeypatch):
    """A registered VeOmni processor must win over a *degraded* AutoProcessor result.

    ``AutoProcessor.from_pretrained`` does not only succeed-or-raise. For a
    checkpoint that ships no ``preprocessor_config.json`` and no
    ``AutoProcessor`` entry in ``auto_map`` (HunyuanImage-3), transformers 5.8
    raised -- which the loader's ``except`` branch caught and rescued -- but
    transformers 5.9 returns the bare tokenizer instead. Accepting that silently
    drops the model's image branch, and the run only dies much later inside a
    dataloader worker with ``TokenizersBackend has no attribute
    image_processor``.
    """
    monkeypatch.setenv("MODELING_BACKEND", "veomni")

    class _BareTokenizer:  # stands in for transformers' TokenizersBackend
        pass

    class _DegradingAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _BareTokenizer()

    monkeypatch.setattr(loader, "AutoProcessor", _DegradingAutoProcessor)

    processor = get_model_processor("./tests/toy_config/hunyuan_image_3_toy")

    assert type(processor).__name__ == "HunyuanImage3Processor"
    assert hasattr(processor, "image_processor")


def test_degraded_autoprocessor_kept_when_no_processor_registered(monkeypatch, tmp_path):
    """The rescue must not fire for models that never registered a processor."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    (tmp_path / "config.json").write_text('{"model_type": "not_a_registered_veomni_model"}')

    class _BareTokenizer:
        pass

    class _DegradingAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _BareTokenizer()

    monkeypatch.setattr(loader, "AutoProcessor", _DegradingAutoProcessor)

    assert isinstance(get_model_processor(str(tmp_path)), _BareTokenizer)


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
