from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from veomni.data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY
from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.modules.module_processing_base import bind_module_assets
from veomni.models.seed_omni.processing_omni import OmniProcessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


class _RecordingPreprocessor:
    def __init__(self, tag: str, store: list[str]) -> None:
        self._tag = tag
        self._store = store

    def __call__(self, batch, inference=False, **kwargs) -> None:
        del batch, inference, kwargs
        self._store.append(self._tag)


def _png(tmp_path, name: str = "frame.png") -> str:
    path = tmp_path / name
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(str(path))
    return str(path)


def test_omni_processor_builds_conversation_and_runs_preprocessors_in_order(tmp_path):
    calls: list[str] = []
    processor = OmniProcessor(
        {
            "a": _RecordingPreprocessor("first", calls),
            "b": _RecordingPreprocessor("second", calls),
        }
    )

    model_input = processor(text="hello", images=[_png(tmp_path)])

    assert calls == ["first", "second"]
    assert "conversation_list" in model_input
    conversation = model_input["conversation_list"]
    assert len(conversation) == 2
    assert conversation[0].type == "image"
    assert conversation[1].type == "text"
    assert conversation[1].value == "hello"


def test_omni_processor_decodes_images_the_way_the_training_path_does(tmp_path):
    """Request building shares the training transform's fetchers.

    It used to have its own PIL-only loader, so the same file became a PIL image
    for inference and a uint8 pixel tensor for training, and no metadata was
    attached either way.
    """
    processor = OmniProcessor({"a": _RecordingPreprocessor("only", [])})

    conversation = processor(text="hello", images=_png(tmp_path))["conversation_list"]

    assert conversation[0].type == "image"
    assert isinstance(conversation[0].value, torch.Tensor)
    assert conversation[0].value.dtype == torch.uint8


def test_omni_processor_honours_audios_rather_than_dropping_them(tmp_path):
    """``audios`` used to be unreachable: the parameter existed on
    ``build_conversation`` but no caller ever passed it, so a request naming a
    clip came back text-only."""
    path = tmp_path / "speech.wav"
    sf.write(str(path), np.zeros(800, dtype=np.float32), 16_000)
    processor = OmniProcessor({"a": _RecordingPreprocessor("only", [])})

    conversation = processor(text="transcribe", audios=str(path))["conversation_list"]

    assert [item.type for item in conversation] == ["audio", "text"]
    assert conversation[0].meta[AUDIO_METADATA_KEY].sampling_rate == 16_000


def test_omni_processor_mm_configs_do_not_leak_into_generation_kwargs(tmp_path):
    """Decode knobs are a named dict because ``**generation_kwargs`` would
    otherwise swallow them and the clip would be decoded at the default rate."""
    seen: list[dict] = []

    class _CapturingPreprocessor:
        def __call__(self, batch, inference=False, **kwargs) -> None:
            del batch, inference
            seen.append(kwargs.get("generation_kwargs"))

    processor = OmniProcessor({"a": _CapturingPreprocessor()})

    processor(
        text="hi",
        images=_png(tmp_path),
        mm_configs={"image_max_pixels": 64},
        max_new_tokens=8,
    )

    assert seen == [{"max_new_tokens": 8}]


def test_omni_processor_preprocess_mutates_existing_conversation():
    calls: list[str] = []
    processor = OmniProcessor({"a": _RecordingPreprocessor("only", calls)})
    conversation = [ConversationItem(type="text", value="hi", role="user")]

    out = processor.preprocess(conversation, inference=True)

    assert calls == ["only"]
    assert out["conversation_list"] is conversation


def test_omni_processor_preprocess_batch_runs_with_inference_false():
    calls: list[tuple[str, bool]] = []

    class _FlagPreprocessor:
        def __call__(self, batch, inference=False, **kwargs) -> None:
            del batch, kwargs
            calls.append(("batch", inference))

    processor = OmniProcessor({"a": _FlagPreprocessor()})
    batches = [[ConversationItem(type="text", value="hi", role="user")]]

    processor.preprocess_batch({"conversation_list": batches}, inference=False)

    assert calls == [("batch", False)]


@patch("veomni.models.seed_omni.processing_omni.OMNI_MODEL_REGISTRY")
@patch("veomni.models.seed_omni.processing_omni.read_model_type", return_value="encoder_type")
@patch("veomni.models.seed_omni.processing_omni.OmniConfig.from_pretrained")
def test_omni_processor_from_pretrained_collects_module_preprocessors(
    mock_from_pretrained,
    mock_read_model_type,
    mock_registry,
    tmp_path,
):
    del mock_read_model_type
    mock_from_pretrained.return_value = OmniConfig(
        _module_entries={"encoder": {"model_path": "encoder"}},
        training_graphs={"default": [{"from": "encoder", "to": "end"}]},
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )
    fake_mod_cls = MagicMock()
    mock_registry.__getitem__.return_value = MagicMock(return_value=fake_mod_cls)

    processor = OmniProcessor.from_pretrained(tmp_path, infer_type="infer_gen")

    mock_from_pretrained.assert_called_once_with(tmp_path, infer_type="infer_gen")
    fake_mod_cls.preprocessor_class.from_pretrained.assert_called_once()
    assert len(processor._preprocessors) == 1


@patch("veomni.models.seed_omni.processing_omni.OMNI_MODEL_REGISTRY")
@patch("veomni.models.seed_omni.processing_omni.read_model_type", return_value="encoder_type")
def test_omni_processor_from_config_forwards_module_model_config_overrides(mock_read_model_type, mock_registry):
    """A module's YAML `model_config:` override (e.g. visual-instruction-tuning's
    `enable_image: true` on `qwen3_text_encoder`) must reach the module's
    `Preprocessor.from_pretrained` — regression: this used to only pass the
    checkpoint path, silently dropping the override the live model itself receives.
    """
    del mock_read_model_type
    fake_mod_cls = MagicMock()
    mock_registry.__getitem__.return_value = MagicMock(return_value=fake_mod_cls)
    config = OmniConfig(
        _module_entries={
            "encoder": {
                "model_path": "encoder",
                "model_config": {"enable_image": True},
            }
        },
        training_graphs={"default": [{"from": "encoder", "to": "end"}]},
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )

    OmniProcessor.from_config(config, checkpoint_root="/tmp/checkpoint_root")

    fake_mod_cls.preprocessor_class.from_pretrained.assert_called_once_with(
        "/tmp/checkpoint_root/encoder", config_overrides={"enable_image": True}
    )


@patch("veomni.models.seed_omni.processing_omni.OMNI_MODEL_REGISTRY")
@patch("veomni.models.seed_omni.processing_omni.read_model_type", return_value="encoder_type")
def test_omni_processor_from_config_forwards_module_processor_config(mock_read_model_type, mock_registry):
    """YAML ``processor_config:`` is splatted as kwargs, matching ``build_processor``."""
    del mock_read_model_type
    fake_mod_cls = MagicMock()
    mock_registry.__getitem__.return_value = MagicMock(return_value=fake_mod_cls)
    config = OmniConfig(
        _module_entries={
            "encoder": {
                "model_path": "encoder",
                "processor_config": {"packed_preprocess": True},
            }
        },
        training_graphs={"default": [{"from": "encoder", "to": "end"}]},
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )

    OmniProcessor.from_config(config, checkpoint_root="/tmp/checkpoint_root")

    fake_mod_cls.preprocessor_class.from_pretrained.assert_called_once_with(
        "/tmp/checkpoint_root/encoder",
        config_overrides={},
        packed_preprocess=True,
    )


class _TwoAssetPreprocessor:
    def __init__(self) -> None:
        self._tokenizer = "preprocessor-tokenizer"
        self._image_processor = "image-processor"


class _AssetHolder:
    """Stand-in for a module model: ``tokenizer`` is settable, as on the real ones."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._image_processor = None


def test_bind_module_assets_fills_the_assets_a_caller_did_not_set():
    """One hand-set asset must not cost the module its others.

    Module models expose a public ``tokenizer`` setter, so ``_tokenizer`` can
    already hold a value when binding runs. Treating "some asset is set" as
    "this module is bound" would skip the whole copy and leave the module
    without its image processor, which only surfaces as a failure much later
    inside ``forward``.
    """
    model = _AssetHolder()
    model._tokenizer = "caller-tokenizer"

    bind_module_assets(model, preprocessor=_TwoAssetPreprocessor())

    assert model._image_processor == "image-processor"
    assert model._tokenizer == "caller-tokenizer"  # the caller's asset wins


def test_bind_module_assets_is_a_noop_the_second_time():
    model = _AssetHolder()
    bind_module_assets(model, preprocessor=_TwoAssetPreprocessor())
    model._image_processor = "swapped-later"

    bind_module_assets(model, preprocessor=_TwoAssetPreprocessor())

    assert model._image_processor == "swapped-later"
