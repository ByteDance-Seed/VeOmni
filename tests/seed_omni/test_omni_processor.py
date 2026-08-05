from unittest.mock import MagicMock, patch

from veomni.models.seed_omni.configuration_omni import OmniConfig
from veomni.models.seed_omni.processing import OmniProcessor
from veomni.models.seed_omni.utils.conversation import ConversationItem


class _RecordingPreprocessor:
    def __init__(self, tag: str, store: list[str]) -> None:
        self._tag = tag
        self._store = store

    def __call__(self, conversation_list, inference=False, **kwargs) -> None:
        del inference, kwargs
        self._store.append(self._tag)


def test_omni_processor_builds_conversation_and_runs_preprocessors_in_order():
    calls: list[str] = []
    processor = OmniProcessor(
        {
            "a": _RecordingPreprocessor("first", calls),
            "b": _RecordingPreprocessor("second", calls),
        }
    )

    with patch("veomni.models.seed_omni.processing.load_image", return_value="img"):
        model_input = processor(text="hello", images=["/tmp/fake.png"])

    assert calls == ["first", "second"]
    assert "conversation_list" in model_input
    conversation = model_input["conversation_list"]
    assert len(conversation) == 2
    assert conversation[0].type == "image"
    assert conversation[1].type == "text"
    assert conversation[1].value == "hello"


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
        def __call__(self, conversation_list, inference=False, **kwargs) -> None:
            del conversation_list, kwargs
            calls.append(("batch", inference))

    processor = OmniProcessor({"a": _FlagPreprocessor()})
    batches = [[ConversationItem(type="text", value="hi", role="user")]]

    processor.preprocess_batch(batches, inference=False)

    assert calls == [("batch", False)]


@patch("veomni.models.seed_omni.processing.OMNI_MODEL_REGISTRY")
@patch("veomni.models.seed_omni.processing.read_model_type", return_value="encoder_type")
@patch("veomni.models.seed_omni.processing.OmniConfig.from_pretrained")
def test_omni_processor_from_pretrained_collects_module_preprocessors(
    mock_from_pretrained,
    mock_read_model_type,
    mock_registry,
    tmp_path,
):
    del mock_read_model_type
    mock_from_pretrained.return_value = OmniConfig(
        modules={"encoder": {"subfolder": "encoder"}},
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )
    fake_mod_cls = MagicMock()
    mock_registry.__getitem__.return_value = MagicMock(return_value=fake_mod_cls)

    processor = OmniProcessor.from_pretrained(tmp_path, infer_type="infer_gen")

    mock_from_pretrained.assert_called_once_with(tmp_path, infer_type="infer_gen")
    fake_mod_cls.preprocessor_class.from_pretrained.assert_called_once()
    assert len(processor._preprocessors) == 1


@patch("veomni.models.seed_omni.processing.OMNI_MODEL_REGISTRY")
@patch("veomni.models.seed_omni.processing.read_model_type", return_value="encoder_type")
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
        modules={
            "encoder": {
                "subfolder": "encoder",
                "model": {"model_config": {"enable_image": True}},
            }
        },
        training_graph=[{"from": "encoder", "to": "end"}],
        generation_graphs={"infer_gen": {"initial": "run", "states": {}}},
    )

    OmniProcessor.from_config(config, checkpoint_root="/tmp/checkpoint_root")

    fake_mod_cls.preprocessor_class.from_pretrained.assert_called_once_with(
        "/tmp/checkpoint_root/encoder", config_overrides={"enable_image": True}
    )
