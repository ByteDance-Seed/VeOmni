"""OpenAI-style VLM normalization must preserve order and supervision."""

import base64

import numpy as np
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.multimodal.openai_messages import normalize_openai_messages


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_mixed_messages_preserve_image_order_and_masks():
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look", "image": None}, {"type": "image", "image": None}],
            },
            {"role": "assistant", "content": "first", "loss_mask": 0},
            {"role": "assistant", "content": [{"type": "text", "text": "second"}], "loss_mask": 1},
        ],
        "images": np.array([b"image"], dtype=object),
    }
    messages, images = normalize_openai_messages(sample)
    assert messages == [
        {"role": "user", "content": [("text", "look"), ("image", None)], "loss_mask": 0},
        {"role": "assistant", "content": [("text", "first")], "loss_mask": 0},
        {"role": "assistant", "content": [("text", "second")], "loss_mask": 1},
    ]
    assert images == [b"image"] and sample["messages"][0]["content"][1]["image"] is None


def test_inline_data_uri_and_text_only():
    uri = "data:image/png;base64," + base64.b64encode(b"bytes").decode()
    messages, images = normalize_openai_messages(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": uri}]},
                {"role": "assistant", "content": "answer"},
            ]
        }
    )
    assert images == [b"bytes"] and messages[1]["loss_mask"] == 1
    _, images = normalize_openai_messages({"messages": [{"role": "user", "content": "text"}]})
    assert images == []


def test_masked_fractional_weight_does_not_change_labels():
    """Masked message weights do not require the weighted renderer."""
    from types import SimpleNamespace

    from veomni.data.multimodal.openai_messages import OpenAIChatMLVisionTemplate

    vision = SimpleNamespace(
        tokenizer=_ToyTokenizer(), image_token_id=1000, video_token_id=1001, image_pattern=lambda count: "I" * count
    )
    template = OpenAIChatMLVisionTemplate(vision, 8192)
    message = {"role": "assistant", "content": "hidden", "loss_mask": 0, "loss_weight": 0.5}
    normalized, _ = normalize_openai_messages({"messages": [message]})
    assert normalized[0]["loss_weight"] == 0.5
    actual = template.encode_messages(normalized)
    baseline, _ = normalize_openai_messages({"messages": [{**message, "loss_weight": 0}]})
    expected = template.encode_messages(baseline)
    assert (actual["input_ids"] == expected["input_ids"]).all()
    assert (actual["labels"] == expected["labels"]).all()
    assert (actual["labels"] == -100).all()


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_masked_weights_must_still_be_valid(weight):
    with pytest.raises(ValueError, match="loss_weight"):
        normalize_openai_messages(
            {"messages": [{"role": "assistant", "content": "hidden", "loss_mask": 0, "loss_weight": weight}]}
        )


@pytest.mark.parametrize(
    "sample",
    [
        {"messages": [{"role": "assistant", "content": [{"type": "image", "image": b"x"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "image": None}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "image": b"x"}]}], "images": [b"y"]},
        {"messages": [{"role": "user", "content": "hi"}], "images": [b"extra"]},
        {"messages": [{"role": "assistant", "content": "hi", "loss_mask": 0.5}]},
        {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "call"}]}]},
        {"messages": [{"role": "user", "content": "hello"}], "image": b"hidden image"},
        {"messages": [{"role": "user", "content": "hello"}], "pixel_values": [1]},
    ],
)
def test_rejects_ambiguous_or_unsupported_inputs(sample):
    with pytest.raises(ValueError):
        normalize_openai_messages(sample)


class _ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith("<|image_pad|>"):
                result.append(1000)
                text = text[len("<|image_pad|>") :]
            elif text.startswith("<|video_pad|>"):
                result.append(1001)
                text = text[len("<|video_pad|>") :]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result


def test_template_matches_text_path_and_masks_image_placeholders():
    from types import SimpleNamespace

    import torch

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.multimodal.openai_messages import OpenAIChatMLVisionTemplate
    from veomni.utils.constants import IMAGE_INPUT_INDEX

    tokenizer = _ToyTokenizer()
    vision = SimpleNamespace(
        tokenizer=tokenizer,
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
    )
    template = OpenAIChatMLVisionTemplate(vision, max_seq_len=8192)
    sample = {
        "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world", "loss_mask": 1}]
    }
    messages, _ = normalize_openai_messages(sample)
    actual = template.encode_messages(messages)
    reference = ChatmlTemplate(tokenizer).encode_messages(sample["messages"])
    for key in reference:
        assert torch.equal(actual[key], torch.tensor(reference[key]))
    image_sample = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": b"x"}]},
            {"role": "assistant", "content": "world"},
        ]
    }
    image_messages, _ = normalize_openai_messages(image_sample)
    encoded = template.encode_messages(image_messages, {"image": [3]})
    image_mask = encoded["input_ids"] == IMAGE_INPUT_INDEX
    assert image_mask.sum() == 3 and torch.all(encoded["labels"][image_mask] == -100)
    with pytest.raises(ValueError, match="unused"):
        template.encode_messages(image_messages, {"image": [3, 1]})


def test_qwen_transform_text_and_image_paths(monkeypatch):
    from types import SimpleNamespace

    import torch

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.data_transform import process_sample_qwen_vl
    from veomni.data.multimodal import image_utils

    tokenizer = _ToyTokenizer()

    class ImageProcessor:
        merge_size = 2

        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(4, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    processor = SimpleNamespace(image_processor=ImageProcessor())
    template = SimpleNamespace(
        tokenizer=tokenizer,
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
        _openai_chatml_vlm=True,
    )
    monkeypatch.setattr(image_utils, "fetch_images", lambda images, **kwargs: images)

    def position_ids(**kwargs):
        return {"position_ids": torch.arange(kwargs["input_ids"].numel()).unsqueeze(0)}

    text = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]}
    feature = process_sample_qwen_vl(text, processor, template, position_ids, source_name="openai_chatml")[0]
    text_feature = {key: value.clone() for key, value in feature.items()}
    automatic = process_sample_qwen_vl(
        {**text, "source_name": "arbitrary-dataset-name"}, processor, template, position_ids
    )[0]
    assert torch.equal(automatic["input_ids"], text_feature["input_ids"])
    assert torch.equal(automatic["labels"], text_feature["labels"])
    collision = process_sample_qwen_vl({**text, "source_name": "sharegpt4v_sft"}, processor, template, position_ids)[0]
    assert torch.equal(collision["input_ids"], text_feature["input_ids"])
    with pytest.raises(ValueError, match="Unknown explicit multimodal source"):
        process_sample_qwen_vl({**text, "source": "mistyped-source"}, processor, template, position_ids)
    with pytest.raises(ValueError, match="only supports OpenAI"):
        process_sample_qwen_vl({"source": "sharegpt4v_sft", "conversations": []}, processor, template, position_ids)
    ref = ChatmlTemplate(tokenizer).encode_messages(text["messages"])
    assert torch.equal(feature["input_ids"], torch.tensor(ref["input_ids"]))
    assert torch.equal(feature["labels"], torch.tensor(ref["labels"]))
    assert not feature["image_mask"].any()
    multimodal = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "image": b"img"}]},
            {"role": "assistant", "content": "answer"},
        ]
    }
    feature = process_sample_qwen_vl(multimodal, processor, template, position_ids, source_name="openai_chatml")[0]
    assert feature["image_mask"].sum() == 1
    assert feature["labels"][feature["image_mask"]].item() == -100
    assert feature["pixel_values"].shape == (4, 2)

    from veomni.data import data_collator

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    lengths = [text_feature["input_ids"].numel(), feature["input_ids"].numel()]
    packed = data_collator.MainCollator()([text_feature, feature])
    assert packed["input_ids"].shape[-1] == sum(lengths)
    assert packed["image_mask"].sum() == 1
    assert packed["labels"][0, lengths[0]].item() == -100


def test_length_limits_and_mtp_fail_closed():
    from types import SimpleNamespace

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.multimodal.openai_messages import OpenAIChatMLVisionTemplate, validate_openai_chatml_model_config

    tokenizer = _ToyTokenizer()
    vision = SimpleNamespace(
        tokenizer=tokenizer,
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
    )
    template = OpenAIChatMLVisionTemplate(vision, 12)
    source = {"messages": [{"role": "assistant", "content": "long answer"}]}
    messages, _ = normalize_openai_messages(source)
    actual = template.encode_messages(messages)
    reference = ChatmlTemplate(tokenizer).encode_messages(source["messages"], max_seq_len=12)
    assert actual["input_ids"].tolist() == reference["input_ids"]
    image_messages, _ = normalize_openai_messages(
        {"messages": [{"role": "user", "content": [{"type": "image", "url": "data:image/png;base64,eA=="}]}]}
    )
    with pytest.raises(ValueError, match="refusing to truncate"):
        template.encode_messages(image_messages, {"image": [30]})
    validate_openai_chatml_model_config(SimpleNamespace(text_config=SimpleNamespace(mtp_num_hidden_layers=1)))
    with pytest.raises(ValueError, match="MTP"):
        validate_openai_chatml_model_config(SimpleNamespace(text_config=SimpleNamespace(mtp_loss_weight=0.1)))
    with pytest.raises(ValueError, match="MTP"):
        validate_openai_chatml_model_config({"text_config": {"mtp_loss_weight": 0.1}})


def test_tools_require_explicit_verified_system_context():
    sample = {
        "messages": [{"role": "system", "content": "Use tool_lookup"}, {"role": "user", "content": "hi"}],
        "tools": '[{"type":"function","function":{"name":"tool_lookup"}}]',
    }
    with pytest.raises(ValueError, match="tools_in_system"):
        normalize_openai_messages(sample)
    assert normalize_openai_messages(sample, tools_in_system=True)[0][0]["role"] == "system"
    sample["messages"][0]["content"] = "missing tool specification"
    with pytest.raises(ValueError, match="tool name"):
        normalize_openai_messages(sample, tools_in_system=True)


def test_chat_template_persisted_with_processor(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    from veomni.data.chat_template import OpenAIChatMLQwenVLTemplate

    words = {"[UNK]": 0, "user": 1, "assistant": 2, "hello": 3, "world": 4}
    backend = Tokenizer(models.WordLevel(words, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ]
        }
    )
    processor = Qwen3VLProcessor(Qwen2VLImageProcessor(), tokenizer, Qwen3VLVideoProcessor())
    OpenAIChatMLQwenVLTemplate(processor)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image", "image": None}]},
        {"role": "assistant", "content": "world"},
    ]
    expected = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    processor.save_pretrained(tmp_path)
    restored = Qwen3VLProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert restored.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) == expected
    assert restored.tokenizer.chat_template == processor.tokenizer.chat_template
    from jinja2.exceptions import TemplateError

    for modality in ("video", "audio"):
        with pytest.raises(TemplateError, match="Unsupported multimodal ChatML content type"):
            restored.apply_chat_template([{"role": "user", "content": [{"type": modality}]}], tokenize=False)


def test_legacy_template_keeps_source_name_precedence(monkeypatch):
    from types import SimpleNamespace

    import torch

    from veomni.data import multimodal
    from veomni.data.data_transform import process_sample_qwen_vl

    seen = []

    def preprocess(source, conversations, **kwargs):
        seen.append(source)
        return []

    monkeypatch.setattr(multimodal, "conv_preprocess", preprocess)
    template = SimpleNamespace(
        encode_messages=lambda *args, **kwargs: {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
            "attention_mask": torch.ones(2),
        }
    )
    process_sample_qwen_vl(
        {"source": "row-metadata", "conversations": [1]},
        SimpleNamespace(),
        template,
        lambda **kwargs: {"position_ids": torch.tensor([[0, 1]])},
        source_name="legacy-preprocessor",
    )
    assert seen == ["legacy-preprocessor"]


def test_fractional_weight_is_not_silently_dropped():
    from types import SimpleNamespace

    from veomni.data.multimodal.openai_messages import OpenAIChatMLVisionTemplate

    tokenizer = _ToyTokenizer()
    vision = SimpleNamespace(
        tokenizer=tokenizer,
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
    )
    template = OpenAIChatMLVisionTemplate(vision, 8192)
    sample = {"messages": [{"role": "assistant", "content": "answer", "loss_weight": 0.25}]}
    messages, _ = normalize_openai_messages(sample)
    weighted = template.text_template.encode_messages(sample["messages"])
    if "loss_weights" in weighted:
        assert (template.encode_messages(messages)["loss_weights"] == 0.25).all()
    else:
        with pytest.raises(ValueError, match="weighted ChatML"):
            template.encode_messages(messages)
    sample["messages"][0]["loss_weight"] = 0
    messages, _ = normalize_openai_messages(sample)
    assert (template.encode_messages(messages)["labels"] == -100).all()


def test_masked_and_active_fractional_weights_with_weighted_renderer():
    """Masked weights stay excluded alongside a real fractional target."""
    from types import SimpleNamespace

    import torch

    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.multimodal.openai_messages import OpenAIChatMLVisionTemplate

    tokenizer = _ToyTokenizer()
    sample = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "masked", "loss_mask": 0, "loss_weight": 0.5},
            {"role": "assistant", "content": "trained", "loss_mask": 1, "loss_weight": 0.25},
        ]
    }
    expected = ChatmlTemplate(tokenizer).encode_messages(sample["messages"])
    if "loss_weights" not in expected:
        pytest.skip("requires the independent weighted-loss feature")
    vision = SimpleNamespace(
        tokenizer=tokenizer, image_token_id=1000, video_token_id=1001, image_pattern=lambda count: "I" * count
    )
    normalized, _ = normalize_openai_messages(sample)
    actual = OpenAIChatMLVisionTemplate(vision, 8192).encode_messages(normalized)
    assert torch.equal(actual["input_ids"], torch.tensor(expected["input_ids"]))
    assert torch.equal(actual["labels"], torch.tensor(expected["labels"]))
    assert torch.equal(actual["loss_weights"], torch.tensor(expected["loss_weights"]))


def test_weighted_text_image_mixed_packing(monkeypatch):
    from types import SimpleNamespace

    import torch

    from veomni.data import data_collator
    from veomni.data.chat_template import ChatmlTemplate
    from veomni.data.data_transform import process_sample_qwen_vl
    from veomni.data.multimodal import image_utils

    tokenizer = _ToyTokenizer()
    assistant = {"role": "assistant", "content": "answer", "loss_weight": 0.25}
    if "loss_weights" not in ChatmlTemplate(tokenizer).encode_messages([assistant]):
        pytest.skip("requires the independent weighted-loss feature")

    class ImageProcessor:
        merge_size = 2

        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(4, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    processor = SimpleNamespace(image_processor=ImageProcessor())
    template = SimpleNamespace(
        tokenizer=tokenizer,
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
        _openai_chatml_vlm=True,
    )
    monkeypatch.setattr(image_utils, "fetch_images", lambda images, **kwargs: images)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    def positions(**kwargs):
        return {"position_ids": torch.arange(kwargs["input_ids"].numel()).unsqueeze(0)}

    text = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "text answer"}]}
    image = {"messages": [{"role": "user", "content": [{"type": "image", "image": b"img"}]}, assistant]}
    first = process_sample_qwen_vl(text, processor, template, positions)[0]
    second = process_sample_qwen_vl(image, processor, template, positions)[0]
    first_labels, second_labels = first["labels"].clone(), second["labels"].clone()
    first_ids, second_ids = first["input_ids"].clone(), second["input_ids"].clone()
    first_image_mask, second_image_mask = first["image_mask"].clone(), second["image_mask"].clone()
    offset = first_ids.numel()
    expected_second_labels = second_labels.clone()
    expected_second_labels[0] = -100
    packed = data_collator.MainCollator()([first, second])
    assert torch.equal(packed["input_ids"][0], torch.cat([first_ids, second_ids]))
    assert torch.equal(packed["labels"][0], torch.cat([first_labels, expected_second_labels]))
    first_valid = packed["labels"][0, :offset] != -100
    second_valid = packed["labels"][0, offset:] != -100
    assert first_valid.any() and second_valid.any()
    assert torch.all(packed["loss_weights"][0, :offset][first_valid] == 1)
    assert torch.all(packed["loss_weights"][0, offset:][second_valid] == 0.25)
    assert packed["image_mask"].sum() == 1
    assert torch.equal(packed["image_mask"][0], torch.cat([first_image_mask, second_image_mask]))
    assert torch.all(packed["loss_weights"][packed["image_mask"].bool()] == 0)


def test_openai_image_supervision_reaches_hf_qwen3_vl_vision(monkeypatch):
    """Use a tiny HF model to check the adapter's input and supervision semantics.

    The HF model expects its native image token IDs; the VeOmni transform
    replaces them with zero and carries their locations in image_mask.
    """
    from types import SimpleNamespace

    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from veomni.data.data_transform import process_sample_qwen_vl
    from veomni.data.multimodal import image_utils

    class ImageProcessor:
        merge_size = 2

        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(4, 3 * 2 * 14 * 14), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 1100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [],
        },
        image_token_id=1000,
        video_token_id=1001,
    )
    torch.manual_seed(42)
    model = Qwen3VLForConditionalGeneration(config).train()
    monkeypatch.setattr(image_utils, "fetch_images", lambda images, **kwargs: images)
    template = SimpleNamespace(
        tokenizer=_ToyTokenizer(),
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
        _openai_chatml_vlm=True,
    )

    def positions(**kwargs):
        length = kwargs["input_ids"].shape[-1]
        return {"position_ids": torch.arange(length).view(1, 1, -1).expand(3, 1, -1)}

    sample = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Look "}, {"type": "image", "image": b"x"}]},
            {"role": "assistant", "content": "This is a picture."},
        ]
    }
    feature = process_sample_qwen_vl(sample, SimpleNamespace(image_processor=ImageProcessor()), template, positions)[0]
    input_ids = feature["input_ids"].clone()
    assert feature["image_mask"].sum().item() == 1
    assert (input_ids[feature["image_mask"]] == 0).all()
    input_ids[feature["image_mask"]] = config.image_token_id
    assert (feature["labels"] != -100).any()
    assert (feature["labels"][feature["image_mask"]] == -100).all()
    result = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=feature["attention_mask"].unsqueeze(0),
        position_ids=feature["position_ids"].unsqueeze(1),
        labels=feature["labels"].unsqueeze(0),
        pixel_values=feature["pixel_values"],
        image_grid_thw=feature["image_grid_thw"],
        use_cache=False,
    )
    result.loss.backward()
    vision_gradients = [param.grad for param in model.model.visual.parameters() if param.grad is not None]
    assert torch.isfinite(result.loss)
    assert vision_gradients and all(torch.isfinite(grad).all() for grad in vision_gradients)
    assert any(torch.count_nonzero(grad).item() for grad in vision_gradients)


def test_qwen4_patched_model_mixed_packing_matches_separate_loss_and_gradients(monkeypatch):
    """A tiny actual VeOmni model sees the same supervision packed or separately."""
    import copy
    from types import SimpleNamespace

    import torch
    from transformers import Qwen4ExpConfig

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.data import data_collator
    from veomni.data.data_transform import process_sample_qwen_vl
    from veomni.data.multimodal import image_utils
    from veomni.models.configuration_base import as_veomni_config
    from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
        Qwen4ExpForConditionalGeneration,
    )

    class ImageProcessor:
        merge_size = 2

        def __call__(self, **kwargs):
            return {"pixel_values": torch.ones(4, 3 * 2 * 4 * 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json")
    as_veomni_config(config, validate=False)._set_ops_implementation(make_eager_ops_config())
    torch.manual_seed(42)
    model = Qwen4ExpForConditionalGeneration(config).float().train()
    monkeypatch.setattr(image_utils, "fetch_images", lambda images, **kwargs: images)
    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    processor = SimpleNamespace(image_processor=ImageProcessor())
    template = SimpleNamespace(
        tokenizer=_ToyTokenizer(),
        image_token_id=1000,
        video_token_id=1001,
        image_pattern=lambda count: "<|image_pad|>" * count,
        _openai_chatml_vlm=True,
    )

    def positions(**kwargs):
        length = kwargs["input_ids"].shape[-1]
        return {"position_ids": torch.arange(length).view(1, 1, -1).expand(4, 1, -1)}

    text = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
    image = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Look "}, {"type": "image", "image": b"x"}]},
            {"role": "assistant", "content": "A picture."},
        ]
    }
    first = process_sample_qwen_vl(text, processor, template, positions)[0]
    second = process_sample_qwen_vl(image, processor, template, positions)[0]
    lengths = [first["input_ids"].numel(), second["input_ids"].numel()]
    assert not first["image_mask"].any() and second["image_mask"].sum() == 1
    assert (second["labels"][second["image_mask"]] == -100).all()
    packed = data_collator.MainCollator()([copy.deepcopy(first), copy.deepcopy(second)])
    assert torch.equal(packed["cu_seq_lens_q"], torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32))
    assert torch.equal(packed["input_ids"][0], torch.cat([first["input_ids"], second["input_ids"]]))
    assert torch.equal(packed["labels"][0], torch.cat([first["labels"], second["labels"]]))
    assert torch.equal(packed["image_mask"][0], torch.cat([first["image_mask"], second["image_mask"]]))
    assert (first["labels"][1:] != -100).any() and (second["labels"][1:] != -100).any()

    def forward(feature, *, is_packed):
        ids = feature["input_ids"] if is_packed else feature["input_ids"].unsqueeze(0)
        position_ids = feature["position_ids"].permute(1, 0, 2) if is_packed else feature["position_ids"].unsqueeze(1)
        kwargs = {
            "input_ids": ids,
            "position_ids": position_ids,
            "attention_mask": None,
            "labels": feature["labels"] if is_packed else feature["labels"].unsqueeze(0),
            "image_mask": feature["image_mask"] if is_packed else feature["image_mask"].unsqueeze(0),
            "video_mask": feature["video_mask"] if is_packed else feature["video_mask"].unsqueeze(0),
            "mm_token_type_ids": feature["mm_token_type_ids"]
            if is_packed
            else feature["mm_token_type_ids"].unsqueeze(0),
            "cu_seq_lens_q": feature["cu_seq_lens_q"]
            if is_packed
            else torch.tensor([0, ids.numel()], dtype=torch.int32),
            "use_cache": False,
        }
        if "pixel_values" in feature:
            kwargs.update(pixel_values=feature["pixel_values"], image_grid_thw=feature["image_grid_thw"])
        return model(**kwargs).loss

    actual = forward(packed, is_packed=True)
    actual.backward()
    actual_grads = {
        name: None if param.grad is None else param.grad.clone() for name, param in model.named_parameters()
    }
    vision_grads = [param.grad for param in model.model.visual.parameters() if param.grad is not None]
    assert torch.isfinite(actual) and vision_grads and all(torch.isfinite(grad).all() for grad in vision_grads)
    assert any(torch.count_nonzero(grad).item() for grad in vision_grads)
    model.zero_grad(set_to_none=True)
    first_loss, second_loss = forward(first, is_packed=False), forward(second, is_packed=False)
    first_count = (first["labels"][1:] != -100).sum()
    second_count = (second["labels"][1:] != -100).sum()
    expected = (first_loss * first_count + second_loss * second_count) / (first_count + second_count)
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert sum(grad is not None for grad in actual_grads.values()) > 0
    for name, param in model.named_parameters():
        actual_grad = actual_grads[name]
        assert (actual_grad is None) == (param.grad is None), name
        if actual_grad is not None:
            torch.testing.assert_close(actual_grad, param.grad, rtol=3e-4, atol=3e-5, msg=name)
