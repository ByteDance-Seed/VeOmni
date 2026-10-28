from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.chat_template import MiniMaxM3VLChatTemplate
from veomni.data.data_transform import process_sample_minimax_m3_vl
from veomni.utils.import_utils import is_ffmpeg_available


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _render_fake_minimax_chat_template(messages, *, add_generation_prompt):
    """Render deterministic test chat text, including an optional assistant generation prefix."""
    rendered = "]~!b[]~b]system\nMiniMax system prompt[e~[\n]~b]developer\nYou are a helpful assistant.[e~[\n"
    for message in messages:
        role = "ai" if message["role"] == "assistant" else message["role"]
        rendered += f"]~b]{role}\n"
        if message["role"] == "assistant":
            rendered += "</mm:think>"
        rendered += f"{message['content']}[e~[\n"
    if add_generation_prompt:
        rendered += "]~b]ai\n"
    return rendered


class _FakeNativeTemplateProcessor:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        """Forward template options to the deterministic native-template renderer."""
        assert tokenize is False
        self.chat_template_calls.append(
            {
                "messages": deepcopy(messages),
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return _render_fake_minimax_chat_template(messages, add_generation_prompt=add_generation_prompt)


class _FakeImageProcessor:
    merge_size = 2

    def __init__(self):
        self.calls = []

    def __call__(self, *, images, return_tensors):
        """Return synthetic image tensors and grid metadata for the requested images."""
        self.calls.append({"images": images, "return_tensors": return_tensors})
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.arange(24, dtype=torch.float32).reshape(4, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class _FakeVideoProcessor:
    merge_size = 2

    def __init__(self):
        self.calls = []

    def __call__(self, *, videos, video_metadata, do_sample_frames, return_tensors, return_metadata):
        """Return synthetic video tensors, grids and optional timing metadata."""
        self.calls.append(
            {
                "videos": videos,
                "video_metadata": video_metadata,
                "do_sample_frames": do_sample_frames,
                "return_tensors": return_tensors,
                "return_metadata": return_metadata,
            }
        )
        assert do_sample_frames is False
        assert return_tensors == "pt"
        assert return_metadata is True
        return {
            "pixel_values_videos": torch.arange(48, dtype=torch.float32).reshape(8, 6),
            "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
            "video_metadata": [{"timestamps": [0.0, 1.0]}],
        }


class _FakeProcessor(_FakeNativeTemplateProcessor):
    image_token_id = 200025
    video_token_id = 200026

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.image_processor = _FakeImageProcessor()
        self.video_processor = _FakeVideoProcessor()
        self.replaced_images = []
        self.replaced_videos = []
        self.chat_template_calls = []

    def replace_image_token(self, image_inputs, image_idx):
        """Validate image replacement metadata and return a fixed image token for the fixture."""
        self.replaced_images.append({"image_inputs": image_inputs, "image_idx": image_idx})
        assert "pixel_values" in image_inputs
        assert "image_grid_thw" in image_inputs
        return MiniMaxM3VLChatTemplate.IMAGE_TOKEN

    def replace_video_token(self, video_inputs, video_idx):
        """Validate video replacement metadata and return a fixed video token for the fixture."""
        self.replaced_videos.append({"video_inputs": video_inputs, "video_idx": video_idx})
        assert "pixel_values_videos" in video_inputs
        assert "video_grid_thw" in video_inputs
        return MiniMaxM3VLChatTemplate.VIDEO_TOKEN


class _FakeMiniMaxTokenizer:
    eos_token = "[e~["

    _token_ids = {
        MiniMaxM3VLChatTemplate.IMAGE_TOKEN: 200025,
        MiniMaxM3VLChatTemplate.VIDEO_TOKEN: 200026,
        "[e~[": 200027,
        "]~!b[": 200028,
        "]~b]": 200029,
        "</mm:think>": 200030,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def encode(self, text, add_special_tokens=False):
        """Tokenize test text while preserving the special image and video token IDs."""
        self.last_encoded_text = text
        token_ids = []
        idx = 0
        special_tokens = (
            MiniMaxM3VLChatTemplate.IMAGE_TOKEN,
            MiniMaxM3VLChatTemplate.VIDEO_TOKEN,
            "</mm:think>",
            "]~!b[",
            "]~b]",
            "[e~[",
        )
        while idx < len(text):
            for token in special_tokens:
                if text.startswith(token, idx):
                    token_ids.append(self._token_ids[token])
                    idx += len(token)
                    break
            else:
                token_ids.append(1000 + ord(text[idx]) % 1000)
                idx += 1
        return token_ids


class _ProcessorWithoutReplacementMethods(_FakeNativeTemplateProcessor):
    def __init__(self):
        self.tokenizer = _FakeMiniMaxTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=2)
        self.chat_template_calls = []


def test_minimax_chat_template_uses_hf_native_template_and_masks_generation_prefix():
    """Check native template rendering and loss masking of the assistant generation prefix."""
    processor = _ProcessorWithoutReplacementMethods()
    chat_template = MiniMaxM3VLChatTemplate(processor)
    encoded = chat_template.encode_messages(
        [["user", ("text", "Question")], ["assistant", ("text", "Answer")]],
        processor=processor,
    )

    expected_messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    expected_text = _render_fake_minimax_chat_template(expected_messages, add_generation_prompt=False)
    expected_ids = processor.tokenizer.encode(expected_text, add_special_tokens=False)
    prompt_text = _render_fake_minimax_chat_template(expected_messages[:1], add_generation_prompt=True)
    prompt_length = len(processor.tokenizer.encode(prompt_text, add_special_tokens=False))

    assert encoded["input_ids"].tolist() == expected_ids
    assert torch.all(encoded["labels"][:prompt_length] == -100)
    assert encoded["labels"][prompt_length:].tolist() == expected_ids[prompt_length:]
    assert [call["add_generation_prompt"] for call in processor.chat_template_calls] == [False, True, False]
    assert processor.chat_template_calls[-1]["messages"] == expected_messages
    assert "]~b]ai\n" in expected_text
    assert "]~b]assistant\n" not in expected_text


def test_minimax_chat_template_expands_media_without_processor_replacement_methods():
    """Check grid-based image and video expansion when the processor has no replacement helpers."""
    processor = _ProcessorWithoutReplacementMethods()
    chat_template = MiniMaxM3VLChatTemplate(processor)
    encoded = chat_template.encode_messages(
        [["user", ("image", None), ("video", None), ("image", None), ("video", None)]],
        processor=processor,
        image_inputs={"image_grid_thw": torch.tensor([[1, 2, 2], [1, 4, 2]])},
        video_inputs={
            "video_grid_thw": torch.tensor([[2, 2, 2], [1, 4, 2]]),
        },
        video_metadata=[VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3]), None],
    )

    assert (encoded["input_ids"] == chat_template.image_token_id).sum().item() == 3
    assert (encoded["input_ids"] == chat_template.video_token_id).sum().item() == 4
    assert "]<]0.0 seconds[>[" in processor.tokenizer.last_encoded_text
    assert "]<]1.0 seconds[>[" in processor.tokenizer.last_encoded_text


@pytest.mark.parametrize("metadata_kwargs", [{"video_metadata": None}, {}], ids=["none", "missing-key"])
def test_minimax_chat_template_expands_video_without_metadata(metadata_kwargs):
    """Check that video expansion works when optional timing metadata is absent."""
    processor = _ProcessorWithoutReplacementMethods()
    chat_template = MiniMaxM3VLChatTemplate(processor)
    video_inputs = {"video_grid_thw": torch.tensor([[2, 2, 2]])}

    encoded = chat_template.encode_messages(
        [["user", ("video", None)]],
        processor=processor,
        video_inputs=video_inputs,
        **metadata_kwargs,
    )

    assert (encoded["input_ids"] == chat_template.video_token_id).sum().item() == 2
    assert "seconds[>[" not in processor.tokenizer.last_encoded_text


@pytest.mark.parametrize(
    ("message", "media_kwargs", "modality"),
    [
        (["user", ("image", None)], {}, "image"),
        (
            ["user", ("image", None), ("image", None)],
            {"image_inputs": {"image_grid_thw": torch.tensor([[1, 2, 2]])}},
            "image",
        ),
        (["user", ("video", None)], {}, "video"),
    ],
    ids=["image-missing", "image-short", "video-missing"],
)
def test_minimax_chat_template_rejects_placeholders_without_processed_media(message, media_kwargs, modality):
    processor = _ProcessorWithoutReplacementMethods()
    chat_template = MiniMaxM3VLChatTemplate(processor)

    with pytest.raises(ValueError, match=f"more {modality} placeholders"):
        chat_template.encode_messages([message], processor=processor, **media_kwargs)


def test_minimax_chat_template_rejects_conversation_starting_with_assistant():
    processor = _ProcessorWithoutReplacementMethods()
    chat_template = MiniMaxM3VLChatTemplate(processor)

    with pytest.raises(ValueError, match="must not start with an assistant message"):
        chat_template.encode_messages([["assistant", ("text", "Answer")]], processor=processor)


class _RealMiniMaxVideoProcessorShim(_FakeNativeTemplateProcessor):
    image_token_id = 200025
    video_token_id = 200026

    def __init__(self):
        from transformers.models.minimax_m3_vl.video_processing_minimax_m3_vl import MiniMaxM3VLVideoProcessor

        self.tokenizer = _FakeMiniMaxTokenizer()
        self.video_processor = MiniMaxM3VLVideoProcessor()
        self.chat_template_calls = []

    def replace_video_token(self, video_inputs, video_idx):
        """Adapt the real video replacement API to the processor test fixture."""
        merge_length = self.video_processor.merge_size**2
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        grid_t = int(grid_thw[0])
        frame_seqlen = int(grid_thw[1:].prod() // merge_length)
        return "".join(
            MiniMaxM3VLChatTemplate.VISION_START_TOKEN
            + MiniMaxM3VLChatTemplate.VIDEO_TOKEN * frame_seqlen
            + MiniMaxM3VLChatTemplate.VISION_END_TOKEN
            for _ in range(grid_t)
        )


def test_minimax_m3_vl_transform_uses_real_chat_template_replacements(monkeypatch):
    """Verify media preprocessing and native-template expansion through the registered data transform."""

    def fake_conv_preprocess(source, conversations, **kwargs):
        """Return a fixed image/video conversation for the media-transform contract test."""
        assert source == "minimax_template_test"
        return [
            ["user", ("image", None), ("video", None), ("text", "Describe both.")],
            ["assistant", ("text", "Done.")],
        ]

    def fake_fetch_images(images, **kwargs):
        """Return fixture image inputs without reading external media."""
        assert images == ["image://one"]
        return ["decoded-image"]

    def fake_fetch_videos_metadata(videos, **kwargs):
        """Return fixture video frames and metadata without decoding external media."""
        assert videos == ["video://one"]
        return ["decoded-video"], [{"fps": 1.0}], None, None

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)
    monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", fake_fetch_images)
    monkeypatch.setattr("veomni.data.multimodal.video_utils.fetch_videos_metadata", fake_fetch_videos_metadata)

    processor = _FakeProcessor(tokenizer=_FakeMiniMaxTokenizer())
    chat_template = MiniMaxM3VLChatTemplate(processor)
    assert chat_template.processor is processor
    assert chat_template.tokenizer is processor.tokenizer
    [example] = process_sample_minimax_m3_vl(
        sample={
            "source_name": "minimax_template_test",
            "conversations": [{"from": "human", "value": "<image><video>Describe both."}],
            "images": ["image://one"],
            "videos": ["video://one"],
        },
        processor=processor,
        chat_template=chat_template,
    )

    assert len(processor.image_processor.calls) == 1
    assert len(processor.video_processor.calls) == 1
    assert processor.replaced_images == []
    assert processor.replaced_videos == []
    assert torch.equal(example["image_grid_thw"], torch.tensor([[1, 2, 2]], dtype=torch.long))
    assert torch.equal(example["video_grid_thw"], torch.tensor([[2, 2, 2]], dtype=torch.long))
    assert example["pixel_values"].shape == (4, 6)
    assert example["pixel_values_videos"].shape == (8, 6)
    assert "video_metadata" not in example
    assert torch.equal(example["position_ids"], torch.arange(example["input_ids"].numel(), dtype=torch.long))
    assert example["image_mask"].sum().item() == 1
    assert example["video_mask"].sum().item() == 2
    assert torch.equal(example["image_mask"], example["input_ids"] == processor.image_token_id)
    assert torch.equal(example["video_mask"], example["input_ids"] == processor.video_token_id)
    assert torch.all(example["labels"][example["image_mask"] | example["video_mask"]] == -100)


def _video_input(kind, tmp_path):
    """Build a small video input in the tensor layout expected by the native processor."""
    if kind == "frames":
        return [Image.new("RGB", (28, 28), color=(idx * 40, 64, 32)) for idx in range(2)]
    if not is_ffmpeg_available():
        pytest.skip("torchcodec or ffmpeg is not available")
    pytest.importorskip("av")
    from veomni.data.multimodal.video_utils import save_video_tensors_to_file

    video = torch.zeros((4, 3, 28, 28), dtype=torch.uint8)
    video[:, 0] = torch.arange(4, dtype=torch.uint8).view(4, 1, 1) * 40
    video[:, 1] = 80
    video[:, 2] = 32
    video_path = tmp_path / "minimax_m3_vl_tiny.mp4"
    save_video_tensors_to_file(video, str(video_path), fps=2)
    return str(video_path) if kind == "path" else video_path.read_bytes()


@pytest.mark.parametrize("video_kind", ["frames", "path", "bytes"])
def test_minimax_m3_vl_transform_with_real_video_processor(monkeypatch, tmp_path, video_kind):
    """Run VeOmni's pre-decoded-frame and str/bytes container video paths through the native processor."""

    def fake_conv_preprocess(source, conversations, **kwargs):
        return [["user", ("video", None), ("text", "Describe video.")], ["assistant", ("text", "Done.")]]

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)

    processor = _RealMiniMaxVideoProcessorShim()
    chat_template = MiniMaxM3VLChatTemplate(processor)

    [example] = process_sample_minimax_m3_vl(
        sample={
            "source_name": "minimax_real_video",
            "conversations": [{"from": "human", "value": "<video>Describe video."}],
            "videos": [_video_input(video_kind, tmp_path)],
        },
        processor=processor,
        chat_template=chat_template,
        fps=1.0,
        use_audio_in_video=False,
    )

    expected_video_tokens = int(example["video_grid_thw"][0].prod() // processor.video_processor.merge_size**2)
    assert example["pixel_values_videos"].shape == (16, 1176)
    assert torch.equal(example["video_grid_thw"], torch.tensor([[1, 4, 4]], dtype=torch.long))
    assert example["video_mask"].sum().item() == expected_video_tokens
    assert torch.all(example["labels"][example["video_mask"]] == -100)
    assert "video_metadata" not in example


# Public MiniMax M3 checkpoints ship their own processing_minimax.py, so which
# class name AutoProcessor resolves depends on the checkpoint's auto_map. Both
# must route to VeOmni's processor.
@pytest.mark.parametrize("processor_class_name", ["MiniMaxM3VLProcessor", "MiniMaxVLProcessor"])
def test_minimax_m3_vl_processor_registry_resolves_to_veomni_class(processor_class_name):
    """Check that MiniMax processor lookup resolves to the VeOmni integration."""
    from veomni.models.registry import MODEL_PROCESSOR_REGISTRY
    from veomni.models.transformers.minimax_m3_vl.processing_minimax_m3_vl import MiniMaxM3VLProcessor

    assert MODEL_PROCESSOR_REGISTRY[processor_class_name]() is MiniMaxM3VLProcessor


def test_minimax_m3_vl_processor_builds_from_native_subprocessors(monkeypatch):
    # The checkpoint's bundled processor is unusable, so from_pretrained must
    # build the native tokenizer / image / video processors by concrete class
    # with trust_remote_code=False, and take the chat template from the tokenizer.
    """Check construction from the checkpoint tokenizer, image processor and video processor."""
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


def test_minimax_m3_vl_processor_refuses_other_minimax_checkpoints(tmp_path):
    # ``MiniMaxVLProcessor`` is a generic remote class name; only substitute for M3 checkpoints.
    import json

    from veomni.models.transformers.minimax_m3_vl.processing_minimax_m3_vl import MiniMaxM3VLProcessor

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "minimax_vl_other"}))
    with pytest.raises(ValueError, match="only supports model_type 'minimax_m3_vl'"):
        MiniMaxM3VLProcessor.from_pretrained(str(tmp_path))
