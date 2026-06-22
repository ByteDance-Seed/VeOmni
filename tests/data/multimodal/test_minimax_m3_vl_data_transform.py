import os

import pytest
import torch
from PIL import Image

from veomni.data.data_transform import process_sample_minimax_m3_vl
from veomni.data.multimodal.multimodal_chat_template import MiniMaxM3VLChatTemplate
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


class _FakeImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, return_tensors):
        self.calls.append({"images": images, "return_tensors": return_tensors})
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.arange(24, dtype=torch.float32).reshape(4, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class _FakeVideoProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, videos, video_metadata, return_tensors, return_metadata):
        self.calls.append(
            {
                "videos": videos,
                "video_metadata": video_metadata,
                "return_tensors": return_tensors,
                "return_metadata": return_metadata,
            }
        )
        assert return_tensors == "pt"
        assert return_metadata is True
        return {
            "pixel_values_videos": torch.arange(48, dtype=torch.float32).reshape(8, 6),
            "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
            "video_metadata": [{"timestamps": [0.0, 1.0]}],
        }


class _FakeProcessor:
    image_token_id = 200025
    video_token_id = 200026

    def __init__(self):
        self.image_processor = _FakeImageProcessor()
        self.video_processor = _FakeVideoProcessor()
        self.replaced_images = []
        self.replaced_videos = []

    def replace_image_token(self, image_inputs, image_idx):
        self.replaced_images.append({"image_inputs": image_inputs, "image_idx": image_idx})
        assert "pixel_values" in image_inputs
        assert "image_grid_thw" in image_inputs
        return MiniMaxM3VLChatTemplate.IMAGE_TOKEN

    def replace_video_token(self, video_inputs, video_idx):
        self.replaced_videos.append({"video_inputs": video_inputs, "video_idx": video_idx})
        assert "pixel_values_videos" in video_inputs
        assert "video_grid_thw" in video_inputs
        assert video_inputs["video_metadata"] == [{"timestamps": [0.0, 1.0]}]
        return MiniMaxM3VLChatTemplate.VIDEO_TOKEN


class _FakeMiniMaxChatTemplate:
    image_token_id = 200025
    video_token_id = 200026

    def __init__(self):
        self.seen = None

    def encode_messages(self, conversations, **kwargs):
        self.seen = {"conversations": conversations, **kwargs}

        image_inputs = kwargs["image_inputs"]
        video_inputs = kwargs["video_inputs"]
        assert "pixel_values" in image_inputs
        assert "image_grid_thw" in image_inputs
        assert "pixel_values_videos" in video_inputs
        assert "video_grid_thw" in video_inputs
        assert video_inputs["video_metadata"] == [{"timestamps": [0.0, 1.0]}]

        return {
            "input_ids": torch.tensor([101, self.image_token_id, 102, self.video_token_id, 103], dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.tensor([101, self.image_token_id, 102, self.video_token_id, 103], dtype=torch.long),
        }


def test_minimax_m3_vl_transform_uses_hf_image_and_video_processors(monkeypatch):
    def fake_conv_preprocess(source, conversations, **kwargs):
        assert source == "minimax_test"
        return [
            ["user", ("image", None), ("video", None), ("text", "Describe both.")],
            ["assistant", ("text", "Done.")],
        ]

    def fake_fetch_images(images, **kwargs):
        assert images == ["image://one"]
        return ["decoded-image"]

    def fake_fetch_videos_metadata(videos, **kwargs):
        assert videos == ["video://one"]
        return ["decoded-video"], [{"fps": 1.0}], None, None

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)
    monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", fake_fetch_images)
    monkeypatch.setattr("veomni.data.multimodal.video_utils.fetch_videos_metadata", fake_fetch_videos_metadata)

    processor = _FakeProcessor()
    chat_template = _FakeMiniMaxChatTemplate()
    [example] = process_sample_minimax_m3_vl(
        sample={
            "source_name": "minimax_test",
            "conversations": [{"from": "human", "value": "<image><video>Describe both."}],
            "images": ["image://one"],
            "videos": ["video://one"],
        },
        processor=processor,
        chat_template=chat_template,
    )

    assert len(processor.image_processor.calls) == 1
    assert len(processor.video_processor.calls) == 1
    assert torch.equal(example["image_grid_thw"], torch.tensor([[1, 2, 2]], dtype=torch.long))
    assert torch.equal(example["video_grid_thw"], torch.tensor([[2, 2, 2]], dtype=torch.long))
    assert example["pixel_values"].shape == (4, 6)
    assert example["pixel_values_videos"].shape == (8, 6)
    assert "video_metadata" not in example
    assert torch.equal(example["image_mask"], example["input_ids"] == processor.image_token_id)
    assert torch.equal(example["video_mask"], example["input_ids"] == processor.video_token_id)
    assert example["labels"][1].item() == -100
    assert example["labels"][3].item() == -100
    assert torch.equal(example["position_ids"], torch.arange(example["input_ids"].numel(), dtype=torch.long))


class _FakeMiniMaxTokenizer:
    eos_token = "<eos>"

    _token_ids = {
        MiniMaxM3VLChatTemplate.IMAGE_TOKEN: 200025,
        MiniMaxM3VLChatTemplate.VIDEO_TOKEN: 200026,
        "<eos>": 200027,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def encode(self, text, add_special_tokens=False):
        token_ids = []
        idx = 0
        special_tokens = (
            MiniMaxM3VLChatTemplate.IMAGE_TOKEN,
            MiniMaxM3VLChatTemplate.VIDEO_TOKEN,
            "<eos>",
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


class _RealMiniMaxVideoProcessorShim:
    image_token_id = 200025
    video_token_id = 200026

    def __init__(self):
        from transformers.models.minimax_m3_vl.video_processing_minimax_m3_vl import MiniMaxM3VLVideoProcessor

        self.video_processor = MiniMaxM3VLVideoProcessor()

    def replace_video_token(self, video_inputs, video_idx):
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


def test_minimax_m3_vl_transform_uses_real_chat_template_processor_replacements(monkeypatch):
    def fake_conv_preprocess(source, conversations, **kwargs):
        assert source == "minimax_template_test"
        return [
            ["user", ("image", None), ("video", None), ("text", "Describe both.")],
            ["assistant", ("text", "Done.")],
        ]

    def fake_fetch_images(images, **kwargs):
        assert images == ["image://one"]
        return ["decoded-image"]

    def fake_fetch_videos_metadata(videos, **kwargs):
        assert videos == ["video://one"]
        return ["decoded-video"], [{"fps": 1.0}], None, None

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)
    monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", fake_fetch_images)
    monkeypatch.setattr("veomni.data.multimodal.video_utils.fetch_videos_metadata", fake_fetch_videos_metadata)

    processor = _FakeProcessor()
    chat_template = MiniMaxM3VLChatTemplate(_FakeMiniMaxTokenizer())
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

    assert processor.replaced_images[0]["image_idx"] == 0
    assert processor.replaced_videos[0]["video_idx"] == 0
    assert "video_metadata" not in example
    assert example["image_mask"].sum().item() == 1
    assert example["video_mask"].sum().item() == 1
    assert torch.equal(example["image_mask"], example["input_ids"] == processor.image_token_id)
    assert torch.equal(example["video_mask"], example["input_ids"] == processor.video_token_id)
    assert torch.all(example["labels"][example["image_mask"] | example["video_mask"]] == -100)


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL video processor requires transformers>=5.12.0",
)
def test_minimax_m3_vl_transform_uses_veomni_video_fetch_with_real_processor(monkeypatch):
    """Exercise MiniMax with VeOmni's default pre-decoded-frame video path."""

    def fake_conv_preprocess(source, conversations, **kwargs):
        assert source == "minimax_real_video_fetch_test"
        return [["user", ("video", None), ("text", "Describe video.")], ["assistant", ("text", "Done.")]]

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)

    frames = [Image.new("RGB", (28, 28), color=(idx * 40, 64, 32)) for idx in range(2)]
    processor = _RealMiniMaxVideoProcessorShim()
    chat_template = MiniMaxM3VLChatTemplate(_FakeMiniMaxTokenizer())

    [example] = process_sample_minimax_m3_vl(
        sample={
            "source_name": "minimax_real_video_fetch_test",
            "conversations": [{"from": "human", "value": "<video>Describe video."}],
            "videos": [frames],
        },
        processor=processor,
        chat_template=chat_template,
        fps=1.0,
    )

    expected_video_tokens = int(example["video_grid_thw"][0].prod() // processor.video_processor.merge_size**2)
    assert example["pixel_values_videos"].shape == (16, 1176)
    assert torch.equal(example["video_grid_thw"], torch.tensor([[1, 4, 4]], dtype=torch.long))
    assert example["video_mask"].sum().item() == expected_video_tokens
    assert torch.all(example["labels"][example["video_mask"]] == -100)
    assert "video_metadata" not in example


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL video processor requires transformers>=5.12.0",
)
@pytest.mark.parametrize("container_kind", ["path", "bytes"])
def test_minimax_m3_vl_transform_uses_video_container_with_real_processor(monkeypatch, tmp_path, container_kind):
    """Exercise MiniMax with VeOmni's str/bytes video-container path."""
    pytest.importorskip("av")
    from veomni.data.multimodal.video_utils import save_video_tensors_to_file

    def fake_conv_preprocess(source, conversations, **kwargs):
        assert source == f"minimax_real_video_container_{container_kind}"
        return [["user", ("video", None), ("text", "Describe video.")], ["assistant", ("text", "Done.")]]

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)

    video = torch.zeros((4, 3, 28, 28), dtype=torch.uint8)
    video[:, 0] = torch.arange(4, dtype=torch.uint8).view(4, 1, 1) * 40
    video[:, 1] = 80
    video[:, 2] = 32
    video_path = tmp_path / "minimax_m3_vl_tiny.mp4"
    save_video_tensors_to_file(video, str(video_path), fps=2)
    video_input = str(video_path) if container_kind == "path" else video_path.read_bytes()

    processor = _RealMiniMaxVideoProcessorShim()
    chat_template = MiniMaxM3VLChatTemplate(_FakeMiniMaxTokenizer())

    [example] = process_sample_minimax_m3_vl(
        sample={
            "source_name": f"minimax_real_video_container_{container_kind}",
            "conversations": [{"from": "human", "value": "<video>Describe video."}],
            "videos": [video_input],
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


class _ModelShapeImageProcessor(_FakeImageProcessor):
    def __init__(self, *, pixel_row_size, merge_size):
        super().__init__()
        self.pixel_row_size = pixel_row_size
        self.merge_size = merge_size

    def __call__(self, *, images, return_tensors):
        self.calls.append({"images": images, "return_tensors": return_tensors})
        assert return_tensors == "pt"
        grid = torch.tensor([[1, self.merge_size, self.merge_size]], dtype=torch.long)
        return {
            "pixel_values": torch.randn((self.merge_size * self.merge_size, self.pixel_row_size)),
            "image_grid_thw": grid,
        }


class _ModelShapeVideoProcessor(_FakeVideoProcessor):
    def __init__(self, *, pixel_row_size, merge_size):
        super().__init__()
        self.pixel_row_size = pixel_row_size
        self.merge_size = merge_size

    def __call__(self, *, videos, video_metadata, return_tensors, return_metadata):
        self.calls.append(
            {
                "videos": videos,
                "video_metadata": video_metadata,
                "return_tensors": return_tensors,
                "return_metadata": return_metadata,
            }
        )
        assert return_tensors == "pt"
        assert return_metadata is True
        grid = torch.tensor([[1, self.merge_size, self.merge_size]], dtype=torch.long)
        return {
            "pixel_values_videos": torch.randn((self.merge_size * self.merge_size, self.pixel_row_size)),
            "video_grid_thw": grid,
            "video_metadata": [{"timestamps": [0.0, 1.0]}],
        }


class _ModelShapeProcessor(_FakeProcessor):
    def __init__(self, *, pixel_row_size, merge_size):
        super().__init__()
        self.image_processor = _ModelShapeImageProcessor(pixel_row_size=pixel_row_size, merge_size=merge_size)
        self.video_processor = _ModelShapeVideoProcessor(pixel_row_size=pixel_row_size, merge_size=merge_size)


@pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.12.0"),
    reason="MiniMax M3 VL generated modeling requires transformers>=5.12.0",
)
def test_minimax_m3_vl_transform_to_collator_to_generated_model_backward(monkeypatch):
    """Exercise transform -> MainCollator -> generated model backward.

    This bridges the fake HF processor outputs through the same VeOmni
    transform/collator contract the trainer uses, then runs a toy generated
    MiniMax model step. The toy config has a tiny vocab, so the test feeds
    `inputs_embeds` plus the transform-produced image/video masks while keeping
    the multimodal tensors and metadata from the actual collated batch.
    """
    from veomni.data.data_collator import MainCollator
    from veomni.models.loader import get_model_class, get_model_config

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12358")

    def fake_conv_preprocess(source, conversations, **kwargs):
        assert source == "minimax_e2e_test"
        return [
            ["user", ("image", None), ("video", None), ("text", "Describe both.")],
            ["assistant", ("text", "Done.")],
        ]

    def fake_fetch_images(images, **kwargs):
        assert images == ["image://one"]
        return ["decoded-image"]

    def fake_fetch_videos_metadata(videos, **kwargs):
        assert videos == ["video://one"]
        return ["decoded-video"], [{"fps": 1.0}], None, None

    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", fake_conv_preprocess)
    monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", fake_fetch_images)
    monkeypatch.setattr("veomni.data.multimodal.video_utils.fetch_videos_metadata", fake_fetch_videos_metadata)

    torch.manual_seed(20260618)
    config = get_model_config("./tests/toy_config/minimax_m3_vl_toy/config.json")
    model_cls = get_model_class(config)
    model = model_cls(config)
    model.train()

    patch_size = config.vision_config.patch_size
    temporal_patch_size = config.vision_config.temporal_patch_size
    num_channels = config.vision_config.num_channels
    merge = config.vision_config.spatial_merge_size
    pixel_row_size = num_channels * temporal_patch_size * patch_size * patch_size

    processor = _ModelShapeProcessor(pixel_row_size=pixel_row_size, merge_size=merge)
    chat_template = MiniMaxM3VLChatTemplate(_FakeMiniMaxTokenizer())
    features = process_sample_minimax_m3_vl(
        sample={
            "source_name": "minimax_e2e_test",
            "conversations": [{"from": "human", "value": "<image><video>Describe both."}],
            "images": ["image://one"],
            "videos": ["video://one"],
        },
        processor=processor,
        chat_template=chat_template,
    )
    batch = MainCollator(metadata_collate_func=model.get_metadata_collate_func())(features)

    assert batch["multimodal_metadata"] == {
        "image_grid_thw_list": batch["image_grid_thw"].tolist(),
        "video_grid_thw_list": batch["video_grid_thw"].tolist(),
    }
    assert batch["image_mask"].sum().item() == 1
    assert batch["video_mask"].sum().item() == 1

    safe_ids = batch["input_ids"].remainder(config.text_config.vocab_size)
    inputs_embeds = model.get_input_embeddings()(safe_ids).detach().clone().requires_grad_(True)
    labels = batch["labels"].clone()
    labels[labels != -100] = labels[labels != -100].remainder(config.text_config.vocab_size)

    outputs = model(
        inputs_embeds=inputs_embeds,
        pixel_values=batch["pixel_values"],
        pixel_values_videos=batch["pixel_values_videos"],
        image_grid_thw=batch["image_grid_thw"],
        video_grid_thw=batch["video_grid_thw"],
        image_mask=batch["image_mask"],
        video_mask=batch["video_mask"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=labels,
        multimodal_metadata=batch["multimodal_metadata"],
    )

    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert model.model.vision_tower.embeddings.proj.weight.grad.abs().sum() > 0
    assert model.model.multi_modal_projector.merge_linear_2.weight.grad.abs().sum() > 0
    assert model.lm_head.weight.grad.abs().sum() > 0
