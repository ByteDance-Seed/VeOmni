# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw Gemma4 conversations reach packed training through the native processor."""

import io
import pickle
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import ops_config_scope
from tests.tools.gemma4_data_utils import gemma4_processor
from tests.tools.training_utils import make_eager_ops_config
from veomni.data import MainCollator, build_data_transform
from veomni.models import build_foundation_model
from veomni.ops.config import get_ops_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def model():
    with ops_config_scope(get_ops_config()):
        yield build_foundation_model(
            "tests/toy_config/gemma4_toy",
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=make_eager_ops_config(attn_implementation="sdpa"),
        )


def _sample(modalities=(), *, audio_length=1600, frames=2):
    contents = [{"type": "text", "text": "What is here?"}]
    sample = {}
    if "image" in modalities:
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
        sample["images"] = [stream.getvalue()]
        contents.append({"type": "image"})
    if "video" in modalities:
        sample["videos"] = [{"video": np.zeros((frames, 8, 8, 3), dtype=np.uint8), "video_fps": 2.0}]
        contents.append({"type": "video"})
    if "audio" in modalities:
        sample["audios"] = [np.zeros(audio_length, dtype=np.float32)]
        contents.append({"type": "audio"})
    sample["messages"] = [
        {"role": "user", "content": contents},
        {"role": "assistant", "content": "A cat."},
    ]
    return sample


def _transform(model, processor, **kwargs):
    return build_data_transform(
        "gemma4", processor=processor, chat_template=None, position_id_func=model.get_position_id_func(), **kwargs
    )


def _collator(model, **kwargs):
    return MainCollator(
        data_collate_info=model.get_extra_collate_infos(),
        metadata_collate_func=model.get_metadata_collate_func(),
        **kwargs,
    )


@pytest.mark.parametrize("modalities", [(), ("image",), ("video",), ("audio",), ("image", "video", "audio")])
def test_raw_gemma4_sample_reaches_model_forward_backward(model, modalities):
    processor = gemma4_processor()
    sample = _sample(modalities)
    transform = _transform(model, processor, min_frames=2, max_frames=2)
    feature = transform(sample)[0]
    ids, labels = feature["input_ids"], feature["labels"]
    assert ids.ndim == labels.ndim == feature["position_ids"].ndim == 1
    torch.testing.assert_close(feature["position_ids"], torch.arange(len(ids)))
    response = processor.tokenizer.decode(ids[labels != -100])
    assert response == "A cat.<turn|>\n"
    for token in (121, 123, 125):
        assert torch.all(labels[ids == token] == -100)
    for modality, key in (("image", "pixel_values"), ("video", "pixel_values_videos"), ("audio", "input_features")):
        assert (key in feature) == (modality in modalities)
    if "image" in modalities:
        assert (ids == model.config.image_token_id).sum() == 1
    if "video" in modalities:
        assert (ids == model.config.video_token_id).sum() == 2
    batch = _collator(model)([feature])
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    assert batch["cu_seq_lens_q"].tolist() == [0, len(ids)]
    output = model(**batch, use_cache=False)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_gemma4_packs_ragged_audio_and_video_without_losing_boundaries(model):
    processor = gemma4_processor()
    first = _transform(model, processor, min_frames=2, max_frames=2)(_sample(("audio", "video")))[0]
    second_sample = _sample(("audio", "video"), audio_length=3200, frames=3)
    second_sample["audios"].append(np.zeros(4800, dtype=np.float32))
    second_sample["messages"][0]["content"].append({"type": "audio"})
    second = _transform(model, processor, min_frames=3, max_frames=3)(second_sample)[0]
    text = _transform(model, processor)(_sample())[0]
    features = [first, second, text]
    batch = _collator(model, pad_to_length=sum(len(f["input_ids"]) for f in features) + 4)(deepcopy(features))
    assert batch["input_features"].shape[:2] == (3, second["input_features"].shape[1])
    assert batch["input_features_mask"].shape == batch["input_features"].shape[:2]
    torch.testing.assert_close(
        batch["input_features"][0, : first["input_features"].shape[1]], first["input_features"][0]
    )
    assert not batch["input_features_mask"][0, first["input_features"].shape[1] :].any()
    assert not batch["input_features"][0, first["input_features"].shape[1] :].any()
    assert batch["pixel_values_videos"].shape[:2] == (5, 1)
    assert batch["video_position_ids"].shape[:2] == (5, 1)
    boundary = len(first["input_ids"])
    assert batch["labels"][0, boundary] == -100
    assert batch["position_ids"][0, boundary] == 0
    assert batch["labels"][0, -4:].eq(-100).all()
    assert batch["mm_token_type_ids"][0, -4:].eq(0).all()
    output = model(**batch, use_cache=False)
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_gemma4_data_hooks_are_picklable_without_capturing_the_model(model):
    position = pickle.loads(pickle.dumps(model.get_position_id_func()))
    ids = torch.tensor([[2, 7, 8]])
    torch.testing.assert_close(position(input_ids=ids)["position_ids"], torch.tensor([[0, 1, 2]]))
    collator = pickle.loads(pickle.dumps(_collator(model)))
    features = _transform(model, gemma4_processor())(_sample())
    assert collator(features)["input_ids"].ndim == 2


def test_gemma4_uses_registered_source_preprocessing(model):
    sample = _sample(("image",))
    sample.pop("messages")
    sample.update(
        source="ArxivQA",
        conversations=[{"from": "human", "value": "<image>\nWhat is here?"}, {"from": "gpt", "value": "A cat."}],
    )
    feature = _transform(model, gemma4_processor())(sample)[0]
    assert (feature["input_ids"] == model.config.image_token_id).sum() == 1
    assert feature["labels"].ne(-100).any()


def test_gemma4_rejects_multimodal_truncation(model):
    with pytest.raises(ValueError, match="max_seq_len"):
        _transform(model, gemma4_processor(), max_seq_len=4)(_sample(("image",)))


def test_gemma4_rejects_missing_media_instead_of_training_orphan_placeholders(model):
    sample = _sample(("image",))
    sample.pop("images")
    with pytest.raises(ValueError, match="image"):
        _transform(model, gemma4_processor())(sample)


@pytest.mark.parametrize("first_loss_mask", [0, 1])
def test_gemma4_multi_turn_labels_follow_native_media_expansions(model, first_loss_mask):
    processor = gemma4_processor()
    sample = _sample(("image",))
    sample["messages"][1]["loss_mask"] = first_loss_mask
    sample["messages"] += [
        {"role": "user", "content": [{"type": "text", "text": "A cat."}, {"type": "image"}]},
        {"role": "assistant", "content": "Again."},
    ]
    sample["images"] *= 2
    feature = _transform(model, processor)(sample)[0]
    decoded = processor.tokenizer.decode(feature["labels"][feature["labels"] != -100])
    assert decoded == ("A cat.<turn|>\n" if first_loss_mask else "") + "Again.<turn|>\n"
    assert (feature["input_ids"] == model.config.image_token_id).sum() == 2


def test_gemma4_rejects_prefix_rewriting_chat_template(model, monkeypatch):
    processor = gemma4_processor()
    native_render = processor.apply_chat_template

    def render(messages, **kwargs):
        text = native_render(messages, **kwargs)
        return text.replace("user", "system", 1) if len(messages) == 1 else text

    monkeypatch.setattr(processor, "apply_chat_template", render)
    with pytest.raises(ValueError, match="prefix-stable"):
        _transform(model, processor)(_sample())


def test_gemma4_frame_packing_preserves_native_video_features(model):
    feature = _transform(model, gemma4_processor(), min_frames=2, max_frames=2)(_sample(("video",)))[0]
    frames, positions = feature["pixel_values_videos"], feature["video_position_ids"]
    model.eval()
    with torch.no_grad():
        packed = model.model.get_video_features(frames, positions, return_dict=True).pooler_output
        native = model.model.get_video_features(
            frames.reshape(1, 2, *frames.shape[2:]), positions.reshape(1, 2, *positions.shape[2:]), return_dict=True
        ).pooler_output
    torch.testing.assert_close(torch.cat(packed), torch.cat(native), rtol=0, atol=0)


def test_gemma4_vlm_collation_rejects_sequence_parallelism_before_packing(model, monkeypatch):
    from types import SimpleNamespace

    from veomni.models.transformers.gemma4.generated import patched_modeling_gemma4_gpu

    monkeypatch.setattr(patched_modeling_gemma4_gpu, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        model.get_extra_collate_infos()
