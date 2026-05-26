# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for SeedOmni V2 ``data_type=seedomni`` transform (Feature D1).

Covers the per-item dict schema (`type` / `value` / `role` / `loss_mask`),
image IO + uint8 tensor shape, ordering between conversation turns and
the supplied image list, mismatch detection, and unsupported-modality
errors.  These tests use the existing sharegpt4v preprocessors as a
realistic upstream and do **not** depend on any model / processor /
tokenizer — the whole point of D1 is that the data layer is
model-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from veomni.data.data_transform import build_data_transform
from veomni.data.multimodal.seedomni_transform import (
    _build_conversation_list,
    _modalities_in,
    _pil_to_uint8_tensor,
    process_seedomni_example,
)


_TEST_IMAGE = Path(__file__).resolve().parents[1].parent / "testdata" / "qwen-vl-demo.jpeg"


# ─────────────────────────── helpers ───────────────────────────


def _solid_image(color: tuple, size=(64, 48)) -> Image.Image:
    """Return a small solid-color RGB PIL image (avoids tying tests to disk)."""
    return Image.new("RGB", size, color)


def _check_item_schema(item: dict) -> None:
    """The per-item dict must carry exactly the four V2 fields and nothing else.

    Extra fields would hint at the data layer leaking model-specific info
    (e.g. token ids, chat-template prefixes) which violates D1's contract.
    """
    assert set(item.keys()) == {"type", "value", "role", "loss_mask"}, (
        f"unexpected keys in conversation_list item: {set(item.keys()) - {'type', 'value', 'role', 'loss_mask'}}"
    )
    assert item["role"] in ("system", "user", "assistant"), item["role"]
    assert item["loss_mask"] in (0, 1), item["loss_mask"]


# ─────────────────────────── _pil_to_uint8_tensor ───────────────────────────


def test_pil_to_uint8_tensor_shape_and_dtype():
    img = _solid_image((10, 20, 30), size=(48, 32))  # PIL is (W, H)
    tensor = _pil_to_uint8_tensor(img)

    assert tensor.dtype == torch.uint8
    assert tensor.shape == (3, 32, 48)  # (C, H, W)
    # solid color should be preserved exactly through the np→torch round-trip
    assert tensor[0, 0, 0].item() == 10
    assert tensor[1, 0, 0].item() == 20
    assert tensor[2, 0, 0].item() == 30


def test_pil_to_uint8_tensor_handles_non_rgb():
    """Greyscale / RGBA inputs must be auto-converted to RGB."""
    grey = Image.new("L", (8, 8), 128)
    tensor = _pil_to_uint8_tensor(grey)

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.uint8


# ─────────────────────────── _build_conversation_list ───────────────────────────


def test_build_conversation_list_text_only():
    constructed = [
        ["user", ("text", "hello")],
        ["assistant", ("text", "hi there")],
    ]
    items = _build_conversation_list(constructed, image_tensors=[])

    assert len(items) == 2
    for it in items:
        _check_item_schema(it)
    assert items[0] == {"type": "text", "value": "hello", "role": "user", "loss_mask": 0}
    assert items[1] == {"type": "text", "value": "hi there", "role": "assistant", "loss_mask": 1}


def test_build_conversation_list_pairs_images_in_order():
    """Image tensors are consumed in source order — turn N's image is image N."""
    constructed = [
        ["user", ("image", None), ("text", "describe")],
        ["assistant", ("text", "a cat"), ("image", None)],
    ]
    img1 = torch.zeros(3, 8, 8, dtype=torch.uint8)
    img2 = torch.full((3, 8, 8), 255, dtype=torch.uint8)
    items = _build_conversation_list(constructed, image_tensors=[img1, img2])

    assert [it["type"] for it in items] == ["image", "text", "text", "image"]
    assert torch.equal(items[0]["value"], img1)
    assert items[1]["role"] == "user" and items[1]["value"] == "describe"
    assert items[2]["role"] == "assistant" and items[2]["loss_mask"] == 1
    assert torch.equal(items[3]["value"], img2)
    assert items[3]["loss_mask"] == 1  # assistant turn → supervised by default


def test_build_conversation_list_missing_image_raises():
    constructed = [["user", ("image", None), ("image", None)]]
    with pytest.raises(ValueError, match="more 'image' turns than supplied images"):
        _build_conversation_list(constructed, image_tensors=[torch.zeros(3, 4, 4, dtype=torch.uint8)])


def test_build_conversation_list_unused_image_raises():
    constructed = [["user", ("text", "no images here")]]
    with pytest.raises(ValueError, match="unused image"):
        _build_conversation_list(constructed, image_tensors=[torch.zeros(3, 4, 4, dtype=torch.uint8)])


def test_build_conversation_list_unsupported_modality():
    """Video / audio etc. must raise NotImplementedError (D1 scope is text + image)."""
    constructed = [["user", ("video", None)]]
    with pytest.raises(NotImplementedError, match="modality type 'video'"):
        _build_conversation_list(constructed, image_tensors=[])


def test_build_conversation_list_text_value_none_becomes_empty_string():
    """Some preprocessors emit ('text', None) for placeholder turns; we
    coerce that to empty string so the downstream tokenizer doesn't choke
    on a None value."""
    items = _build_conversation_list([["user", ("text", None)]], image_tensors=[])
    assert items[0]["value"] == ""


# ─────────────────────────── _modalities_in ───────────────────────────


def test_modalities_in_detects_image():
    assert _modalities_in([["user", ("image", None), ("text", "x")]]) == (True, False)
    assert _modalities_in([["user", ("vq_image", None)]]) == (True, False)
    assert _modalities_in([["user", ("text", "x")]]) == (False, False)


# ─────────────────────────── process_seedomni_example ───────────────────────────


def test_process_seedomni_example_with_real_image():
    """End-to-end through the registered transform with a real PIL image
    on disk — exercises ``fetch_images`` + smart_resize + tensor conversion."""
    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [
            {"from": "human", "value": "<image>What's in this image?"},
            {"from": "gpt", "value": "A scene from a demo."},
        ],
        "images": [str(_TEST_IMAGE)],
    }
    out = process_seedomni_example(sample)

    assert isinstance(out, list) and len(out) == 1
    raw_batch_entry = out[0]
    assert set(raw_batch_entry.keys()) == {"conversation_list"}

    conv_list = raw_batch_entry["conversation_list"]
    types = [it["type"] for it in conv_list]
    roles = [it["role"] for it in conv_list]
    assert "image" in types, "image turn was lost"
    assert "assistant" in roles
    for it in conv_list:
        _check_item_schema(it)

    img_items = [it for it in conv_list if it["type"] == "image"]
    assert len(img_items) == 1
    img_tensor = img_items[0]["value"]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dtype == torch.uint8
    assert img_tensor.dim() == 3 and img_tensor.shape[0] == 3, img_tensor.shape


def test_process_seedomni_example_text_only():
    """No 'images' field, no media IO — purely synthetic conversation
    routed through ``sharegpt4v_sft`` preprocessor."""
    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [
            {"from": "human", "value": "What is 2 + 2?"},
            {"from": "gpt", "value": "4"},
        ],
    }
    out = process_seedomni_example(sample)
    conv_list = out[0]["conversation_list"]

    assert len(conv_list) == 2
    assert conv_list[0] == {"type": "text", "value": "What is 2 + 2?", "role": "user", "loss_mask": 0}
    assert conv_list[1] == {"type": "text", "value": "4", "role": "assistant", "loss_mask": 1}


def test_process_seedomni_example_resize_kwargs_propagate(tmp_path):
    """``smart_resize`` kwargs (image_max_pixels) should be honoured —
    ensures the IO + resize stage exposes the size cap to upstream YAML."""
    img_path = tmp_path / "big.jpg"
    Image.new("RGB", (1024, 768), (50, 100, 150)).save(img_path)

    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [{"from": "human", "value": "<image>"}],
        "images": [str(img_path)],
    }
    # Cap at 64*64 = 4096 pixels — far below 1024*768.
    out = process_seedomni_example(sample, image_max_pixels=64 * 64)
    img_tensor = out[0]["conversation_list"][0]["value"]

    assert img_tensor.shape[1] * img_tensor.shape[2] <= 64 * 64, img_tensor.shape


def test_process_seedomni_example_missing_source_raises():
    sample = {"conversations": [{"from": "human", "value": "hi"}]}
    with pytest.raises(KeyError, match="source_name"):
        process_seedomni_example(sample)


def test_process_seedomni_example_silently_ignores_legacy_kwargs():
    """``OmniTrainer._build_data_transform`` injects tokenizer / max_seq_len /
    text_keys from the V1 contract.  V2 transform must accept and discard
    them — the model-agnostic data layer no longer cares about tokenizer."""
    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}],
    }
    out = process_seedomni_example(
        sample,
        tokenizer="<dummy_tokenizer_should_be_ignored>",
        max_seq_len=2048,
        text_keys="messages",
    )
    assert len(out[0]["conversation_list"]) == 2


# ─────────────────────────── registry wiring ───────────────────────────


def test_seedomni_is_registered_and_callable_via_build():
    """``build_data_transform("seedomni", ...)`` must return a callable
    that handles tokenizer / max_seq_len kwargs without raising —
    ``OmniTrainer`` builds it that way."""
    transform = build_data_transform("seedomni", tokenizer=None, max_seq_len=1024, text_keys="messages")
    sample = {
        "source_name": "sharegpt4v_sft",
        "conversations": [{"from": "human", "value": "hello"}],
    }
    out = transform(sample)
    assert out[0]["conversation_list"][0]["value"] == "hello"
