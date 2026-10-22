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

from types import SimpleNamespace

import pytest
import torch

from veomni.data.data_transform import _replace_gemma3_image_tokens, process_sample_gemma3
from veomni.utils.constants import IGNORE_INDEX


def test_replace_gemma3_image_tokens_expands_original_placeholders_once():
    processor = SimpleNamespace(
        boi_token="<image>",
        replace_image_token=lambda _image_inputs, image_idx: f"<image><tokens-{image_idx}>",
    )

    text = _replace_gemma3_image_tokens(
        "before<image>middle<image>after",
        processor,
        torch.tensor([0, 0]),
        image_count=2,
    )

    assert text == "before<image><tokens-0>middle<image><tokens-1>after"


def test_replace_gemma3_image_tokens_rejects_placeholder_mismatch():
    processor = SimpleNamespace(
        boi_token="<image>",
        replace_image_token=lambda _image_inputs, image_idx: f"<image><tokens-{image_idx}>",
    )

    with pytest.raises(ValueError, match="1 image placeholders for 2 images"):
        _replace_gemma3_image_tokens("before<image>after", processor, torch.tensor([0, 0]), image_count=2)


def test_process_sample_gemma3_keeps_one_bos_and_assistant_label_boundary(monkeypatch):
    token_calls = []

    class FakeTokenizer:
        image_token_id = 99

        def __call__(self, text, *, add_special_tokens, return_tensors):
            token_calls.append(add_special_tokens)
            token_ids = {
                "<bos>": 1,
                "<user>": 2,
                "hello": 3,
                "<assistant>": 4,
                "answer": 5,
            }
            ids = [token_ids[token] for token in text.split()]
            if add_special_tokens:
                ids.insert(0, token_ids["<bos>"])
            return {"input_ids": torch.tensor([ids])}

    processor = SimpleNamespace(
        boi_token="<image>",
        image_seq_length=256,
        tokenizer=FakeTokenizer(),
        apply_chat_template=lambda messages, **_kwargs: " ".join(
            ["<bos>"]
            + [
                token
                for message in messages
                for token in (f"<{message['role']}>", message["content"][0]["text"])
            ]
        ),
    )
    conversations = [
        ["user", ("text", "hello")],
        ["assistant", ("text", "answer")],
    ]
    monkeypatch.setattr("veomni.data.multimodal.conv_preprocess", lambda *_args, **_kwargs: conversations)
    monkeypatch.setattr("veomni.data.multimodal.image_utils.fetch_images", lambda *_args, **_kwargs: [])

    result = process_sample_gemma3(
        sample={"conversations": []},
        processor=processor,
        position_id_func=lambda input_ids, **_kwargs: {
            "position_ids": torch.arange(input_ids.shape[-1]).unsqueeze(0)
        },
    )[0]

    assert token_calls == [False, False, False]
    assert result["input_ids"].tolist() == [1, 2, 3, 4, 5]
    assert result["input_ids"].tolist().count(1) == 1
    assert result["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]


def test_process_sample_gemma3_handles_interleaved_tulu_sample_with_null_columns(monkeypatch):
    class FakeTokenizer:
        image_token_id = 99

        def __call__(self, text, *, add_special_tokens, return_tensors):
            del add_special_tokens, return_tensors
            ids = [1 if token == "<user>" else 2 if token == "<assistant>" else 3 for token in text.split()]
            return {"input_ids": torch.tensor([ids])}

    processor = SimpleNamespace(
        boi_token="<image>",
        image_seq_length=256,
        tokenizer=FakeTokenizer(),
        apply_chat_template=lambda messages, **_kwargs: " ".join(
            f"<{message['role']}> {message['content'][0]['text']}" for message in messages
        ),
    )
    fetched_images = []
    monkeypatch.setattr(
        "veomni.data.multimodal.image_utils.fetch_images",
        lambda images, **_kwargs: fetched_images.extend(images) or [],
    )

    result = process_sample_gemma3(
        sample={
            "source_name": "tulu-3-sft-mixture",
            "conversations": None,
            "images": None,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
            ],
        },
        processor=processor,
        position_id_func=lambda input_ids, **_kwargs: {
            "position_ids": torch.arange(input_ids.shape[-1]).unsqueeze(0)
        },
    )[0]

    assert fetched_images == []
    assert result["input_ids"].tolist() == [1, 3, 2, 3]
    assert result["labels"].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 2, 3]
