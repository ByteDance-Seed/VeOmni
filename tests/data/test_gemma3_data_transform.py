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

from veomni.data.data_transform import _replace_gemma3_image_tokens


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
