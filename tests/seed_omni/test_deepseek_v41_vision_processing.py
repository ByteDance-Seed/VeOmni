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

"""The DeepSeek-V4.1 image processor must accept what the training data layer emits.

Inference hands the processor a PIL image, but the SeedOmni V2 data layer decodes
every image to a ``(C, H, W) uint8`` tensor before it reaches a module
(``veomni.data.seed_omni.image_utils.fetch_images``). Both inputs have to produce
the same patches: the resize plan runs on PIL, so the tensor path is only correct
if it reproduces the PIL image exactly.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from veomni.data.seed_omni.image_utils import fetch_images
from veomni.models.seed_omni.modules.deepseek_v41.vision.processing import (
    DeepseekV41VisionImageProcessor,
    DeepseekV41VisionPreprocessor,
    image_token_types,
)
from veomni.models.seed_omni.utils.conversation import ConversationItem


def _png_bytes(width: int, height: int) -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def _processor() -> DeepseekV41VisionImageProcessor:
    return DeepseekV41VisionImageProcessor(patch_size=14, downsample_ratio=3, max_image_tokens=64, min_pixels=0)


@pytest.mark.parametrize("size", [(100, 60), (37, 211)])
def test_data_layer_tensor_matches_pil(size):
    raw = _png_bytes(*size)
    (tensor,) = fetch_images([raw])
    assert tensor.dtype == torch.uint8 and tensor.shape[0] == 3

    processor = _processor()
    from_tensor = processor.preprocess_one(tensor)
    from_pil = processor.preprocess_one(Image.open(io.BytesIO(raw)))

    assert from_tensor[1:] == from_pil[1:]
    torch.testing.assert_close(from_tensor[0], from_pil[0], rtol=0, atol=0)


def test_preprocess_conversations_accepts_data_layer_images():
    (tensor,) = fetch_images([_png_bytes(100, 60)])
    item = ConversationItem(type="image", value=tensor, role="user")
    preprocessor = DeepseekV41VisionPreprocessor(_processor(), dtype=torch.bfloat16)

    preprocessor.preprocess_conversations([[item]])

    n_vit_h, n_vit_w = item.meta["_omni_vit_grid"]
    assert item.value.shape == (n_vit_h * n_vit_w, 3, 14, 14)
    assert item.value.dtype == torch.bfloat16
    n_llm_h, n_llm_w = -(-n_vit_h // 3), -(-n_vit_w // 3)
    assert torch.equal(item.meta["_omni_token_types"], image_token_types(n_llm_h, n_llm_w))


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(3, 8, 8, dtype=torch.float32),
        torch.zeros(8, 8, dtype=torch.uint8),
        torch.zeros(8, 8, 3, dtype=torch.uint8),
    ],
)
def test_rejects_tensors_outside_the_data_layer_contract(bad):
    with pytest.raises(TypeError):
        _processor().preprocess_one(bad)
