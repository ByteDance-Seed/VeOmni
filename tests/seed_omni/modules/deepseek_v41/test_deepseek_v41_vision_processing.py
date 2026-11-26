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
(``veomni.data.seed_omni.utils.image.fetch_images``). Both inputs have to produce
the same patches: the resize plan runs on PIL, so the tensor path is only correct
if it reproduces the PIL image exactly.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from veomni.data.seed_omni.utils.image import fetch_images
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


def test_preprocess_conversations_expands_video_into_frame_images():
    from veomni.data.seed_omni.utils.video import VideoInputs

    frames = torch.randint(0, 256, (4, 3, 84, 84), dtype=torch.uint8)
    video_item = ConversationItem(
        type="video",
        value=VideoInputs(video=frames),
        role="user",
        meta={"video_metadata": "sentinel"},
    )
    text_item = ConversationItem(type="text", value="Describe this video.", role="user")
    sample = [video_item, text_item]

    preprocessor = DeepseekV41VisionPreprocessor(_processor(), dtype=torch.bfloat16)
    preprocessor.preprocess_conversations([sample])

    assert len(sample) == 5
    for frame_item in sample[:4]:
        assert frame_item.type == "image"
        assert frame_item.role == "user"
        assert "video_metadata" not in frame_item.meta
        assert frame_item.meta["_omni_token_len"] > 0
        assert frame_item.value.dtype == torch.bfloat16
    assert sample[4] is text_item


def test_vision_encoder_gradient_checkpointing_matches_uncheckpointed():
    from veomni.models.seed_omni.modules.deepseek_v41.vision.configuration import DeepseekV41VisionEncoderConfig
    from veomni.models.seed_omni.modules.deepseek_v41.vision.modeling import (
        DeepseekV41VisionEncoder,
        build_deepseek_v41_vit_metadata,
    )

    torch.manual_seed(0)
    cfg = DeepseekV41VisionEncoderConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        out_hidden_size=64,
        patch_size=14,
        downsample_ratio=3,
        ops_implementation={"attn_implementation": "sdpa", "rms_norm_implementation": "eager"},
    )
    ref = DeepseekV41VisionEncoder(cfg).train()
    gc_model = DeepseekV41VisionEncoder(cfg).train()
    gc_model.load_state_dict(ref.state_dict())
    gc_model.gradient_checkpointing_enable()
    assert gc_model.vision.gradient_checkpointing is True
    assert all(b.gradient_checkpointing is True for b in gc_model.vision.blocks)

    patches = torch.randn(9, 3, 14, 14, requires_grad=True)
    patches_gc = patches.detach().clone().requires_grad_(True)
    grids = [[3, 3]]
    types = [image_token_types(1, 1)]
    meta = build_deepseek_v41_vit_metadata(grids, cfg.rope_dim, cfg.rope_theta, patches.device)

    out_ref = ref(pixel_values=patches, grids=grids, token_types=types, vit_metadata=meta)["image_embeds"]
    out_gc = gc_model(pixel_values=patches_gc, grids=grids, token_types=types, vit_metadata=meta)["image_embeds"]
    torch.testing.assert_close(out_gc, out_ref, rtol=0, atol=0)

    out_ref.sum().backward()
    out_gc.sum().backward()
    torch.testing.assert_close(patches_gc.grad, patches.grad, rtol=1e-5, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), gc_model.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-5, atol=1e-5)
