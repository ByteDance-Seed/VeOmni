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
# See the License for the specific language governing limitations
# under the License.

"""Qwen2.5-Omni models_kernel consume tests.

Direct-import the generated Thinker class. Do not register or use
``build_foundation_model``. Compare a toy Thinker against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniVisionEncoderConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration as HFQwen2_5OmniThinker,
)

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121
AUDIO_TOKEN_ID = 122


def _tiny_thinker_config() -> Qwen2_5OmniThinkerConfig:
    text = Qwen2_5OmniTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_section": [4, 2, 2], "rope_type": "default"},
        tie_word_embeddings=False,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_sliding_window=False,
    )
    vision = Qwen2_5OmniVisionEncoderConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=16,
        out_hidden_size=64,
        fullatt_block_indexes=[0],
        hidden_act="silu",
    )
    audio = Qwen2_5OmniAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=64,
        n_window=4,
        max_source_positions=16,
    )
    return Qwen2_5OmniThinkerConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        audio_config=audio.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
    )


def _build_ours(config: Qwen2_5OmniThinkerConfig, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.transformers.qwen2_5_omni.generated.patched_modeling_qwen2_5_omni_gpu import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return Qwen2_5OmniThinkerForConditionalGeneration(config)
    finally:
        set_kernels_config(previous)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
        "audio_mask": zeros,
    }


def _image_inputs(config: Qwen2_5OmniThinkerConfig, input_ids: torch.Tensor) -> dict:
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grid_t, grid_h, grid_w = 1, merge, merge
    num_patches = grid_t * grid_h * grid_w
    n_tokens = num_patches // (merge**2)
    feat_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    pixel_values = torch.randn(num_patches, feat_dim)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    ids = input_ids.clone()
    ids[0, :n_tokens] = config.image_token_id
    image_mask = ids == config.image_token_id
    video_mask = torch.zeros_like(ids, dtype=torch.bool)
    audio_mask = torch.zeros_like(ids, dtype=torch.bool)
    return {
        "input_ids": ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": image_mask.int(),
        "image_mask": image_mask,
        "video_mask": video_mask,
        "audio_mask": audio_mask,
    }


def test_qwen2_5_omni_constructs_local_kernels():
    model = _build_ours(_tiny_thinker_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"


def test_qwen2_5_omni_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_thinker_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_thinker_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen2_5_omni_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen2_5OmniThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs=_mask_kwargs(input_ids),
    )


def test_qwen2_5_omni_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen2_5OmniThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    ours_masks = {
        "image_mask": image.pop("image_mask"),
        "video_mask": image.pop("video_mask"),
        "audio_mask": image.pop("audio_mask"),
    }
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        fwd_kwargs=image,
        ours_fwd_kwargs=ours_masks,
    )
