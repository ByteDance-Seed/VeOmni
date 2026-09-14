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

"""Qwen2.5-VL models consume tests.

Direct-import the generated class. Compare a toy model against HuggingFace on
both the text-only and image+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalGeneration,
)

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from tests.models.tiny_configs import tiny_qwen2_5_vl_config as _tiny_config
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _build_ours(config: Qwen2_5_VLConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
        Qwen2_5_VLForConditionalGeneration,
    )

    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return Qwen2_5_VLForConditionalGeneration(config)
    finally:
        set_ops_config(previous)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
    }


def _image_inputs(config: Qwen2_5_VLConfig, input_ids: torch.Tensor) -> dict:
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
    return {
        "input_ids": ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": image_mask.int(),
        "image_mask": image_mask,
        "video_mask": video_mask,
    }


def test_qwen2_5_vl_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"


def test_qwen2_5_vl_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen2_5_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2_5_VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs=_mask_kwargs(input_ids),
    )


def test_qwen2_5_vl_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2_5_VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    ours_masks = {
        "image_mask": image.pop("image_mask"),
        "video_mask": image.pop("video_mask"),
    }
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        fwd_kwargs=image,
        ours_fwd_kwargs=ours_masks,
    )
