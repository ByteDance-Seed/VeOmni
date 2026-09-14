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

"""Qwen3-VL models consume tests.

Direct-import the generated classes. Compare a toy model against HuggingFace on
both the text-only and image+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from tests.models.tiny_configs import tiny_qwen3_vl_config as _tiny_config
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _qwen3_vl_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import (
            Qwen3VLForConditionalGeneration,
        )
    else:
        from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
            Qwen3VLForConditionalGeneration,
        )
    return Qwen3VLForConditionalGeneration


def _build_ours(config: Qwen3VLConfig, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return _qwen3_vl_cls()(config)
    finally:
        set_ops_config(previous)


def _image_inputs(config: Qwen3VLConfig, input_ids: torch.Tensor) -> dict:
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
        "image_mask": image_mask,
        "video_mask": video_mask,
        "mm_token_type_ids": image_mask.int(),
    }


def test_qwen3_vl_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    layer = model.model.language_model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"


def test_qwen3_vl_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen3_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen3_vl_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    assert_eager_matches_hf(hf, ours, input_ids=ids, fwd_kwargs=image)
