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

import pytest
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
    ops_config_scope,
    qwen_image_inputs,
    qwen_video_inputs,
)
from tests.models.tiny_configs import tiny_qwen3_vl_config as _tiny_config


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
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _qwen3_vl_cls()(config)


def test_qwen3_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_qwen3_vl_eager_matches_hf_vision_and_text(modality):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    vision_inputs = (
        qwen_image_inputs(config, input_ids)
        if modality == "image"
        else qwen_video_inputs(config, input_ids, split_frames=True)
    )
    ids = vision_inputs.pop("input_ids")
    labels = vision_inputs.pop("labels")
    assert_eager_matches_hf(hf, ours, input_ids=ids, labels=labels, fwd_kwargs=vision_inputs)
