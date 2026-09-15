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

import pytest
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
    ops_config_scope,
    qwen_image_inputs,
    qwen_video_inputs,
)
from tests.models.tiny_configs import tiny_qwen2_5_vl_config as _tiny_config
from veomni.ops import VeomniOp


def _build_ours(config: Qwen2_5_VLConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
        Qwen2_5_VLForConditionalGeneration,
    )

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return Qwen2_5_VLForConditionalGeneration(config)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
    }


def test_qwen2_5_vl_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"


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


@pytest.mark.parametrize("modality", ["image", "video"])
def test_qwen2_5_vl_eager_matches_hf_vision_and_text(modality):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2_5_VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    vision_inputs = (
        qwen_image_inputs(config, input_ids)
        if modality == "image"
        else qwen_video_inputs(config, input_ids, split_frames=False)
    )
    ids = vision_inputs.pop("input_ids")
    labels = vision_inputs.pop("labels")
    ours_masks = {
        "image_mask": vision_inputs.pop("image_mask"),
        "video_mask": vision_inputs.pop("video_mask"),
    }
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        labels=labels,
        fwd_kwargs=vision_inputs,
        ours_fwd_kwargs=ours_masks,
    )
