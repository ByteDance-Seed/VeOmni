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

"""Qwen2-VL models consume tests.

Direct-import the generated class. Compare a toy model against HuggingFace on
both the text-only and image+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration as HFQwen2VLForConditionalGeneration,
)

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    qwen_image_inputs,
)
from tests.models.tiny_configs import tiny_qwen2_vl_config as _tiny_config
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _build_ours(config: Qwen2VLConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_vl.generated.patched_modeling_qwen2_vl_gpu import (
        Qwen2VLForConditionalGeneration,
    )

    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return Qwen2VLForConditionalGeneration(config)
    finally:
        set_ops_config(previous)


def test_qwen2_vl_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"


def test_qwen2_vl_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen2_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen2_vl_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    labels = image.pop("labels")
    assert_eager_matches_hf(hf, ours, input_ids=ids, labels=labels, fwd_kwargs=image)
