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

"""Qwen3-VL-MoE registry hooks, auxiliary-loss, and multimodal parity tests.

Direct-import the generated classes. Compare a toy model against Hugging Face on
both the text-only and image+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration,
)

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
    pin_eager_attn_implementation,
    qwen_image_inputs,
    stamp_attn_implementation,
)
from tests.models.tiny_configs import tiny_qwen3_vl_moe_config as _tiny_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _qwen3_vl_moe_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_npu import (
            Qwen3VLMoeForConditionalGeneration,
        )
    else:
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeForConditionalGeneration,
        )
    return Qwen3VLMoeForConditionalGeneration


def _build_ours(config: Qwen3VLMoeConfig, ops: SimpleNamespace | None = None):
    ops = ops if ops is not None else eager_ops_config()
    stamp_attn_implementation(config, ops.attn_implementation)
    with ops_config_scope(ops):
        return _qwen3_vl_moe_cls()(config)


@torch.no_grad()
def test_qwen3_vl_moe_registry_installs_checkpoint_and_lora_hooks():
    from veomni.models import get_model_class

    cases = (
        ("Qwen3VLMoeForConditionalGeneration", "model.language_model."),
        ("Qwen3VLMoeModel", "language_model."),
        ("Qwen3VLMoeTextModel", ""),
    )
    for architecture, prefix in cases:
        model_cls = get_model_class(_tiny_config(architecture))
        assert callable(model_cls._create_checkpoint_tensor_converter)
        modules, parameters = model_cls._convert_lora_targets_to_parameters(
            None,
            ["q_proj", "gate_proj", "up_proj", "down_proj"],
            [],
        )
        assert modules == ["q_proj"]
        assert parameters == [
            f"{prefix}layers.*.mlp.experts.gate_up_proj",
            f"{prefix}layers.*.mlp.experts.down_proj",
        ]


def test_qwen3_vl_moe_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    # Merged-expert loop ULP (~1e-7); not bitwise vs Hugging Face.
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen3_vl_moe_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    labels = image.pop("labels")
    # Merged-expert loop ULP (~1e-7); not bitwise vs Hugging Face.
    assert_eager_matches_hf(hf, ours, input_ids=ids, labels=labels, fwd_kwargs=image)


def test_qwen3_vl_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())
    pin_eager_attn_implementation(hf)
    pin_eager_attn_implementation(ours)

    input_ids = torch.randint(3, 100, (2, 8))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
