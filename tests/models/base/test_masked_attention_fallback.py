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

"""Masked attention binds SDPA when the requested impl has no mask path."""

from __future__ import annotations

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_flux_config, tiny_qwen_image_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _flux_joint_attention(ops):
    from veomni.models.transformers.flux.modeling_flux import FluxJointAttention

    cfg = tiny_flux_config()
    cfg._set_ops_implementation(ops)
    return FluxJointAttention(32, 32, 4, 8, config=cfg)


def _qwen_image_processor(ops):
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageSPAttnProcessor,
    )

    cfg = tiny_qwen_image_config()
    cfg._set_ops_implementation(ops)
    return QwenImageSPAttnProcessor(cfg)


@pytest.mark.parametrize(
    "build",
    (
        pytest.param(_flux_joint_attention, id="flux"),
        pytest.param(_qwen_image_processor, id="qwen_image"),
    ),
)
def test_masked_attention_falls_back_to_veomni_sdpa(available_nvidia_ops, build):
    ops = eager_ops_config()
    ops.attn_implementation = "flash_attention_2"
    module = build(ops)

    assert module.veomni_attn.impl == "flash_attention_2"
    assert module.veomni_attn_masked.impl == "veomni_sdpa"
    assert module.veomni_attn_masked is not module.veomni_attn
