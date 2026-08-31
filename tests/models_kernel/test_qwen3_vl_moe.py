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

"""Qwen3-VL-MoE models_kernel consume tests.

Direct-import the generated classes. Do not register or use
``build_foundation_model``. Compare a toy model against HuggingFace on
both the text-only and image+text paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration,
)

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    assert_no_ops_or_old_models_import,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _tiny_config() -> Qwen3VLMoeConfig:
    text = Qwen3VLMoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_scaling={"mrope_interleaved": True, "mrope_section": [8, 4, 4], "rope_type": "default"},
        tie_word_embeddings=False,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        _experts_implementation="eager",
    )
    vision = Qwen3VLMoeVisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh",
    )
    return Qwen3VLMoeConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )


def _qwen3_vl_moe_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_npu import (
            Qwen3VLMoeForConditionalGeneration,
        )
    else:
        from veomni.models_kernel.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeForConditionalGeneration,
        )
    return Qwen3VLMoeForConditionalGeneration


def _build_ours(config: Qwen3VLMoeConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return _qwen3_vl_moe_cls()(config)
    finally:
        set_kernels_config(previous)


def _image_inputs(config: Qwen3VLMoeConfig, input_ids: torch.Tensor) -> dict:
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


def test_qwen3_vl_moe_modeling_has_no_opslot_or_ops_import():
    from veomni.models_kernel.transformers.qwen3_vl_moe.generated import (
        patched_modeling_qwen3_vl_moe_gpu as gpu,
    )
    from veomni.models_kernel.transformers.qwen3_vl_moe.generated import (
        patched_modeling_qwen3_vl_moe_npu as npu,
    )

    assert_no_ops_or_old_models_import(gpu, npu)


def test_qwen3_vl_moe_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniKernel)
    assert model.veomni_lb.impl == "eager"
    layer = model.model.language_model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.experts.veomni_moe.impl == "eager"


def test_qwen3_vl_moe_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    fused_cfg = eager_kernels_config()
    fused_cfg.moe_implementation = "fused_triton"
    fused = _build_ours(_tiny_config(), fused_cfg)

    assert eager.model.language_model.layers[0].mlp.experts.veomni_moe.impl == "eager"
    assert fused.model.language_model.layers[0].mlp.experts.veomni_moe.impl == "triton"

    set_kernels_config(fused_cfg)
    assert eager.model.language_model.layers[0].mlp.experts.veomni_moe.impl == "eager"


def test_qwen3_vl_moe_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen3_vl_moe_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    assert_eager_matches_hf(hf, ours, input_ids=ids, fwd_kwargs=image)


def test_qwen3_vl_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3VLMoeForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
