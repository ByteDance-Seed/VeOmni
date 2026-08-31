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

"""Qwen3-Omni-MoE models_kernel consume tests.

Direct-import the generated Thinker class. Do not register or use
``build_foundation_model``. Compare a toy Thinker against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration as HFQwen3OmniMoeThinker,
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
AUDIO_TOKEN_ID = 122


def _tiny_thinker_config() -> Qwen3OmniMoeThinkerConfig:
    text = Qwen3OmniMoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    )
    text._experts_implementation = "eager"
    vision = Qwen3OmniMoeVisionEncoderConfig(
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
    audio = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=64,
        downsample_hidden_size=16,
        n_window=4,
        max_source_positions=16,
    )
    return Qwen3OmniMoeThinkerConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        audio_config=audio.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
    )


def _qwen3_omni_moe_thinker_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_npu import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    else:
        from veomni.models_kernel.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_gpu import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    return Qwen3OmniMoeThinkerForConditionalGeneration


def _build_ours(config: Qwen3OmniMoeThinkerConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return _qwen3_omni_moe_thinker_cls()(config)
    finally:
        set_kernels_config(previous)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
        "audio_mask": zeros,
    }


def _image_inputs(config: Qwen3OmniMoeThinkerConfig, input_ids: torch.Tensor) -> dict:
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


def test_qwen3_omni_moe_modeling_has_no_opslot_or_ops_import():
    from pathlib import Path

    generated = Path("veomni/models_kernel/transformers/qwen3_omni_moe/generated")
    gpu = SimpleNamespace(__file__=str(generated / "patched_modeling_qwen3_omni_moe_gpu.py"))
    # NPU generated file imports torch_npu at module level. Read it as text
    # so GPU hosts can still check the consume boundary.
    npu = SimpleNamespace(__file__=str(generated / "patched_modeling_qwen3_omni_moe_npu.py"))
    assert_no_ops_or_old_models_import(gpu, npu)


def test_qwen3_omni_moe_constructs_local_kernels():
    model = _build_ours(_tiny_thinker_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniKernel)
    assert model.veomni_lb.impl == "eager"
    layer = model.model.layers[0]
    assert layer.mlp.experts.veomni_moe.impl == "eager"
    assert layer.mlp.experts.veomni_moe.kernel == "moe_experts"


def test_qwen3_omni_moe_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_thinker_config(), eager_kernels_config())
    fused_cfg = eager_kernels_config()
    fused_cfg.moe_implementation = "fused_triton"
    fused = _build_ours(_tiny_thinker_config(), fused_cfg)

    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"
    assert fused.model.layers[0].mlp.experts.veomni_moe.impl == "triton"

    set_kernels_config(fused_cfg)
    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"


def test_qwen3_omni_moe_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen3OmniMoeThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs=_mask_kwargs(input_ids),
    )


def test_qwen3_omni_moe_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen3OmniMoeThinker(config)
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


def test_qwen3_omni_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_thinker_config()
    hf = HFQwen3OmniMoeThinker(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    labels = input_ids.clone()
    masks = _mask_kwargs(input_ids)
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        output_router_logits=True,
        **masks,
    )
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
