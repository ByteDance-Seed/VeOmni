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

"""Qwen3.5-MoE models_kernel consume tests.

Direct-import the generated classes. Do not register or use
``build_foundation_model``. Compare a toy model against HuggingFace on
full-attention text, linear-attention (GDN) text, and image+text.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as HFQwen3_5MoeForCausalLM,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeForConditionalGeneration,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNormGated,
    torch_chunk_gated_delta_rule,
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


def _tiny_text_config(*, layer_types: list[str]) -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
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
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attn_implementation="eager",
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        experts_implementation="eager",
    )


def _tiny_vl_config(*, layer_types: list[str]) -> Qwen3_5MoeConfig:
    text = _tiny_text_config(layer_types=layer_types)
    vision = Qwen3_5MoeVisionConfig(
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
        hidden_act="gelu_pytorch_tanh",
    )
    config = Qwen3_5MoeConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )
    # CondGen's generated file dropped ``@use_experts_implementation``, so
    # HF's grouped_mm default raises. Pin eager on both the wrapper and the
    # text tower that owns the experts.
    config._experts_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    return config


def _qwen3_5_moe_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from veomni.models_kernel.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    return Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration


def _build_causal(config: Qwen3_5MoeTextConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        causal_cls, _ = _qwen3_5_moe_classes()
        return causal_cls(config)
    finally:
        set_kernels_config(previous)


def _build_vlm(config: Qwen3_5MoeConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        _, vlm_cls = _qwen3_5_moe_classes()
        return vlm_cls(config)
    finally:
        set_kernels_config(previous)


def _empty_cu_seq_lens() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int32)


def _pin_hf_gdn_to_torch(model: torch.nn.Module) -> None:
    """Force HF GatedDeltaNet onto the torch path our eager kernels match."""
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else []
    language = getattr(getattr(model, "model", None), "language_model", None)
    if language is not None:
        layers = language.layers
    for layer in layers:
        gdn = getattr(layer, "linear_attn", None)
        if gdn is None:
            continue
        gdn.causal_conv1d_fn = None
        gdn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        if not isinstance(gdn.norm, Qwen3_5MoeRMSNormGated):
            device = gdn.out_proj.weight.device
            replacement = Qwen3_5MoeRMSNormGated(gdn.head_v_dim, eps=gdn.layer_norm_epsilon).to(device)
            replacement.weight.data.copy_(gdn.norm.weight.detach().to(device))
            gdn.norm = replacement


def _image_inputs(config: Qwen3_5MoeConfig, input_ids: torch.Tensor) -> dict:
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


def test_qwen3_5_moe_modeling_has_no_opslot_or_ops_import():
    from veomni.models_kernel.transformers.qwen3_5_moe.generated import (
        patched_modeling_qwen3_5_moe_gpu as gpu,
    )
    from veomni.models_kernel.transformers.qwen3_5_moe.generated import (
        patched_modeling_qwen3_5_moe_npu as npu,
    )

    assert_no_ops_or_old_models_import(gpu, npu)


def test_qwen3_5_moe_constructs_local_kernels():
    model = _build_causal(_tiny_text_config(layer_types=["linear_attention", "full_attention"]))
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniKernel)
    assert model.veomni_lb.impl == "eager"
    layer0 = model.model.layers[0]
    assert layer0.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer0.input_layernorm.veomni_rms_norm.variant == "qwen3_5"
    assert layer0.linear_attn.veomni_rms_norm_gated.impl == "eager"
    assert layer0.mlp.experts.veomni_moe.impl == "eager"
    assert layer0.mlp.experts.veomni_moe.kernel == "moe_experts"


def test_qwen3_5_moe_instances_keep_distinct_impls():
    eager = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]))
    fused_cfg = eager_kernels_config()
    fused_cfg.moe_implementation = "fused_triton"
    fused = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]), fused_cfg)

    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"
    assert fused.model.layers[0].mlp.experts.veomni_moe.impl == "triton"

    set_kernels_config(fused_cfg)
    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"


def test_qwen3_5_moe_eager_matches_hf_full_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["full_attention", "full_attention"])
    hf = HFQwen3_5MoeForCausalLM(config)
    ours = _build_causal(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        fwd_kwargs={"cu_seq_lens_q": torch.tensor([0, 8], dtype=torch.int32)},
    )


def test_qwen3_5_moe_eager_matches_hf_linear_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["linear_attention", "linear_attention"])
    hf = HFQwen3_5MoeForCausalLM(config)
    ours = _build_causal(config)
    ours.load_state_dict(hf.state_dict())

    _pin_hf_gdn_to_torch(hf)
    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs={"cu_seq_lens_q": _empty_cu_seq_lens()},
    )


def test_qwen3_5_moe_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_vl_config(layer_types=["full_attention", "full_attention"])
    hf = HFQwen3_5MoeForConditionalGeneration(config)
    ours = _build_vlm(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        fwd_kwargs=image,
        ours_fwd_kwargs={"cu_seq_lens_q": _empty_cu_seq_lens()},
    )


def test_qwen3_5_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_vl_config(layer_types=["full_attention", "full_attention"])
    hf = HFQwen3_5MoeForConditionalGeneration(config)
    ours = _build_vlm(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    labels = input_ids.clone()
    extra = {"output_router_logits": True, "cu_seq_lens_q": _empty_cu_seq_lens()}
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, **extra)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, **extra)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
