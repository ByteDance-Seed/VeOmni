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

"""Qwen3.5 models_kernel consume tests.

Direct-import the generated classes. Do not register or use
``build_foundation_model``. Compare a toy model against HuggingFace on
full-attention text, linear-attention (GDN) text, and image+text.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated, torch_chunk_gated_delta_rule

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _tiny_text_config(*, layer_types: list[str]) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
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
    )


def _tiny_vl_config(*, layer_types: list[str]) -> Qwen3_5Config:
    text = _tiny_text_config(layer_types=layer_types)
    vision = Qwen3_5VisionConfig(
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
    return Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )


def _qwen3_5_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3_5.generated.patched_modeling_qwen3_5_npu import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )
    else:
        from veomni.models_kernel.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )
    return Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration


def _build_causal(config: Qwen3_5TextConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        causal_cls, _ = _qwen3_5_classes()
        return causal_cls(config)
    finally:
        set_kernels_config(previous)


def _build_vlm(config: Qwen3_5Config, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        _, vlm_cls = _qwen3_5_classes()
        return vlm_cls(config)
    finally:
        set_kernels_config(previous)


def _empty_cu_seq_lens() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int32)


def _pin_hf_gdn_to_torch(model: torch.nn.Module) -> None:
    """Force HF GatedDeltaNet onto the torch path our eager kernels match.

    This environment has ``fla`` but not ``causal_conv1d``. HF then binds FLA
    chunk / fused gated-norm while still using torch conv. Pin all three to
    the torch modules so the toy compare is the HF eager math, not FLA.
    """
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
        if not isinstance(gdn.norm, Qwen3_5RMSNormGated):
            device = gdn.out_proj.weight.device
            replacement = Qwen3_5RMSNormGated(gdn.head_v_dim, eps=gdn.layer_norm_epsilon).to(device)
            replacement.weight.data.copy_(gdn.norm.weight.detach().to(device))
            gdn.norm = replacement


def _image_inputs(config: Qwen3_5Config, input_ids: torch.Tensor) -> dict:
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


def test_qwen3_5_constructs_local_kernels():
    model = _build_causal(_tiny_text_config(layer_types=["linear_attention", "full_attention"]))
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    layer0 = model.model.layers[0]
    assert layer0.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer0.input_layernorm.veomni_rms_norm.variant == "qwen3_5"
    assert layer0.linear_attn.veomni_rms_norm_gated.impl == "eager"
    assert layer0.linear_attn.veomni_causal_conv1d.impl == "eager"
    assert layer0.linear_attn.veomni_chunk_gated_delta_rule.impl == "eager"


def test_qwen3_5_instances_keep_distinct_impls():
    eager = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]))
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen3_5_eager_matches_hf_full_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["full_attention", "full_attention"])
    hf = HFQwen3_5ForCausalLM(config)
    ours = _build_causal(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        fwd_kwargs={"cu_seq_lens_q": torch.tensor([0, 8], dtype=torch.int32)},
    )


def test_qwen3_5_eager_matches_hf_linear_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["linear_attention", "linear_attention"])
    hf = HFQwen3_5ForCausalLM(config)
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


def test_qwen3_5_eager_matches_hf_mixed_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["linear_attention", "linear_attention", "full_attention"])
    hf = HFQwen3_5ForCausalLM(config)
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


def test_qwen3_5_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_vl_config(layer_types=["linear_attention", "linear_attention", "full_attention"])
    hf = HFQwen3_5ForConditionalGeneration(config)
    ours = _build_vlm(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    image = _image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    _pin_hf_gdn_to_torch(hf)
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        fwd_kwargs=image,
        ours_fwd_kwargs={"cu_seq_lens_q": _empty_cu_seq_lens()},
    )
