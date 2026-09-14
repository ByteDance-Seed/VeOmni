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

"""Qwen3.5-MoE models consume tests.

Direct-import the generated classes. Compare a toy model against HuggingFace on
full-attention text, linear-attention (GDN) text, and image+text.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
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

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    pin_eager_attn_implementation,
    qwen_image_inputs,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_config as _tiny_vl_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_text_config as _tiny_text_config,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _qwen3_5_moe_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    return Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration


def _build_causal(config: Qwen3_5MoeTextConfig, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        causal_cls, _ = _qwen3_5_moe_classes()
        return causal_cls(config)
    finally:
        set_ops_config(previous)


def _build_vlm(config: Qwen3_5MoeConfig, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        _, vlm_cls = _qwen3_5_moe_classes()
        return vlm_cls(config)
    finally:
        set_ops_config(previous)


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


def test_qwen3_5_moe_constructs_local_kernels():
    model = _build_causal(_tiny_text_config(layer_types=["linear_attention", "full_attention"]))
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniOp)
    assert model.veomni_lb.impl == "eager"
    layer0 = model.model.layers[0]
    assert layer0.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer0.input_layernorm.veomni_rms_norm.variant == "qwen3_5"
    assert layer0.linear_attn.veomni_rms_norm_gated.impl == "eager"
    assert layer0.mlp.experts.veomni_moe.impl == "eager"
    assert layer0.mlp.experts.veomni_moe.op == "moe_experts"


def test_qwen3_5_moe_instances_keep_distinct_impls():
    eager = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]))
    fused_cfg = eager_ops_config()
    fused_cfg.moe_implementation = "fused_triton"
    fused = _build_causal(_tiny_text_config(layer_types=["full_attention", "full_attention"]), fused_cfg)

    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"
    assert fused.model.layers[0].mlp.experts.veomni_moe.impl == "fused_triton"

    set_ops_config(fused_cfg)
    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"


def test_qwen3_5_moe_parallel_plans_cover_multimodal_and_text_wrappers():
    causal_cls, conditional_cls = _qwen3_5_moe_classes()

    conditional_ep_plan = conditional_cls.get_parallel_plan(None).extra_parallel_plan["ep"]
    causal_ep_plan = causal_cls.get_parallel_plan(None).extra_parallel_plan["ep"]

    assert set(conditional_ep_plan) == {
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
    }
    assert set(causal_ep_plan) == {
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    }


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


def test_qwen3_5_moe_eager_matches_hf_mixed_attention():
    torch.manual_seed(0)
    config = _tiny_text_config(layer_types=["linear_attention", "linear_attention", "full_attention"])
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
    config = _tiny_vl_config(layer_types=["linear_attention", "linear_attention", "full_attention"])
    hf = HFQwen3_5MoeForConditionalGeneration(config)
    ours = _build_vlm(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    labels = image.pop("labels")
    _pin_hf_gdn_to_torch(hf)
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        labels=labels,
        fwd_kwargs=image,
        ours_fwd_kwargs={"cu_seq_lens_q": _empty_cu_seq_lens()},
    )


def test_qwen3_5_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_vl_config(layer_types=["linear_attention", "linear_attention", "full_attention"])
    hf = HFQwen3_5MoeForConditionalGeneration(config)
    ours = _build_vlm(config)
    ours.load_state_dict(hf.state_dict())

    _pin_hf_gdn_to_torch(hf)
    pin_eager_attn_implementation(hf)
    pin_eager_attn_implementation(ours)
    input_ids = torch.randint(3, 100, (2, 8))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        output_router_logits=True,
        cu_seq_lens_q=_empty_cu_seq_lens(),
    )
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
