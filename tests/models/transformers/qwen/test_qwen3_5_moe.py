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

"""Qwen3.5-MoE parallel-plan, auxiliary-loss, and multimodal parity tests.

Direct-import the generated classes. Compare a toy model against Hugging Face on
full-attention text, linear-attention (GDN) text, and image+text.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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
    ops_config_scope,
    pin_eager_attn_implementation,
    qwen_image_inputs,
    stamp_attn_implementation,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_config as _tiny_vl_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_text_config as _tiny_text_config,
)


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
    ops = ops if ops is not None else eager_ops_config()
    stamp_attn_implementation(config, ops.attn_implementation)
    with ops_config_scope(ops):
        causal_cls, _ = _qwen3_5_moe_classes()
        return causal_cls(config)


def _build_vlm(config: Qwen3_5MoeConfig, ops: SimpleNamespace | None = None):
    ops = ops if ops is not None else eager_ops_config()
    stamp_attn_implementation(config, ops.attn_implementation)
    with ops_config_scope(ops):
        _, vlm_cls = _qwen3_5_moe_classes()
        return vlm_cls(config)


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
    # Merged-expert loop ULP (~1e-7); not bitwise vs Hugging Face.
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
    assert ours.model.layers[0].input_layernorm.veomni_rms_norm.variant == "qwen3_5"
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


def test_qwen3_5_moe_sdpa_full_attention_accepts_single_sequence_cu_seq_lens_q():
    ops = eager_ops_config()
    ops.attn_implementation = "sdpa"
    config = _tiny_text_config(layer_types=["full_attention", "full_attention"])
    ours = _build_causal(config, ops).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 32))
    with ops_config_scope(ops), torch.no_grad():
        output = ours(
            input_ids=input_ids,
            use_cache=False,
            cu_seq_lens_q=torch.tensor([0, 32], dtype=torch.int32),
        )
    assert torch.isfinite(output.logits).all()


def _sdpa_simple_ops():
    ops = eager_ops_config()
    ops.attn_implementation = "sdpa"
    return ops


def _sdpa_public_ops():
    from tests.tools.training_utils import make_eager_ops_config

    return make_eager_ops_config(attn_implementation="sdpa")


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: _sdpa_simple_ops(),
        lambda: _sdpa_public_ops(),
    ],
    ids=["sdpa", "veomni_sdpa"],
)
def test_qwen3_5_moe_sdpa_packed_sequences_are_isolated(ops_factory):
    torch.manual_seed(0)
    ops = ops_factory()
    config = _tiny_text_config(layer_types=["full_attention", "full_attention"])
    model = _build_causal(config, ops).eval()
    first_input_ids = torch.tensor([[5, 6, 7, 8]])
    second_input_ids = torch.tensor([[9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)
    packed_position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    packed_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    with ops_config_scope(ops), torch.no_grad():
        packed_logits = model(
            input_ids=packed_input_ids,
            attention_mask=torch.ones_like(packed_input_ids),
            position_ids=packed_position_ids,
            cu_seq_lens_q=packed_cu,
            use_cache=False,
        ).logits
        first_logits = model(
            input_ids=first_input_ids,
            attention_mask=torch.ones_like(first_input_ids),
            position_ids=torch.arange(4).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 4], dtype=torch.int32),
            use_cache=False,
        ).logits
        second_logits = model(
            input_ids=second_input_ids,
            attention_mask=torch.ones_like(second_input_ids),
            position_ids=torch.arange(4).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 4], dtype=torch.int32),
            use_cache=False,
        ).logits
    torch.testing.assert_close(packed_logits[:, :4], first_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(packed_logits[:, 4:], second_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: _sdpa_simple_ops(),
        lambda: _sdpa_public_ops(),
    ],
    ids=["sdpa", "veomni_sdpa"],
)
def test_qwen3_5_moe_sdpa_editing_first_packed_sample_does_not_move_second(ops_factory):
    torch.manual_seed(0)
    ops = ops_factory()
    config = _tiny_text_config(layer_types=["full_attention", "full_attention"])
    model = _build_causal(config, ops).eval()
    packed_input_ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    edited_input_ids = packed_input_ids.clone()
    edited_input_ids[0, 0] = 20
    packed_position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    packed_cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    attention_mask = torch.ones_like(packed_input_ids)
    with ops_config_scope(ops), torch.no_grad():
        base_logits = model(
            input_ids=packed_input_ids,
            attention_mask=attention_mask,
            position_ids=packed_position_ids,
            cu_seq_lens_q=packed_cu,
            use_cache=False,
        ).logits
        edited_logits = model(
            input_ids=edited_input_ids,
            attention_mask=attention_mask,
            position_ids=packed_position_ids,
            cu_seq_lens_q=packed_cu,
            use_cache=False,
        ).logits
    torch.testing.assert_close(base_logits[:, 4:], edited_logits[:, 4:], rtol=1e-5, atol=1e-5)
    assert not torch.equal(base_logits[:, :4], edited_logits[:, :4])
