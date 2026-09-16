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

"""GPT-OSS sliding-attention, auxiliary-loss, and Hugging Face parity tests.

Direct-import the generated class. Compare a toy CausalLM against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFGptOssForCausalLM

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    named_trainable,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_gpt_oss_config as _tiny_config
from tests.ops.tol import (
    MOE_FUSED_SWIGLU_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC1_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC1_RTOL,
    MOE_FUSED_SWIGLU_GRAD_FC2_ATOL,
    MOE_FUSED_SWIGLU_GRAD_FC2_RTOL,
    MOE_FUSED_SWIGLU_RTOL,
)
from veomni.utils.import_utils import is_quack_gemm_available


def _build_ours(config: GptOssConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.gpt_oss.generated.patched_modeling_gpt_oss_gpu import (
        GptOssForCausalLM,
    )

    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return GptOssForCausalLM(config)


@pytest.mark.parametrize(
    "seq_len,partial_labels", [(7, False), (17, True)], ids=["within-window", "padded-beyond-window"]
)
def test_gpt_oss_eager_matches_hf(seq_len, partial_labels):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGptOssForCausalLM(config)
    ours = _build_ours(config)
    assert ours.model.layers[0].mlp.experts.veomni_moe.variant == "gpt_oss"
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    if partial_labels:
        attention_mask[1, :3] = 0
        input_ids[1, :3] = config.pad_token_id
        labels[:, :4] = -100  # Prompt tokens are visible but not supervised.
        labels[0, 8:10] = -100
    assert_eager_matches_hf(
        hf, ours, input_ids=input_ids, labels=labels, fwd_kwargs={"attention_mask": attention_mask}, logits_equal=True
    )


def test_gpt_oss_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGptOssForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 17))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not is_quack_gemm_available(), reason="GPT-OSS fused_quack needs SM90+ quack")
def test_gpt_oss_fused_quack_matches_eager():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    config = _tiny_config()
    eager_ops = eager_ops_config()
    fused_ops = eager_ops_config()
    fused_ops.moe_implementation = "fused_quack"
    eager = _build_ours(config, eager_ops).to(device=device, dtype=dtype)
    fused = _build_ours(config, fused_ops).to(device=device, dtype=dtype)
    fused.load_state_dict(eager.state_dict())
    assert fused.model.layers[0].mlp.experts.veomni_moe.impl == "fused_quack"

    input_ids = torch.randint(3, config.vocab_size, (2, 8), device=device)
    labels = input_ids.clone()
    eager.train()
    fused.train()
    eager_logits = eager(input_ids=input_ids, use_cache=False).logits
    fused_logits = fused(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(
        fused_logits.float(),
        eager_logits.float(),
        atol=MOE_FUSED_SWIGLU_ATOL,
        rtol=MOE_FUSED_SWIGLU_RTOL,
    )

    eager_out = eager(input_ids=input_ids, labels=labels, use_cache=False)
    fused_out = fused(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(
        fused_out.loss.float(),
        eager_out.loss.float(),
        atol=MOE_FUSED_SWIGLU_ATOL,
        rtol=MOE_FUSED_SWIGLU_RTOL,
    )
    eager_out.loss.backward()
    fused_out.loss.backward()
    eager_grads = named_trainable(eager)
    fused_grads = named_trainable(fused)
    for name, param in eager_grads.items():
        if "experts" not in name or param.grad is None:
            continue
        fused_grad = fused_grads[name].grad
        assert fused_grad is not None, name
        if "gate_up" in name:
            atol, rtol = MOE_FUSED_SWIGLU_GRAD_FC1_ATOL, MOE_FUSED_SWIGLU_GRAD_FC1_RTOL
        else:
            atol, rtol = MOE_FUSED_SWIGLU_GRAD_FC2_ATOL, MOE_FUSED_SWIGLU_GRAD_FC2_RTOL
        torch.testing.assert_close(fused_grad.float(), param.grad.float(), atol=atol, rtol=rtol, msg=name)
