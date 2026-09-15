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
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_gpt_oss_config as _tiny_config


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
        hf, ours, input_ids=input_ids, labels=labels, fwd_kwargs={"attention_mask": attention_mask}
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
