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

"""Llama models consume tests.

Direct-import the generated classes. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForTokenClassification as HFLlamaForTokenClassification
from transformers.models.llama.modeling_llama import LlamaModel as HFLlamaModel

from tests.models.compare import (
    assert_eager_matches_hf,
    assert_sequence_classification_matches_hf,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_llama_config as _tiny_config


def _llama_classes():
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import (
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaForTokenClassification,
        LlamaModel,
    )

    return LlamaForCausalLM, LlamaForSequenceClassification, LlamaForTokenClassification, LlamaModel


def _build_ours(config: LlamaConfig, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        causal_cls, _, _, _ = _llama_classes()
        return causal_cls(config)


def test_llama_rope_reads_selected_impl(monkeypatch):
    from veomni.models.transformers.llama.generated import patched_modeling_llama_gpu as modeling

    selected: list[tuple[str, str, str]] = []

    class StubOp:
        def __init__(self, op: str, variant: str, impl: str):
            selected.append((op, variant, impl))

        def __call__(self, q, k, *_args, **_kwargs):
            return q, k

    monkeypatch.setattr(modeling, "VeomniOp", StubOp)
    ops = eager_ops_config()
    ops.rotary_pos_emb_implementation = "test_impl"
    with ops_config_scope(ops):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn_like(q)
        cos = torch.randn(1, 4, 8)
        sin = torch.randn_like(cos)
        q_out, k_out = modeling.apply_rotary_pos_emb(q, k, cos, sin)

    assert selected == [("rope", "full", "test_impl")]
    assert q_out is q
    assert k_out is k


def test_llama_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFLlamaForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_llama_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFLlamaModel(config)
    with ops_config_scope(eager_ops_config()):
        *_, model_cls = _llama_classes()
        ours = model_cls(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False)
    ours_out = ours(input_ids=input_ids, use_cache=False)
    torch.testing.assert_close(ours_out.last_hidden_state, hf_out.last_hidden_state)


@pytest.mark.parametrize("supervision", ["last-valid", "selected-tokens"])
def test_llama_seq_cls_matches_hf(supervision):
    torch.manual_seed(0)
    _, seq_cls, _, _ = _llama_classes()
    with ops_config_scope(eager_ops_config()):
        model = seq_cls(_tiny_config(num_labels=4))
    assert_sequence_classification_matches_hf(model, supervision=supervision)


def test_llama_token_cls_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFLlamaForTokenClassification(config)
    with ops_config_scope(eager_ops_config()):
        _, _, token_cls, _ = _llama_classes()
        ours = token_cls(config)
    ours.load_state_dict(hf.state_dict())
    hf.eval()
    ours.eval()

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.randint(0, config.num_labels, input_ids.shape)
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.logits, hf_out.logits)
    torch.testing.assert_close(ours_out.loss, hf_out.loss)
