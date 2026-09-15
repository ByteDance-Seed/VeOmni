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

"""Qwen3-MoE registry hooks, operator selection, and task-head parity tests.

Direct-import the generated classes. Compare a toy model against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForQuestionAnswering as HFQwen3MoeForQuestionAnswering,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForTokenClassification as HFQwen3MoeForTokenClassification,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel as HFQwen3MoeModel

from tests.models.compare import (
    assert_eager_matches_hf,
    assert_sequence_classification_matches_hf,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_qwen3_moe_config as _tiny_config


def _qwen3_moe_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForSequenceClassification,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )
    else:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForSequenceClassification,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )
    return (
        Qwen3MoeForCausalLM,
        Qwen3MoeForSequenceClassification,
        Qwen3MoeForTokenClassification,
        Qwen3MoeForQuestionAnswering,
        Qwen3MoeModel,
    )


def _construct_ours(model_cls, config: Qwen3MoeConfig, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return model_cls(config)


def _build_ours(config: Qwen3MoeConfig, ops: SimpleNamespace | None = None):
    causal_cls, *_ = _qwen3_moe_classes()
    return _construct_ours(causal_cls, config, ops)


def test_qwen3_moe_rope_reads_selected_impl(monkeypatch):
    from veomni.models.transformers.qwen3_moe.generated import patched_modeling_qwen3_moe_gpu as modeling

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


def test_qwen3_moe_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3MoeForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen3_moe_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3MoeModel(config)
    *_, model_cls = _qwen3_moe_classes()
    ours = _construct_ours(model_cls, config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False)
    ours_out = ours(input_ids=input_ids, use_cache=False)
    torch.testing.assert_close(ours_out.last_hidden_state, hf_out.last_hidden_state)


@pytest.mark.parametrize("supervision", ["last-valid", "selected-tokens"])
def test_qwen3_moe_sequence_classification_matches_hf(supervision):
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    _, model_cls, *_ = _qwen3_moe_classes()
    model = _construct_ours(model_cls, config)
    assert_sequence_classification_matches_hf(model, supervision=supervision)


def test_qwen3_moe_token_classification_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFQwen3MoeForTokenClassification(config)
    _, _, model_cls, *_ = _qwen3_moe_classes()
    ours = _construct_ours(model_cls, config)
    ours.load_state_dict(hf.state_dict())
    hf.eval()
    ours.eval()

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.randint(0, config.num_labels, input_ids.shape)
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.logits, hf_out.logits)
    torch.testing.assert_close(ours_out.loss, hf_out.loss)


def test_qwen3_moe_question_answering_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3MoeForQuestionAnswering(config)
    _, _, _, model_cls, _ = _qwen3_moe_classes()
    ours = _construct_ours(model_cls, config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    start_positions = torch.tensor([1, 2])
    end_positions = torch.tensor([3, 4])
    hf_out = hf(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions, use_cache=False)
    ours_out = ours(
        input_ids=input_ids,
        start_positions=start_positions,
        end_positions=end_positions,
        use_cache=False,
    )
    torch.testing.assert_close(ours_out.start_logits, hf_out.start_logits)
    torch.testing.assert_close(ours_out.end_logits, hf_out.end_logits)
    torch.testing.assert_close(ours_out.loss, hf_out.loss)


@pytest.mark.parametrize(
    ("architecture", "target_prefix"),
    [
        ("Qwen3MoeForCausalLM", "model"),
        ("Qwen3MoeForSequenceClassification", "model"),
        ("Qwen3MoeForTokenClassification", "model"),
        ("Qwen3MoeForQuestionAnswering", "transformer"),
        ("Qwen3MoeModel", ""),
    ],
)
def test_qwen3_moe_registry_installs_checkpoint_and_lora_hooks(architecture: str, target_prefix: str):
    from veomni.models import get_model_class

    model_cls = get_model_class(_tiny_config(architectures=[architecture]))
    assert callable(model_cls._create_checkpoint_tensor_converter)
    assert callable(model_cls._convert_fqn_to_index_mapping)

    modules, parameters = model_cls._convert_lora_targets_to_parameters(
        None,
        ["q_proj", "gate_proj", "up_proj", "down_proj"],
        [],
    )
    prefix = f"{target_prefix}." if target_prefix else ""
    assert modules == ["q_proj"]
    assert parameters == [
        f"{prefix}layers.*.mlp.experts.gate_up_proj",
        f"{prefix}layers.*.mlp.experts.down_proj",
    ]


def test_qwen3_moe_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3MoeForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False, output_router_logits=True)
    assert ours_out.aux_loss is not None
    assert hf_out.aux_loss is not None
    torch.testing.assert_close(ours_out.aux_loss, hf_out.aux_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=1e-6, rtol=1e-6)
