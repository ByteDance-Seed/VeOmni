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

"""Qwen3-MoE models_kernel consume tests.

Direct-import the generated classes. Compare a toy model against HuggingFace.
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

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _tiny_config(**overrides) -> Qwen3MoeConfig:
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "attn_implementation": "eager",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "experts_implementation": "eager",
    }
    kwargs.update(overrides)
    return Qwen3MoeConfig(**kwargs)


def _qwen3_moe_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForSequenceClassification,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )
    else:
        from veomni.models_kernel.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (
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
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return model_cls(config)
    finally:
        set_ops_config(previous)


def _build_ours(config: Qwen3MoeConfig, ops: SimpleNamespace | None = None):
    causal_cls, *_ = _qwen3_moe_classes()
    return _construct_ours(causal_cls, config, ops)


def test_qwen3_moe_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniOp)
    assert model.veomni_lb.impl == "eager"
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.experts.veomni_moe.impl == "eager"
    assert layer.mlp.experts.veomni_moe.op == "moe_experts"


def test_qwen3_moe_rope_reads_selected_impl(monkeypatch):
    from veomni.models_kernel.transformers.qwen3_moe.generated import patched_modeling_qwen3_moe_gpu as modeling

    selected: list[tuple[str, str, str]] = []

    class StubOp:
        def __init__(self, op: str, variant: str, impl: str):
            selected.append((op, variant, impl))

        def __call__(self, q, k, *_args, **_kwargs):
            return q, k

    monkeypatch.setattr(modeling, "VeomniOp", StubOp)
    ops = eager_ops_config()
    ops.rotary_pos_emb_implementation = "test_impl"
    previous = get_ops_config()
    set_ops_config(ops)
    try:
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn_like(q)
        cos = torch.randn(1, 4, 8)
        sin = torch.randn_like(cos)
        q_out, k_out = modeling.apply_rotary_pos_emb(q, k, cos, sin)
    finally:
        set_ops_config(previous)

    assert selected == [("rope", "full", "test_impl")]
    assert q_out is q
    assert k_out is k


def test_qwen3_moe_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    fused_cfg = eager_ops_config()
    fused_cfg.moe_implementation = "fused_triton"
    fused = _build_ours(_tiny_config(), fused_cfg)

    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"
    assert fused.model.layers[0].mlp.experts.veomni_moe.impl == "fused_triton"

    set_ops_config(fused_cfg)
    assert eager.model.layers[0].mlp.experts.veomni_moe.impl == "eager"


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


def test_qwen3_moe_sequence_classification_forward():
    config = _tiny_config(num_labels=4)
    _, model_cls, *_ = _qwen3_moe_classes()
    model = _construct_ours(model_cls, config)
    assert "models_kernel" in model.model.__class__.__module__
    assert model.veomni_ce.impl == "eager"

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.full(input_ids.shape, -100, dtype=torch.long)
    labels[:, -1] = torch.tensor([1, 2])
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.logits is not None


def test_qwen3_moe_token_classification_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFQwen3MoeForTokenClassification(config)
    _, _, model_cls, *_ = _qwen3_moe_classes()
    ours = _construct_ours(model_cls, config)
    assert "models_kernel" in ours.model.__class__.__module__
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
    assert "models_kernel" in ours.transformer.__class__.__module__
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
    from veomni.models_kernel import get_model_class

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
