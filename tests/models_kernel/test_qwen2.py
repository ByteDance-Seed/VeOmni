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

"""Qwen2 models_kernel consume tests.

Direct-import the generated class. Compare a toy model against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as HFQwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForQuestionAnswering as HFQwen2ForQuestionAnswering,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForTokenClassification as HFQwen2ForTokenClassification,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2Model

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from tests.models_kernel.tiny_configs import tiny_qwen2_config as _tiny_config
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _qwen2_classes():
    from veomni.models_kernel.transformers.qwen2.generated.patched_modeling_qwen2_gpu import (
        Qwen2ForCausalLM,
        Qwen2ForQuestionAnswering,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
        Qwen2Model,
    )

    return (
        Qwen2ForCausalLM,
        Qwen2ForSequenceClassification,
        Qwen2ForTokenClassification,
        Qwen2ForQuestionAnswering,
        Qwen2Model,
    )


def _construct_ours(model_cls, config: Qwen2Config, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        return model_cls(config)
    finally:
        set_ops_config(previous)


def _build_ours(config: Qwen2Config, ops: SimpleNamespace | None = None):
    causal_cls, *_ = _qwen2_classes()
    return _construct_ours(causal_cls, config, ops)


def test_qwen2_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"


def test_qwen2_rope_reads_selected_impl(monkeypatch):
    from veomni.models_kernel.transformers.qwen2.generated import patched_modeling_qwen2_gpu as modeling

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


def test_qwen2_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen2_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_qwen2_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2Model(config)
    *_, model_cls = _qwen2_classes()
    ours = _construct_ours(model_cls, config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False)
    ours_out = ours(input_ids=input_ids, use_cache=False)
    torch.testing.assert_close(ours_out.last_hidden_state, hf_out.last_hidden_state)


def test_qwen2_sequence_classification_forward():
    config = _tiny_config(num_labels=4)
    _, model_cls, *_ = _qwen2_classes()
    model = _construct_ours(model_cls, config)
    assert model.veomni_ce.impl == "eager"

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.full(input_ids.shape, -100, dtype=torch.long)
    labels[:, -1] = torch.tensor([1, 2])
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.logits is not None


def test_qwen2_token_classification_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFQwen2ForTokenClassification(config)
    _, _, model_cls, *_ = _qwen2_classes()
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


def test_qwen2_question_answering_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2ForQuestionAnswering(config)
    _, _, _, model_cls, _ = _qwen2_classes()
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
