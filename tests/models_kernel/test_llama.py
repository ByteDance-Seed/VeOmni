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

"""Llama models_kernel consume tests.

Direct-import the generated classes. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForTokenClassification as HFLlamaForTokenClassification
from transformers.models.llama.modeling_llama import LlamaModel as HFLlamaModel

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config, set_ops_config


def _tiny_config(**overrides) -> LlamaConfig:
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "attention_bias": False,
        "mlp_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def _llama_classes():
    from veomni.models_kernel.transformers.llama.generated.patched_modeling_llama_gpu import (
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaForTokenClassification,
        LlamaModel,
    )

    return LlamaForCausalLM, LlamaForSequenceClassification, LlamaForTokenClassification, LlamaModel


def _build_ours(config: LlamaConfig, ops: SimpleNamespace | None = None):
    previous = get_ops_config()
    set_ops_config(ops if ops is not None else eager_ops_config())
    try:
        causal_cls, _, _, _ = _llama_classes()
        return causal_cls(config)
    finally:
        set_ops_config(previous)


def test_llama_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"


def test_llama_rope_reads_selected_impl(monkeypatch):
    from veomni.models_kernel.transformers.llama.generated import patched_modeling_llama_gpu as modeling

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


def test_llama_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_ops_config())
    chunk_cfg = eager_ops_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_ops_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


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
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        *_, model_cls = _llama_classes()
        ours = model_cls(config)
    finally:
        set_ops_config(previous)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_out = hf(input_ids=input_ids, use_cache=False)
    ours_out = ours(input_ids=input_ids, use_cache=False)
    torch.testing.assert_close(ours_out.last_hidden_state, hf_out.last_hidden_state)


def test_llama_seq_cls_forward():
    _, seq_cls, _, _ = _llama_classes()
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        model = seq_cls(_tiny_config(num_labels=4))
    finally:
        set_ops_config(previous)
    assert model.veomni_ce.impl == "eager"

    input_ids = torch.randint(3, 128, (2, 6))
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, -1] = torch.tensor([1, 2])
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.logits is not None


def test_llama_token_cls_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config(num_labels=4)
    hf = HFLlamaForTokenClassification(config)
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        _, _, token_cls, _ = _llama_classes()
        ours = token_cls(config)
    finally:
        set_ops_config(previous)
    ours.load_state_dict(hf.state_dict())
    hf.eval()
    ours.eval()

    input_ids = torch.randint(3, config.vocab_size, (2, 6))
    labels = torch.randint(0, config.num_labels, input_ids.shape)
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.logits, hf_out.logits)
    torch.testing.assert_close(ours_out.loss, hf_out.loss)
