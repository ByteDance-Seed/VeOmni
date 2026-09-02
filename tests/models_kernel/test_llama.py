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

Direct-import the generated classes. Do not register or use
``build_foundation_model``. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


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
    )

    return LlamaForCausalLM, LlamaForSequenceClassification


def _build_ours(config: LlamaConfig, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        causal_cls, _ = _llama_classes()
        return causal_cls(config)
    finally:
        set_kernels_config(previous)


def test_llama_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"


def test_llama_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_llama_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFLlamaForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_llama_seq_cls_forward():
    _, seq_cls = _llama_classes()
    previous = get_kernels_config()
    set_kernels_config(eager_kernels_config())
    try:
        model = seq_cls(_tiny_config(num_labels=4))
    finally:
        set_kernels_config(previous)
    assert model.veomni_ce.impl == "eager"

    input_ids = torch.randint(3, 128, (2, 6))
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[:, -1] = torch.tensor([1, 2])
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.logits is not None
