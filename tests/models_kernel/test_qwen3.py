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

"""Qwen3 models_kernel consume tests.

Direct-import the generated classes. Do not register or use
``build_foundation_model``. Compare a toy model against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _eager_kernels_config() -> SimpleNamespace:
    return SimpleNamespace(
        attn_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
    )


def _tiny_config(**overrides) -> Qwen3Config:
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
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "attn_implementation": "eager",
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def _qwen3_classes():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.qwen3.generated.patched_modeling_qwen3_npu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
        )
    else:
        from veomni.models_kernel.transformers.qwen3.generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
        )
    return Qwen3ForCausalLM, Qwen3ForSequenceClassification


def _build_qwen3(config: Qwen3Config, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else _eager_kernels_config())
    try:
        causal_cls, _ = _qwen3_classes()
        return causal_cls(config)
    finally:
        set_kernels_config(previous)


def _named_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def test_qwen3_modeling_has_no_opslot_or_ops_import():
    from veomni.models_kernel.transformers.qwen3.generated import patched_modeling_qwen3_gpu as gpu
    from veomni.models_kernel.transformers.qwen3.generated import patched_modeling_qwen3_npu as npu

    for module in (gpu, npu):
        source = module.__file__
        assert source is not None
        text = open(source, encoding="utf-8").read()
        assert "use_non_eager_impl" not in text
        assert "OpSlot" not in text
        assert "veomni.ops" not in text
        assert "from veomni.models." not in text
        assert "from veomni.models import" not in text
        assert "from veomni.models_kernel.utils.loss_utils import" in text


def test_qwen3_constructs_local_kernels():
    model = _build_qwen3(_tiny_config())
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_ce, VeomniKernel)
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"
    assert layer.self_attn.veomni_rope.impl == "eager"


def test_qwen3_instances_keep_distinct_impls():
    eager = _build_qwen3(_tiny_config(), _eager_kernels_config())
    chunk_cfg = _eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_qwen3(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_qwen3_eager_matches_hf_logits_and_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen3ForCausalLM(config)
    ours = _build_qwen3(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_logits = hf(input_ids=input_ids, use_cache=False).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert ours_out.logits is None

    hf_out.loss.backward()
    ours_out.loss.backward()
    hf_grads = _named_trainable(hf)
    ours_grads = _named_trainable(ours)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        assert param.grad is not None, name
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(
            ours_grads[name].grad, param.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL, msg=name
        )


def test_qwen3_seq_cls_forward():
    _, seq_cls = _qwen3_classes()
    previous = get_kernels_config()
    set_kernels_config(_eager_kernels_config())
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
