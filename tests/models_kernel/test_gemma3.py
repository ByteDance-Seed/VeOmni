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

"""Gemma 3 text models_kernel consume tests.

Direct-import the generated CausalLM. Do not register or use
``build_foundation_model``. Compare a toy model against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM as HFGemma3ForCausalLM

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
    named_trainable,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _tiny_config(
    *,
    layer_types: list[str] | None = None,
    final_logit_softcapping: float | None = None,
    attn_logit_softcapping: float | None = None,
) -> Gemma3TextConfig:
    if layer_types is None:
        layer_types = ["sliding_attention", "sliding_attention", "full_attention"]
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=16,
        sliding_window=8,
        layer_types=layer_types,
        final_logit_softcapping=final_logit_softcapping,
        attn_logit_softcapping=attn_logit_softcapping,
        use_bidirectional_attention=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )


def _build_ours(config: Gemma3TextConfig, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.transformers.gemma3.generated.patched_modeling_gemma3_gpu import (
        Gemma3ForCausalLM,
    )

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return Gemma3ForCausalLM(config)
    finally:
        set_kernels_config(previous)


def test_gemma3_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"


def test_gemma3_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"


def test_gemma3_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGemma3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_gemma3_eager_matches_hf_softcap():
    """Softcap-on labeled path keeps logits. The helper sees `logits=`, not fused hidden+weight."""
    torch.manual_seed(1)
    config = _tiny_config(final_logit_softcapping=30.0)
    hf = HFGemma3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    hf_logits = hf(input_ids=input_ids, use_cache=False).logits
    ours_logits = ours(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    labels = input_ids.clone()
    hf_out = hf(input_ids=input_ids, labels=labels, use_cache=False)
    ours_out = ours(input_ids=input_ids, labels=labels, use_cache=False)
    torch.testing.assert_close(ours_out.loss, hf_out.loss, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    assert ours_out.logits is not None
    torch.testing.assert_close(ours_out.logits, hf_out.logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    hf_out.loss.backward()
    ours_out.loss.backward()
    hf_grads = named_trainable(hf)
    ours_grads = named_trainable(ours)
    assert hf_grads.keys() == ours_grads.keys()
    for name, param in hf_grads.items():
        if param.grad is None:
            assert ours_grads[name].grad is None, name
            continue
        assert ours_grads[name].grad is not None, name
        torch.testing.assert_close(
            ours_grads[name].grad,
            param.grad,
            atol=EAGER_GRAD_ATOL,
            rtol=EAGER_GRAD_RTOL,
            msg=name,
        )


def test_gemma3_packed_eager_matches_independent_samples():
    """Packed ``cu_seq_lens_q`` uses ``packed_causal_mask`` / ``sliding_window_mask``."""
    torch.manual_seed(123)
    config = _tiny_config()
    ours = _build_ours(config).eval()
    first_input_ids = torch.tensor([[5, 6, 7]])
    second_input_ids = torch.tensor([[8, 9, 10, 11, 12]])
    packed_input_ids = torch.cat((first_input_ids, second_input_ids), dim=1)

    with torch.no_grad():
        packed_logits = ours(
            input_ids=packed_input_ids,
            attention_mask=torch.ones_like(packed_input_ids),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
            cu_seq_lens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
            use_cache=False,
        ).logits
        first_logits = ours(
            input_ids=first_input_ids,
            attention_mask=torch.ones_like(first_input_ids),
            position_ids=torch.arange(3).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
            use_cache=False,
        ).logits
        second_logits = ours(
            input_ids=second_input_ids,
            attention_mask=torch.ones_like(second_input_ids),
            position_ids=torch.arange(5).unsqueeze(0),
            cu_seq_lens_q=torch.tensor([0, 5], dtype=torch.int32),
            use_cache=False,
        ).logits

    torch.testing.assert_close(packed_logits[:, :3], first_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
    torch.testing.assert_close(packed_logits[:, 3:], second_logits, atol=EAGER_ATOL, rtol=EAGER_RTOL)
