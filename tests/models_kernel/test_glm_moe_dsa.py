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

"""GLM-MoE-DSA models_kernel consume tests.

Direct-import the generated class. Do not register or use
``build_foundation_model``. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM as HFGlmMoeDsaForCausalLM

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _tiny_config() -> GlmMoeDsaConfig:
    """Official GlmMoeDsaConfig fields, sized down for a toy.

    Omit ``mlp_layer_types`` / ``indexer_types`` so official ``__post_init__``
    fills them: first 3 dense then sparse, first layer full indexer then
    every ``index_topk_freq`` (default 1) layer full. Four layers therefore
    keep one sparse MoE layer. ``index_topk`` is shrunk below the toy
    sequence so the official top-k additive mask is actually sparse.
    """
    return GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        attn_implementation="eager",
        experts_implementation="eager",
    )


def _build_ours(config: GlmMoeDsaConfig, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.transformers.glm_moe_dsa.generated.patched_modeling_glm_moe_dsa_gpu import (
        GlmMoeDsaForCausalLM,
    )

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return GlmMoeDsaForCausalLM(config)
    finally:
        set_kernels_config(previous)


def test_glm_moe_dsa_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    attn = model.model.layers[0].self_attn
    assert attn.veomni_dsa_attention.kernel == "dsa_attention"
    assert attn.veomni_dsa_attention.variant == "glm"
    assert attn.veomni_dsa_attention.impl == "eager"
    assert attn.indexer.veomni_dsa_indexer.kernel == "dsa_indexer"
    assert attn.indexer.veomni_dsa_indexer.variant == "glm"
    assert attn.indexer.veomni_dsa_indexer.impl == "eager"


def test_glm_moe_dsa_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"
    assert eager.model.layers[0].self_attn.veomni_dsa_attention.impl == "eager"


def test_glm_moe_dsa_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)
