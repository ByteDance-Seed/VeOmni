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

"""DeepSeek-V4 models_kernel consume tests.

Direct-import the generated class. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFDeepseekV4ForCausalLM

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _tiny_config() -> DeepseekV4Config:
    """Official DeepseekV4Config fields, sized down for a toy.

    Omit schedule / router / mHC / RoPE / window fields so ``__post_init__``
    keeps the official defaults: 2× HCA bootstrap then CSA/HCA interleave,
    3× ``hash_moe`` then ``moe``, ``scoring_func="sqrtsoftplus"``,
    ``hc_mult=4``, ``sliding_window=128``, CSA=4 / HCA=128.
    Four layers therefore include a CSA indexer layer and one routed MoE layer.
    """
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        max_position_embeddings=64,
        o_groups=8,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=16,
        attn_implementation="eager",
        experts_implementation="eager",
    )


def _dsv4_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_npu import (
            DeepseekV4ForCausalLM,
        )
    else:
        from veomni.models_kernel.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
            DeepseekV4ForCausalLM,
        )
    return DeepseekV4ForCausalLM


def _build_ours(config: DeepseekV4Config, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return _dsv4_cls()(config)
    finally:
        set_kernels_config(previous)


def test_deepseek_v4_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    assert isinstance(model.veomni_lb, VeomniKernel)
    assert model.config.layer_types == [
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    assert model.config.mlp_layer_types == ["hash_moe", "hash_moe", "hash_moe", "moe"]
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.attn_hc.veomni_mhc_pre.kernel == "mhc"
    assert layer.veomni_mhc_post.variant == "post"
    assert layer.self_attn.veomni_dsa_attention.kernel == "dsa_attention"
    assert layer.self_attn.veomni_dsa_attention.variant == "deepseek_v4"
    csa = model.model.layers[3].self_attn.compressor
    assert csa.indexer.veomni_dsa_indexer.kernel == "dsa_indexer"
    assert csa.indexer.veomni_dsa_indexer.variant == "deepseek_v4"
    assert layer.mlp.experts.veomni_moe.kernel == "moe_experts"
    assert layer.mlp.shared_experts.veomni_swiglu_mlp.kernel == "swiglu_mlp"
    assert layer.mlp.shared_experts.limit == model.config.swiglu_limit
    assert model.model.hc_head.veomni_mhc_head.variant == "head"


def test_deepseek_v4_shared_mlp_passes_swiglu_limit():
    config = _tiny_config()
    model = _build_ours(config)
    shared = model.model.layers[0].mlp.shared_experts
    captured: dict = {}

    def record(x, *args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(x)

    shared.veomni_swiglu_mlp = record
    shared(torch.randn(2, 8, config.hidden_size))
    assert captured["swiglu_limit"] == config.swiglu_limit


def test_deepseek_v4_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"
    assert eager.model.layers[0].self_attn.veomni_dsa_attention.impl == "eager"


def test_deepseek_v4_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_deepseek_v4_eager_matches_hf_aux_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV4ForCausalLM(config)
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
