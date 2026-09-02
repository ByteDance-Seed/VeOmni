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

"""DeepSeek-V3 models_kernel consume tests.

Direct-import the generated class. Do not register or use
``build_foundation_model``. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from tests.models_kernel.compare import (
    assert_eager_matches_hf,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _tiny_config() -> DeepseekV3Config:
    """Official DeepseekV3Config fields, sized down for a toy.

    Algorithm defaults stay official: ``rope_interleave=True``,
    ``first_k_dense_replace=3``, ``n_group=8``, ``topk_group=4``,
    ``routed_scaling_factor=2.5``. Sixteen routed experts keep the official
    grouped router (``experts_per_group >= 2``). Four layers keep one routed
    MoE layer after the official dense prefix.
    """
    return DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=16,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attn_implementation="eager",
        experts_implementation="eager",
    )


def _dsv3_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models_kernel.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_npu import (
            DeepseekV3ForCausalLM,
        )
    else:
        from veomni.models_kernel.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_gpu import (
            DeepseekV3ForCausalLM,
        )
    return DeepseekV3ForCausalLM


def _build_ours(config: DeepseekV3Config, kernels: SimpleNamespace | None = None):
    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return _dsv3_cls()(config)
    finally:
        set_kernels_config(previous)


def test_deepseek_v3_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniKernel)
    assert model.veomni_ce.impl == "eager"
    dense = model.model.layers[0]
    assert dense.input_layernorm.veomni_rms_norm.impl == "eager"
    assert dense.mlp.veomni_swiglu_mlp.kernel == "swiglu_mlp"
    moe = model.model.layers[3]
    assert moe.mlp.experts.veomni_moe.kernel == "moe_experts"
    assert moe.mlp.experts.veomni_moe.impl == "eager"
    assert moe.mlp.shared_experts.veomni_swiglu_mlp.kernel == "swiglu_mlp"


def test_deepseek_v3_instances_keep_distinct_impls():
    eager = _build_ours(_tiny_config(), eager_kernels_config())
    chunk_cfg = eager_kernels_config()
    chunk_cfg.cross_entropy_loss_implementation = "chunk_loss"
    chunk = _build_ours(_tiny_config(), chunk_cfg)

    assert eager.veomni_ce.impl == "eager"
    assert chunk.veomni_ce.impl == "chunk_loss"

    set_kernels_config(chunk_cfg)
    assert eager.veomni_ce.impl == "eager"
    assert eager.model.layers[3].mlp.experts.veomni_moe.impl == "eager"


def test_deepseek_v3_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)
