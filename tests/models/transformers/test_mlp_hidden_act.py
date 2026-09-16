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

"""MLP hidden_act routing: silu/swish use swiglu_mlp, anything else uses act_fn."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Experts as HFDeepseekV3Experts
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP as HFDeepseekV3MLP
from transformers.models.llama.modeling_llama import LlamaMLP as HFLlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as HFQwen3MLP
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts as HFQwen3_5MoeExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts as HFQwen3MoeExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP as HFQwen3MoeMLP
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts as HFQwen3OmniMoeThinkerTextExperts,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts as HFQwen3VLMoeTextExperts
from transformers.models.seed_oss.modeling_seed_oss import SeedOssMLP as HFSeedOssMLP

from tests.models.compare import assert_module_forward_and_grads_match, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import (
    tiny_deepseek_v3_config,
    tiny_llama_config,
    tiny_qwen2_config,
    tiny_qwen3_5_moe_text_config,
    tiny_qwen3_config,
    tiny_qwen3_moe_config,
    tiny_qwen3_omni_moe_text_config,
    tiny_qwen3_vl_moe_config,
    tiny_seed_oss_config,
)
from veomni.models.utils.op_utils import uses_swiglu_mlp
from veomni.utils.device import IS_NPU_AVAILABLE


HIDDEN_ACTS = ("silu", "swish", "gelu")


class _CallCounter:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


def _raise_if_called(*_args, **_kwargs):
    raise AssertionError("non-silu hidden_act must not call the fused SwiGLU / silu MoE kernel")


def _init_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        nn.init.normal_(param, std=0.02)


def _pair(ours_cls, hf_cls, config):
    hf = hf_cls(config)
    _init_parameters(hf)
    with ops_config_scope(eager_ops_config()):
        ours = ours_cls(config)
    ours.load_state_dict(hf.state_dict())
    counter = None
    if uses_swiglu_mlp(config.hidden_act):
        handle_name = "veomni_swiglu_mlp" if hasattr(ours, "veomni_swiglu_mlp") else "veomni_moe"
        counter = _CallCounter(getattr(ours, handle_name))
        setattr(ours, handle_name, counter)
    else:
        if hasattr(ours, "veomni_swiglu_mlp"):
            ours.veomni_swiglu_mlp = _raise_if_called
        if hasattr(ours, "veomni_moe"):
            ours.veomni_moe = _raise_if_called
    return hf, ours, counter


def _assert_matches(hf, ours, counter, *args):
    assert_module_forward_and_grads_match(hf, ours, *args)
    if counter is not None:
        assert counter.calls == 1


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_uses_swiglu_mlp_matches_silu_family(hidden_act):
    assert uses_swiglu_mlp(hidden_act) is (hidden_act in {"silu", "swish"})


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen2_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    from veomni.models.transformers.qwen2.generated.patched_modeling_qwen2_gpu import Qwen2MLP

    config = tiny_qwen2_config(hidden_act=hidden_act)
    hf, ours, counter = _pair(Qwen2MLP, HFQwen2MLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_npu import Qwen3MLP
    else:
        from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3MLP

    config = tiny_qwen3_config(hidden_act=hidden_act)
    hf, ours, counter = _pair(Qwen3MLP, HFQwen3MLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_moe_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import Qwen3MoeMLP
    else:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeMLP

    config = tiny_qwen3_moe_config(hidden_act=hidden_act)
    hf, ours, counter = _pair(Qwen3MoeMLP, HFQwen3MoeMLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_moe_merged_experts_match_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_npu import Qwen3MoeExperts
    else:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import Qwen3MoeExperts

    config = tiny_qwen3_moe_config(hidden_act=hidden_act)
    hf, ours, counter = _pair(Qwen3MoeExperts, HFQwen3MoeExperts, config)
    tokens = 6
    hidden = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.long)
    top_k_weights = torch.rand(tokens, config.num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    _assert_matches(hf, ours, counter, hidden, top_k_index, top_k_weights)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_llama_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaMLP

    config = tiny_llama_config(hidden_act=hidden_act)
    hf, ours, counter = _pair(LlamaMLP, HFLlamaMLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_seed_oss_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.seed_oss.generated.patched_modeling_seed_oss_npu import SeedOssMLP
    else:
        from veomni.models.transformers.seed_oss.generated.patched_modeling_seed_oss_gpu import SeedOssMLP

    config = tiny_seed_oss_config()
    config.hidden_act = hidden_act
    hf, ours, counter = _pair(SeedOssMLP, HFSeedOssMLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_deepseek_v3_dense_mlp_matches_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_npu import DeepseekV3MLP
    else:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_gpu import DeepseekV3MLP

    config = tiny_deepseek_v3_config()
    config.hidden_act = hidden_act
    hf, ours, counter = _pair(DeepseekV3MLP, HFDeepseekV3MLP, config)
    x = torch.randn(2, 5, config.hidden_size)
    _assert_matches(hf, ours, counter, x)


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_deepseek_v3_merged_experts_match_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_npu import DeepseekV3Experts
    else:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_gpu import DeepseekV3Experts

    config = tiny_deepseek_v3_config()
    config.hidden_act = hidden_act
    hf, ours, counter = _pair(DeepseekV3Experts, HFDeepseekV3Experts, config)
    tokens = 6
    hidden = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.long)
    top_k_weights = torch.rand(tokens, config.num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    _assert_matches(hf, ours, counter, hidden, top_k_index, top_k_weights)


def _moe_expert_inputs(config):
    tokens = 6
    hidden = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.long)
    top_k_weights = torch.rand(tokens, config.num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    return hidden, top_k_index, top_k_weights


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_5_moe_merged_experts_match_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_npu import Qwen3_5MoeExperts
    else:
        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import Qwen3_5MoeExperts

    config = tiny_qwen3_5_moe_text_config(layer_types=["full_attention", "full_attention"])
    config.hidden_act = hidden_act
    hf, ours, counter = _pair(Qwen3_5MoeExperts, HFQwen3_5MoeExperts, config)
    _assert_matches(hf, ours, counter, *_moe_expert_inputs(config))


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_vl_moe_merged_experts_match_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_npu import (
            Qwen3VLMoeTextExperts,
        )
    else:
        from veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu import (
            Qwen3VLMoeTextExperts,
        )

    config = tiny_qwen3_vl_moe_config()
    text = config.text_config
    text.hidden_act = hidden_act
    hf, ours, counter = _pair(Qwen3VLMoeTextExperts, HFQwen3VLMoeTextExperts, text)
    _assert_matches(hf, ours, counter, *_moe_expert_inputs(text))


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
def test_qwen3_omni_moe_merged_experts_match_hf_for_hidden_act(hidden_act):
    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_npu import (
            Qwen3OmniMoeThinkerTextExperts,
        )
    else:
        from veomni.models.transformers.qwen3_omni_moe.generated.patched_modeling_qwen3_omni_moe_gpu import (
            Qwen3OmniMoeThinkerTextExperts,
        )

    config = tiny_qwen3_omni_moe_text_config()
    config.hidden_act = hidden_act
    hf, ours, counter = _pair(Qwen3OmniMoeThinkerTextExperts, HFQwen3OmniMoeThinkerTextExperts, config)
    _assert_matches(hf, ours, counter, *_moe_expert_inputs(config))
