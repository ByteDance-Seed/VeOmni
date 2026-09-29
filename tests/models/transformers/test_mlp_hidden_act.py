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

import importlib
from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts as HFDeepseekV4Experts
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts as HFQwen3MoeExperts

from tests.models.compare import assert_module_forward_and_grads_match, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import (
    tiny_deepseek_v4_config,
    tiny_qwen2_config,
    tiny_qwen3_moe_config,
)
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


def _generated_cls(family: str, class_name: str):
    suffix = "npu" if IS_NPU_AVAILABLE and family != "qwen2" else "gpu"
    module = importlib.import_module(
        f"veomni.models.transformers.{family}.generated.patched_modeling_{family}_{suffix}"
    )
    return getattr(module, class_name)


def _pair(ours_cls, hf_cls, config):
    hf = hf_cls(config)
    _init_parameters(hf)
    with ops_config_scope(eager_ops_config()):
        ours = ours_cls(config)
    ours.load_state_dict(hf.state_dict())
    counter = None
    if config.hidden_act in {"silu", "swish"}:
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


def _moe_expert_inputs(config):
    tokens = 6
    hidden = torch.randn(tokens, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.long)
    top_k_weights = torch.rand(tokens, config.num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    return hidden, top_k_index, top_k_weights


def _config_with_hidden_act(factory: Callable[..., object], hidden_act: str):
    try:
        return factory(hidden_act=hidden_act)
    except TypeError:
        config = factory()
        config.hidden_act = hidden_act
        return config


@pytest.mark.parametrize("hidden_act", HIDDEN_ACTS)
@pytest.mark.parametrize(
    ("family", "class_name", "hf_cls", "config_factory", "kind"),
    (
        pytest.param("qwen2", "Qwen2MLP", HFQwen2MLP, tiny_qwen2_config, "dense", id="qwen2-dense"),
        pytest.param(
            "qwen3_moe",
            "Qwen3MoeExperts",
            HFQwen3MoeExperts,
            tiny_qwen3_moe_config,
            "moe",
            id="qwen3-moe-experts",
        ),
        pytest.param(
            "deepseek_v4",
            "DeepseekV4Experts",
            HFDeepseekV4Experts,
            tiny_deepseek_v4_config,
            "moe",
            id="dsv4-experts",
        ),
    ),
)
def test_hidden_act_routes_silu_to_fused_kernel(family, class_name, hf_cls, config_factory, kind, hidden_act):
    config = _config_with_hidden_act(config_factory, hidden_act)
    hf, ours, counter = _pair(_generated_cls(family, class_name), hf_cls, config)
    args = (torch.randn(2, 5, config.hidden_size),) if kind == "dense" else _moe_expert_inputs(config)
    _assert_matches(hf, ours, counter, *args)


def test_merged_experts_act_fn_forward_rejects_ep_sharded_weights_without_ep():
    from transformers.activations import ACT2FN

    from veomni.models.utils.moe_utils import merged_experts_act_fn_forward

    hidden = torch.randn(3, 4)
    top_k_index = torch.tensor([[3], [3], [3]], dtype=torch.long)
    top_k_weights = torch.ones(3, 1)
    gate_up_proj = torch.randn(2, 8, 4)
    down_proj = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="expert-parallel sharded weights"):
        merged_experts_act_fn_forward(
            hidden,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            ACT2FN["gelu"],
            num_experts=4,
        )
