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

"""DeepSeek-V3 registry-hook and Hugging Face parity tests.

Direct-import the generated class. Compare a toy CausalLM against Hugging Face.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_deepseek_v3_config as _tiny_config


def _dsv3_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_npu import (
            DeepseekV3ForCausalLM,
        )
    else:
        from veomni.models.transformers.deepseek_v3.generated.patched_modeling_deepseek_v3_gpu import (
            DeepseekV3ForCausalLM,
        )
    return DeepseekV3ForCausalLM


def _build_ours(config: DeepseekV3Config, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _dsv3_cls()(config)


def test_deepseek_v3_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFDeepseekV3ForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    # MLA + MoE expert-loop ULP (~1e-7); not bitwise vs Hugging Face.
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_deepseek_v3_experts_cast_router_scores_to_hidden_dtype():
    config = _tiny_config()
    model = _build_ours(config)
    experts = next(layer.mlp.experts for layer in model.model.layers if hasattr(layer.mlp, "experts"))
    hidden_states = torch.linspace(-0.7, 0.8, steps=4 * config.hidden_size).reshape(4, config.hidden_size)
    selected_experts = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]], dtype=torch.long)
    top_k_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.55, 0.45], [0.8, 0.2]],
        dtype=torch.float32,
    )
    captured = {}

    def record(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.zeros_like(hidden_states)

    experts.veomni_moe = record
    actual = experts(hidden_states, selected_experts, top_k_weights)
    args = captured["args"]

    assert args[0] is hidden_states
    torch.testing.assert_close(args[1], top_k_weights.to(hidden_states.dtype), rtol=0, atol=0)
    assert args[2] is selected_experts
    assert captured["kwargs"] == {"num_experts": experts.num_experts}
    torch.testing.assert_close(actual, torch.zeros_like(hidden_states), rtol=0, atol=0)


def test_deepseek_v3_registry_installs_checkpoint_hooks():
    from veomni.models import get_model_class

    for architecture in (
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForSequenceClassification",
        "DeepseekV3ForTokenClassification",
        "DeepseekV3Model",
    ):
        model_cls = get_model_class(_tiny_config(architecture))
        assert callable(model_cls._create_checkpoint_tensor_converter)
        assert callable(model_cls._convert_fqn_to_index_mapping)
