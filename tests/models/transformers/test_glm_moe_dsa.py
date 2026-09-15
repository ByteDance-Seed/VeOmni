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

"""GLM-MoE-DSA registry, parallel-plan, and Hugging Face parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM as HFGlmMoeDsaForCausalLM
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel as HFGlmMoeDsaModel

from tests.models.compare import (
    assert_eager_matches_hf,
    assert_outputs_and_grads_match,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_glm_moe_dsa_config as _tiny_config


def _glm_cls(architecture: str):
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.glm_moe_dsa.generated import patched_modeling_glm_moe_dsa_npu as gen
    else:
        from veomni.models.transformers.glm_moe_dsa.generated import patched_modeling_glm_moe_dsa_gpu as gen
    return getattr(gen, architecture)


def _build_ours(
    config: GlmMoeDsaConfig,
    ops: SimpleNamespace | None = None,
    architecture: str = "GlmMoeDsaForCausalLM",
):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _glm_cls(architecture)(config)


def _assert_dsa_wiring(attn) -> None:
    """GPU patches bind DSA ops; NPU generated modeling keeps the HF attention path."""
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        assert not hasattr(attn, "veomni_dsa_attention")
        assert attn.indexer is not None
        assert not hasattr(attn.indexer, "veomni_dsa_indexer")
        return
    assert attn.veomni_dsa_attention.variant == "glm"
    assert attn.indexer.veomni_dsa_indexer.variant == "glm"


def test_glm_moe_dsa_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFGlmMoeDsaForCausalLM(config)
    ours = _build_ours(config)
    _assert_dsa_wiring(ours.model.layers[0].self_attn)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)


def test_glm_moe_dsa_base_model_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config("GlmMoeDsaModel")
    hf = HFGlmMoeDsaModel(config)
    ours = _build_ours(config, architecture="GlmMoeDsaModel")
    _assert_dsa_wiring(ours.layers[0].self_attn)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    gradient_weights = torch.randn(2, 8, config.hidden_size)
    assert_outputs_and_grads_match(
        hf,
        ours,
        lambda model: model(input_ids=input_ids, use_cache=False).last_hidden_state * gradient_weights,
    )


def test_glm_moe_dsa_registry_installs_checkpoint_hooks_and_ep_plan():
    from veomni.models import get_model_class

    for architecture in ("GlmMoeDsaForCausalLM", "GlmMoeDsaModel"):
        model_cls = get_model_class(_tiny_config(architecture))
        assert callable(model_cls._create_checkpoint_tensor_converter)
        assert callable(model_cls._convert_fqn_to_index_mapping)

    causal_cls = get_model_class(_tiny_config())
    ep_plan = causal_cls.get_parallel_plan(None).extra_parallel_plan["ep"]
    assert set(ep_plan) == {
        "model.layers.*.mlp.experts.gate_up_proj",
        "model.layers.*.mlp.experts.down_proj",
    }
    assert all(placement.dim == 0 for placement in ep_plan.values())
