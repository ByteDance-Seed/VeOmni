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

"""Model registration, public construction, and instance-local op selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter

import pytest
from transformers import PretrainedConfig

from tests.models.compare import eager_ops_config, ops_config_scope, stamp_attn_implementation
from tests.models.tiny_configs import (
    tiny_deepseek_v3_config as _tiny_deepseek_v3_config,
)
from tests.models.tiny_configs import (
    tiny_deepseek_v4_config as _tiny_deepseek_v4_config,
)
from tests.models.tiny_configs import (
    tiny_flux_config as _tiny_flux_config,
)
from tests.models.tiny_configs import (
    tiny_gemma3_text_config as _tiny_gemma3_text_config,
)
from tests.models.tiny_configs import (
    tiny_glm_moe_dsa_config as _tiny_glm_moe_dsa_config,
)
from tests.models.tiny_configs import (
    tiny_gpt_oss_config as _tiny_gpt_oss_config,
)
from tests.models.tiny_configs import (
    tiny_llama_config as _tiny_llama_config,
)
from tests.models.tiny_configs import (
    tiny_ltx2_3_config as _tiny_ltx2_3_config,
)
from tests.models.tiny_configs import (
    tiny_minimax_h3_config as _tiny_minimax_h3_config,
)
from tests.models.tiny_configs import (
    tiny_movqgan_config as _tiny_movqgan_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_5_omni_config as _tiny_qwen2_5_omni_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_5_omni_text_config as _tiny_qwen2_5_omni_text_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_5_omni_thinker_config as _tiny_qwen2_5_omni_thinker_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_5_vl_config as _tiny_qwen2_5_vl_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_config as _tiny_qwen2_config,
)
from tests.models.tiny_configs import (
    tiny_qwen2_vl_config as _tiny_qwen2_vl_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_config as _tiny_qwen3_5_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_config as _tiny_qwen3_5_moe_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_moe_text_config as _tiny_qwen3_5_moe_text_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_5_text_config as _tiny_qwen3_5_text_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_config as _tiny_qwen3_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_moe_config as _tiny_qwen3_moe_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_omni_moe_config as _tiny_qwen3_omni_moe_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_omni_moe_text_config as _tiny_qwen3_omni_moe_text_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_omni_moe_thinker_config as _tiny_qwen3_omni_moe_thinker_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_vl_config as _tiny_qwen3_vl_config,
)
from tests.models.tiny_configs import (
    tiny_qwen3_vl_moe_config as _tiny_qwen3_vl_moe_config,
)
from tests.models.tiny_configs import (
    tiny_qwen_image_config as _tiny_qwen_image_config,
)
from tests.models.tiny_configs import (
    tiny_seed_oss_config as _tiny_seed_oss_config,
)
from tests.models.tiny_configs import (
    tiny_wan_config as _tiny_wan_config,
)
from tests.models.tiny_configs import (
    tiny_wan_t2v_config as _tiny_wan_t2v_config,
)
from veomni.models import (
    MODEL_CONFIG_REGISTRY,
    MODEL_PROCESSOR_REGISTRY,
    MODELING_REGISTRY,
    build_config,
    build_foundation_model,
    check_context_parallel_supported,
    check_model_build_prerequisites,
    get_model_class,
)
from veomni.ops import VeomniOp
from veomni.ops.config import get_ops_config


class _UnregisteredConfig(PretrainedConfig):
    model_type = "unregistered_test_model"

    def __init__(self):
        super().__init__()
        self.architectures = ["UnregisteredForCausalLM"]


@dataclass(frozen=True)
class _ModelCase:
    model_type: str
    config_factory: Callable[[str], PretrainedConfig]
    architectures: tuple[str, ...]
    has_registered_config: bool = False
    registered_config_aliases: tuple[str, ...] = ()
    registered_model_aliases: tuple[str, ...] = ()
    processor_class_name: str | None = None
    # Representative integration points, not an exhaustive snapshot of model internals.
    eager_ops: tuple[tuple[str, str], ...] = (("veomni_ce", "cross_entropy_loss"),)
    isolation_op_path: str | None = None
    # Attention that binds from ``config._attn_implementation`` needs the
    # ops selection stamped before HF construction. resolve_op_impl families
    # leave HF attn as eager; stamping sdpa makes PreTrainedModel reject them.
    stamps_hf_attn: bool = True


_MODEL_CASES = (
    _ModelCase(
        model_type="deepseek_v3",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.layers.3.mlp.experts.veomni_moe", "moe_experts"),
            ("model.layers.3.mlp.shared_experts.veomni_swiglu_mlp", "swiglu_mlp"),
        ),
        config_factory=_tiny_deepseek_v3_config,
        architectures=(
            "DeepseekV3ForCausalLM",
            "DeepseekV3ForSequenceClassification",
            "DeepseekV3ForTokenClassification",
            "DeepseekV3Model",
        ),
    ),
    _ModelCase(
        model_type="deepseek_v4",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.self_attn.q_b_norm.veomni_unweighted_rms_norm", "rms_norm"),
            ("model.layers.0.attn_hc.veomni_mhc_pre", "mhc"),
            ("model.layers.0.veomni_mhc_post", "mhc"),
            ("model.layers.0.self_attn.veomni_dsa_attention", "dsa_attention"),
            ("model.layers.3.self_attn.compressor.indexer.veomni_dsa_indexer", "dsa_indexer"),
            ("model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
            ("model.layers.0.mlp.shared_experts.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.hc_head.veomni_mhc_head", "mhc"),
        ),
        config_factory=_tiny_deepseek_v4_config,
        architectures=("DeepseekV4ForCausalLM", "DeepseekV4Model"),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="flux",
        eager_ops=(("blocks.0.attn.norm_q_a.veomni_rms_norm", "rms_norm"),),
        config_factory=_tiny_flux_config,
        architectures=("FluxModel",),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="gemma3_text",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_gemma3_text_config,
        architectures=("Gemma3ForCausalLM", "Gemma3TextModel"),
    ),
    _ModelCase(
        model_type="gpt_oss",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        # HF rejects sdpa for GptOss, and gpt_oss moe has no fused_triton row.
        isolation_op_path="veomni_ce",
        config_factory=_tiny_gpt_oss_config,
        architectures=(
            "GptOssForCausalLM",
            "GptOssForSequenceClassification",
            "GptOssForTokenClassification",
            "GptOssModel",
        ),
    ),
    _ModelCase(
        model_type="glm_moe_dsa",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.self_attn.veomni_dsa_attention", "dsa_attention"),
            ("model.layers.0.self_attn.indexer.veomni_dsa_indexer", "dsa_indexer"),
        ),
        config_factory=_tiny_glm_moe_dsa_config,
        architectures=("GlmMoeDsaForCausalLM", "GlmMoeDsaModel"),
    ),
    _ModelCase(
        model_type="llama",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_llama_config,
        architectures=(
            "LlamaForCausalLM",
            "LlamaForTokenClassification",
            "LlamaForSequenceClassification",
            "LlamaModel",
        ),
    ),
    _ModelCase(
        model_type="movqgan",
        config_factory=_tiny_movqgan_config,
        architectures=("MoVQGAN",),
        has_registered_config=True,
        processor_class_name="MoVQGANProcessor",
        eager_ops=(),
    ),
    _ModelCase(
        model_type="seed_oss",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_seed_oss_config,
        architectures=(
            "SeedOssForCausalLM",
            "SeedOssForQuestionAnswering",
            "SeedOssForSequenceClassification",
            "SeedOssForTokenClassification",
            "SeedOssModel",
        ),
    ),
    _ModelCase(
        model_type="qwen2",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_qwen2_config,
        architectures=(
            "Qwen2ForCausalLM",
            "Qwen2ForTokenClassification",
            "Qwen2ForSequenceClassification",
            "Qwen2ForQuestionAnswering",
            "Qwen2Model",
        ),
    ),
    _ModelCase(
        model_type="qwen2_vl",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.visual.blocks.0.attn.veomni_attn", "attention"),
            ("model.language_model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.language_model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_qwen2_vl_config,
        architectures=("Qwen2VLForConditionalGeneration", "Qwen2VLModel"),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="qwen2_5_vl",
        config_factory=_tiny_qwen2_5_vl_config,
        architectures=("Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel"),
        has_registered_config=True,
        registered_config_aliases=("qwen2_5_vl_text",),
    ),
    _ModelCase(
        model_type="qwen2_5_omni",
        eager_ops=(("thinker.veomni_ce", "cross_entropy_loss"),),
        config_factory=_tiny_qwen2_5_omni_config,
        architectures=("Qwen2_5OmniForConditionalGeneration",),
        has_registered_config=True,
        processor_class_name="Qwen2_5OmniProcessor",
    ),
    _ModelCase(
        model_type="qwen2_5_omni_thinker",
        config_factory=_tiny_qwen2_5_omni_thinker_config,
        architectures=("Qwen2_5OmniThinkerForConditionalGeneration",),
    ),
    _ModelCase(
        model_type="qwen2_5_omni_text",
        config_factory=_tiny_qwen2_5_omni_text_config,
        architectures=("Qwen2_5OmniThinkerTextModel",),
        eager_ops=(),
    ),
    _ModelCase(
        model_type="qwen3",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.veomni_swiglu_mlp", "swiglu_mlp"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        config_factory=_tiny_qwen3_config,
        architectures=(
            "Qwen3ForCausalLM",
            "Qwen3ForTokenClassification",
            "Qwen3ForSequenceClassification",
            "Qwen3Model",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.language_model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.language_model.layers.0.linear_attn.veomni_rms_norm_gated", "rms_norm_gated"),
            ("model.language_model.layers.0.linear_attn.veomni_causal_conv1d", "causal_conv1d"),
            ("model.language_model.layers.0.linear_attn.veomni_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
        ),
        config_factory=_tiny_qwen3_5_config,
        architectures=(
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5Model",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5_text",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.linear_attn.veomni_rms_norm_gated", "rms_norm_gated"),
            ("model.layers.0.linear_attn.veomni_causal_conv1d", "causal_conv1d"),
            ("model.layers.0.linear_attn.veomni_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
        ),
        config_factory=_tiny_qwen3_5_text_config,
        architectures=(
            "Qwen3_5ForCausalLM",
            "Qwen3_5TextModel",
        ),
    ),
    _ModelCase(
        model_type="qwen3_5_moe",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.language_model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.language_model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
            ("model.language_model.layers.0.linear_attn.veomni_rms_norm_gated", "rms_norm_gated"),
            ("model.language_model.layers.0.linear_attn.veomni_causal_conv1d", "causal_conv1d"),
            ("model.language_model.layers.0.linear_attn.veomni_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
        ),
        isolation_op_path="model.language_model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_5_moe_config,
        architectures=("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeModel"),
    ),
    _ModelCase(
        model_type="qwen3_5_moe_text",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
            ("model.layers.0.linear_attn.veomni_rms_norm_gated", "rms_norm_gated"),
            ("model.layers.0.linear_attn.veomni_causal_conv1d", "causal_conv1d"),
            ("model.layers.0.linear_attn.veomni_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
        ),
        isolation_op_path="model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_5_moe_text_config,
        architectures=("Qwen3_5MoeForCausalLM", "Qwen3_5MoeTextModel"),
    ),
    _ModelCase(
        model_type="qwen3_moe",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
            ("model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.layers.0.self_attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_moe_config,
        architectures=(
            "Qwen3MoeForCausalLM",
            "Qwen3MoeForTokenClassification",
            "Qwen3MoeForSequenceClassification",
            "Qwen3MoeForQuestionAnswering",
            "Qwen3MoeModel",
        ),
    ),
    _ModelCase(
        model_type="qwen3_omni_moe",
        eager_ops=(
            ("thinker.veomni_ce", "cross_entropy_loss"),
            ("thinker.veomni_lb", "load_balancing_loss"),
            ("thinker.model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
        ),
        isolation_op_path="thinker.model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_omni_moe_config,
        architectures=("Qwen3OmniMoeForConditionalGeneration",),
        has_registered_config=True,
        processor_class_name="Qwen3OmniMoeProcessor",
    ),
    _ModelCase(
        model_type="qwen3_omni_moe_thinker",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
        ),
        isolation_op_path="model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_omni_moe_thinker_config,
        architectures=("Qwen3OmniMoeThinkerForConditionalGeneration",),
    ),
    _ModelCase(
        model_type="qwen3_omni_moe_text",
        config_factory=_tiny_qwen3_omni_moe_text_config,
        architectures=("Qwen3OmniMoeThinkerTextModel",),
        eager_ops=(),
    ),
    _ModelCase(
        model_type="qwen3_vl",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("model.language_model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.language_model.layers.0.self_attn.veomni_rope", "rope"),
            ("model.language_model.layers.0.self_attn.veomni_attn", "attention"),
            ("model.visual.blocks.0.attn.veomni_rope", "rope"),
            ("model.visual.blocks.0.attn.veomni_attn", "attention"),
        ),
        isolation_op_path="model.language_model.layers.0.self_attn.veomni_attn",
        config_factory=_tiny_qwen3_vl_config,
        architectures=("Qwen3VLForConditionalGeneration", "Qwen3VLModel"),
    ),
    _ModelCase(
        model_type="qwen3_vl_moe",
        eager_ops=(
            ("veomni_ce", "cross_entropy_loss"),
            ("veomni_lb", "load_balancing_loss"),
            ("model.language_model.layers.0.input_layernorm.veomni_rms_norm", "rms_norm"),
            ("model.language_model.layers.0.mlp.experts.veomni_moe", "moe_experts"),
        ),
        isolation_op_path="model.language_model.layers.0.mlp.experts.veomni_moe",
        config_factory=_tiny_qwen3_vl_moe_config,
        architectures=(
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3VLMoeModel",
            "Qwen3VLMoeTextModel",
        ),
    ),
    _ModelCase(
        model_type="LTXVideoTransformerModel",
        config_factory=_tiny_ltx2_3_config,
        architectures=("LTXVideoTransformerModel",),
        has_registered_config=True,
        registered_config_aliases=("LTXVideoConditionModel",),
        registered_model_aliases=("LTXVideoConditionModel",),
        eager_ops=(("transformer_blocks.0.veomni_rms_norm_unweighted", "rms_norm"),),
        isolation_op_path="transformer_blocks.0.veomni_rms_norm_unweighted",
    ),
    _ModelCase(
        model_type="MiniMaxH3DiTModel",
        config_factory=_tiny_minimax_h3_config,
        architectures=("MiniMaxH3DiTModel",),
        has_registered_config=True,
        registered_config_aliases=("MiniMaxH3ConditionModel",),
        registered_model_aliases=("MiniMaxH3ConditionModel",),
        eager_ops=(("dit.blocks.0.attn.veomni_attn", "attention"),),
        isolation_op_path="dit.blocks.0.attn.veomni_attn",
        stamps_hf_attn=False,
    ),
    _ModelCase(
        model_type="QwenImageTransformer2DModel",
        config_factory=_tiny_qwen_image_config,
        architectures=("QwenImageTransformer2DModel",),
        has_registered_config=True,
        registered_config_aliases=("QwenImageConditionModel",),
        registered_model_aliases=("QwenImageConditionModel",),
        eager_ops=(),
    ),
    _ModelCase(
        model_type="wan",
        eager_ops=(
            ("blocks.0.self_attn.norm_q.veomni_rms_norm", "rms_norm"),
            ("blocks.0.self_attn.attn.veomni_attn", "attention"),
            ("blocks.0.self_attn.veomni_rope", "rope"),
        ),
        # wan rope has no liger row; isolate via attention (sdpa) and keep rope in eager_ops
        isolation_op_path="blocks.0.self_attn.attn.veomni_attn",
        stamps_hf_attn=False,
        config_factory=_tiny_wan_config,
        architectures=("WanModel",),
        has_registered_config=True,
    ),
    _ModelCase(
        model_type="WanTransformer3DModel",
        config_factory=_tiny_wan_t2v_config,
        architectures=("WanTransformer3DModel",),
        has_registered_config=True,
        registered_config_aliases=("WanTransformer3DConditionModel",),
        registered_model_aliases=("WanTransformer3DConditionModel",),
        # Portable attention is sdpa; there is no model-local eager forward.
        eager_ops=(),
    ),
)

_ARCHITECTURE_CASES = tuple(
    pytest.param(model_case, architecture, id=f"{model_case.model_type}-{architecture}")
    for model_case in _MODEL_CASES
    for architecture in model_case.architectures
)


def test_get_model_class_unknown_type_raises():
    with pytest.raises(RuntimeError, match="unregistered_test_model.*not registered in veomni.models"):
        get_model_class(_UnregisteredConfig())


def test_get_model_class_hf_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "hf")
    from transformers import AutoModelForCausalLM

    assert get_model_class(_tiny_qwen3_config()) is AutoModelForCausalLM


def test_build_foundation_model_requires_ops_config():
    with ops_config_scope(None):
        with pytest.raises(ValueError, match="ops_implementation"):
            build_foundation_model(_tiny_qwen3_config())


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda model_case: model_case.model_type)
def test_build_foundation_model_constructs_registered_model(model_case: _ModelCase):
    cfg = eager_ops_config()
    with ops_config_scope(None):
        model = build_foundation_model(
            model_case.config_factory(model_case.architectures[0]),
            torch_dtype="float32",
            init_device="cpu",
            ops_implementation=cfg,
        )
        assert get_ops_config() is cfg
    assert model.__class__.__name__ == model_case.architectures[0]
    for path, expected_op in model_case.eager_ops:
        op = attrgetter(path)(model)
        assert isinstance(op, VeomniOp), path
        assert op.op == expected_op, path
        assert op.impl == "eager", path


_ALTERNATE_OP_IMPLS = {
    "cross_entropy_loss": ("cross_entropy_loss_implementation", "chunk_loss"),
    "moe_experts": ("moe_implementation", "fused_triton"),
    "rms_norm": ("rms_norm_implementation", "liger_kernel"),
    "attention": ("attn_implementation", "sdpa"),
    "rope": ("rotary_pos_emb_implementation", "liger_kernel"),
}


def _op_bindings(model, selected_path):
    """Snapshot handles and selected implementations, including unrelated model ops."""
    bindings = {
        (module_name, attribute): (value, value.impl)
        for module_name, module in model.named_modules()
        for attribute, value in vars(module).items()
        if isinstance(value, VeomniOp)
    }
    # Diffusers processors can hold the selected op outside the nn.Module tree.
    selected = attrgetter(selected_path)(model)
    module_name, _, attribute = selected_path.rpartition(".")
    bindings[module_name, attribute] = (selected, selected.impl)
    return bindings


@pytest.mark.parametrize(
    "model_case",
    [case for case in _MODEL_CASES if case.eager_ops],
    ids=lambda case: case.model_type,
)
def test_model_instances_keep_distinct_impls(model_case: _ModelCase, available_nvidia_ops):
    """Construction and later config changes must not rebind existing instances.

    This checks selection only; platform gates are stubbed and no optimized
    kernel executes. Family parity and ops tests own numerical execution.
    """
    previous = get_ops_config()
    eager_config = eager_ops_config()
    selected_path = model_case.isolation_op_path or model_case.eager_ops[0][0]
    selected_op_name = next(op for path, op in model_case.eager_ops if path == selected_path)

    def construct(config):
        model_config = model_case.config_factory(model_case.architectures[0])
        # HF-interface attention binds from ``config._attn_implementation``.
        # VL nested ``to_dict()`` also drops the field and HF then defaults
        # to sdpa, so eager construction must stamp as well.
        if selected_op_name == "attention" and model_case.stamps_hf_attn:
            stamp_attn_implementation(model_config, config.attn_implementation)
        with ops_config_scope(config):
            return get_model_class(model_config)(model_config)

    eager = construct(eager_config)
    selected_op = attrgetter(selected_path)
    assert selected_op(eager).impl == "eager"
    eager_bindings = _op_bindings(eager, selected_path)
    field, alternate_impl = _ALTERNATE_OP_IMPLS[selected_op(eager).op]
    alternate_config = eager_ops_config()
    setattr(alternate_config, field, alternate_impl)
    alternate = construct(alternate_config)
    assert selected_op(alternate).impl == alternate_impl
    alternate_bindings = _op_bindings(alternate, selected_path)

    for config in (alternate_config, eager_config, None):
        with ops_config_scope(config):
            assert _op_bindings(eager, selected_path) == eager_bindings
            assert _op_bindings(alternate, selected_path) == alternate_bindings
    assert get_ops_config() is previous


@pytest.mark.parametrize("model_case", _MODEL_CASES, ids=lambda model_case: model_case.model_type)
def test_model_registry_entries(model_case: _ModelCase):
    assert (model_case.model_type in MODEL_CONFIG_REGISTRY.valid_keys()) is model_case.has_registered_config
    assert set(model_case.registered_config_aliases) <= set(MODEL_CONFIG_REGISTRY.valid_keys())
    assert model_case.model_type in MODELING_REGISTRY.valid_keys()
    assert set(model_case.registered_model_aliases) <= set(MODELING_REGISTRY.valid_keys())
    if model_case.processor_class_name is not None:
        assert model_case.processor_class_name in MODEL_PROCESSOR_REGISTRY.valid_keys()


@pytest.mark.parametrize(("model_case", "architecture"), _ARCHITECTURE_CASES)
def test_get_model_class_returns_registered_architecture(model_case: _ModelCase, architecture: str):
    model_cls = get_model_class(model_case.config_factory(architecture))
    assert model_cls.__name__ == architecture


def test_get_model_config_uses_the_registered_dsv4_subclass(tmp_path):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as UpstreamConfig

    from veomni.models.registry import get_model_config
    from veomni.models.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=["compressed_sparse_attention"] * 2,
    ).save_pretrained(tmp_path)
    config = get_model_config(str(tmp_path), dsa_indexer_loss=True, dsa_indexer_loss_coef=0.25)
    assert type(config) is DeepseekV4Config
    assert type(config) is not UpstreamConfig
    assert config.dsa_indexer_loss is True
    assert config.dsa_indexer_loss_coef == 0.25
    assert type(build_config(str(tmp_path), dsa_indexer_loss=True)) is DeepseekV4Config


def test_a_config_that_cannot_ask_for_the_objective_is_left_alone():
    from transformers import AutoConfig

    other = AutoConfig.for_model("llama", num_hidden_layers=2)
    assert not hasattr(other, "dsa_indexer_loss")
    assert not hasattr(other, "validate_build_prerequisites")
    check_model_build_prerequisites(other)


def test_the_generic_hook_reaches_the_model_that_implements_it():
    from veomni.models.transformers.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    cfg = eager_ops_config()
    cfg.dsa_indexer_implementation = "eager"
    with ops_config_scope(cfg):
        config = DeepseekV4Config(
            num_hidden_layers=2,
            layer_types=["compressed_sparse_attention"] * 2,
            dsa_indexer_loss=True,
        )
        with pytest.raises(ValueError, match="dsa_indexer_implementation"):
            check_model_build_prerequisites(config)


def test_context_parallel_is_refused_on_npu(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    import veomni.models.auto as auto

    monkeypatch.setattr(auto, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(auto, "get_parallel_state", lambda: SimpleNamespace(cp_enabled=True))
    monkeypatch.setattr(auto, "is_torch_npu_available", lambda: True)
    with pytest.raises(NotImplementedError, match="GPU-only"):
        check_context_parallel_supported(_tiny_qwen3_config())


def test_context_parallel_gate_is_inert_when_no_parallel_state_was_installed(monkeypatch: pytest.MonkeyPatch):
    from veomni.distributed import parallel_state as parallel_state_module

    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    monkeypatch.setattr(parallel_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state_module.dist, "get_world_size", lambda: 2)
    check_context_parallel_supported(_tiny_qwen3_config())
