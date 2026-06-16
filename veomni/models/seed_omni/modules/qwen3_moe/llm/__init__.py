"""Qwen3-MoE AR backbone OmniModule."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3_moe_llm")
def register_qwen3_moe_llm_config():
    from .configuration import Qwen3MoeLlmConfig

    return Qwen3MoeLlmConfig


@OMNI_MODEL_REGISTRY.register("qwen3_moe_llm")
def register_qwen3_moe_llm_model():
    from .modeling import Qwen3MoeLlm

    return Qwen3MoeLlm
