"""Qwen3-Omni thinker AR backbone OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3omni_llm")
def register_qwen3omni_llm_config():
    from .configuration import Qwen3OmniLlmConfig

    return Qwen3OmniLlmConfig


@OMNI_MODEL_REGISTRY.register("qwen3omni_llm")
def register_qwen3omni_llm_model():
    from .modeling import Qwen3OmniLlm

    return Qwen3OmniLlm


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3omni_llm")
def register_qwen3omni_llm_accelerated_model():
    from .accelerated import Qwen3OmniLlmAccelerated

    return Qwen3OmniLlmAccelerated
