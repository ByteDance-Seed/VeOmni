"""Qwen3 AR backbone OmniModule."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3_llm")
def register_qwen3_llm_config():
    from .configuration import Qwen3LlmConfig

    return Qwen3LlmConfig


@OMNI_MODEL_REGISTRY.register("qwen3_llm")
def register_qwen3_llm_model():
    from .modeling import Qwen3Llm

    return Qwen3Llm
