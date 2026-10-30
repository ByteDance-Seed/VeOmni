"""Qwen3-VL AR backbone OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("qwen3vl_llm")
def register_qwen3vl_llm_config():
    from .configuration import Qwen3VLLlmConfig

    return Qwen3VLLlmConfig


@OMNI_MODEL_REGISTRY.register("qwen3vl_llm")
def register_qwen3vl_llm_model():
    from .modeling import Qwen3VLLlm

    return Qwen3VLLlm


@OMNI_ACCELERATED_MODEL_REGISTRY.register("qwen3vl_llm")
def register_qwen3vl_llm_accelerated_model():
    from .accelerated import Qwen3VLLlmAccelerated

    return Qwen3VLLlmAccelerated
