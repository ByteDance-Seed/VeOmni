"""DeepSeek-V4.1 CED backbone OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("deepseek_v41_llm")
def register_deepseek_v41_llm_config():
    from .configuration import DeepseekV41LlmConfig

    return DeepseekV41LlmConfig


@OMNI_MODEL_REGISTRY.register("deepseek_v41_llm")
def register_deepseek_v41_llm_model():
    from .modeling import DeepseekV41Llm

    return DeepseekV41Llm


@OMNI_ACCELERATED_MODEL_REGISTRY.register("deepseek_v41_llm")
def register_deepseek_v41_llm_accelerated_model():
    from .accelerated import DeepseekV41LlmAccelerated

    return DeepseekV41LlmAccelerated
