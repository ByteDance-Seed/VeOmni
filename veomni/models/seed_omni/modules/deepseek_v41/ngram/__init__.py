"""DeepSeek-V4.1 n-gram memory lookup OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("deepseek_v41_ngram")
def register_deepseek_v41_ngram_config():
    from .configuration import DeepseekV41NgramConfig

    return DeepseekV41NgramConfig


@OMNI_MODEL_REGISTRY.register("deepseek_v41_ngram")
def register_deepseek_v41_ngram_model():
    from .modeling import DeepseekV41Ngram

    return DeepseekV41Ngram


@OMNI_ACCELERATED_MODEL_REGISTRY.register("deepseek_v41_ngram")
def register_deepseek_v41_ngram_accelerated_model():
    from .accelerated import DeepseekV41NgramAccelerated

    return DeepseekV41NgramAccelerated
