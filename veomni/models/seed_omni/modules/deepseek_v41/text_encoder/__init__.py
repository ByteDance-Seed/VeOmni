"""DeepSeek-V4.1 text encoder OmniModule (wte + lm_head + chat template)."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("deepseek_v41_text_encoder")
def register_deepseek_v41_text_encoder_config():
    from .configuration import DeepseekV41TextEncoderConfig

    return DeepseekV41TextEncoderConfig


@OMNI_MODEL_REGISTRY.register("deepseek_v41_text_encoder")
def register_deepseek_v41_text_encoder_model():
    from .modeling import DeepseekV41TextEncoder

    return DeepseekV41TextEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("deepseek_v41_text_encoder")
def register_deepseek_v41_text_encoder_accelerated_model():
    from .accelerated import DeepseekV41TextEncoderAccelerated

    return DeepseekV41TextEncoderAccelerated
