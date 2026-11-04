"""DeepSeek-V4.1 vision tower OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("deepseek_v41_vision")
def register_deepseek_v41_vision_config():
    from .configuration import DeepseekV41VisionEncoderConfig

    return DeepseekV41VisionEncoderConfig


@OMNI_MODEL_REGISTRY.register("deepseek_v41_vision")
def register_deepseek_v41_vision_model():
    from .modeling import DeepseekV41VisionEncoder

    return DeepseekV41VisionEncoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("deepseek_v41_vision")
def register_deepseek_v41_vision_accelerated_model():
    from .accelerated import DeepseekV41VisionEncoderAccelerated

    return DeepseekV41VisionEncoderAccelerated
