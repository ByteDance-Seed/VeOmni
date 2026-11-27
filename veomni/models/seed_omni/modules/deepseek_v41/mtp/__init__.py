"""DeepSeek-V4.1 DSpark / MTP OmniModule."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("deepseek_v41_mtp")
def register_deepseek_v41_mtp_config():
    from .configuration import DeepseekV41MtpConfig

    return DeepseekV41MtpConfig


@OMNI_MODEL_REGISTRY.register("deepseek_v41_mtp")
def register_deepseek_v41_mtp_model():
    from .modeling import DeepseekV41Mtp

    return DeepseekV41Mtp


@OMNI_ACCELERATED_MODEL_REGISTRY.register("deepseek_v41_mtp")
def register_deepseek_v41_mtp_accelerated_model():
    from .accelerated import DeepseekV41MtpAccelerated

    return DeepseekV41MtpAccelerated
