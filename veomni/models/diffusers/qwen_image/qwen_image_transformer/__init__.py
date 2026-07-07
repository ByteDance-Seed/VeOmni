from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_transformer_config():
    from .configuration_qwen_image_transformer import QwenImageTransformer2DModelConfig

    return QwenImageTransformer2DModelConfig


@MODELING_REGISTRY.register("QwenImageTransformer2DModel")
def register_qwen_image_transformer_modeling(architecture: str = None):
    from .modeling_qwen_image_transformer import QwenImageTransformer2DModel as VeOmniQwenImageTransformer2DModel
    from .modeling_qwen_image_transformer import apply_veomni_qwen_image_transformer_patch

    apply_veomni_qwen_image_transformer_patch()

    return VeOmniQwenImageTransformer2DModel
