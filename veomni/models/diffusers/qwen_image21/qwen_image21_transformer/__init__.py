from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_config():
    from .configuration_qwen_image21_transformer import QwenImage21Transformer2DModelConfig

    return QwenImage21Transformer2DModelConfig


@MODELING_REGISTRY.register("QwenImage21Transformer2DModel")
def register_qwen_image21_transformer_modeling(architecture: str = None):
    from .modeling_qwen_image21_transformer import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel
