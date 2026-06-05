from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("QwenImageConditionModel")
def register_qwen_image_condition_config():
    from .configuration_qwen_image_condition import QwenImageConditionModelConfig

    return QwenImageConditionModelConfig


@MODELING_REGISTRY.register("QwenImageConditionModel")
def register_qwen_image_condition_modeling(architecture: str = None):
    from .modeling_qwen_image_condition import QwenImageConditionModel

    return QwenImageConditionModel
