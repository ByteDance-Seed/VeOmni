from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("HunyuanVideo15Model")
def register_model_config():
    from .configuration_hunyuanvideo15 import HunyuanVideo15Config

    return HunyuanVideo15Config


@MODELING_REGISTRY.register("HunyuanVideo15Model")
def register_model(architecture=None):
    from .modeling_hunyuanvideo15 import HunyuanVideo15Model

    return HunyuanVideo15Model


@MODEL_CONFIG_REGISTRY.register("HunyuanVideo15ConditionModel")
def register_condition_config():
    from .configuration_hunyuanvideo15 import HunyuanVideo15ConditionConfig

    return HunyuanVideo15ConditionConfig


@MODELING_REGISTRY.register("HunyuanVideo15ConditionModel")
def register_condition(architecture=None):
    from .modeling_hunyuanvideo15_condition import HunyuanVideo15ConditionModel

    return HunyuanVideo15ConditionModel
