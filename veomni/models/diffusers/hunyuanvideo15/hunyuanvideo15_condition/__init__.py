"""Register the local HunyuanVideo 1.5 condition config and model."""

from ....registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("HunyuanVideo15ConditionModel")
def register_condition_config():
    from .configuration_hunyuanvideo15_condition import HunyuanVideo15ConditionConfig

    return HunyuanVideo15ConditionConfig


@MODELING_REGISTRY.register("HunyuanVideo15ConditionModel")
def register_condition(architecture=None):
    from .modeling_hunyuanvideo15_condition import HunyuanVideo15ConditionModel

    return HunyuanVideo15ConditionModel
