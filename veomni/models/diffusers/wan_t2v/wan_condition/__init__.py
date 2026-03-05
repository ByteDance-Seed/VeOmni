from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("wan_condition")
def register_wan_condition_config():
    from .configuration_wan_condition import WanConditionConfig

    return WanConditionConfig


@MODELING_REGISTRY.register("wan_condition")
def register_wan_condition_modeling(architecture: str):
    from .modeling_wan_condition import WanConditionModel

    return WanConditionModel
