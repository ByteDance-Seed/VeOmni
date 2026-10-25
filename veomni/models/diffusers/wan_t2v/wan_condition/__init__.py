"""Register the local Wan T2V condition config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("WanTransformer3DConditionModel")
def register_wan_condition_config():
    from .configuration_wan_condition import WanTransformer3DConditionModelConfig

    return WanTransformer3DConditionModelConfig


@MODELING_REGISTRY.register("WanTransformer3DConditionModel")
def register_wan_condition_modeling(_architecture: str | None = None):
    from .modeling_wan_condition import WanTransformer3DConditionModel

    return WanTransformer3DConditionModel
