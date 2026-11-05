"""Register the local Cosmos3 condition config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Cosmos3ConditionModel")
def register_cosmos3_condition_config():
    from .configuration_cosmos3_condition import Cosmos3ConditionModelConfig

    return Cosmos3ConditionModelConfig


@MODELING_REGISTRY.register("Cosmos3ConditionModel")
def register_cosmos3_condition_modeling(_architecture: str | None = None):
    from .modeling_cosmos3_condition import Cosmos3ConditionModel

    return Cosmos3ConditionModel
