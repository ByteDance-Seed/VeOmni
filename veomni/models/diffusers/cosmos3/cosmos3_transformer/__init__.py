"""Register the local Cosmos3 DiT config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Cosmos3MoTTransformer")
def register_cosmos3_transformer_config():
    from .configuration_cosmos3_transformer import Cosmos3MoTTransformerConfig

    return Cosmos3MoTTransformerConfig


@MODELING_REGISTRY.register("Cosmos3MoTTransformer")
def register_cosmos3_transformer_modeling(_architecture: str | None = None):
    from .modeling_cosmos3_transformer import Cosmos3MoTTransformer

    return Cosmos3MoTTransformer
