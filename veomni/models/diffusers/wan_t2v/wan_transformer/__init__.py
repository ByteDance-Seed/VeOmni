from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("WanTransformer3DModel")
def register_wan_diffusers_transformer_config():
    from .configuration_wan_transformer import WanTransformer3DModelConfig

    return WanTransformer3DModelConfig


@MODELING_REGISTRY.register("WanTransformer3DModel")
def register_wan_diffusers_transformer_modeling(architecture: str):
    from .modeling_wan_transformer import WanTransformer3DModel

    return WanTransformer3DModel
