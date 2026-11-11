"""Register the local HunyuanVideo 1.5 transformer config and model."""

from ....registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("HunyuanVideo15Model")
def register_model_config():
    from .configuration_hunyuanvideo15_transformer import HunyuanVideo15Config

    return HunyuanVideo15Config


@MODELING_REGISTRY.register("HunyuanVideo15Model")
def register_model(architecture=None):
    from .modeling_hunyuanvideo15_transformer import HunyuanVideo15Model

    return HunyuanVideo15Model
