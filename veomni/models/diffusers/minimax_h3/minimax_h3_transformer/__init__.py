from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("MiniMaxH3DiTModel")
def register_minimax_h3_transformer_config():
    from .configuration_minimax_h3_transformer import MiniMaxH3DiTModelConfig

    return MiniMaxH3DiTModelConfig


@MODELING_REGISTRY.register("MiniMaxH3DiTModel")
def register_minimax_h3_transformer_modeling(architecture: str = None):
    from .modeling_minimax_h3_transformer import MiniMaxH3DiTModel

    return MiniMaxH3DiTModel
