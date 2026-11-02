from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("MiniMaxH3ConditionModel")
def register_minimax_h3_condition_config():
    from .configuration_minimax_h3_condition import MiniMaxH3ConditionModelConfig

    return MiniMaxH3ConditionModelConfig


@MODELING_REGISTRY.register("MiniMaxH3ConditionModel")
def register_minimax_h3_condition_modeling(architecture: str = None):
    from .modeling_minimax_h3_condition import MiniMaxH3ConditionModel

    return MiniMaxH3ConditionModel
