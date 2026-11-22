"""Register the MiniMax-H3-Image condition model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("MiniMaxH3ImageConditionModel")
def register_minimax_h3_image_condition_config():
    from .configuration_minimax_h3_image_condition import MiniMaxH3ImageConditionModelConfig

    return MiniMaxH3ImageConditionModelConfig


@MODELING_REGISTRY.register("MiniMaxH3ImageConditionModel")
def register_minimax_h3_image_condition_modeling(_architecture: str | None = None):
    from .modeling_minimax_h3_image_condition import MiniMaxH3ImageConditionModel

    return MiniMaxH3ImageConditionModel
