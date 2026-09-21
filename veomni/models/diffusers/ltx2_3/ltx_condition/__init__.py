from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("LTXVideoConditionModel")
def register_ltx_condition_config():
    from .configuration_ltx2_3_condition import LTXVideoConditionModelConfig

    return LTXVideoConditionModelConfig


@MODELING_REGISTRY.register("LTXVideoConditionModel")
def register_ltx_condition_modeling(architecture: str = None):
    from .modeling_ltx2_3_condition import LTXVideoConditionModel

    return LTXVideoConditionModel
