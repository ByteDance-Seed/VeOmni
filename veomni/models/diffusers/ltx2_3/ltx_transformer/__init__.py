from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("LTXVideoTransformerModel")
def register_ltx_transformer_config():
    from .configuration_ltx2_3_transformer import LTXVideoTransformerModelConfig

    return LTXVideoTransformerModelConfig


@MODELING_REGISTRY.register("LTXVideoTransformerModel")
def register_ltx_transformer_modeling(architecture: str):
    from .modeling_ltx2_3_transformer import LTXVideoTransformerModel as VeOmniLTXVideoTransformerModel
    from .modeling_ltx2_3_transformer import apply_veomni_ltx_transformer_patch

    apply_veomni_ltx_transformer_patch()

    return VeOmniLTXVideoTransformerModel
