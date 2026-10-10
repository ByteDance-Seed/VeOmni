from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("glm_moe_dsa")
def register_glm_moe_dsa_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_glm_moe_dsa_fqn_to_index_mapping,
        create_glm_moe_dsa_checkpoint_tensor_converter,
    )

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_glm_moe_dsa_npu import (
            GlmMoeDsaForCausalLM,
            GlmMoeDsaModel,
        )
    else:
        from .generated.patched_modeling_glm_moe_dsa_gpu import (
            GlmMoeDsaForCausalLM,
            GlmMoeDsaModel,
        )

    for model_cls in (GlmMoeDsaForCausalLM, GlmMoeDsaModel):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_glm_moe_dsa_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_glm_moe_dsa_fqn_to_index_mapping)

    if "ForCausalLM" in architecture:
        return GlmMoeDsaForCausalLM
    elif "Model" in architecture:
        return GlmMoeDsaModel
    else:
        return GlmMoeDsaForCausalLM
