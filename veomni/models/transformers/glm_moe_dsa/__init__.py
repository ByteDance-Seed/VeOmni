from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("glm_moe_dsa")
def register_glm_moe_dsa_modeling(architecture: str):
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

    if "ForCausalLM" in architecture:
        return GlmMoeDsaForCausalLM
    elif "Model" in architecture:
        return GlmMoeDsaModel
    else:
        return GlmMoeDsaForCausalLM
