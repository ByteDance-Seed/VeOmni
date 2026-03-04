from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODELING_REGISTRY


@MODELING_REGISTRY.register("glm_moe_dsa")
def register_glm_moe_dsa_modeling(architecture: str):
    if is_transformers_version_greater_or_equal_to("5.2.0"):
        from .generated.patched_modeling_glm_moe_dsa_npu import (
            GlmMoeDsaForCausalLM,
            GlmMoeDsaModel,
        )
    else:
        raise RuntimeError("glm_moe_dsa not available. Please make sure transformers version >= 5.2.0")

    if "ForCausalLM" in architecture:
        return GlmMoeDsaForCausalLM
    elif "Model" in architecture:
        return GlmMoeDsaModel
    else:
        return GlmMoeDsaForCausalLM
