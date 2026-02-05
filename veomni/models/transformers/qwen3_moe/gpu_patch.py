import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe

from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def apply_veomni_qwen3_moe_gpu_patch():
    # ================================================================
    # PATCH: apply_rotary_pos_emb, Qwen3MoeRMSNorm, Qwen3MoeMLP
    # 1. Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available():
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen3_moe.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_qwen3_moe.Qwen3MoeRMSNorm = LigerRMSNorm
        hf_qwen3_moe.Qwen3MoeMLP = LigerSwiGLUMLP

        logger.info_rank0("Apply liger kernel to qwen3_moe.")
