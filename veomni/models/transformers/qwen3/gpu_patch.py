import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def apply_veomni_qwen3_gpu_patch():
    # ================================================================
    # PATCH: apply_rotary_pos_emb, Qwen3RMSNorm, Qwen3MLP
    # 1. Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available():
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_qwen3.Qwen3RMSNorm = LigerRMSNorm
        hf_qwen3.Qwen3MLP = LigerSwiGLUMLP

        logger.info_rank0("Apply liger kernel to Qwen3.")
