import transformers.models.qwen2_vl.modeling_qwen2_vl as hf_qwen2_vl

from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def apply_veomni_qwen2vl_gpu_patch():
    # ================================================================
    # PATCH: apply_rotary_pos_emb, Qwen2VLModelRMSNorm, Qwen2VLModelMLP
    # 1. Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available():
        from liger_kernel.transformers.layer_norm import LigerLayerNorm
        from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen2_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
        hf_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
        hf_qwen2_vl.LayerNorm = LigerLayerNorm
        hf_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP

        logger.info_rank0("Apply liger kernel to qwen2_vl.")
