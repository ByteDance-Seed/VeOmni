import transformers.models.qwen2_vl.modeling_qwen2_vl as hf_qwen2_vl

from ....ops.ops_config import get_ops_config
from ....utils import logging


logger = logging.get_logger(__name__)


def apply_veomni_qwen2vl_gpu_patch():
    ops_config = get_ops_config()
    if ops_config is None:
        return

    applied = []

    if ops_config.rotary_pos_emb_implementation == "liger_kernel":
        from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb

        hf_qwen2_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
        applied.append("M-RoPE")

    if ops_config.rms_norm_implementation == "liger_kernel":
        from liger_kernel.transformers.layer_norm import LigerLayerNorm
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        hf_qwen2_vl.Qwen2RMSNorm = LigerRMSNorm
        hf_qwen2_vl.LayerNorm = LigerLayerNorm
        applied.append("RMSNorm+LayerNorm")

    if ops_config.swiglu_mlp_implementation == "liger_kernel":
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP
        applied.append("SwiGLU")

    if applied:
        logger.info_rank0(f"Apply liger kernel to Qwen2-VL: {', '.join(applied)}.")
