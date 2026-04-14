import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

from ....ops.ops_config import get_ops_config
from ....utils import logging


logger = logging.get_logger(__name__)


def apply_veomni_qwen3_gpu_patch():
    ops_config = get_ops_config()
    if ops_config is None:
        return

    applied = []

    if ops_config.rotary_pos_emb_implementation == "liger_kernel":
        from liger_kernel.transformers.rope import liger_rotary_pos_emb

        hf_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
        applied.append("RoPE")

    if ops_config.rms_norm_implementation == "liger_kernel":
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        hf_qwen3.Qwen3RMSNorm = LigerRMSNorm
        applied.append("RMSNorm")

    if ops_config.swiglu_mlp_implementation == "liger_kernel":
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen3.Qwen3MLP = LigerSwiGLUMLP
        applied.append("SwiGLU")

    if applied:
        logger.info_rank0(f"Apply liger kernel to Qwen3: {', '.join(applied)}.")
