import torch_npu
import transformers.models.qwen3.modeling_qwen3 as hf_qwen3

from ....ops.npu_patch import npu_fused_operator
from ....utils import logging
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_qwen3_npu_patch():
    # Patches for Qwen3 Model
    hf_qwen3.apply_rotary_pos_emb = npu_fused_operator.apply_rotary_pos_emb_npu
    hf_qwen3.Qwen3RMSNorm.forward = npu_fused_operator.rms_norm_forward_npu

    # ================================================================
    # PATCH: apply_rotary_pos_emb, Qwen3RMSNorm, Qwen3MLP
    # Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available():
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_qwen3.Qwen3RMSNorm = LigerRMSNorm
        hf_qwen3.Qwen3MLP = LigerSwiGLUMLP

        logger.info_rank0("Apply liger kernel to Qwen3.")
