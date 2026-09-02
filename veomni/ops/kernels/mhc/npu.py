"""VeOmni OpSlot adapters for Ascend DeepSeek-V4 mHC kernels."""

import torch
import torch.nn.functional as F

from ..deepseek_v4.npu_mhc import npu_mhc_post, npu_mhc_pre_sinkhorn


def mhc_pre_npu(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_streams.device.type != "npu" or hidden_streams.dtype != torch.bfloat16:
        raise ValueError("Ascend mHC requires NPU BF16 activations")
    return npu_mhc_pre_sinkhorn(
        hidden_streams,
        fn.float().contiguous(),
        scale.float().contiguous(),
        base.float().contiguous(),
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        norm_eps,
    )


def mhc_post_npu(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return npu_mhc_post(output, residual, post, comb)


def mhc_head_npu(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    hc_eps: float,
) -> torch.Tensor:
    # cann_ops_transformer has fused pre/post mHC operators but no final-head
    # operator. Keep the final collapse in the same FP32 math as VeOmni eager.
    flat = F.rms_norm(hidden_streams.flatten(2).float(), (hidden_streams.shape[-2] * hidden_streams.shape[-1],), eps=norm_eps)
    mixes = F.linear(flat, fn.float())
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)


__all__ = ["mhc_head_npu", "mhc_post_npu", "mhc_pre_npu"]
