# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Ascend fused MHC operator wrappers for the FSDP2 DeepSeek-V4 model."""

import torch


def _mhc_ops():
    try:
        import cann_ops_transformer
    except ImportError as exc:
        raise ImportError("cann_ops_transformer is required when use_ascend_mhc is enabled.") from exc
    return cann_ops_transformer.ops


def _view_mhc_output(
    tensor: torch.Tensor,
    batch_size: int,
    seq_length: int,
    hc_mult: int | None,
    name: str,
) -> torch.Tensor:
    if tensor.dim() == 2:
        tensor = tensor.view(batch_size, seq_length, tensor.shape[-1])
    if hc_mult is not None and tensor.dim() == 3:
        if tensor.shape[-1] != hc_mult * hc_mult:
            raise RuntimeError(f"{name} last dim must be {hc_mult * hc_mult}, got {tensor.shape[-1]}")
        tensor = tensor.view(batch_size, seq_length, hc_mult, hc_mult)
    expected_dim = 4 if hc_mult is not None else 3
    if tensor.dim() != expected_dim:
        raise RuntimeError(f"{name} must be {expected_dim}D, got {tensor.dim()}D")
    return tensor


def npu_mhc_pre_sinkhorn(
    hidden_streams: torch.Tensor,
    phi: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    num_iters: int,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused MHC projection and Sinkhorn on tensors in [B, S, ...] layout."""
    collapsed, post, comb = _mhc_ops().mhc_pre_sinkhorn(
        hidden_streams.contiguous(),
        phi,
        scale,
        base,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
    )[:3]
    batch_size, seq_length = hidden_streams.shape[:2]
    collapsed = _view_mhc_output(collapsed, batch_size, seq_length, None, "collapsed")
    post = _view_mhc_output(post, batch_size, seq_length, None, "post")
    comb = _view_mhc_output(comb, batch_size, seq_length, hc_mult, "comb")
    return post, comb, collapsed.to(hidden_streams.dtype)


def npu_mhc_post(
    block_output: torch.Tensor,
    residual_streams: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run fused MHC residual mixing on tensors in [B, S, ...] layout."""
    output = _mhc_ops().mhc_post(
        residual_streams.contiguous(),
        comb.contiguous(),
        block_output.contiguous(),
        post.contiguous(),
    )
    return output.view_as(residual_streams)
