# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Packed-sample boundaries for PLE history and its causal short convolution."""

import torch
import torch.nn.functional as F


def local_packed_positions(
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    local_length: int,
    sp_rank: int = 0,
    sp_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return each local token's distance from its own sample's first token.

    Cumulative lengths describe the flattened global batch, before contiguous
    SP slicing. They must retain every batch-row boundary. Rotary positions
    are intentionally not used: multimodal RoPE need not count token offsets.
    """
    if cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2:
        raise ValueError("Packed PLE requires one-dimensional cumulative sequence lengths.")
    if cu_seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Packed PLE cumulative lengths must be integers.")
    if batch_size < 1 or local_length < 1 or not 0 <= sp_rank < sp_size:
        raise ValueError("Invalid packed PLE local shape or SP rank.")
    global_length = local_length * sp_size
    boundaries = cu_seq_lens.to(device=device, dtype=torch.long)
    starts, ends = boundaries[:-1], boundaries[1:]
    valid = (boundaries[0] == 0) & (boundaries[-1] == batch_size * global_length)
    valid = valid & (ends > starts).all()
    valid = (
        valid
        & (
            torch.div(starts, global_length, rounding_mode="floor")
            == torch.div(ends - 1, global_length, rounding_mode="floor")
        ).all()
    )
    torch._assert_async(valid, "Packed PLE segments must cover the global batch without crossing rows.")
    positions = (
        torch.arange(local_length, device=boundaries.device)[None, :]
        + torch.arange(batch_size, device=boundaries.device)[:, None] * global_length
        + sp_rank * local_length
    )
    segments = torch.searchsorted(boundaries[1:].contiguous(), positions.contiguous(), right=True)
    return positions - boundaries[segments]


def local_packed_boundary_indices(
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    local_length: int,
    reach: int,
    sp_rank: int = 0,
    sp_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build fixed-size boundary indices without a device-to-host nonzero.

    Invalid candidates get distinct disposable output rows. This makes the
    eventual overwrite deterministic even for short samples or nonlocal
    boundaries, and their discarded rows cannot contribute a gradient.
    """
    bounds = cu_seq_lens.to(device=device, dtype=torch.long)
    global_length = local_length * sp_size
    candidates = bounds[:-1, None] + torch.arange(reach, device=bounds.device)[None, :]
    batch = torch.div(candidates, global_length, rounding_mode="floor")
    within_row = candidates.remainder(global_length)
    local = within_row - sp_rank * local_length
    valid = (candidates < bounds[1:, None]) & (local >= 0) & (local < local_length)
    flat = batch * local_length + local
    sinks = batch_size * local_length + torch.arange(candidates.numel(), device=bounds.device).view_as(candidates)
    return torch.where(valid, flat, sinks).flatten()


def correct_packed_ple_convolution(
    conv_input: torch.Tensor,
    conv_output: torch.Tensor,
    conv: torch.nn.Conv1d,
    packed_positions: torch.Tensor,
    boundary_indices: torch.Tensor,
) -> torch.Tensor:
    """Recompute only boundary outputs, excluding taps from previous samples.

    Keep the original convolution for interior tokens. The input already has
    the ordinary causal padding or differentiable SP halo; the new metadata
    only suppresses out-of-sample taps. This needs no additional collective.
    """
    batch_size, channels, length = conv_output.shape
    kernel = conv.kernel_size[0]
    dilation = conv.dilation[0]
    reach = (kernel - 1) * dilation
    if conv.stride != (1,) or conv.padding != (0,):
        raise ValueError("Packed PLE correction requires stride one and no implicit padding.")
    if packed_positions.shape != (batch_size, length) or conv_input.shape != (
        batch_size,
        channels,
        length + reach,
    ):
        raise ValueError("Packed PLE convolution metadata does not match its local input/output shape.")
    if reach == 0:
        return conv_output
    border = boundary_indices
    if border.numel() == 0:
        return conv_output
    in_range = border < batch_size * length
    safe_border = border.clamp_max(batch_size * length - 1)
    batch_indices = torch.div(safe_border, length, rounding_mode="floor")
    local_indices = safe_border.remainder(length)
    taps = torch.arange(kernel, device=conv_input.device)
    input_indices = local_indices[:, None] + taps[None, :] * dilation
    windows = conv_input.transpose(1, 2)[batch_indices[:, None], input_indices]
    distances = (kernel - 1 - taps) * dilation
    allowed = (packed_positions.flatten()[safe_border, None] >= distances[None, :]) & in_range[:, None]
    windows = torch.where(allowed[:, :, None], windows, 0).transpose(1, 2).contiguous()
    corrected = F.conv1d(windows, conv.weight, conv.bias, groups=conv.groups).squeeze(-1)
    flat_output = conv_output.transpose(1, 2).reshape(-1, channels)
    padded_output = F.pad(flat_output, (0, 0, 0, border.numel()))
    corrected_output = padded_output.index_copy(0, border, corrected)[: batch_size * length]
    return corrected_output.view(batch_size, length, channels).transpose(1, 2)
