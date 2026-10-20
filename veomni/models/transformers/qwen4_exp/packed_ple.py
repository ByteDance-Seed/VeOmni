# Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Packed-sample boundaries for PLE history and its causal short convolution."""

import torch
import torch.nn.functional as F


_BOUNDARY_CORRECTION_BUDGET_BYTES = 64 * 1024 * 1024


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
    packed_seq_lens: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Build fixed-size boundary indices without a device-to-host nonzero.

    Invalid candidates get distinct disposable output rows. This makes the
    eventual overwrite deterministic even for short samples or nonlocal
    boundaries, and their discarded rows cannot contribute a gradient.

    When the collator's host-side packed lengths are available, build only the
    local rows that actually overlap a sample boundary. This keeps correction
    work proportional to local tokens instead of every global segment, without
    synchronizing the device tensor back to the host.
    """
    global_length = local_length * sp_size
    if batch_size < 1 or local_length < 1 or reach < 0 or not 0 <= sp_rank < sp_size:
        raise ValueError("Invalid packed PLE boundary shape, reach, or SP rank.")
    if packed_seq_lens is not None:
        lengths = tuple(int(length) for length in packed_seq_lens)
        if not lengths or any(length <= 0 for length in lengths) or sum(lengths) != batch_size * global_length:
            raise ValueError("Packed PLE host lengths must be positive and cover the complete global batch.")
        local_start = sp_rank * local_length
        local_end = local_start + local_length
        indices = []
        global_start = 0
        for length in lengths:
            global_end = global_start + length
            if global_start // global_length != (global_end - 1) // global_length:
                raise ValueError("Packed PLE host lengths must not cross batch rows.")
            batch_index = global_start // global_length
            row_start = batch_index * global_length
            overlap_start = max(global_start, row_start + local_start)
            overlap_end = min(global_start + reach, global_end, row_start + local_end)
            for global_index in range(overlap_start, overlap_end):
                local_index = global_index - row_start - local_start
                indices.append(batch_index * local_length + local_index)
            global_start = global_end
        return torch.tensor(indices, dtype=torch.long, device=device)

    bounds = cu_seq_lens.to(device=device, dtype=torch.long)
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
    # Tensor-only callers may encode invalid candidates as sinks. Put those
    # writes on one disposable row: clamping them onto a real token both
    # corrupts its corrected value and double-counts its gradient.
    window_border = border.clamp_max(batch_size * length - 1)
    write_border = torch.where(in_range, border, border.new_full((), batch_size * length))
    flat_output = conv_output.transpose(1, 2).reshape(-1, channels)
    corrected_output = F.pad(flat_output, (0, 0, 0, 1))
    taps = torch.arange(kernel, device=conv_input.device)
    distances = (kernel - 1 - taps) * dilation
    # Windows dominate correction memory: batch_size x kernel x channels.
    # Bound each temporary to a fixed budget; the final output is accumulated
    # in place so a fragmented global pack cannot allocate one giant tensor.
    window_bytes = max(1, kernel * channels * conv_input.element_size())
    chunk_rows = max(1, _BOUNDARY_CORRECTION_BUDGET_BYTES // window_bytes)
    for start in range(0, border.numel(), chunk_rows):
        end = min(start + chunk_rows, border.numel())
        write_chunk = write_border[start:end]
        window_chunk = window_border[start:end]
        in_range_chunk = in_range[start:end]
        batch_indices = torch.div(window_chunk, length, rounding_mode="floor")
        local_indices = window_chunk.remainder(length)
        input_indices = local_indices[:, None] + taps[None, :] * dilation
        windows = conv_input.transpose(1, 2)[batch_indices[:, None], input_indices]
        allowed = (packed_positions.flatten()[window_chunk, None] >= distances[None, :]) & in_range_chunk[:, None]
        windows = torch.where(allowed[:, :, None], windows, 0).transpose(1, 2).contiguous()
        corrected = F.conv1d(windows, conv.weight, conv.bias, groups=conv.groups).squeeze(-1)
        replacement = torch.where(
            in_range_chunk[:, None],
            corrected,
            corrected_output.index_select(0, write_chunk),
        )
        corrected_output.index_copy_(0, write_chunk, replacement)
    return corrected_output[: batch_size * length].view(batch_size, length, channels).transpose(1, 2)
