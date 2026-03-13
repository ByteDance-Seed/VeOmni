"""Dummy data generators for VeOmni tests.

Provides ``get_dummy_data`` which builds deterministic input tensors in the
three attention layouts used by VeOmni: ``padded_bsh``, ``cu_seqlens``, and
``position_ids``.
"""

from typing import Dict

import torch
from transformers import set_seed


def get_dummy_data(
    layout: str = "padded_bsh",
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 1000,
    num_labels: int = -1,
    seed: int = 42,
    dtype: torch.dtype = torch.long,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Generate dummy input data for a specific attention layout.

    Parameters
    ----------
    layout : str
        One of ``"padded_bsh"``, ``"cu_seqlens"``, or ``"position_ids"``.
    batch_size : int
        Number of sequences.
    seq_len : int
        Maximum sequence length.
    vocab_size : int
        Range for random token IDs.
    num_labels : int
        If > 0, generate classification labels instead of LM labels.
    seed : int
        Random seed for reproducibility.
    dtype : torch.dtype
        Data type for token tensors.
    device : str
        Target device.

    Returns
    -------
    dict
        Dictionary with keys like ``input_ids``, ``attention_mask``,
        ``labels``, and layout-specific keys (``cu_seqlens_q``,
        ``position_ids``, etc.).
    """
    set_seed(seed)
    ignore_index = -100

    if layout == "padded_bsh":
        return _padded_bsh(batch_size, seq_len, vocab_size, num_labels, ignore_index, dtype, device)
    elif layout == "cu_seqlens":
        return _cu_seqlens(batch_size, seq_len, vocab_size, num_labels, ignore_index, dtype, device)
    elif layout == "position_ids":
        return _position_ids(batch_size, seq_len, vocab_size, num_labels, ignore_index, dtype, device)
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Choose from padded_bsh, cu_seqlens, position_ids.")


def _padded_bsh(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_labels: int,
    ignore_index: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Standard padded batch layout: (batch, seq_len)."""
    actual_lens = [torch.randint(seq_len // 2, seq_len + 1, (1,)).item() for _ in range(batch_size)]

    input_ids = torch.full((batch_size, seq_len), 0, dtype=dtype, device=device)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=dtype, device=device)
    labels = torch.full((batch_size, seq_len), ignore_index, dtype=dtype, device=device)

    for i, length in enumerate(actual_lens):
        input_ids[i, :length] = torch.randint(1, vocab_size, (length,), dtype=dtype)
        attention_mask[i, :length] = 1
        if num_labels > 0:
            labels[i, length - 1] = torch.randint(0, num_labels, (1,)).item()
        else:
            labels[i, :length] = torch.randint(0, vocab_size, (length,), dtype=dtype)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _cu_seqlens(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_labels: int,
    ignore_index: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Packed layout with cumulative sequence lengths."""
    actual_lens = [torch.randint(seq_len // 2, seq_len + 1, (1,)).item() for _ in range(batch_size)]
    total_len = sum(actual_lens)

    input_ids = torch.randint(1, vocab_size, (1, total_len), dtype=dtype, device=device)
    attention_mask = torch.ones(1, total_len, dtype=dtype, device=device)
    labels = torch.randint(0, vocab_size, (1, total_len), dtype=dtype, device=device)

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, length in enumerate(actual_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
    }


def _position_ids(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_labels: int,
    ignore_index: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Packed layout with explicit position IDs (used by VLMs like Qwen2-VL)."""
    actual_lens = [torch.randint(seq_len // 2, seq_len + 1, (1,)).item() for _ in range(batch_size)]
    total_len = sum(actual_lens)

    input_ids = torch.randint(1, vocab_size, (1, total_len), dtype=dtype, device=device)
    attention_mask = torch.ones(1, total_len, dtype=dtype, device=device)
    labels = torch.randint(0, vocab_size, (1, total_len), dtype=dtype, device=device)

    position_ids = torch.zeros(1, total_len, dtype=dtype, device=device)
    offset = 0
    for length in actual_lens:
        position_ids[0, offset : offset + length] = torch.arange(length, dtype=dtype)
        offset += length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
    }


def pos2culen(position_ids: torch.Tensor) -> torch.Tensor:
    """Convert packed position IDs to cumulative sequence lengths.

    Parameters
    ----------
    position_ids : torch.Tensor
        1-D or 2-D position ID tensor where sequences restart at 0.

    Returns
    -------
    torch.Tensor
        Cumulative sequence lengths (int32).
    """
    pos = position_ids.view(-1)
    boundaries = (pos[1:] < pos[:-1]).nonzero(as_tuple=False).view(-1) + 1
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=pos.device),
            boundaries.to(torch.int32),
            torch.tensor([pos.numel()], dtype=torch.int32, device=pos.device),
        ]
    )
    return cu_seqlens
