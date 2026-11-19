"""Unit tests for the sequence-parallel shard-handoff primitives.

These are pure tensor/arithmetic helpers over ``conversation_list``, so they are
tested without a mesh: the window arithmetic (:func:`sp_local_len`,
:func:`sp_valid_in_window`, :func:`split_sp_sharded_tensor`) and the varlen row
metadata (:func:`sp_varlen_row`) are what every module's ``post_forward`` /
``forward_pre`` relies on, and a mistake there is a silent wrong-length row or a
kernel launched with the wrong maximum.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.models.seed_omni.modules.base import sp_shard
from veomni.models.seed_omni.modules.base.sp_shard import (
    sp_local_len,
    sp_valid_in_window,
    split_sp_sharded_tensor,
)


@pytest.mark.parametrize(
    ("valid_len", "sp_size", "expected"),
    [
        (8, 1, [8]),
        (8, 2, [4, 4]),
        (8, 4, [2, 2, 2, 2]),
        # 7 over 2 ranks pads to 8: the last rank owns 1 real row and 3 pad rows.
        (7, 2, [4, 3]),
        # 7 over 4 ranks pads to 8: window 2, real rows 2/2/2/1.
        (7, 4, [2, 2, 2, 1]),
        # Shorter than the group: only the first ranks are inside the data.
        (2, 4, [1, 1, 0, 0]),
    ],
)
def test_window_arithmetic(valid_len: int, sp_size: int, expected: list[int]) -> None:
    """``sp_local_len`` is the padded window; the real rows shrink in the last window."""
    window = sp_local_len(valid_len, sp_size)
    assert window == (valid_len + (-valid_len % sp_size)) // sp_size
    assert [sp_valid_in_window(valid_len, rank, sp_size) for rank in range(sp_size)] == expected


@pytest.mark.parametrize(
    ("lengths", "sp_rank", "sp_size"),
    [
        ((5, 3), 0, 2),
        ((5, 3), 1, 2),
        ((7,), 0, 2),
        ((7,), 1, 2),
        ((2, 5), 0, 4),
        ((2, 5), 3, 4),
    ],
)
def test_split_keeps_only_real_rows_and_reassembles(lengths: tuple[int, ...], sp_rank: int, sp_size: int) -> None:
    """No slice carries padding, and concatenating every rank's slices rebuilds the input."""
    valid_len = sum(lengths)
    window = sp_local_len(valid_len, sp_size)
    full = torch.arange(valid_len, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)
    # This rank's window of the padded sequence, with the pad tail filled by -1.
    padded = torch.cat([full, torch.full((window * sp_size - valid_len, 2), -1.0)])
    local = padded[sp_rank * window : (sp_rank + 1) * window]

    slices = split_sp_sharded_tensor(local, lengths, sp_rank, sp_size)
    # Each slice is exactly this segment's overlap with the window, and they add up
    # to the window's real rows — no pad row reaches any item.
    assert sum(int(s.size(0)) for s in slices) == sp_valid_in_window(valid_len, sp_rank, sp_size)
    assert not any(bool((s == -1).any()) for s in slices), "a slice carried the SP pad tail"

    rebuilt = torch.cat(
        [
            chunk
            for rank in range(sp_size)
            for chunk in split_sp_sharded_tensor(padded[rank * window : (rank + 1) * window], lengths, rank, sp_size)
        ],
        dim=0,
    )
    torch.testing.assert_close(rebuilt, full)


@pytest.mark.parametrize(("sample_seqlens", "tail_padding"), [((5, 3), 1), ((5, 3), 5), ((5, 3), 7), ((2, 2), 8)])
def test_varlen_row_max_covers_the_pad_segment(
    sample_seqlens: tuple[int, ...], tail_padding: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_q``/``max_k`` must describe the longest segment in ``cu_seqlens``.

    ``cu_q`` and ``cu_k`` are the same tensor, so a single maximum has to cover
    both. The pad tail is a segment like any other: when it is longer than every
    real sample it — not ``max(sample_seqlens)`` — is the longest, and
    ``flash_attn_varlen_func`` launches from ``max_seqlen_q``.
    """
    valid_len = sum(sample_seqlens)
    padded_len = valid_len + tail_padding
    # ``sp_pad`` pads to a multiple of sp_size; drive the helper through a stub
    # parallel state so the arithmetic under test is the only thing running.
    sp_size = padded_len  # one row per rank keeps sp_pad/slice_input_tensor trivial
    monkeypatch.setattr(sp_shard, "get_parallel_state", lambda: SimpleNamespace(sp_size=sp_size, sp_rank=0))
    monkeypatch.setattr(sp_shard, "sp_pad", lambda t, dim, pad_value: t)
    monkeypatch.setattr(sp_shard, "slice_input_tensor", lambda t, dim, padding, group: t)

    pids = torch.cat([torch.arange(length) for length in sample_seqlens] + [torch.zeros(tail_padding)])
    mask = torch.ones(padded_len, dtype=torch.long)

    _, _, (cu_q, cu_k, max_q, max_k) = sp_shard.sp_varlen_row(
        pids.unsqueeze(0), mask.unsqueeze(0), valid_len, pos_dim=1, group=None, sample_seqlens=sample_seqlens
    )

    true_max = int((cu_q[1:] - cu_q[:-1]).max())
    assert true_max == max(max(sample_seqlens), tail_padding)
    assert max_q == max_k == true_max, (sample_seqlens, tail_padding, cu_q.tolist(), max_q, max_k)
    # The tail is one coalesced segment, not one per padding token.
    assert cu_q[-1] == padded_len
    assert len(cu_q) <= len(sample_seqlens) + 2


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_varlen_row_pads_and_coalesces_with_a_stub_group(monkeypatch: pytest.MonkeyPatch):
    """The unmocked call pads the row, narrows the positions and coalesces the tail.

    ``slice_input_tensor`` is stubbed out because narrowing to a rank's window is
    a collective, and this test is about the arithmetic around it: the padded row
    keeps full length for the attention all-to-all, the positions are windowed, and
    the pad tail becomes exactly one segment.
    """
    # 5 real rows over 2 ranks pads to 6: window 3, so this rank keeps 3 positions.
    monkeypatch.setattr(sp_shard, "get_parallel_state", lambda: SimpleNamespace(sp_size=2, sp_rank=0))
    monkeypatch.setattr(sp_shard, "slice_input_tensor", lambda t, dim, padding, group: t[..., :3])

    def _pad_to_window(t: torch.Tensor, dim: int, pad_value: int) -> torch.Tensor:
        pad = (-t.size(dim)) % 2
        if not pad:
            return t
        shape = list(t.shape)
        shape[dim] = pad
        return torch.cat([t, torch.full(shape, pad_value, dtype=t.dtype)], dim=dim)

    monkeypatch.setattr(sp_shard, "sp_pad", _pad_to_window)

    pids = torch.cat([torch.arange(3), torch.arange(2)]).unsqueeze(0)
    mask = torch.ones(1, 5, dtype=torch.long)
    out_mask, pids_window, (cu_q, cu_k, max_q, max_k) = sp_shard.sp_varlen_row(
        pids, mask, 5, pos_dim=1, group=None, sample_seqlens=[3, 2]
    )
    assert pids_window.size(1) == sp_local_len(5, 2) == 3
    assert out_mask.size(1) == 6, "the mask stays full-length for the attention all-to-all"
    assert out_mask.tolist() == [[1, 1, 1, 1, 1, 1]], "pad rows must be attendable (mask value 1)"
    assert cu_q.tolist() == [0, 3, 5, 6], "real samples then exactly one coalesced pad segment"
    assert cu_q is cu_k
    assert max_q == max_k == 3
