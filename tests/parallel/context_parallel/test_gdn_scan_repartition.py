"""CP-A gate: zigzag↔contiguous repartition round-trip (bit-exact).

Must pass before wiring GDN kernels (implementation contract).
"""

from __future__ import annotations

import pytest
import torch

from veomni.distributed.context_parallel.gdn_scan_cp import (
    contiguous_block_chunk_indices,
    gdn_replication_factor_for_impl,
    normalize_gdn_cp_impl,
    simulate_block_to_zigzag,
    simulate_zigzag_to_block,
    zigzag_half_destination,
    zigzag_owner_of_chunk,
)
from veomni.distributed.context_parallel.sharding import balanced_cp_slice


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_simulate_repartition_round_trip_bit_exact(cp_size: int, seq_len: int):
    assert seq_len % (2 * cp_size) == 0
    # Distinct token ids so permutation errors cannot cancel.
    full = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len, 1)
    zigzag = [balanced_cp_slice(full, cp_size=cp_size, cp_rank=r, dim=1) for r in range(cp_size)]
    blocks = simulate_zigzag_to_block(zigzag, cp_size=cp_size, seq_dim=1)

    local = seq_len // cp_size
    for r, block in enumerate(blocks):
        assert block.shape[1] == local
        expected = full[:, r * local : (r + 1) * local, :]
        assert torch.equal(block, expected), f"cp={cp_size} rank={r} contig mismatch"

    zigzag_back = simulate_block_to_zigzag(blocks, cp_size=cp_size, seq_dim=1)
    for r in range(cp_size):
        assert torch.equal(zigzag_back[r], zigzag[r]), f"cp={cp_size} rank={r} zigzag mismatch"


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_chunk_routing_consistency(cp_size: int):
    for chunk_idx in range(2 * cp_size):
        z_rank, half = zigzag_owner_of_chunk(chunk_idx, cp_size)
        dst = zigzag_half_destination(z_rank, half, cp_size)
        assert dst == chunk_idx // 2
    for c_rank in range(cp_size):
        i0, i1 = contiguous_block_chunk_indices(c_rank, cp_size)
        assert i0 == 2 * c_rank and i1 == 2 * c_rank + 1


def test_impl_normalize_and_replication_factor():
    assert normalize_gdn_cp_impl("serial") == "state_passing_serial"
    assert normalize_gdn_cp_impl("gather_full") == "gather_full_replicated"
    assert gdn_replication_factor_for_impl("gather_full_replicated", cp_size=4) == 4
    assert gdn_replication_factor_for_impl("state_passing_serial", cp_size=4) == 1
    assert gdn_replication_factor_for_impl("kcp", cp_size=4) == 1


@pytest.mark.parametrize("cp_size", [2, 4])
def test_repartition_preserves_numel_and_never_s_full(cp_size: int):
    seq_len = 128
    full = torch.arange(seq_len).view(2, seq_len, 3, 5)
    zigzag = [balanced_cp_slice(full, cp_size=cp_size, cp_rank=r, dim=1) for r in range(cp_size)]
    blocks = simulate_zigzag_to_block(zigzag, cp_size=cp_size, seq_dim=1)
    for block in blocks:
        assert block.shape[1] == seq_len // cp_size
        assert block.shape[1] * cp_size == seq_len
        assert block.numel() == zigzag[0].numel()
