"""Integer-only QSA layout reuse must preserve packed samples and owner order."""

import pytest
import torch

from veomni.models.transformers.qwen4_exp.qsa_metadata import cached_host_metadata


@pytest.mark.parametrize("segments", [((0, 0, 16),), ((0, 0, 3), (0, 3, 16)), ((0, 0, 16), (1, 0, 16))])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_cached_layout_matches_reference(segments, world):
    """All ranks retain the exact uncoalesced owner-major metadata contract."""
    local = 16 // world
    for rank in range(world):
        blocks, owned, gathered, sections = cached_host_metadata(segments, 4, local, rank, world)
        reference = [(i, b, x) for i, (b, start, end) in enumerate(segments) for x in range(start, end - 3, 4)]
        assert blocks == tuple(reference)
        assert owned == tuple(x for x in reference if x[2] // local == rank)
        assert gathered == tuple(x for owner in range(world) for x in reference if x[2] // local == owner)
        assert sections == (
            tuple(sum(x[2] // local == owner for x in reference) for owner in range(world)),
            tuple(x[1] for x in owned),
            tuple(x[2] for x in owned),
            tuple(x[0] for x in gathered),
            tuple(x[2] for x in gathered),
        )


def test_cache_is_bounded_and_packing_sensitive():
    """Changing boundaries or ranks must not reuse a previous sample's layout."""
    cached_host_metadata.cache_clear()
    for end in range(5, 16):
        result = cached_host_metadata(((0, 0, end),), 4, 16, 0, 1)
        assert not any(isinstance(value, torch.Tensor) for section in result[-1] for value in section)
    assert cached_host_metadata.cache_info().currsize == 2
    first = cached_host_metadata(((0, 0, 16),), 4, 8, 0, 2)
    assert first is cached_host_metadata(((0, 0, 16),), 4, 8, 0, 2)
    assert first != cached_host_metadata(((0, 0, 3), (0, 3, 16)), 4, 8, 0, 2)
    assert first != cached_host_metadata(((0, 0, 16),), 4, 8, 1, 2)
