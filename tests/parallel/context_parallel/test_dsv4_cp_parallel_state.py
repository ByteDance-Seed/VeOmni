"""Context parallelism is admitted, but only without Ulysses."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from veomni.distributed.parallel_state import ParallelState


def test_context_parallel_alone_is_accepted(monkeypatch):
    # ParallelState.world_size reads torch.distributed directly; simulate a
    # 2-rank world so the world-size product check (cp_size=2, everything
    # else 1) is satisfied without a real process group. sp_enabled is True
    # here (cp_size > 1), which requires a non-None device_mesh -- a mock
    # stands in since nothing here touches the mesh itself.
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 2)
    state = ParallelState(dp_size=1, cp_size=2, ulysses_size=1, device_type="cpu", device_mesh=MagicMock())
    assert state.cp_size == 2
    assert state.cp_enabled


def test_context_parallel_is_disabled_at_size_one():
    state = ParallelState(dp_size=1, cp_size=1, ulysses_size=1, device_type="cpu")
    assert not state.cp_enabled


@pytest.mark.parametrize("cp_size", [0, -1], ids=["zero", "negative"])
def test_non_positive_context_parallel_size_is_rejected(monkeypatch, cp_size):
    """A negative cp_size survives the world-size product check by cancelling.

    ``dp_size=-1`` with ``cp_size=-1`` multiplies out to +1, so the product check
    would admit the topology and ``cp_enabled`` would then report False -- CP
    silently off, on a state that looks validated. ``dp_shard_size=-1`` is set for
    the same reason: it keeps the dp_replicate * dp_shard == dp_size check from
    being the thing that fires, so the rejection is attributable to the new guard.
    """
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 1)
    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        ParallelState(
            dp_size=-1,
            cp_size=cp_size,
            ulysses_size=1,
            dp_replicate_size=1,
            dp_shard_size=-1,
            device_type="cpu",
            device_mesh=MagicMock(),
        )


def test_hybrid_context_and_ulysses_is_rejected(monkeypatch):
    # cp_size=2 * ulysses_size=2 * (dp/tp/pp=1) == world_size=4, so the
    # rejection below is attributable to the new CP+Ulysses guard rather
    # than the pre-existing world-size product ValueError.
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.is_initialized", lambda: True)
    monkeypatch.setattr("veomni.distributed.parallel_state.dist.get_world_size", lambda: 4)
    with pytest.raises(NotImplementedError, match="ulysses_size"):
        ParallelState(dp_size=1, cp_size=2, ulysses_size=2, device_type="cpu", device_mesh=MagicMock())
