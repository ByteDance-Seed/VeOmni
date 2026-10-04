import pytest
import torch

from veomni.distributed.sequence_parallel import async_ulysses as sp_async_ulysses
from veomni.distributed.sequence_parallel import comm as sp_comm
from veomni.distributed.sequence_parallel import data as sp_data
from veomni.distributed.sequence_parallel import utils as sp_utils


class TestSliceInputTensor:
    """Unit tests for slice_input_tensor function."""

    def test_no_group_returns_input_unchanged(self):
        """When group is None and no unified group exists, input should be returned unchanged."""
        x = torch.randn(2, 8, 4)
        result = sp_data.slice_input_tensor(x, dim=1, padding=False, group=None)
        assert torch.equal(result, x)

    @pytest.mark.parametrize("rank, expected_slice", [(0, slice(0, 3)), (1, slice(3, 6))])
    def test_slice_with_mocked_group_no_padding(self, monkeypatch, rank, expected_slice):
        """Slice into contiguous chunks when SP group is mocked and padding is disabled."""
        x = torch.arange(10).reshape(2, 5)
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: rank)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 2)

        result = sp_data.slice_input_tensor(x, dim=1, padding=False, group=None)
        assert torch.equal(result, x[:, expected_slice])
        assert result.is_contiguous()

    def test_slice_with_mocked_group_padding_value(self, monkeypatch):
        """Padding inserts the requested value for uneven splits."""
        x = torch.tensor([[1, 2, 3, 4, 5]])
        group = object()
        monkeypatch.setattr(sp_data, "get_unified_sequence_parallel_group", lambda: group)
        monkeypatch.setattr(sp_data.dist, "get_rank", lambda g: 2)
        monkeypatch.setattr(sp_data.dist, "get_world_size", lambda g: 4)

        result = sp_data.slice_input_tensor(x, dim=1, padding=True, padding_value=9, group=None)
        expected = torch.tensor([[5, 9]])
        assert torch.equal(result, expected)


class TestRemoveLastRankPadding:
    """Unit tests for remove_last_rank_padding."""

    @staticmethod
    def _shards(monkeypatch, unpad_dim_size, sp_world):
        padded = -(-unpad_dim_size // sp_world) * sp_world
        full = torch.arange(padded).reshape(1, padded)
        local_len = padded // sp_world
        out = []
        for rank in range(sp_world):
            monkeypatch.setattr(sp_utils, "get_ulysses_sequence_parallel_rank", lambda g, r=rank: r)
            local = full[:, rank * local_len : (rank + 1) * local_len]
            out.append(sp_utils.remove_last_rank_padding(local, dim=1, unpad_dim_size=unpad_dim_size, group=object()))
        return out

    @pytest.mark.parametrize("unpad_dim_size, sp_world", [(8, 4), (10, 4), (1, 4), (5, 4), (7, 2)])
    def test_shards_concat_to_unpadded_sequence(self, monkeypatch, unpad_dim_size, sp_world):
        shards = self._shards(monkeypatch, unpad_dim_size, sp_world)
        assert torch.equal(torch.cat(shards, dim=1), torch.arange(unpad_dim_size).reshape(1, -1))

    def test_divisible_last_rank_keeps_data(self, monkeypatch):
        # Previously the last rank dropped sp_world real tokens when no padding existed.
        shards = self._shards(monkeypatch, unpad_dim_size=8, sp_world=4)
        assert [s.shape[1] for s in shards] == [2, 2, 2, 2]


def test_context_parallel_world_size_defaults_to_one_without_dist(monkeypatch):
    monkeypatch.setattr(sp_comm.dist, "is_initialized", lambda: False)
    assert sp_comm.get_context_parallel_world_size() == 1


def test_async_ulysses_rejects_kv_heads_not_divisible_by_ulysses(monkeypatch):
    """Async QKV must refuse kv heads > ulysses_size that do not divide evenly, like the sync path."""
    monkeypatch.setattr(sp_async_ulysses, "get_ulysses_sequence_parallel_world_size", lambda: 4)
    head_dim, hidden = 4, 16
    hidden_states = torch.randn(1, 3, hidden)
    q_w = torch.randn(8 * head_dim, hidden)
    kv_w = torch.randn(6 * head_dim, hidden)
    with pytest.raises(AssertionError, match="num_key_value_heads"):
        sp_async_ulysses.async_ulysses_qkv_projection(
            hidden_states=hidden_states,
            seq_dimension=1,
            head_dimension=2,
            q_weight=q_w,
            k_weight=kv_w,
            v_weight=kv_w,
            unpadded_dim_size=12,
            head_dim=head_dim,
            group=object(),
        )
