import pytest
import torch

from veomni.distributed.sequence_parallel import data as sp_data


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
