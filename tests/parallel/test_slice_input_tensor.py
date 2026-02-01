import warnings

import torch

from veomni.distributed.sequence_parallel.data import slice_input_tensor


class TestSliceInputTensor:
    """Unit tests for slice_input_tensor function."""

    def test_no_group_returns_input_unchanged(self):
        """When group is None and no unified group exists, input should be returned unchanged."""
        x = torch.randn(2, 8, 4)
        result = slice_input_tensor(x, dim=1, padding=False, group=None)
        assert torch.equal(result, x)

    def test_no_deprecation_warning(self):
        """Ensure no DeprecationWarning is raised for multidimensional indexing."""
        x = torch.randn(2, 8, 4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = slice_input_tensor(x, dim=1, padding=False, group=None)
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, (
                f"DeprecationWarning raised: {[str(warning.message) for warning in deprecation_warnings]}"
            )

    def test_returns_tensor(self):
        """Result should be a tensor."""
        x = torch.randn(2, 8, 4)
        result = slice_input_tensor(x, dim=1, padding=False, group=None)
        assert isinstance(result, torch.Tensor)

    def test_contiguous_output(self):
        """Result should be contiguous."""
        x = torch.randn(2, 8, 4)
        result = slice_input_tensor(x, dim=1, padding=False, group=None)
        assert result.is_contiguous()

    def test_various_dimensions(self):
        """Test slicing along different dimensions returns unchanged tensor when no group."""
        for dim in [0, 1, 2]:
            x = torch.randn(4, 8, 16)
            result = slice_input_tensor(x, dim=dim, padding=False, group=None)
            assert torch.equal(result, x), f"Failed for dim={dim}"

    def test_negative_dimension(self):
        """Test slicing with negative dimension index."""
        x = torch.randn(2, 8, 4)
        result = slice_input_tensor(x, dim=-1, padding=False, group=None)
        assert torch.equal(result, x)

    def test_with_padding_flag(self):
        """Test with padding=True (should still return unchanged when no group)."""
        x = torch.randn(2, 8, 4)
        result = slice_input_tensor(x, dim=1, padding=True, padding_value=0, group=None)
        assert torch.equal(result, x)
