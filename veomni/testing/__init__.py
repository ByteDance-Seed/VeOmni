"""VeOmni testing utilities.

Reusable test helpers for comparing tensors, generating dummy data,
and loading toy configs. These utilities are designed to be importable
by downstream projects via ``import veomni.testing``.
"""

from .comparison_utils import (
    TensorComparator,
    assert_close,
    assert_exact,
    compare_metrics,
    print_comparison_table,
)
from .data_generators import get_dummy_data
from .toy_config_utils import get_toy_config_path, validate_toy_config


__all__ = [
    "TensorComparator",
    "assert_close",
    "assert_exact",
    "compare_metrics",
    "print_comparison_table",
    "get_dummy_data",
    "get_toy_config_path",
    "validate_toy_config",
]
