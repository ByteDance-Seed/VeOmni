"""VeOmni testing utilities.

Reusable test helpers for comparing tensors and cluster setup.
These utilities are designed to be importable by downstream projects
via ``import veomni.testing``.
"""

from .cluster_utils import find_free_port
from .comparison_utils import (
    TensorComparator,
    assert_close,
    assert_exact,
    compare_metrics,
    print_comparison_table,
)


__all__ = [
    "TensorComparator",
    "assert_close",
    "assert_exact",
    "compare_metrics",
    "find_free_port",
    "print_comparison_table",
]
