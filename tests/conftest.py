"""Top-level conftest for VeOmni tests.

Registers pytest markers and provides shared fixtures/helpers.
"""

import pytest

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to


def pytest_configure(config):
    config.addinivalue_line("markers", "L0: Level 0 - Data processing & transform correctness (no GPU)")
    config.addinivalue_line("markers", "L1: Level 1 - Kernel & operator consistency (single GPU)")
    config.addinivalue_line("markers", "L2: Level 2 - Single-GPU vs FSDP equivalence (2+ GPUs)")
    config.addinivalue_line("markers", "L3: Level 3 - Parallelism combination consistency (2-8 GPUs)")
    config.addinivalue_line("markers", "L4: Level 4 - Corner cases & robustness")
    config.addinivalue_line("markers", "L5: Level 5 - Integration / smoke test")
    config.addinivalue_line("markers", "v4_only: Requires transformers < 5.0.0")
    config.addinivalue_line("markers", "v5_only: Requires transformers >= 5.0.0")
    config.addinivalue_line("markers", "multi_gpu: Requires multiple GPUs")
    config.addinivalue_line("markers", "moe: MoE model specific test")


# Version skip helpers
_is_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
v4_only = pytest.mark.skipif(_is_v5, reason="Not compatible with transformers >= 5.0.0")
v5_only = pytest.mark.skipif(not _is_v5, reason="Requires transformers >= 5.0.0")
