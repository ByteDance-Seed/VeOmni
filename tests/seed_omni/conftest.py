"""Apply model CPU isolation to SeedOmni module and runtime tests."""

import pytest
import torch

from tests.tools.model_fixtures import cpu_model_attention_kernel_guard as cpu_model_attention_kernel_guard


@pytest.fixture(autouse=True)
def reset_compiled_graphs_between_tests():
    """Drop traced graphs after each test so one test's cache cannot break another.

    Several SeedOmni module tests build ``torch.compile`` graphs (FlexAttention block
    masks, MoT block masks) with different shapes. Dynamo keys its cache on guards that
    can be satisfied by a later test, and inductor then compiles C++ against the wrong
    shape constants, failing with ``cur_qSplitSize1 was not declared in this scope``.
    Resetting per test keeps each compilation local to the test that triggered it.
    """
    yield
    torch._dynamo.reset()
