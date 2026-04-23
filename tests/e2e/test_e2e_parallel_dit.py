"""E2E parallel-alignment tests for diffusion transformers (DiT).

These tests are gated behind `pytest.mark.dit` so the GPU e2e workflow
can run them in a separate step after syncing with `--extra dit`
(diffusers). The main v4/v5 matrix skips them at collection time via
`dit_only` because diffusers is not installed there.
"""

import pytest

from ._harness import DEFAULT_ATOL, DEFAULT_RTOL, dit_only, main


pytestmark = [pytest.mark.e2e, pytest.mark.dit]


wan_dit_test_cases = [
    pytest.param(
        "wan_t2v",
        "./tests/fixtures/toy_config/wan_t2v_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        marks=dit_only,
    ),
]


@pytest.mark.parametrize("model_name, config_path, is_moe, rtol, atol", wan_dit_test_cases)
def test_wan_dit_parallel_align(
    model_name: str, config_path: str, is_moe: bool, rtol: float, atol: float, dummy_wan_t2v_dataset
):
    """Validate that WanTransformer3DModel loss and grad_norm are identical with
    and without Ulysses sequence-parallelism at equal DP sizes.
    """
    main(
        task_name="train_dit_test",
        model_name=model_name,
        config_path=config_path,
        is_moe=is_moe,
        rtol=rtol,
        atol=atol,
        train_path=dummy_wan_t2v_dataset,
    )
