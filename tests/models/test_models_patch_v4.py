"""HF↔VeOmni forward/backward parity tests for transformers-v4-only models.

Counterpart to ``test_models_patch.py`` (the v5 default lane). Models that
have not yet been migrated to the v5 patchgen pipeline (currently
``llama3_1`` and ``qwen2_5_omni``) are exercised here against the
``transformers-v4-legacy`` extra installed in the dedicated
``gpu_unit_tests_v4`` job. Test infrastructure (``TrainerTest`` + helpers)
is reused from ``test_models_patch`` so the two files only differ in the
case list.

When a model gains a v5 patchgen migration it should move from this file
into ``test_models_patch.py`` and stop being collected here.
"""

import pytest

from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from .test_models_patch import DEFAULT_ATOL, DEFAULT_RTOL, run_models_patch_fwd_bwd


# Skip the whole module on transformers >= 5.0.0; these models are
# v4-only and ``raise_if_not_migrated_to_v5`` would explicitly fail at
# model build time. ``allow_module_level=True`` is OK here because the
# v5 default lane runs ``test_models_patch.py`` (which always collects
# at least the v5 cases), so pytest exiting with code 5 from this file
# alone is not possible.
if is_transformers_version_greater_or_equal_to("5.0.0"):
    pytest.skip(
        "test_models_patch_v4.py only covers v4-only models; run on the transformers-v4-legacy lane instead.",
        allow_module_level=True,
    )


TEST_CASES_V4 = [
    pytest.param(
        "./tests/toy_config/llama31_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        id="llama3.1",
    ),
    pytest.param(
        "./tests/toy_config/qwen25omni_toy",
        False,
        DEFAULT_RTOL,
        DEFAULT_ATOL,
        id="qwen2_5_omni",
    ),
]


@pytest.mark.parametrize("config_path, is_moe, rtol, atol", TEST_CASES_V4)
def test_models_patch_fwd_bwd_v4(
    request: pytest.FixtureRequest,
    config_path: str,
    is_moe: bool,
    rtol: float,
    atol: float,
):
    run_models_patch_fwd_bwd(request, config_path, is_moe, rtol, atol)
