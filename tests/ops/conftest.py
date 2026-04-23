"""Collection-time guard for GPU-backend-only op tests on NPU CI.

Several files under ``tests/ops/`` hard-import GPU-only backends
(``triton``, ``flash_attn``, quack/CUTLASS) at module scope — either
directly, or transitively through ``veomni.ops.kernels.moe._kernels``.
Those packages are not installed in the Ascend CI image, so collecting
these files on NPU raises ``ModuleNotFoundError`` before pytest ever
gets a chance to honour ``pytest.mark.gpu_only`` or any ``skipif``.

Pre-refactor the NPU workflow avoided the problem by listing test paths
explicitly (only ``tests/ops/test_comp.py`` was included). Now that the
workflows drive selection by ``-m unit tests/``, we reproduce that
exclusion here via ``collect_ignore`` — keyed on the
``VEOMNI_TEST_DEVICE`` env var that the NPU reusable workflow sets.

Keep this list aligned with whatever subset of ``tests/ops/`` NPU CI is
expected to skip. Any file that gates its GPU imports *inside* a test
body (e.g. ``test_flash_attn_varlen_padding.py``) does not need an
entry here — its module is importable even when the backend is missing.
"""

import os


if os.environ.get("VEOMNI_TEST_DEVICE", "").lower() == "npu":
    collect_ignore = [
        # triton import at module top (transitively via veomni.ops.kernels.moe._kernels)
        "test_fused_moe_split_vs_merged.py",
        # quack / CUTLASS kernels — SM90 only
        "test_quack_fused_moe.py",
        # triton kernels for load-balancing loss
        "test_fused_load_balancing_loss.py",
        # cross-entropy / seqcls loss kernels pull in triton
        "test_seqcls_loss.py",
    ]
