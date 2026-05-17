"""Implicit CUDA-sync gate for VeOmni-patched v5 modeling code.

Runs each model's forward under ``torch.cuda.set_sync_debug_mode("warn")``
and fails if any implicit host<->device sync originates from
``veomni/models/transformers/<model>/generated/``.

Why this matters
----------------
Patched modeling can quietly serialise the host against the device by
turning a 0-D GPU scalar into a Python int (``.item()`` / ``int(t)``),
slicing with a GPU tensor, calling ``repeat_interleave(GPU_repeats)``,
``if gpu_tensor:``, etc. Invisible during compute-heavy steps; expensive
under SP/EP and small micro-batches, and can cascade into NCCL watchdog
timeouts. See the ``debug-cuda-sync`` skill for the manual investigation
flow against real weights.

This test is the *unit-level ratchet* for the same property: it runs on
toy configs (so only catches sync sites reachable from a tiny forward),
but it's cheap and catches "someone added a ``.item()`` to a patch" at
PR time. Real-model SP/EP coverage stays with the skill.

Extending to more models
------------------------
Append a ``Case`` to ``CASES`` — either reuse one from
``test_models_logits_equal_v5.CASES`` via ``_logits_case("...")``, or
declare a new one inline (for cases that don't have an HF-parity
counterpart, e.g. fused-MoE on the production path). Add a
``_MOE_IMPL_BY_CASE`` entry to override the default ``"eager"``
backend for MoE cases.

Acknowledging a necessary sync
------------------------------
``_ALLOWED_SYNCS`` is a per-case dict from ``(basename, lineno)`` to a
short reason string — pre-existing acknowledged syncs (HF-inherited
code paths, eager-only fallback loops, EP dispatch with data-dependent
split sizes). The semantics are *ratchet*: the test fails if a new
``(file, lineno)`` shows up in generated/ that isn't in the allowlist.
Removing a site (by fixing the patch) is encouraged.

Line numbers are into the **generated** file, so if patchgen
regenerates and shifts lines, the allowlist must be re-checked. Treat
this the same as the checked-in ``.diff`` snapshot under each model's
``generated/`` — both are line-aware.
"""

import importlib
import importlib.util
import os
import re
import warnings

import pytest
import torch

from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device, synchronize

from .test_models_logits_equal_v5 import (
    _DTYPE_MAP,
    Case,
    _apply_determinism,
    _forward_target,
    _make_config,
    _make_inputs,
    _release,
)
from .test_models_logits_equal_v5 import (
    CASES as _ALL_CASES,
)


def _logits_case(case_id: str) -> Case:
    """Pull a ``Case`` out of the logits-equal CASES list by ``case_id``."""
    for c in _ALL_CASES:
        if c.case_id == case_id:
            return c
    raise KeyError(f"{case_id!r} not in test_models_logits_equal_v5.CASES")


def _toy(name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "toy_config", name
    )


# Per-case ``moe_implementation`` for VeOmni's ``apply_ops_config``. Defaults
# to ``"eager"``. Set to ``"fused_triton"`` for production-path coverage
# (A100/SM80+); the fused dispatch in ``patched_modeling_*_moe_gpu.py``
# short-circuits the eager expert loop and replaces it with a single
# Triton kernel call — no Python-level sync sites.
_MOE_IMPL_BY_CASE: dict[str, str] = {
    "qwen3_5_moe-text-fa2-fused": "fused_triton",
}


# Cases this gate covers. Built explicitly (rather than imported wholesale
# from logits_equal) because we want a different MoE backend than the
# logits test forces for HF parity, and we want to drop SDPA in favour of
# FA2. To extend, add a ``Case`` here (and optionally a ``_MOE_IMPL_BY_CASE``
# entry for fused-MoE coverage).
CASES = [
    # qwen3_5 (non-MoE, text-only sub-config) — both attention paths.
    _logits_case("qwen3_5-text-eager"),
    _logits_case("qwen3_5-text-fa2"),
    # qwen3_5_moe-text — eager path (kept for non-MoE-code coverage) and
    # the production FA2 + fused-Triton MoE path.
    _logits_case("qwen3_5_moe-text-eager"),
    Case(
        "qwen3_5_moe-text-fa2-fused",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
]

# Pre-existing acknowledged sync sites in generated/. Keyed by
# ``Case.case_id``; value maps ``(basename, lineno)`` -> reason.
#
# Two flavours show up in the current baseline:
#   - ``_update_linear_attn_mask`` (HF-inherited): a Python ``if`` on a
#     GPU scalar (``cache_position[0]``) plus ``torch.all(mask == 1)``.
#     The patch doesn't touch this method; the toy config forces all
#     layers to ``full_attention`` so the linear-attention branch is
#     unreachable, but the guard still evaluates each forward.
#   - Eager expert loop in ``Qwen3_5MoeExperts.forward``: ``.nonzero()``
#     → Python ``for`` over GPU tensor → ``if expert_idx == N`` →
#     ``torch.where`` indexing. Production dispatches via
#     ``veomni_moe_experts_forward`` and skips this path; the loop is
#     only reachable under ``_experts_implementation="eager"`` (which
#     the test forces for determinism).
_LINEAR_ATTN_REASON = (
    "HF-inherited _update_linear_attn_mask: `if cache_position[0] > 0 or "
    "torch.all(mask == 1)`; unreachable under toy full_attention config."
)
_EAGER_EXPERTS_REASON = (
    "Eager Qwen3_5MoeExperts.forward fallback loop (.nonzero() + Python for "
    "over experts); production runs fused via veomni_moe_experts_forward."
)
_ALLOWED_SYNCS: dict[str, dict[tuple[str, int], str]] = {
    "qwen3_5-text-eager": {
        ("patched_modeling_qwen3_5_gpu.py", 1750): _LINEAR_ATTN_REASON,
    },
    "qwen3_5-text-fa2": {
        ("patched_modeling_qwen3_5_gpu.py", 1750): _LINEAR_ATTN_REASON,
    },
    "qwen3_5_moe-text-eager": {
        ("patched_modeling_qwen3_5_moe_gpu.py", 1052): _EAGER_EXPERTS_REASON,
        ("patched_modeling_qwen3_5_moe_gpu.py", 1056): _EAGER_EXPERTS_REASON,
        ("patched_modeling_qwen3_5_moe_gpu.py", 1058): _EAGER_EXPERTS_REASON,
        ("patched_modeling_qwen3_5_moe_gpu.py", 1060): _EAGER_EXPERTS_REASON,
        ("patched_modeling_qwen3_5_moe_gpu.py", 1062): _EAGER_EXPERTS_REASON,
        ("patched_modeling_qwen3_5_moe_gpu.py", 1955): _LINEAR_ATTN_REASON,
    },
    # qwen3_5_moe-text-fa2-fused: production path (fused_triton MoE
    # bypasses the eager expert loop). Only the HF-inherited
    # linear-attention guard remains.
    "qwen3_5_moe-text-fa2-fused": {
        ("patched_modeling_qwen3_5_moe_gpu.py", 1955): _LINEAR_ATTN_REASON,
    },
}


# torch's implicit-sync warning message; emitted by
# ``set_sync_debug_mode("warn")`` from various ATen ops.
_SYNC_RE = re.compile(r"called a synchronizing")


def _is_generated_path(filename: str) -> bool:
    """True if ``filename`` lives under ``veomni/models/transformers/*/generated/``."""
    norm = filename.replace(os.sep, "/")
    return "/veomni/models/transformers/" in norm and "/generated/" in norm


# NCCL bootstrap env so this module is runnable on its own (``pytest
# tests/models/test_model_forward_no_implicit_sync.py``). In a same-process
# pytest run the sibling logits_equal test is usually imported first and
# its ``setdefault`` block already populated these; the fixture below
# also no-ops if the PG is already initialised.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12357")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(scope="module", autouse=True)
def _single_rank_process_group():
    """1-rank NCCL group for VeOmni's SP-aware attention wrappers.

    Duplicated rather than imported from the sibling logits test because
    pytest doesn't apply autouse fixtures across modules.
    """
    from veomni.utils.device import get_dist_comm_backend
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not IS_CUDA_AVAILABLE or not is_transformers_version_greater_or_equal_to("5.2.0"):
        yield
        return

    import torch.distributed as dist

    we_initialised = False
    if not dist.is_initialized():
        get_torch_device().set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend=get_dist_comm_backend(), rank=0, world_size=1)
        we_initialised = True
    try:
        yield
    finally:
        if we_initialised and dist.is_initialized():
            dist.destroy_process_group()


def _build_veomni_model(case, config):
    """Random-init VeOmni model — we only need forward to run, not match HF."""
    from veomni.models.auto import build_foundation_model
    from veomni.ops import apply_ops_config

    training_utils = importlib.import_module("tests.tools.training_utils")
    apply_ops_config(
        training_utils.make_eager_ops_config(
            attn_implementation=case.attn_implementation,
            moe_implementation=_MOE_IMPL_BY_CASE.get(case.case_id, "eager"),
        )
    )

    torch.manual_seed(0)
    get_torch_device().manual_seed_all(0)
    return build_foundation_model(
        config_path=config,
        weights_path=None,
        torch_dtype=case.dtype,
        attn_implementation=case.attn_implementation,
        init_device=get_device_type(),
    ).eval()


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_no_implicit_sync_in_generated_forward(case):
    """No implicit CUDA sync should originate from generated/ during forward."""
    from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

    if not is_transformers_version_greater_or_equal_to("5.2.0"):
        pytest.skip("Scope is transformers v5 model definition only (v5 stack pins >= 5.2.0).")
    if not IS_CUDA_AVAILABLE:
        pytest.skip("CUDA required.")
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    if case.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn package not installed.")
    if _MOE_IMPL_BY_CASE.get(case.case_id) == "fused_triton":
        from veomni.utils.import_utils import is_fused_moe_available

        if not is_fused_moe_available():
            pytest.skip("fused_triton MoE requires triton + CUDA SM70+.")

    _apply_determinism()

    device = get_device_type()
    dtype = _DTYPE_MAP[case.dtype]
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)

    model = _build_veomni_model(case, config)
    target = _forward_target(model, case)

    # Warmup outside debug mode: rotary cos/sin cache fill, kernel
    # autotuning, lazy buffer materialisation — these fire once and
    # aren't relevant to steady-state.
    with torch.no_grad():
        target(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
    synchronize()

    # ``torch.cuda.{get,set}_sync_debug_mode`` are the actual API for the
    # debug knob this test relies on — no ``veomni.utils.device`` helper
    # exists for it. CUDA-only by design; the ``IS_CUDA_AVAILABLE`` skip
    # above gates the whole test, so this isn't an NPU-compat hazard.
    prev_mode = torch.cuda.get_sync_debug_mode()
    captured: list[tuple[str, int, str]] = []
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with torch.no_grad():
                target(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)
        for w in wlist:
            if w.category is UserWarning and _SYNC_RE.search(str(w.message)):
                captured.append((w.filename, w.lineno, str(w.message)))
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)

    del model
    _release()

    allowed = _ALLOWED_SYNCS.get(case.case_id, {})
    captured_sites = {(os.path.basename(f), ln) for (f, ln, _) in captured if _is_generated_path(f)}
    offending = [
        (f, ln, msg) for (f, ln, msg) in captured if _is_generated_path(f) and (os.path.basename(f), ln) not in allowed
    ]
    # Dead allowlist entries: listed but no longer observed. Usually means
    # the patch was fixed (good — delete the entry) or that patchgen
    # shifted line numbers (re-check the underlying code, then update).
    dead = sorted(k for k in allowed if k not in captured_sites)

    if offending or dead:
        problems: list[str] = []
        if offending:
            # Dedup ``(basename, lineno)`` — even with ``simplefilter("always")``
            # one site can fire many times (per layer, per expert, ...);
            # the reliable signal is the *set* of sites.
            unique = {(os.path.basename(f), ln): m.splitlines()[0] for (f, ln, m) in offending}
            formatted = "\n".join(f"  {f}:{ln}  ::  {m}" for (f, ln), m in sorted(unique.items()))
            problems.append(
                f"{len(unique)} new implicit CUDA sync site(s) in patched modeling:\n"
                f"{formatted}\n"
                f"Each line is into the generated modeling file. Fix the patch to derive the "
                f"value host-side (typical sources: 0-D GPU tensor used as a shape/index arg, "
                f".item() for a Python int, repeat_interleave with GPU repeats, if/bool on a "
                f"GPU scalar). If the sync is genuinely needed (e.g. EP dispatch sizes), add "
                f"the (basename, lineno) to _ALLOWED_SYNCS[{case.case_id!r}] with a one-line "
                f"reason."
            )
        if dead:
            dead_fmt = "\n".join(f"  {f}:{ln}" for (f, ln) in dead)
            problems.append(
                f"{len(dead)} dead _ALLOWED_SYNCS entr{'y' if len(dead) == 1 else 'ies'} "
                f"for {case.case_id!r} (allowlisted but no longer observed):\n"
                f"{dead_fmt}\n"
                f"Either the patch was fixed (drop the entry) or patchgen shifted lines "
                f"(re-check the source and update the lineno)."
            )
        raise AssertionError(f"[{case.case_id}]\n" + "\n\n".join(problems))
