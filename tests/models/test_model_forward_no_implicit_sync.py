"""Implicit CUDA-sync gate for VeOmni-patched v5 modeling code.

Runs each model's forward under ``torch.cuda.set_sync_debug_mode("warn")``
and fails if any new implicit host<->device sync site shows up in
``veomni/models/transformers/<model>/generated/``.

The principle — two axes
------------------------
For each surfaced sync site, decide along two independent axes:

  1. **Owner** — VeOmni-patched (the line appears in the patchgen ``.diff``)
     vs HF-verbatim (the line is unchanged from upstream HF, even though it
     lives inside generated/).
  2. **Path** — production (code runs every real training/inference step,
     no override above it) vs eager-only fallback (code only runs under a
     specific dev/fallback setting; production bypasses it via an OpSlot
     or other VeOmni override above).

The rule:

  - Production-path + VeOmni-patched  →  fix the patch (derive host-side,
    precompute in the collator, etc.).
  - Production-path + HF-verbatim     →  add an ``override_method`` patch
    and fix there (now becomes ours). **Don't leave a production sync in
    just because the line originated upstream** — that's an unreasonable
    cost in our hot path.
  - Eager-only fallback + HF-verbatim (production bypasses via OpSlot/
    override)                          →  leave alone, allowlist if needed.
  - Algorithm-essential (EP dispatch sizes, variable per-rank counts) →
    accept.

``_ALLOWED_SYNCS`` therefore holds two flavours of entries, distinguished
by a tag prefix in the reason string:

  - ``"HF-eager-only: ..."``          — accepted long-term; production
                                        bypasses this code.
  - ``"HF-prod-pending-fix: ..."``    — production-path HF-verbatim site
                                        currently tracked for a fix
                                        (follow-up PR named in the reason).
  - ``"algorithm-essential: ..."``    — accepted by design.

The dead-entry detection ensures the pending-fix entries get cleaned up
when the fix lands.

Why this matters
----------------
Patched modeling can quietly serialise the host against the device by
turning a 0-D GPU scalar into a Python int (``.item()`` / ``int(t)``),
slicing with a GPU tensor, calling ``repeat_interleave(GPU_repeats)``,
``if gpu_tensor:``, etc. Invisible during compute-heavy steps; expensive
under SP/EP and small micro-batches, and can cascade into NCCL watchdog
timeouts. See the ``debug-cuda-sync`` skill for the manual investigation
flow against real weights.

This test is the *unit-level ratchet*: it runs on toy configs (so only
catches sync sites reachable from a tiny forward) and is cheap enough
to gate PRs. Real-model SP/EP coverage stays with the skill.

Extending to more models
------------------------
Append a ``Case`` to ``CASES`` — either reuse one from
``test_models_logits_equal_v5.CASES`` via ``_logits_case("...")``, or
declare a new one inline (for cases that don't have an HF-parity
counterpart, e.g. fused-MoE on the production path). Add a
``_MOE_IMPL_BY_CASE`` entry to override the default ``"eager"``
backend for MoE cases.

When a new model's first run surfaces sync sites, classify each per the
two-axis rule above:

- VeOmni-patched line (appears in the patchgen ``.diff``) → fix the patch.
- HF-verbatim line on a production path (code runs unconditionally, no
  VeOmni override above) → add an ``override_method`` patch to fix it;
  in the meantime allowlist with ``"HF-prod-pending-fix: ..."`` and
  reference the follow-up.
- HF-verbatim line on an eager-only fallback path that production
  bypasses (e.g. inside the eager experts loop, when production
  dispatches to the fused MoE OpSlot) → allowlist with
  ``"HF-eager-only: ..."``.

Line numbers are into the **generated** file, so a ``make patchgen``
that shifts lines requires re-checking the allowlist. Dead allowlist
entries (listed but no longer observed) also fail the test, so drift
can't silently rot the gate.
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
    _toy,
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
#
# ``qwen3_5_moe-text-eager`` is intentionally absent: the eager experts
# loop is HF-verbatim and not the production path (production uses
# ``veomni_moe_experts_forward`` via OpSlot, exercised by the fa2-fused
# case below). Gating MoE on that case alone keeps the allowlist focused
# on VeOmni-patched code.
CASES = [
    # qwen3_5 (non-MoE, text-only sub-config) — both attention paths through
    # our patched Qwen3_5Model.forward.
    _logits_case("qwen3_5-text-eager"),
    _logits_case("qwen3_5-text-fa2"),
    # qwen3_5_moe-text — production FA2 + fused-Triton MoE. The fused
    # short-circuit (line ~1044 in patched_modeling_qwen3_5_moe_gpu.py)
    # is VeOmni's; the HF eager loop body that it bypasses is verbatim
    # and not exercised in this case.
    Case(
        "qwen3_5_moe-text-fa2-fused",
        _toy("qwen3_5_moe_toy"),
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_text",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
]

# Acknowledged sync sites in generated/. Keyed by ``Case.case_id``;
# value maps ``(basename, lineno)`` -> one-line reason.
#
# Reasons MUST start with one of the category tags below — see the
# module docstring's "principle" section for the two-axis triage. The
# gate's pass/fail behaviour is the same across categories; the tags
# encode follow-up state:
#
#   "HF-eager-only: ..."           accepted long-term; production
#                                  bypasses this code via an OpSlot or
#                                  override above.
#   "HF-prod-pending-fix: ..."     production-path HF-verbatim site
#                                  currently tracked for a fix; the
#                                  reason names the follow-up branch /
#                                  PR. Dead-entry detection forces
#                                  cleanup once the fix lands.
#   "algorithm-essential: ..."     accepted by design (EP dispatch
#                                  sizes, variable per-rank counts).
#
# Currently empty: the previous ``HF-prod-pending-fix`` entries for
# ``_update_linear_attn_mask`` in qwen3_5 / qwen3_5_moe were cleared by
# the override_method patch in this branch (the method is now
# VeOmni-patched and host-side, no sync).
_ALLOWED_SYNCS: dict[str, dict[tuple[str, int], str]] = {}


# torch's implicit-sync warning message; emitted by
# ``set_sync_debug_mode("warn")`` from various ATen ops.
_SYNC_RE = re.compile(r"called a synchronizing")


def _is_generated_path(filename: str) -> bool:
    """True if ``filename`` lives under ``veomni/models/transformers/*/generated/``."""
    norm = filename.replace(os.sep, "/")
    # No leading slash on the first substring: ``WarningMessage.filename`` is
    # almost always absolute, but relative-path edge cases (zip imports,
    # custom loaders) shouldn't silently bypass the gate.
    return "veomni/models/transformers/" in norm and "/generated/" in norm


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
            # Filter by message text only — the regex is specific to torch's
            # sync warning, and an exact-category check would silently drop
            # the warning if torch ever switches to a UserWarning subclass.
            if _SYNC_RE.search(str(w.message)):
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
                f"{len(unique)} new implicit CUDA sync site(s) in generated modeling:\n"
                f"{formatted}\n"
                f"Each line is into the generated file. Triage along two axes (owner + path):\n"
                f"  1. Check the patchgen .diff next to the generated file.\n"
                f"     Line is *in the .diff* (added/modified by VeOmni) -> ours.\n"
                f"     Line is *unchanged from HF* -> HF-verbatim.\n"
                f"  2. Check whether the code is on the production path or only on an\n"
                f"     eager/fallback path that production bypasses (e.g. via an OpSlot).\n"
                f"Then act:\n"
                f"  - Production + VeOmni-patched -> fix the patch (derive host-side,\n"
                f"    precompute in the collator).\n"
                f"  - Production + HF-verbatim -> add an override_method patch and fix\n"
                f"    there. Don't leave a production sync in because it came from upstream.\n"
                f"    In the meantime, allowlist with reason 'HF-prod-pending-fix: ...'.\n"
                f"  - Eager-only fallback + HF-verbatim -> allowlist with reason\n"
                f"    'HF-eager-only: ...'.\n"
                f"Add the (basename, lineno) to _ALLOWED_SYNCS[{case.case_id!r}] with the\n"
                f"appropriate tag prefix."
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
