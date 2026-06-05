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

The allowlist is keyed by ``(generated-file, function qualname)``: each
sync warning's line number is resolved to its enclosing function via AST,
so a ``make patchgen`` that shifts lines does **not** rot the gate. Dead
allowlist entries (a function that no longer issues a sync) still fail
the test, so a landed fix can't silently leave a stale entry behind.
"""

import ast
import importlib
import importlib.util
import os
import re
import warnings
from functools import lru_cache

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
    "qwen3_vl_moe-fa2-fused": "fused_triton",
    "qwen3_omni_moe-fa2-fused": "fused_triton",
}


# Cases this gate covers. Built explicitly (rather than imported wholesale
# from logits_equal) because we want a different MoE backend than the
# logits test forces for HF parity, and we want to drop SDPA in favour of
# FA2. To extend, add a ``Case`` here (and optionally a ``_MOE_IMPL_BY_CASE``
# entry for fused-MoE coverage).
#
# Eager-MoE cases are intentionally absent: the eager experts loop body is
# HF-verbatim and not the production path (production uses
# ``veomni_moe_experts_forward`` via OpSlot, exercised by the fa2-fused
# cases below). Gating MoE on the fused path alone keeps the allowlist
# focused on VeOmni-patched code.
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
    # qwen3_5_vl (non-MoE VLM) — exercises the qwen3_5-family ViT
    # (``Qwen3_5VisionModel.forward`` + ``fast_pos_embed_interpolate`` +
    # ``rot_pos_emb``) and the VLM path of ``Qwen3_5Model.forward``. The
    # qwen3_5-text-* cases above are text-only sub-configs and never reach
    # the vision tower, so this case is what gates the qwen3_5 ViT precompute
    # consumer. qwen3_5_moe's ViT forward is the *same* imported function, so
    # one non-MoE case covers the shared ViT.
    #
    # SDPA (not FA2) here: FA2+bf16 produces NaN on the qwen3_5 toy config
    # (an upstream FA-on-tiny-shape issue — see test_models_logits_equal_v5).
    # The ViT metadata syncs this case gates (cu_seqlens build, .tolist(),
    # rot_pos_emb) are attention-implementation-independent, so SDPA covers
    # them exactly as well.
    _logits_case("qwen3_5_vl-sdpa"),
    # qwen3_vl (non-MoE VLM) — full multimodal forward with a dummy 2x2
    # image patch; exercises patched ``Qwen3VLModel.forward`` +
    # ``get_image_features`` + the vision tower.
    _logits_case("qwen3_vl-fa2"),
    # qwen3_vl_moe — production FA2 + fused-Triton MoE on the VLM path.
    Case(
        "qwen3_vl_moe-fa2-fused",
        _toy("qwen3vlmoe_toy"),
        "Qwen3VLMoeForConditionalGeneration",
        "vlm_full",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
    ),
    # qwen3_omni_moe — forward on ``model.thinker`` (talker stays out of
    # scope); production FA2 + fused-Triton MoE.
    Case(
        "qwen3_omni_moe-fa2-fused",
        _toy("qwen3omni_toy"),
        "Qwen3OmniMoeForConditionalGeneration",
        "omni_thinker",
        attn_implementation="flash_attention_2",
        dtype="bfloat16",
        forward_attr="thinker",
    ),
    # qwen2_vl — full multimodal forward; exercises patched
    # ``Qwen2VLModel.forward`` + the (non-window) ViT precompute consumer.
    _logits_case("qwen2_vl-fa2"),
    # qwen2_5_vl — full multimodal forward; exercises the window-attention
    # ViT precompute consumer (cu_seqlens + cu_window_seqlens + the
    # get_window_index permutation, all collator-derived).
    _logits_case("qwen2_5_vl-fa2"),
    # qwen2_5_omni — forward on ``model.thinker``; shares the window-attention
    # ViT layout with qwen2_5_vl.
    _logits_case("qwen2_5_omni-fa2"),
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
# Entries for qwen3_vl{,_moe} + qwen3_omni_moe populated by this commit.
# qwen3_5 / qwen3_5_moe entries (the previous ``HF-prod-pending-fix`` ones
# for ``_update_linear_attn_mask``) were cleared by the override_method
# patch in PR #762; that method is now VeOmni-patched and host-side.
#
# ``get_rope_index`` in qwen3_vl{,_moe} (called unconditionally from
# ``Model.forward`` when ``image_grid_thw`` is present, production path).
# The method itself is HF-verbatim, but the syncs it issues are inherent to
# the algorithm — per-sample padding strip (``input_ids[attention_mask[i] ==
# 1]`` GPU boolean indexing, variable-size output), ``torch.argwhere`` for
# vision_start positions (variable-size), and a one-shot ``input_ids.tolist()``
# for the inner Python loop. An in-place ``override_method`` doesn't help:
# trying it (see commit history) actually *increased* the reported sync site
# count because PyTorch's ``set_sync_debug_mode("warn")`` over-reports inside
# multi-line Python override bodies (warnings land on comment / blank / pure
# Python lines), giving more allowlist noise than wins.
#
# The clean fix is **out-of-scope for this PR**: precompute the mrope
# position_ids in the data transform (``get_position_id_func`` already runs
# there) and short-circuit the ``self.get_rope_index(...)`` call in
# ``Model.forward`` when position_ids is provided. (rope_deltas is NOT needed
# — it is generation-only; the training forward never reads it.)
# Tracked as follow-up: requires transform + forward signature changes
# across qwen3_vl / qwen3_vl_moe (GPU + NPU).
_ALG_ESSENTIAL_VL_GET_ROPE_INDEX = (
    "algorithm-essential: get_rope_index does per-sample padding strip (GPU boolean indexing) + "
    "argwhere for vision_start positions + a one-shot input_ids.tolist() — intrinsic to the "
    "variable-shape mrope algorithm. Clean fix is to precompute position_ids in the data "
    "transform and skip the call; out-of-scope for this PR."
)
# Keyed by ``(generated-file basename, enclosing-function qualname)`` — NOT a
# raw line number. The qualname is resolved from the sync warning's line via
# AST (`_enclosing_qualname`), so patchgen line shifts no longer rot the
# allowlist, and each entry reads as the function it accepts. One entry
# therefore covers *all* the sync sites inside that function.
_ALLOWED_SYNCS: dict[str, dict[tuple[str, str], str]] = {
    # qwen3_5_vl-sdpa: like qwen3_vl-fa2 below, the ViT forward consumes the
    # precomputed multimodal metadata (fast path) — so the ViT's own
    # `.tolist()` + host-side cu_seqlens build do not appear here. What remains:
    #  - Qwen3_5VisionAttention.forward: the non-FA varlen-attention branch
    #    does `torch.split(t, lengths.tolist(), ...)`, an HF-verbatim D2H.
    #    This is eager/SDPA-only — production qwen3_5-VL uses FA2 (the
    #    `is_flash_attention_requested` branch, which hands cu_seqlens to
    #    flash_attn_varlen_func directly, no `.tolist()`). This case is forced
    #    onto SDPA only because FA2 NaNs on the toy config, so the sync is on
    #    a path production bypasses → tagged HF-eager-only.
    #  - get_rope_index: the HF-verbatim mrope algorithm (same as qwen3_vl).
    "qwen3_5_vl-sdpa": {
        ("patched_modeling_qwen3_5_gpu.py", "Qwen3_5VisionAttention.forward"): (
            "HF-eager-only: the non-FA branch's `torch.split(t, lengths.tolist(), ...)` "
            "varlen split is an HF-verbatim D2H. Production qwen3_5-VL uses FA2, which "
            "passes cu_seqlens to flash_attn_varlen_func directly and bypasses this; the "
            "case runs SDPA only because FA2 NaNs on the toy config."
        ),
        ("patched_modeling_qwen3_5_gpu.py", "Qwen3_5Model.get_rope_index"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
        ("patched_modeling_qwen3_5_gpu.py", "Qwen3_5Model.get_vision_position_ids"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
    },
    # qwen3_vl-fa2: the ViT forward consumes the precomputed multimodal
    # metadata — Model.forward selects the per-modality `vit_metadata` sub-dict
    # from `multimodal_metadata` (which the toy test injects via
    # _attach_multimodal_metadata above) and threads it to the ViT. So this
    # case runs the precompute *fast path* and skips the fallback `.tolist()`
    # + host-side cu_seqlens build that would otherwise fire ~4 syncs. The
    # entries below are therefore the residual syncs that survive even on the
    # fast path — not eliminable by precompute:
    #  - rot_pos_emb: an algorithm-essential `rot_pos_ids(...).to(device)` H2D
    #    copy (CPU-side lru_cached helper output, over-reported by sync-debug).
    #  - get_rope_index: the HF-verbatim mrope algorithm; see the long comment
    #    above _ALG_ESSENTIAL_VL_GET_ROPE_INDEX for why the in-place override
    #    was reverted and the collator-side fix tracked as follow-up.
    "qwen3_vl-fa2": {
        ("patched_modeling_qwen3_vl_gpu.py", "Qwen3VLVisionModel.rot_pos_emb"): (
            "algorithm-essential: `rot_pos_ids(...).to(device)` H2D copy of a CPU tensor "
            "returned by the lru_cached helper; over-reported by torch sync-debug mode."
        ),
        ("patched_modeling_qwen3_vl_gpu.py", "Qwen3VLModel.get_rope_index"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
        ("patched_modeling_qwen3_vl_gpu.py", "Qwen3VLModel.get_vision_position_ids"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
    },
    # qwen3_vl_moe-fa2-fused: mirrors qwen3_vl (the moe config imports the
    # vision forward / rot_pos_emb / fast_pos_embed_interpolate helpers from
    # qwen3_vl and registers its own Model.forward + get_rope_index).
    "qwen3_vl_moe-fa2-fused": {
        ("patched_modeling_qwen3_vl_moe_gpu.py", "Qwen3VLMoeVisionModel.rot_pos_emb"): (
            "algorithm-essential: see qwen3_vl-fa2 (rot_pos_ids H2D copy)."
        ),
        ("patched_modeling_qwen3_vl_moe_gpu.py", "Qwen3VLMoeModel.get_rope_index"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
        (
            "patched_modeling_qwen3_vl_moe_gpu.py",
            "Qwen3VLMoeModel.get_vision_position_ids",
        ): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
    },
    # qwen3_omni_moe-fa2-fused: clean — Stage 2 (multimodal_metadata_precompute)
    # wired the omni ViT forward to consume the precomputed cu_seqlens /
    # max_seqlen from the collator, and this PR's transformers 5.9 port
    # replaced the deprecated `rot_pos_emb` shim with a host-driven
    # `position_ids` build (no `grid_thw.tolist()` left). The toy test
    # pre-supplies `position_ids` so the patched `get_rope_index` /
    # upstream `get_vision_position_ids` are not exercised either.
    "qwen3_omni_moe-fa2-fused": {},
    # qwen2_vl-fa2: the (non-window) ViT forward consumes the precomputed
    # multimodal metadata — Model.forward selects the per-modality
    # `vit_metadata` sub-dict and threads it to the ViT, so the fallback
    # `.tolist()` + host-side cu_seqlens build do not fire. The patched ViT
    # forward also builds rotary `position_ids` host-driven from
    # `grid_thw_list` (mirroring upstream's `get_vision_position_ids`), so
    # the 5.9-deprecated `self.rot_pos_emb(...)` shim is bypassed entirely —
    # no rotary-path syncs remain. Residual sync: the HF-verbatim mrope
    # algorithm in `get_rope_index` (same as qwen3_vl).
    "qwen2_vl-fa2": {
        ("patched_modeling_qwen2_vl_gpu.py", "Qwen2VLModel.get_rope_index"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
        ("patched_modeling_qwen2_vl_gpu.py", "Qwen2VLModel.get_vision_position_ids"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
    },
    # qwen2_5_vl-fa2: the window-attention ViT forward consumes the
    # precomputed metadata (cu_seqlens + cu_window_seqlens + the
    # get_window_index permutation, all collator-derived), so the in-forward
    # `get_window_index` (`grid_thw.tolist()` + per-image `cu.tolist()`) and
    # the two `.max().cpu()` reductions do not fire. The rotary path is
    # host-driven (same approach as qwen2_vl), so the 5.9-deprecated
    # `rot_pos_emb` shim is bypassed. The patched Model.forward now threads
    # the caller-supplied `mm_token_type_ids` into `compute_3d_position_ids`
    # (previously it was silently dropped, leaving position_ids=None via
    # upstream's `can_compute_mrope` short-circuit), so the M-RoPE branch
    # now actually runs in this test and exposes the HF-verbatim mrope
    # algorithm in `get_rope_index` / `get_vision_position_ids`, same as
    # qwen2_vl.
    "qwen2_5_vl-fa2": {
        ("patched_modeling_qwen2_5_vl_gpu.py", "Qwen2_5_VLModel.get_rope_index"): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
        (
            "patched_modeling_qwen2_5_vl_gpu.py",
            "Qwen2_5_VLModel.get_vision_position_ids",
        ): _ALG_ESSENTIAL_VL_GET_ROPE_INDEX,
    },
    # qwen2_5_omni-fa2: shares the window-attention ViT layout with
    # qwen2_5_vl — the precompute consumer eliminates the same get_window_index
    # / cu_seqlens syncs, and the rotary path is host-driven (no deprecated
    # `rot_pos_emb` call). The omni ViT computes its varlen `.max()` seqlen
    # on-device (no sync) and the omni `get_rope_index` is VeOmni-patched and
    # host-side, so no syncs remain on the production path.
    "qwen2_5_omni-fa2": {},
}

# Cases that are *declared* in CASES but skipped at runtime because they
# currently surface VeOmni-touched sync sites we haven't fixed yet.
# Under the gate's principle (VeOmni-patched syncs get fixed, not
# allowlisted), it would be misleading to add these to ``_ALLOWED_SYNCS``;
# the skip keeps the case visible in pytest output as a reminder. The
# skip reason should name the offending functions and the follow-up.
# Currently empty — all declared cases pass or are fully allowlisted.
_PENDING_FIX_CASES: dict[str, str] = {}


# Cases that have been wired to consume ``multimodal_metadata`` via the
# patched ViT / Model forwards (see
# .agents/knowledge/multimodal_metadata.md). For these, the test attaches
# a metadata dict to ``fwd_kwargs`` mirroring what VeOmni's MainCollator
# would emit in real training, so the consumer fast path is exercised
# and the corresponding fallback-path syncs disappear from the allowlist.
_MM_METADATA_WIRED_CASES: set[str] = {
    "qwen3_5_vl-sdpa",
    "qwen3_vl-fa2",
    "qwen3_vl_moe-fa2-fused",
    "qwen3_omni_moe-fa2-fused",
    "qwen2_vl-fa2",
    "qwen2_5_vl-fa2",
    "qwen2_5_omni-fa2",
}


def _attach_multimodal_metadata(model, case: Case, fwd_kwargs: dict) -> None:
    """Inject ``multimodal_metadata`` by running the model's real collate hook.

    Calls ``model.get_metadata_collate_func()`` — the exact picklable hook the
    VeOmni collator invokes in real training — on a synthetic packed batch of
    the toy ``*_grid_thw`` tensors. So the test feeds the model precisely what
    production would, and the precompute consumer path is exercised with the
    real per-model metadata derivation rather than a test reimplementation
    (which would be circular — a bug shared by both would pass silently).

    No-op for cases not in ``_MM_METADATA_WIRED_CASES`` or models without the
    hook. The toy test has no SP, so the sp-pad counts are zero.
    """
    if case.case_id not in _MM_METADATA_WIRED_CASES:
        return
    get_hook = getattr(model, "get_metadata_collate_func", None)
    if get_hook is None:
        return
    hook = get_hook()
    if hook is None:
        return
    batch = {k: fwd_kwargs[k] for k in ("image_grid_thw", "video_grid_thw") if k in fwd_kwargs}
    if not batch:
        return
    hook(batch, {"pixel_values": 0, "pixel_values_videos": 0})
    md = batch.get("multimodal_metadata")
    if md is not None:
        fwd_kwargs["multimodal_metadata"] = md


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


@lru_cache(maxsize=None)
def _def_spans(path: str) -> tuple[tuple[int, int, str], ...]:
    """``(start_lineno, end_lineno, qualname)`` for every def in a Python file.

    AST-parsed once per file (lru_cached). Used to map a sync warning's raw
    line number onto the *qualified name* of the function/method that
    contains it — a key that survives patchgen line shifts, unlike the line
    number itself.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = prefix + child.name
                # ``lineno`` is the ``def`` line, not the decorator line: a
                # sync warning on a decorator *above* the def would resolve to
                # the enclosing scope. None of the allowlisted functions are
                # decorated, so this is not a live concern — revisit if a
                # patched/HF def with a sync ever gains a decorator.
                spans.append((child.lineno, child.end_lineno, qualname))
                walk(child, qualname + ".")  # nested defs / closures
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                # Recurse through non-def blocks (``if`` / ``try`` / ``with``
                # version gates, etc.) keeping the same prefix — a def inside
                # them is still at the enclosing scope, and missing it would
                # misattribute its syncs to ``<module>``.
                walk(child, prefix)

    walk(tree, "")
    return tuple(spans)


def _enclosing_qualname(path: str, lineno: int) -> str:
    """Qualified name of the innermost def at ``lineno`` (``"<module>"`` if none).

    "Innermost" = the smallest line span covering ``lineno``, so a sync inside
    a nested helper resolves to the helper, not its parent.
    """
    best, best_span = "<module>", None
    for start, end, qualname in _def_spans(path):
        if start <= lineno <= end and (best_span is None or end - start < best_span):
            best, best_span = qualname, end - start
    return best


def test_enclosing_qualname_resolution(tmp_path):
    """`_enclosing_qualname` maps a line onto its innermost def — the property
    the line-shift-proof allowlist keying depends on. CPU-only, no GPU."""
    src = (
        "import torch\n"  # 1  module-level
        "\n"  # 2
        "def helper():\n"  # 3
        "    x = 1\n"  # 4  helper
        "    return x\n"  # 5  helper
        "\n"  # 6
        "class Foo:\n"  # 7
        "    def bar(self):\n"  # 8  Foo.bar
        "        def inner():\n"  # 9  Foo.bar.inner
        "            return 0\n"  # 10 Foo.bar.inner
        "        return inner()\n"  # 11 Foo.bar
        "\n"  # 12
        "if True:\n"  # 13
        "    def gated():\n"  # 14 gated (def inside an `if` block)
        "        return 1\n"  # 15 gated
    )
    f = str(tmp_path / "m.py")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(src)
    assert _enclosing_qualname(f, 1) == "<module>"
    assert _enclosing_qualname(f, 4) == "helper"
    assert _enclosing_qualname(f, 10) == "Foo.bar.inner"  # innermost def wins
    assert _enclosing_qualname(f, 11) == "Foo.bar"
    # def inside an `if` block: still resolved (not misattributed to <module>).
    assert _enclosing_qualname(f, 15) == "gated"


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
    if case.case_id in _PENDING_FIX_CASES:
        pytest.skip(f"Pending fix: {_PENDING_FIX_CASES[case.case_id]}")

    _apply_determinism()

    device = get_device_type()
    dtype = _DTYPE_MAP[case.dtype]
    config = _make_config(case)
    input_ids, fwd_kwargs = _make_inputs(case, config, device, dtype)

    model = _build_veomni_model(case, config)
    target = _forward_target(model, case)

    # Augment toy inputs with the multimodal_metadata that VeOmni's
    # MainCollator would produce in real training (built by the model's own
    # collate hook). This exercises the precompute consumer path in patched
    # Model.forward / ViT.forward, mirroring the contract documented in
    # .agents/knowledge/multimodal_metadata.md. Cases that opt in here have
    # their ViT-side fallback syncs removed from the allowlist below; cases
    # not yet wired keep the fallback entries in ``_ALLOWED_SYNCS``.
    _attach_multimodal_metadata(model, case, fwd_kwargs)

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

    # Resolve each sync warning from generated/ to the *qualified name* of the
    # function it fired in (``(basename, qualname)``). Keying on the qualname
    # rather than a raw line number means patchgen line shifts no longer rot
    # the allowlist — and one entry covers every sync site inside a function.
    # One representative line number per key is kept for the failure report.
    observed: dict[tuple[str, str], tuple[int, str]] = {}
    for f, ln, msg in captured:
        if not _is_generated_path(f):
            continue
        key = (os.path.basename(f), _enclosing_qualname(f, ln))
        observed.setdefault(key, (ln, msg.splitlines()[0]))

    offending = sorted(k for k in observed if k not in allowed)
    # Dead allowlist entries: a function that no longer issues any sync.
    # Means the patch was fixed (good — delete the entry); line shifts can
    # no longer cause this, since the key is the qualname.
    dead = sorted(k for k in allowed if k not in observed)

    if offending or dead:
        problems: list[str] = []
        if offending:
            formatted = "\n".join(
                f"  {bn} :: {qn}  (e.g. line {observed[bn, qn][0]})  ::  {observed[bn, qn][1]}" for bn, qn in offending
            )
            problems.append(
                f"{len(offending)} new implicit CUDA sync site(s) in generated modeling:\n"
                f"{formatted}\n"
                f"Each entry is the function the sync fires in. Triage along two axes (owner + path):\n"
                f"  1. Check the patchgen .diff next to the generated file.\n"
                f"     Code is *in the .diff* (added/modified by VeOmni) -> ours.\n"
                f"     Code is *unchanged from HF* -> HF-verbatim.\n"
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
                f"Add the (basename, qualname) to _ALLOWED_SYNCS[{case.case_id!r}] with the\n"
                f"appropriate tag prefix."
            )
        if dead:
            dead_fmt = "\n".join(f"  {bn} :: {qn}" for bn, qn in dead)
            problems.append(
                f"{len(dead)} dead _ALLOWED_SYNCS entr{'y' if len(dead) == 1 else 'ies'} "
                f"for {case.case_id!r} (allowlisted but no longer observed):\n"
                f"{dead_fmt}\n"
                f"That function no longer issues a sync — drop the entry."
            )
        raise AssertionError(f"[{case.case_id}]\n" + "\n\n".join(problems))


# Cases the equivalence test below covers: every multimodal-metadata-wired
# case. Built once so the parametrize id list and the body agree.
_MM_EQUIV_CASES = [c for c in CASES if c.case_id in _MM_METADATA_WIRED_CASES]


@pytest.mark.parametrize("case", _MM_EQUIV_CASES, ids=[c.case_id for c in _MM_EQUIV_CASES])
def test_multimodal_metadata_path_matches_fallback(case):
    """The precompute fast path must be bitwise-equal to the runtime fallback.

    Runs the *same* model on the *same* inputs twice — once with
    ``multimodal_metadata`` injected (the collator-precompute consumer path),
    once without (the in-forward fallback derivation) — and asserts the logits
    are identical.

    This is the numeric gate on each model's collate hook. The metadata is
    produced by the model's real ``get_metadata_collate_func`` hook (see
    ``_attach_multimodal_metadata``), so a silent bug in e.g. qwen2_5_vl's
    ported ``get_window_index`` — which feeds a ``window_index`` permutation
    that reorders ``hidden_states`` — would make the precompute-path logits
    diverge from the fallback and fail here. The sync gate above only checks
    *that* the fast path is sync-free, not that it is *correct*; this test
    closes that gap.
    """
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

    # Fallback path: no multimodal_metadata → ViT derives cu_seqlens / window
    # metadata in-forward.
    with torch.no_grad():
        out_fallback = target(input_ids=input_ids.clone(), use_cache=False, **fwd_kwargs)

    # Precompute path: the model's own collate hook builds multimodal_metadata.
    fwd_precompute = dict(fwd_kwargs)
    _attach_multimodal_metadata(model, case, fwd_precompute)
    assert "multimodal_metadata" in fwd_precompute, (
        f"[{case.case_id}] get_metadata_collate_func produced no multimodal_metadata — "
        f"the case is in _MM_METADATA_WIRED_CASES but the hook is missing or returned nothing."
    )
    with torch.no_grad():
        out_precompute = target(input_ids=input_ids.clone(), use_cache=False, **fwd_precompute)

    del model
    _release()

    # Bitwise-equal: both forwards run the same weights; the only difference is
    # where the ViT metadata came from. Any mismatch means the collate hook's
    # derivation diverged from the model's in-forward derivation.
    torch.testing.assert_close(
        out_precompute.logits,
        out_fallback.logits,
        rtol=0,
        atol=0,
        msg=lambda m: (
            f"[{case.case_id}] precompute-path logits differ from the fallback path — "
            f"the collate hook's metadata derivation does not match the in-forward "
            f"derivation.\n{m}"
        ),
    )
