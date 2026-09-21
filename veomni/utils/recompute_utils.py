# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block-level activation recomputation, shared by every model.

The framework owns the strategy, models stay unchanged:

* :func:`build_policy` turns ``model.accelerator.gradient_checkpointing`` into one immutable
  :class:`RecomputePolicy` per training run (no process-global switch).
* :func:`apply_recompute_policy` runs right after HF's
  ``gradient_checkpointing_enable`` and binds a per-block decision onto every
  checkpoint entry point of the model.
  Blocks are discovered from class names the framework already knows
  (``_no_split_modules`` / ``basic_modules``), so a model only has to be written
  in one of the two supported styles — HF ``GradientCheckpointingLayer`` blocks,
  or a block loop calling ``self._gradient_checkpointing_func`` — to get both
  layer selection and SAC from configuration alone.
* :func:`checkpoint_forward` is the model-side entry, kept for out-of-tree
  models: the same execution rule as above, for a model that owns a bare
  ``torch.utils.checkpoint`` call and finds the block itself. No in-tree model
  needs it — a block loop calls ``self._gradient_checkpointing_func``, which the
  install step replaces.

``recompute_last_n_layers`` picks the recompute range from the last block
(``-1`` = every layer, the default). Within that range, ``selective_n_layers``
picks from the front the blocks that run selective activation checkpointing
(SAC) instead of full recomputation (``0`` = off, the default). The two are
different modes, not a switch and its refinement.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, Callable, Sequence
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

from veomni.utils import logging


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Selective activation checkpointing (SAC) — operator selection
# ---------------------------------------------------------------------------
#
# Wraps PyTorch's native SAC (``create_selective_checkpoint_contexts``): a per-op
# policy decides which forward outputs are saved instead of recomputed.
# Default policy (Megatron/Flash selective style): attention ops keep their
# outputs (``MUST_SAVE``), everything else is recomputed.
#
# Known restrictions: needs non-reentrant checkpointing, and is not combined with
# activation offload, Ulysses sequence parallel or torch.compile (see build_policy).

#: Substring tokens matched against an operator's ``namespace::name``: a match
#: means the operator's forward output is kept (``MUST_SAVE``) instead of being
#: recomputed. One entry per attention family VeOmni can select, on CUDA and on
#: NPU; a backend whose operator *name* is not listed here is still covered when
#: its namespace matches (see ``_selective_namespaces``), and ``selective_ops``
#: in the config is the escape hatch for anything left over.
#:
#: Only the *attention core* belongs here: operators whose output is the
#: attention result (O(S·H·D)) and whose recompute costs a full attention pass.
#: Fused kernels that produce a much larger intermediate (qkv projections, MLA
#: prolog) are vetoed instead of saved — saving them costs more memory than the
#: recompute they would avoid, see ``_SELECTIVE_VETO_TOKENS``.
DEFAULT_SELECTIVE_TOKENS: tuple[str, ...] = (
    "attention",  # every attention_* / *_attention operator: torch_npu families, xformers
    # cutlass, flex, flash-attention, nsa, floyd, quant-fusion, multi-head, cuDNN
    "attn",  # abbreviated names: npu_attn_softmax_, npu_advance_step_flashattn
    "scaled_dot_product",  # aten SDPA family, including the composite entry point
    "flash_attn",  # FlashAttention 2/3/4 operator names (no "attention" in them)
    "flashattn",  # written without the underscore
    "mla",  # FlashMLA / sparse MLA kernels
    "indexer",  # DSA / lightning indexer (torch_npu and cuDNN FE)
    "sageattn",  # SageAttention entry point
)

#: Substrings that veto any match: never saved, whichever branch matched. Two
#: reasons: the implementation materializes the attention matrix (exactly what
#: we do not want to persist), or the fused kernel's output is far larger than
#: an attention output, so saving it would cost more than the SAC budget.
_SELECTIVE_VETO_TOKENS: tuple[str, ...] = (
    "_efficient_attention_forward",  # xformers decomposition, materializes scores
    "_efficient_attention_backward",
    "qkv",  # npu_fused_attention_qkv_grad / _layernorm_qkv_fwd: output is qkv or its grad
    "mla_prolog",  # npu_mla_prolog*: output is q + the compressed KV
)

#: Candidate operator namespaces probed for custom attention kernels. Extended
#: at runtime with every registered namespace matching a default token, so a
#: backend whose namespace is not listed here (``flash_attn_3``) still resolves.
#: Only namespaces that match no token need an entry here (``DSA``, ``npu``).
_SELECTIVE_NS_CANDIDATES: tuple[str, ...] = (
    "npu",
    "npu_extension",
    "DSA",  # cuDNN FE sparse attention / indexer wrappers
    "flashattn",
    "flash_attn",
    "flash_attn_3",
    "flash_attn_4",
    "flash_mla",
    "sageattention",
    "xformers",
)


def string_to_op(op_string: str) -> Any:
    """Resolve ``"aten.addmm.default"`` style strings to a PyTorch op object."""
    clean_string = op_string.strip()

    if clean_string.startswith("torch.ops."):
        clean_string = clean_string[len("torch.ops.") :]

    parts = clean_string.split(".")

    if not hasattr(torch, "ops"):
        raise AttributeError("torch.ops not available in this PyTorch version")

    current = torch.ops

    # Special handling: ensure accessing aten operations by first trying to trigger registration
    if parts[0] == "aten":
        try:
            _ = torch.ops.aten.add
        except AttributeError:
            try:
                importlib.import_module("torch._C._dispatch")
            except ImportError:
                pass

    for i, part in enumerate(parts):
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            current_path = ".".join(parts[:i])
            available_attrs = dir(current) if hasattr(current, "__dict__") else []
            raise AttributeError(
                f"Operation '{op_string}' not found. "
                f"Missing attribute: '{part}' at path 'torch.ops.{current_path}'. "
                f"Available attributes: {available_attrs[:10]}{'...' if len(available_attrs) > 10 else ''}"
            )

    return current


def _op_qualname(op: Any) -> str:
    """Best-effort ``namespace::name`` for an OpOverload / HOP / raw op."""
    schema = getattr(op, "_schema", None)
    if schema is not None and hasattr(schema, "name"):
        return str(schema.name)
    return str(op).replace("torch.ops.", "").split(".", 1)[-1]


def _matches_tokens(qualname: str, tokens: Sequence[str]) -> bool:
    """True when any token is a substring of the operator's ``namespace::name``."""
    lowered = qualname.lower()
    return any(t in lowered for t in tokens)


def _is_vetoed(qualname: str) -> bool:
    """True when the operator must never be saved (see ``_SELECTIVE_VETO_TOKENS``)."""
    return _matches_tokens(qualname, _SELECTIVE_VETO_TOKENS)


def _registered_namespaces() -> list[str]:
    """Namespaces of every operator the dispatcher knows about.

    ``dir(torch.ops)`` only lists namespaces already touched in this process, so
    an out-of-tree library that registered its schemas without being accessed
    (``flash_attn_3::_flash_attn_forward``) would be invisible. The schemas are
    therefore read from the dispatcher instead.
    """
    try:
        schemas = torch._C._jit_get_all_schemas()
    except AttributeError:  # pragma: no cover - very old or stripped builds
        return [name for name in dir(torch.ops) if not name.startswith("_")]
    return sorted({schema.name.split("::", 1)[0] for schema in schemas if "::" in schema.name})


def _selective_namespaces(registered: Sequence[str]) -> list[str]:
    """Namespaces to probe: the fixed candidates plus any registered one whose
    own name matches a default attention token."""
    namespaces = set(_SELECTIVE_NS_CANDIDATES)
    for name in registered:
        if not name.startswith("_") and _matches_tokens(name, DEFAULT_SELECTIVE_TOKENS):
            namespaces.add(name)
    return sorted(namespaces)


def resolve_exact_ops(extra_op_names: Sequence[str] | None = None) -> tuple[list[Any], list[str]]:
    """Resolve the default attention operators plus user extras to OpOverloads.

    Returns ``(resolved_ops, failed_names)``; never raises. ``failed_names``
    are extra names that could not be resolved (missing registration) — callers
    may fall back to substring matching.
    """
    ops: list[Any] = []
    failed: list[str] = []

    # aten SDPA family: every packet whose name starts with _scaled_dot_product.
    for name in sorted(dir(torch.ops.aten)):
        if not name.startswith("_scaled_dot_product"):
            continue
        default = getattr(getattr(torch.ops.aten, name), "default", None)
        if default is not None:
            ops.append(default)

    # Custom namespaces: probe candidate namespaces for token-matching ops.
    for ns_name in _selective_namespaces(_registered_namespaces()):
        try:
            ns = getattr(torch.ops, ns_name)
        except AttributeError:
            continue
        for op_name in dir(ns):
            qualname = f"{ns_name}::{op_name}"
            if not _matches_tokens(qualname, DEFAULT_SELECTIVE_TOKENS) or _is_vetoed(qualname):
                continue
            default = getattr(getattr(ns, op_name), "default", None)
            if default is not None:
                ops.append(default)

    # User extras: full "torch.ops.<ns>.<name>.default" strings.
    for op_str in extra_op_names or ():
        try:
            op = string_to_op(op_str)
            if not isinstance(op, torch._ops.OpOverload):
                raise TypeError("op string must end with an overload name")
            ops.append(op)
        except (AttributeError, TypeError) as exc:
            logger.warning(f"cannot resolve extra selective op {op_str!r} ({exc})")
            failed.append(op_str)

    # De-duplicate while preserving order (OpOverloads are hashable singletons).
    seen = set()
    unique = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            unique.append(op)
    return unique, failed


def _make_selective_policy(exact_ops: Sequence[Any]) -> Callable[[Any, Any, tuple, dict], Any]:
    """Policy fn for ``create_selective_checkpoint_contexts``.

    Returns MUST_SAVE for resolved attention ops (exact identity) *and* for any
    op whose qualname matches a default token; otherwise PREFER_RECOMPUTE. The
    name match is kept alongside the exact list so an operator that resolved
    neither precisely nor at probe time (a backend registering after the policy
    was built) is still saved. Both directions of mismatch are numerically safe
    (they only change how much is saved vs recomputed).

    ``_SELECTIVE_VETO_TOKENS`` wins over both branches: a fused kernel whose
    output is much larger than an attention output is recomputed even when it
    matched by identity or by name.
    """
    op_set = set(exact_ops)

    def policy(ctx, op, *args, **kwargs):
        del ctx, args, kwargs  # unused
        qualname = _op_qualname(op)
        if _is_vetoed(qualname):
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        if op in op_set or _matches_tokens(qualname, DEFAULT_SELECTIVE_TOKENS):
            return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _build_context_fn(extra_op_names: Sequence[str] | None) -> Callable[[], tuple[Any, Any]]:
    """Build the ``context_fn`` for ``torch.utils.checkpoint.checkpoint``."""
    exact_ops, failed = resolve_exact_ops(extra_op_names)
    if failed:
        logger.warning_once(
            f"cannot resolve extra selective op(s) {failed!r}; they are covered by the name-substring policy only"
        )
    logger.info_rank0(f"SAC enabled with {len(exact_ops)} exact attention ops plus the name-substring fallback")
    return functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, _make_selective_policy(exact_ops)
    )


# ---------------------------------------------------------------------------
# Policy layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecomputePolicy:
    """Immutable recomputation policy for one training run.

    ``context_fn`` is None whenever SAC is unavailable (disabled, reentrant, or
    activation offload owns the checkpoint boundary); the layer counts stay
    meaningful in that case — they gate full recomputation too.
    """

    recompute_last_n_layers: int = -1  # -1 = every layer, N = the last N, 0 = none
    selective_n_layers: int = 0  # 0 = SAC off, N = the first N of the recomputed layers
    context_fn: Callable[[], tuple[Any, Any]] | None = None
    early_stop: bool = True
    use_reentrant: bool = False

    @property
    def active(self) -> bool:
        """Whether the policy asks for anything the default does not already do."""
        return self.recompute_last_n_layers != -1 or self.selective_n_layers > 0


def build_policy(
    *,
    enabled: bool,
    enable_reentrant: bool,
    early_stop: bool,
    extra_op_names: Sequence[str] | None = None,
    recompute_last_n_layers: int = -1,
    selective_n_layers: int = 0,
    offload_active: bool = False,
    compile_enabled: bool = False,
) -> RecomputePolicy:
    """Build the run's :class:`RecomputePolicy` from ``model.accelerator.gradient_checkpointing``.

    SAC requires ``enabled`` and non-reentrant checkpointing; whenever it cannot
    be honoured the counts are kept and only ``context_fn`` is dropped, so those
    layers fall back to full recomputation with a warning. Never raises: invalid
    counts are clamped instead.
    """
    if recompute_last_n_layers < -1:
        logger.warning_once(
            f"recompute_last_n_layers={recompute_last_n_layers} invalid (< -1), using -1 (every layer)"
        )
        recompute_last_n_layers = -1
    if selective_n_layers < 0:
        logger.warning_once(
            f"selective_n_layers={selective_n_layers} invalid (< 0), using 0 (SAC off); "
            "use a value >= the model depth for every recomputed layer"
        )
        selective_n_layers = 0
    if selective_n_layers > 0 and recompute_last_n_layers == 0:
        logger.warning_once(
            "selective_n_layers is set but recompute_last_n_layers=0 recomputes no layer at all, so SAC never applies"
        )

    context_fn = None
    if selective_n_layers > 0:
        if not enabled:
            logger.warning_once(
                f"selective_n_layers={selective_n_layers} ignored: SAC needs "
                "model.accelerator.gradient_checkpointing.enable=True; those layers fall back to full recomputation"
            )
        elif enable_reentrant:
            logger.warning_once(
                f"selective_n_layers={selective_n_layers} ignored: SAC needs "
                "enable_reentrant=False; those layers fall back to full recomputation"
            )
        elif offload_active:
            logger.warning_once(
                f"selective_n_layers={selective_n_layers} ignored: activation offload owns the "
                "checkpoint boundary; those layers fall back to full recomputation"
            )
        else:
            context_fn = _build_context_fn(extra_op_names)

    if context_fn is not None and compile_enabled:
        logger.warning_once("torch.compile is enabled; SAC with torch.compile is unverified")

    return RecomputePolicy(
        recompute_last_n_layers=recompute_last_n_layers,
        selective_n_layers=selective_n_layers,
        context_fn=context_fn,
        early_stop=early_stop,
        use_reentrant=enable_reentrant,
    )


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------


class Decision(IntEnum):
    """What one block does in the forward pass."""

    DIRECT = auto()  # call the block, keep its activations
    FULL = auto()  # checkpoint the block, recompute everything in backward
    SAC = auto()  # checkpoint the block, keep the selected ops' outputs


@dataclass(frozen=True)
class BlockPlan:
    """Pre-computed checkpoint behaviour of a single block.

    ``checkpoint_kwargs`` is empty for ``DIRECT``, which never checkpoints;
    otherwise it carries ``use_reentrant``, plus ``early_stop`` on the
    non-reentrant paths and ``context_fn`` for SAC.
    """

    decision: Decision
    checkpoint_kwargs: dict[str, Any]


def _clamp_recompute_n(count: int, total: int) -> int:
    """Clamp ``recompute_last_n_layers`` against the model depth; -1 = every layer."""
    if count < 0:
        return -1
    if count > total:
        logger.warning_once(f"recompute_last_n_layers={count} >= {total} blocks, using every layer")
        return -1
    return count


def _recompute_start(count: int, total: int) -> int:
    """First block of the recompute range; ``-1`` (every block) starts at 0."""
    return 0 if count < 0 else total - count


def plan_block(policy: RecomputePolicy, index: int, total: int) -> BlockPlan:
    """Decide how block ``index`` of ``total`` is recomputed.

    ``recompute_last_n_layers`` picks the range from the end, and within it
    ``selective_n_layers`` picks from the front. Index 0 is the first block the
    model runs, so the range is always the blocks nearest the loss — the same
    convention for every checkpoint style.
    """
    if not 0 <= index < total:
        raise ValueError(f"block index {index} out of range for a stack of {total} blocks")

    recompute_n = _clamp_recompute_n(policy.recompute_last_n_layers, total)
    recompute_start = _recompute_start(recompute_n, total)
    if index < recompute_start:
        return BlockPlan(Decision.DIRECT, {})

    if policy.use_reentrant:
        # torch rejects context_fn (and early_stop) on the reentrant path.
        return BlockPlan(Decision.FULL, {"use_reentrant": True})

    if policy.context_fn is not None and index - recompute_start < policy.selective_n_layers:
        return BlockPlan(
            Decision.SAC,
            {"use_reentrant": False, "context_fn": policy.context_fn, "early_stop": policy.early_stop},
        )
    return BlockPlan(Decision.FULL, {"use_reentrant": False, "early_stop": policy.early_stop})


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockStack:
    """A run of transformer blocks in execution order.

    ``parts`` names the containers it was folded from, so the startup log can
    show e.g. ``blocks+single_blocks (57 blocks, folded blocks=19, single_blocks=38)``.
    """

    parts: tuple[tuple[str, int], ...]
    modules: tuple[nn.Module, ...]

    @property
    def total(self) -> int:
        return len(self.modules)

    @property
    def fqn(self) -> str:
        return "+".join(fqn for fqn, _ in self.parts)

    def describe(self) -> str:
        counts = ", ".join(f"{fqn}={count}" for fqn, count in self.parts)
        folded = f", folded {counts}" if len(self.parts) > 1 else ""
        return f"{self.fqn} ({self.total} blocks{folded})"


@dataclass(frozen=True)
class _Candidate:
    """One container of matching blocks, before folding."""

    container_fqn: str
    parent_fqn: str
    modules: tuple[nn.Module, ...]

    @property
    def total(self) -> int:
        return len(self.modules)


def _target_class_names(model: nn.Module, basic_modules: Sequence[str] | None) -> tuple[str, ...]:
    """Class names that identify a transformer block in ``model``, in priority order.

    ``_no_split_modules`` plus ``basic_modules`` is what FSDP shards on, so both
    frameworks agree on what a block is. Only ``basic_modules`` carries an order
    worth reading — it comes from the configuration, so the operator's own
    ordering is kept. The model's ``_no_split_modules`` is a membership test:
    HF turns it into a set in ``post_init``, and set iteration order follows the
    process hash seed, so it is sorted to stay stable between runs. Models that
    declare neither fall back to the HF marker attribute, which checkpointing
    layers always carry.
    """
    names: list[str] = [name for name in dict.fromkeys(basic_modules or []) if name]
    names.extend(name for name in sorted(getattr(model, "_no_split_modules", None) or []) if name not in names)
    if names:
        return tuple(names)
    return tuple(
        dict.fromkeys(
            type(child).__name__
            for container in model.modules()
            if isinstance(container, (nn.ModuleList, nn.Sequential))
            for child in container
            if hasattr(child, "gradient_checkpointing")
        )
    )


def discover_block_stack(
    model: nn.Module, basic_modules: Sequence[str] | None = None
) -> tuple[BlockStack | None, list[BlockStack]]:
    """Find ``model``'s main block stack; report the stacks left out of the filter.

    Containers of matching blocks are folded into one sequence when they share a
    parent — flux runs ``blocks`` then ``single_blocks`` — while containers under
    another parent (a vision tower next to the text layers, say) are excluded:
    a single "last N layers" count cannot speak about two independent stacks.

    Which stack becomes the main one is decided from the model's structure alone,
    never from set iteration order, so two processes of the same run always
    schedule recomputation the same way — see :func:`_stack_rank`.
    """
    class_names = _target_class_names(model, basic_modules)
    declared_rank = {name: index for index, name in enumerate(basic_modules or []) if name}
    class_set = set(class_names)
    candidates: list[_Candidate] = []
    for fqn, container in model.named_modules():
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            continue
        blocks = _matching_children(container, class_set)
        if len(blocks) >= 2:
            candidates.append(
                _Candidate(
                    container_fqn=fqn or type(container).__name__,
                    parent_fqn=_parent_fqn(fqn),
                    modules=blocks,
                )
            )
    if not candidates:
        return None, []

    # named_modules() visits a container right before its children, in
    # registration order, so same-parent candidates are already in the order the
    # parent runs them.
    by_parent: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_parent.setdefault(candidate.parent_fqn, []).append(candidate)

    main_parent = max(by_parent, key=lambda parent: (_stack_rank(by_parent[parent], declared_rank), parent))
    main = by_parent.pop(main_parent)
    excluded = [_as_stack(candidate) for group in by_parent.values() for candidate in group]
    return _as_stack(*main), excluded


def _stack_rank(candidates: Sequence[_Candidate], declared_rank: dict[str, int]) -> tuple[int, int, int]:
    """Order candidate containers from the model's structure, never from a set.

    Highest wins, in this order:

    1. a block class named in ``basic_modules`` — an explicit statement about
       which stack the layer counts are meant for;
    2. a stack holding trainable parameters — a frozen tower runs without
       gradients, so there is nothing there to recompute;
    3. the stack with more blocks, and past that the parent's qualified name
       (applied by the caller), which no ordering can depend on.
    """
    undeclared = len(declared_rank)
    declared = min(
        (
            declared_rank.get(type(module).__name__, undeclared)
            for candidate in candidates
            for module in candidate.modules
        ),
        default=undeclared,
    )
    trainable = any(
        parameter.requires_grad
        for candidate in candidates
        for module in candidate.modules
        for parameter in module.parameters()
    )
    return (-declared, int(trainable), sum(candidate.total for candidate in candidates))


def _parent_fqn(fqn: str) -> str:
    return fqn.rsplit(".", 1)[0] if "." in fqn else ""


def _matching_children(container: nn.Module, class_names: set) -> tuple[nn.Module, ...]:
    return tuple(child for child in container if type(child).__name__ in class_names)


def _as_stack(*candidates: _Candidate) -> BlockStack:
    return BlockStack(
        parts=tuple((candidate.container_fqn, len(candidate.modules)) for candidate in candidates),
        modules=tuple(module for candidate in candidates for module in candidate.modules),
    )


# ---------------------------------------------------------------------------
# Install layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockBinding:
    """Where a block sits in its stack, and how it should be checkpointed."""

    index: int
    total: int
    plan: BlockPlan


@dataclass(frozen=True)
class RecomputeReport:
    """What :func:`apply_recompute_policy` did — returned to callers and logged."""

    policy: RecomputePolicy
    stack: BlockStack | None = None
    excluded: tuple[BlockStack, ...] = ()
    bound_blocks: int = 0
    patched_blocks: int = 0
    patched_containers: int = 0

    @property
    def bound(self) -> bool:
        return self.bound_blocks > 0

    @property
    def covered(self) -> bool:
        """Whether the model has a checkpoint entry point the policy could reach."""
        return self.patched_blocks > 0 or self.patched_containers > 0


#: Block -> binding, keyed by module so a weak reference keeps no model alive.
_block_bindings: WeakKeyDictionary[nn.Module, BlockBinding] = WeakKeyDictionary()


def apply_recompute_policy(
    model: nn.Module, policy: RecomputePolicy, *, basic_modules: Sequence[str] | None = None
) -> RecomputeReport:
    """Bind ``policy`` onto every checkpoint entry point of ``model``.

    Called by the framework after HF's ``gradient_checkpointing_enable`` and
    before FSDP sharding. Two entry points are replaced, which together cover
    every in-tree checkpointing style: a block's own
    ``_gradient_checkpointing_func`` (HF ``GradientCheckpointingLayer``), and the
    container's one that hand-written block loops call with the block as
    argument. Blocks without either are still registered, so
    :func:`checkpoint_forward` can resolve them.

    Never raises and touches nothing when the policy is inactive, so it is safe
    to call for every model and in tests that only assert HF's own kwargs.
    """
    if not policy.active:
        return RecomputeReport(policy=policy)

    # ``basic_modules`` is the configured list, so its order decides between
    # several declared block classes; the model's own ``_no_split_modules`` is
    # read as a set of names inside ``discover_block_stack``.
    stack, excluded = discover_block_stack(model, basic_modules)
    if stack is None:
        logger.warning_once(
            f"no block stack found in {type(model).__name__}; the layer counts and SAC are not applied"
        )
        return RecomputeReport(policy=policy)

    for index, block in enumerate(stack.modules):
        _block_bindings[block] = BlockBinding(
            index=index, total=stack.total, plan=plan_block(policy, index, stack.total)
        )

    patched_blocks = 0
    patched_containers = 0
    for module in model.modules():
        if "_gradient_checkpointing_func" not in vars(module):
            continue
        binding = _block_bindings.get(module)
        if binding is not None:
            patched_blocks += 1
        else:
            # A non-block holding the entry point is the container of a
            # hand-written loop (flux style) or a model that arms the function
            # itself (MiniMax-H3 style); either way the policy now reaches it.
            patched_containers += 1
        module._gradient_checkpointing_func = (
            _block_checkpoint_func(binding) if binding is not None else _container_checkpoint_func(policy)
        )

    report = RecomputeReport(
        policy=policy,
        stack=stack,
        excluded=tuple(excluded),
        bound_blocks=stack.total,
        patched_blocks=patched_blocks,
        patched_containers=patched_containers,
    )
    _log_recompute_report(report, model)
    return report


def _block_checkpoint_func(binding: BlockBinding) -> Callable[..., Any]:
    """Replacement for a block's own ``_gradient_checkpointing_func``.

    HF's ``GradientCheckpointingLayer.__call__`` hands us
    ``partial(super().__call__, **kwargs)`` plus the positional inputs, so the
    direct branch stays inside ``nn.Module.__call__`` and keeps every hook.
    """
    plan = binding.plan

    def checkpointed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if plan.decision is Decision.DIRECT:
            return func(*args, **kwargs)
        return _run_checkpoint(func, args, kwargs, plan.checkpoint_kwargs)

    checkpointed._veomni_layer_index = binding.index
    checkpointed._veomni_decision = plan.decision
    return checkpointed


def _container_checkpoint_func(policy: RecomputePolicy) -> Callable[..., Any]:
    """Replacement for a container's ``_gradient_checkpointing_func``.

    Hand-written block loops call the container with the block itself (or its
    ``__call__``) as first argument; the block is resolved from there and its
    binding decides. Blocks that were never bound keep full recomputation, which
    is what those loops did before.
    """
    fallback_kwargs: dict[str, Any] = {"use_reentrant": policy.use_reentrant}
    if not policy.use_reentrant:
        fallback_kwargs["early_stop"] = policy.early_stop

    def checkpointed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        block = _resolve_block(func, args)
        binding = _block_bindings.get(block) if block is not None else None
        if binding is None:
            return _run_checkpoint(func, args, kwargs, fallback_kwargs)
        if binding.plan.decision is Decision.DIRECT:
            return func(*args, **kwargs)
        return _run_checkpoint(func, args, kwargs, binding.plan.checkpoint_kwargs)

    return checkpointed


def _resolve_block(func: Any, args: Sequence[Any]) -> nn.Module | None:
    """Find the module a container-level checkpoint call is wrapping."""
    for candidate in (func, args[0] if args else None):
        if isinstance(candidate, nn.Module):
            return candidate
        if isinstance(candidate, functools.partial):
            inner = _resolve_block(candidate.func, ())
            if inner is not None:
                return inner
        owner = getattr(candidate, "__self__", None)
        if isinstance(owner, nn.Module):
            return owner
    return None


def _log_recompute_report(report: RecomputeReport, model: nn.Module) -> None:
    total = report.stack.total
    decisions = [plan_block(report.policy, index, total).decision for index in range(total)]
    logger.info_rank0(
        f"policy applied to {type(model).__name__}: stack={report.stack.describe()}; {_describe_decisions(decisions)}"
    )
    if not report.covered:
        # The stack was found but nothing on the model calls a checkpoint
        # function the framework can replace: a model driving
        # torch.utils.checkpoint itself never sees the policy.
        logger.info_rank0(
            f"no _gradient_checkpointing_func in {type(model).__name__}; blocks are registered for "
            "checkpoint_forward only, so a model that runs torch.utils.checkpoint itself is not covered "
            "unless it calls that entry point"
        )
    if Decision.DIRECT in decisions:
        logger.info_rank0(
            f"blocks outside recompute_last_n_layers={report.policy.recompute_last_n_layers} keep their "
            "activations (memory grows); selective_n_layers only says which recomputed blocks run SAC"
        )
    for stack in report.excluded:
        logger.warning_once(
            f"stack {stack.describe()} is not a sibling of {report.stack.fqn} and stays outside the layer counts"
        )


def _describe_decisions(decisions: Sequence[Decision]) -> str:
    """Render a per-layer decision list as ``blocks 0-9 full recompute, 10-19 SAC``."""
    labels = {Decision.DIRECT: "no recompute", Decision.FULL: "full recompute", Decision.SAC: "SAC"}
    spans: list[tuple[Decision, int, int]] = []
    for index, decision in enumerate(decisions):
        if spans and spans[-1][0] is decision:
            spans[-1] = (decision, spans[-1][1], index)
        else:
            spans.append((decision, index, index))
    return ", ".join(f"blocks {start}-{end} {labels[decision]}" for decision, start, end in spans)


# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------


def _create_custom_forward(function: Callable[..., Any]) -> Callable[..., Any]:
    def custom_forward(*inputs, **kwargs):
        return function(*inputs, **kwargs)

    return custom_forward


def _run_checkpoint(function: Callable[..., Any], args: tuple, kwargs: dict, checkpoint_kwargs: dict[str, Any]) -> Any:
    """Checkpoint one call, refusing the shape reentrant checkpointing cannot carry.

    Reentrant checkpointing re-runs the function from the tensors it was handed
    positionally: a keyword argument does not cross the boundary. Torch raises on
    its own, and folding the keywords into a closure instead would drop the
    gradient of every keyword tensor that needs one — so the combination is
    reported where it is decided, naming the key that turns it off.
    """
    if kwargs and checkpoint_kwargs.get("use_reentrant"):
        block = _resolve_block(function, args)
        raise ValueError(
            "model.accelerator.gradient_checkpointing.enable_reentrant=True cannot checkpoint a block "
            f"that is called with keyword arguments ({', '.join(sorted(kwargs))}); "
            f"{type(block).__name__ if block is not None else 'this block'} is. Reentrant checkpointing "
            "saves positional tensors only, so a keyword tensor would lose its gradient silently. "
            "Set enable_reentrant=False (the default), or pass those arguments positionally."
        )
    return torch.utils.checkpoint.checkpoint(_create_custom_forward(function), *args, **kwargs, **checkpoint_kwargs)


def checkpoint_forward(
    block: nn.Module,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``block`` with the recomputation strategy bound to it.

    Model-side entry point for a hand-written loop that owns its checkpoint
    call and finds the block itself; the framework binds the strategy, so the
    loop passes no layer information. No in-tree model needs it — loops that
    call ``self._gradient_checkpointing_func`` are patched directly — but it
    stays the documented way for a model to keep driving its own checkpointing.
    A block that was never bound is checkpointed in full, the default
    behaviour, and warns once when that happens while a policy is in force
    elsewhere.

    Highest priority first, and the same rule for every block:

    1. the block is outside the recompute range (``DIRECT``) — called directly;
    2. activation offload — checkpointed with ``save_on_cpu``, which is the
       offload switch's own boundary and carries no SAC context;
    3. checkpointing on — checkpointed with the block's plan, SAC or full;
    4. checkpointing off — called directly.
    """
    plan = _plan_for(block)

    if plan.decision is Decision.DIRECT:
        return block(*args, **kwargs)
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return _run_checkpoint(block, args, kwargs, {"use_reentrant": False})
    if use_gradient_checkpointing:
        return _run_checkpoint(block, args, kwargs, plan.checkpoint_kwargs)
    return block(*args, **kwargs)


def _plan_for(block: nn.Module) -> BlockPlan:
    binding = _block_bindings.get(block)
    if binding is not None:
        return binding.plan
    if _block_bindings:
        # Bindings exist, so a policy is in force somewhere and this block
        # escaped it: worth saying. With no binding at all the run is on the
        # defaults, where full recomputation is the documented behaviour.
        logger.warning_once(
            f"{type(block).__name__} has no recompute binding; recomputing it in full on every call "
            "(the framework binds blocks during model build)"
        )
    # Only ``use_reentrant`` is pinned here: this entry point is handed the block
    # alone, so no policy is in reach and the remaining checkpoint options stay
    # at torch's defaults (``early_stop=True``), rather than echoing a config
    # value that may say otherwise. ``early_stop`` only affects recomputation
    # speed, never the numbers.
    return BlockPlan(Decision.FULL, {"use_reentrant": False})
