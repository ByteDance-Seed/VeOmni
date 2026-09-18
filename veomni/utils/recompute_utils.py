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
* :func:`checkpoint_forward` is the model-side entry for hand-written block loops
  (MiniMax-H3 style): it looks up the binding applied to the block.

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

from veomni.utils import helper


logger = helper.create_logger(__name__)

_LOG_PREFIX = "recompute: "

#: Concerns already reported, so the per-layer hot path logs at most once each.
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    """Log ``message`` at most once per process; ``key`` identifies the concern."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning("%s%s", _LOG_PREFIX, message)


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

#: Substring tokens matched against an operator's ``namespace::name`` as a
#: fallback when exact ``OpOverload`` resolution misses an operator (e.g. a
#: backend custom op not registered at configure time). Over-matching only
#: saves more activations (memory), never changes numerics.
DEFAULT_SELECTIVE_TOKENS: tuple[str, ...] = (
    "_scaled_dot_product",  # aten SDPA family (fused or math variants)
    "npu_fusion_attention",  # torch_npu fused flash attention
    "flash_attn",  # flash-attn-2/3 namespaces
    "flash_attention",
    "sageattn",
    "memory_efficient_attention",  # xformers
)

#: Substrings that must never be saved even if they match a token (e.g. the
#: decomposed efficient-attention helpers materialize the attention matrix,
#: which is exactly what we do not want to persist).
_SELECTIVE_IGNORE_TOKENS: tuple[str, ...] = ("_efficient_attention",)

#: Candidate operator namespaces probed for custom attention kernels.
_SELECTIVE_NS_CANDIDATES: tuple[str, ...] = (
    "npu",
    "npu_extension",
    "flashattn",
    "flash_attn",
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


def _token_matches_qualname(qualname: str, tokens: Sequence[str]) -> bool:
    lowered = qualname.lower()
    return any(t in lowered for t in tokens) and not any(i in lowered for i in _SELECTIVE_IGNORE_TOKENS)


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
    for ns_name in _SELECTIVE_NS_CANDIDATES:
        try:
            ns = getattr(torch.ops, ns_name)
        except AttributeError:
            continue
        for op_name in dir(ns):
            if not _token_matches_qualname(op_name, DEFAULT_SELECTIVE_TOKENS):
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
            logger.warning("%scannot resolve extra selective op %r (%s)", _LOG_PREFIX, op_str, exc)
            failed.append(op_str)

    # De-duplicate while preserving order (OpOverloads are hashable singletons).
    seen = set()
    unique = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            unique.append(op)
    return unique, failed


def _make_selective_policy(exact_ops: Sequence[Any], prefix_mode: bool) -> Callable[[Any, Any, tuple, dict], Any]:
    """Policy fn for ``create_selective_checkpoint_contexts``.

    Returns MUST_SAVE for resolved attention ops (exact identity) or, in prefix
    mode, for any op whose qualname matches a default token; otherwise
    PREFER_RECOMPUTE. Both directions of mismatch are numerically safe (they
    only change how much is saved vs recomputed).
    """
    op_set = set(exact_ops)

    def policy(ctx, op, *args, **kwargs):
        del ctx, args, kwargs  # unused
        if prefix_mode:
            if _token_matches_qualname(_op_qualname(op), DEFAULT_SELECTIVE_TOKENS):
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        elif op in op_set:
            return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _build_context_fn(extra_op_names: Sequence[str] | None) -> Callable[[], tuple[Any, Any]]:
    """Build the ``context_fn`` for ``torch.utils.checkpoint.checkpoint``."""
    exact_ops, failed = resolve_exact_ops(extra_op_names)
    if exact_ops:
        logger.info_rank0("%sSAC enabled with %d exact attention ops", _LOG_PREFIX, len(exact_ops))
        return functools.partial(torch.utils.checkpoint.create_selective_checkpoint_contexts, list(exact_ops))

    _warn_once(
        "sac-prefix",
        f"no attention operator resolved (failed extras: {failed or 'none'}); using the name-substring policy",
    )
    return functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, _make_selective_policy((), prefix_mode=True)
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
        _warn_once(
            "recompute-count",
            f"recompute_last_n_layers={recompute_last_n_layers} invalid (< -1), using -1 (every layer)",
        )
        recompute_last_n_layers = -1
    if selective_n_layers < 0:
        _warn_once(
            "selective-count",
            f"selective_n_layers={selective_n_layers} invalid (< 0), using 0 (SAC off); "
            "use a value >= the model depth for every recomputed layer",
        )
        selective_n_layers = 0
    if selective_n_layers > 0 and recompute_last_n_layers == 0:
        _warn_once(
            "selective-without-recompute",
            "selective_n_layers is set but recompute_last_n_layers=0 recomputes no layer at all, so SAC never applies",
        )

    context_fn = None
    if selective_n_layers > 0:
        if not enabled:
            _warn_once(
                "sac-disabled",
                f"selective_n_layers={selective_n_layers} ignored: SAC needs "
                "model.accelerator.gradient_checkpointing.enable=True; those layers fall back to full recomputation",
            )
        elif enable_reentrant:
            _warn_once(
                "sac-reentrant",
                f"selective_n_layers={selective_n_layers} ignored: SAC needs "
                "enable_reentrant=False; those layers fall back to full recomputation",
            )
        elif offload_active:
            _warn_once(
                "sac-offload",
                f"selective_n_layers={selective_n_layers} ignored: activation offload owns the "
                "checkpoint boundary; those layers fall back to full recomputation",
            )
        else:
            context_fn = _build_context_fn(extra_op_names)

    if context_fn is not None and compile_enabled:
        _warn_once("sac-compile", "torch.compile is enabled; SAC with torch.compile is unverified")

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

    ``checkpoint_kwargs`` always carries ``use_reentrant``; SAC adds
    ``context_fn``, non-reentrant full recomputation adds ``early_stop``.
    """

    decision: Decision
    checkpoint_kwargs: dict[str, Any]


def _clamp_recompute_n(count: int, total: int) -> int:
    """Clamp ``recompute_last_n_layers`` against the model depth; -1 = every layer."""
    if count < 0:
        return -1
    if count > total:
        _warn_once("clamped:recompute", f"recompute_last_n_layers={count} >= {total} blocks, using every layer")
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


def _target_class_names(model: nn.Module, target_classes: Sequence[str]) -> tuple[str, ...]:
    """Class names that identify a transformer block in ``model``, in priority order.

    ``_no_split_modules`` plus ``basic_modules`` is what FSDP shards on, so both
    frameworks agree on what a block is. The order is kept: a model declares its
    main block first, which is what tells a vision tower and a text decoder
    apart. Models that declare neither fall back to the HF marker attribute,
    which checkpointing layers always carry.
    """
    names: list[str] = []
    for name in target_classes:
        if name and name not in names:
            names.append(name)
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
    model: nn.Module, target_classes: Sequence[str]
) -> tuple[BlockStack | None, list[BlockStack]]:
    """Find ``model``'s main block stack; report the stacks left out of the filter.

    Containers of matching blocks are folded into one sequence when they share a
    parent — flux runs ``blocks`` then ``single_blocks`` — while containers under
    another parent (a vision tower next to the text layers, say) are excluded:
    a single "last N layers" count cannot speak about two independent stacks. The
    main stack is the one holding the block class the model declared first, and
    among those the deepest — so a vision tower with more blocks than the decoder
    still loses.
    """
    class_names = _target_class_names(model, target_classes)
    class_rank = {name: index for index, name in enumerate(class_names)}
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

    main_parent = max(by_parent, key=lambda parent: _stack_rank(by_parent[parent], class_rank))
    main = by_parent.pop(main_parent)
    excluded = [_as_stack(candidate) for group in by_parent.values() for candidate in group]
    return _as_stack(*main), excluded


def _stack_rank(candidates: Sequence[_Candidate], class_rank: dict[str, int]) -> tuple[int, int]:
    """Order candidate containers: declared block class first, then block count."""
    undeclared = len(class_rank)
    declared = min(
        class_rank.get(type(module).__name__, undeclared) for candidate in candidates for module in candidate.modules
    )
    return (-declared, sum(candidate.total for candidate in candidates))


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

    @property
    def bound(self) -> bool:
        return self.bound_blocks > 0


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

    # The model's own list comes first: it declares the main block class first,
    # and the framework's ``basic_modules`` is a set union by the time it arrives.
    stack, excluded = discover_block_stack(
        model, list(getattr(model, "_no_split_modules", None) or []) + list(basic_modules or [])
    )
    if stack is None:
        _warn_once(
            "no-stack",
            f"no block stack found in {type(model).__name__}; the layer counts and SAC are not applied",
        )
        return RecomputeReport(policy=policy)

    for index, block in enumerate(stack.modules):
        _block_bindings[block] = BlockBinding(
            index=index, total=stack.total, plan=plan_block(policy, index, stack.total)
        )

    patched_blocks = 0
    for module in model.modules():
        if "_gradient_checkpointing_func" not in vars(module):
            continue
        binding = _block_bindings.get(module)
        if binding is not None:
            patched_blocks += 1
        module._gradient_checkpointing_func = (
            _block_checkpoint_func(binding) if binding is not None else _container_checkpoint_func(policy)
        )

    report = RecomputeReport(
        policy=policy,
        stack=stack,
        excluded=tuple(excluded),
        bound_blocks=stack.total,
        patched_blocks=patched_blocks,
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
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs, **plan.checkpoint_kwargs)

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
            return torch.utils.checkpoint.checkpoint(func, *args, **kwargs, **fallback_kwargs)
        if binding.plan.decision is Decision.DIRECT:
            return func(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs, **binding.plan.checkpoint_kwargs)

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
        "%spolicy applied to %s: stack=%s; %s",
        _LOG_PREFIX,
        type(model).__name__,
        report.stack.describe(),
        _describe_decisions(decisions),
    )
    if report.patched_blocks == 0:
        # Two models land here and they cannot be told apart from the outside: one
        # reads the bindings through checkpoint_forward (MiniMax-H3 style), the
        # other drives torch.utils.checkpoint itself and never sees the policy.
        logger.info_rank0(
            "%sno block-level _gradient_checkpointing_func in %s; blocks are registered for "
            "checkpoint_forward only, so a model that runs torch.utils.checkpoint itself is not covered",
            _LOG_PREFIX,
            type(model).__name__,
        )
    if Decision.DIRECT in decisions:
        logger.info_rank0(
            "%sblocks outside recompute_last_n_layers=%d keep their activations (memory grows); "
            "selective_n_layers only says which recomputed blocks run SAC",
            _LOG_PREFIX,
            report.policy.recompute_last_n_layers,
        )
    for stack in report.excluded:
        _warn_once(
            f"excluded:{stack.fqn}",
            f"stack {stack.describe()} is not a sibling of {report.stack.fqn} and stays outside the layer counts",
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


def _create_custom_forward(module: nn.Module) -> Callable[..., Any]:
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def _run_checkpoint(block: nn.Module, args: tuple, kwargs: dict, checkpoint_kwargs: dict[str, Any]) -> Any:
    return torch.utils.checkpoint.checkpoint(_create_custom_forward(block), *args, **kwargs, **checkpoint_kwargs)


def checkpoint_forward(
    block: nn.Module,
    use_gradient_checkpointing: bool,
    use_gradient_checkpointing_offload: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``block`` with the recomputation strategy bound to it.

    Model-side entry point for hand-written block loops (MiniMax-H3 style); the
    framework binds the strategy, so the loop passes no layer information. A
    block that was never bound is checkpointed in full — the default behaviour —
    and warns once when that happens while a policy is in force elsewhere.

    Priority, same as the per-block entry points the framework set up: activation
    offload > SAC > full checkpoint > direct call.
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
        _warn_once(
            "unbound-block",
            f"{type(block).__name__} has no recompute binding; recomputing it in full on every call "
            "(the framework binds blocks during model build)",
        )
    # Only ``use_reentrant`` is pinned here: this entry point is handed the block
    # alone, so no policy is in reach and the remaining checkpoint options stay
    # at torch's defaults (``early_stop=True``), rather than echoing a config
    # value that may say otherwise. ``early_stop`` only affects recomputation
    # speed, never the numbers.
    return BlockPlan(Decision.FULL, {"use_reentrant": False})
