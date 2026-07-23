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

"""Unified step-level context for the Omni train / offline-cache paths.

Instead of hand-rolling a long, duplicated ``with (...)`` stack plus scattered
helper calls around every model forward/backward, the orchestrator opens a single
:func:`veomni_context` per phase:

    with veomni_context("forward", "train", micro_step, num_micro_steps):
        result = base.model(profiler=profiler, **micro_batch)

**Wiring is decided at setup, from args — like model construction.** There is a
static **catalog** of context *definitions* (populated at import by
:func:`register_veomni_context`). :func:`setup_veomni_context` runs once and asks
each catalog builder — given the live :class:`~veomni.arguments.OmniArguments`
(plus ``module_trainers``) — whether and how its context applies to *this* run. A
builder returns a **provider** to wire the context into the active set, or ``None``
to leave it out entirely. So an off-by-args context (offloading disabled,
batch-invariant off) is simply *absent* from the active set — no runtime flag
checks in the hot path.

At step time :func:`veomni_context` is an :class:`~contextlib.ExitStack` over the
**active** contexts. ``phase`` / ``mode`` are the only runtime gating dimensions
and they are supplied once, to the outer call; each context declares its applicable
``phases`` / ``modes`` at registration, and the composer skips a context whose
declared sets don't include the current ``phase`` / ``mode``. For the applicable
ones it calls the provider ``(micro_step, num_micro_steps)`` to get the actual
context manager (or ``None`` to skip this step, e.g. reshard without grad-accum).
Catalog (dict insertion) order is the nesting order — outermost first.

Only :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer` sets this up (via
``_setup_context``); the active set is wired from the run's args. Inference does
not use this composer at all — it needs nothing but ``no_grad``, which it wraps
directly around ``generate()``.
"""

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Literal, Mapping, Optional

import torch

from ...distributed.offloading import build_activation_offloading_context
from ...ops.batch_invariant_ops import set_batch_invariant_mode


if TYPE_CHECKING:
    from ...arguments import OmniArguments


Phase = Literal["forward", "backward"]
Mode = Literal["train", "offline_cache"]


# A provider produces the actual context manager (or ``None`` to skip this step)
# each time it is invoked; the loop values arrive as keywords, so a provider that
# ignores them takes ``**kwargs``.
#   provider(*, micro_step, num_micro_steps) -> CM | None
ContextProvider = Callable[..., Optional[ContextManager]]

# A builder is called ONCE at setup with the live args + module_trainers and decides
# whether this context is wired into the run: it returns a provider (to wire it in)
# or ``None`` (to leave it out — args-driven wiring, like model construction).
#   builder(args, module_trainers) -> provider | None
ContextBuilder = Callable[..., Optional[ContextProvider]]


@dataclass(frozen=True)
class _CatalogEntry:
    """A context *definition*: how to build it + the phases / modes it applies to."""

    build: ContextBuilder
    phases: frozenset  # subset of Phase
    modes: frozenset  # subset of Mode


@dataclass(frozen=True)
class _ActiveContext:
    """A context wired into the current run: its provider + the phases / modes it applies to."""

    provider: ContextProvider
    phases: frozenset  # subset of Phase
    modes: frozenset  # subset of Mode

    def applies(self, phase: Phase, mode: Mode) -> bool:
        return phase in self.phases and mode in self.modes


# Static catalog of all known context definitions (populated at import by the
# decorators below). Insertion order = nesting order (outermost first).
VEOMNI_CONTEXT_CATALOG: "OrderedDict[str, _CatalogEntry]" = OrderedDict()

# The contexts wired in for the current run, chosen from the catalog at setup by
# each builder from the live args. Empty until setup runs.
_ACTIVE_CONTEXTS: "OrderedDict[str, _ActiveContext]" = OrderedDict()


def register_veomni_context(
    name: str,
    *,
    phases: Iterable[Phase],
    modes: Iterable[Mode],
) -> Callable[[ContextBuilder], ContextBuilder]:
    """Register a context *builder* under ``name`` with its applicable phases / modes.

    The decorated function is a builder ``(args, module_trainers) -> provider | None``
    called once by :func:`setup_veomni_context`; returning ``None`` leaves the
    context out of the run entirely. ``phases`` / ``modes`` are the runtime gating
    sets. Catalog insertion order is the nesting order (outermost first).
    """

    def deco(build_fn: ContextBuilder) -> ContextBuilder:
        if name in VEOMNI_CONTEXT_CATALOG:
            raise ValueError(f"Duplicate veomni_context: {name!r}")
        VEOMNI_CONTEXT_CATALOG[name] = _CatalogEntry(build_fn, frozenset(phases), frozenset(modes))
        return build_fn

    return deco


def setup_veomni_context(
    args: Optional["OmniArguments"],
    module_trainers: "Optional[Mapping[str, Any]]" = None,
) -> None:
    """Wire the run's active contexts from the catalog, based on ``args`` (once, at init).

    Like model construction, this reads the live args and asks each catalog builder
    whether — and how — its context applies to this run; the ones that return a
    provider are wired into the active set (others are simply absent, so there are
    no runtime flag checks). Called by the trainer (``_setup_context``). Rebuilds the
    active set from scratch each call.
    """
    module_trainers = module_trainers if module_trainers is not None else {}
    _ACTIVE_CONTEXTS.clear()
    for name, entry in VEOMNI_CONTEXT_CATALOG.items():
        provider = entry.build(args, module_trainers)
        if provider is not None:
            _ACTIVE_CONTEXTS[name] = _ActiveContext(provider, entry.phases, entry.modes)


def format_active_contexts(stages: "Iterable[tuple[Phase, Mode]]") -> str:
    """Render the wired active contexts as a ``stage → contexts`` table.

    Each row is one execution ``stage`` (``phase/mode``) and the active contexts
    that apply to it, listed in nesting order (outermost first). Meant for a
    one-shot log right after :func:`setup_veomni_context`, so a run's actual
    context wiring is visible at a glance.
    """
    rows = []
    for phase, mode in stages:
        names = [name for name, entry in _ACTIVE_CONTEXTS.items() if entry.applies(phase, mode)]
        rows.append((f"{phase}/{mode}", ", ".join(names) if names else "(none)"))

    stage_header, ctx_header = "stage", "contexts (outer → inner)"
    stage_w = max([len(stage_header)] + [len(stage) for stage, _ in rows])
    ctx_w = max([len(ctx_header)] + [len(ctxs) for _, ctxs in rows])
    rule = f"+-{'-' * stage_w}-+-{'-' * ctx_w}-+"
    lines = [rule, f"| {stage_header.ljust(stage_w)} | {ctx_header.ljust(ctx_w)} |", rule]
    lines += [f"| {stage.ljust(stage_w)} | {ctxs.ljust(ctx_w)} |" for stage, ctxs in rows]
    lines.append(rule)
    return "\n".join(lines)


@contextmanager
def veomni_context(
    phase: Phase,
    mode: Mode,
    micro_step: int = 0,
    num_micro_steps: int = 1,
) -> Iterator[None]:
    """Compose the active contexts applicable to ``phase`` / ``mode``.

    ``phase`` / ``mode`` gate against each active context's declared sets; the
    provider produces the actual CM (or ``None`` to skip this step, e.g. reshard
    without grad-accum). Active-set (catalog) order is the nesting order (outermost
    entered first).
    """
    with ExitStack() as stack:
        for entry in _ACTIVE_CONTEXTS.values():  # insertion order = outer→inner
            if not entry.applies(phase, mode):
                continue
            # micro_step / num_micro_steps passed by keyword so providers that don't
            # need them can just swallow them with **kwargs.
            cm = entry.provider(micro_step=micro_step, num_micro_steps=num_micro_steps)
            if cm is not None:
                stack.enter_context(cm)
        yield


# ── Built-in context definitions ─────────────────────────────────────────────────
#
# Catalog order below IS the nesting order (outermost → innermost); do not reorder
# casually. Each builder decides from args whether — and how — its context is wired
# into the run, so the hot path carries no args-flag branching.


@register_veomni_context("no_grad", phases=("forward",), modes=("offline_cache",))
def _build_no_grad(args, module_trainers):
    # Always wired for its phase/mode: the offline-cache forward just encodes, no
    # backward, so grad is off. Training keeps grad on for the backward pass. A
    # fresh CM per step (torch.no_grad() is cheap; kept per-call for uniformity).
    return lambda **kwargs: torch.no_grad()


@register_veomni_context("activation_offloading_forward", phases=("forward",), modes=("train", "offline_cache"))
def _build_activation_offloading_forward(args, module_trainers):
    # Wired in only when offloading is enabled; the forward context is built ONCE
    # here and reused every step (matches BaseTrainer — the saved_tensors_hooks
    # instance is reusable across passes).
    offload = args.train.accelerator.offload_config if args is not None else None
    if offload is None or not offload.enable_activation:
        return None
    fwd_context, _ = build_activation_offloading_context(
        offload.enable_activation, args.train.gradient_checkpointing.enable, offload.activation_gpu_limit
    )
    return lambda **kwargs: fwd_context


@register_veomni_context("activation_offloading_backward", phases=("backward",), modes=("train", "offline_cache"))
def _build_activation_offloading_backward(args, module_trainers):
    # Wired in only when offloading is enabled; the backward context is built ONCE
    # here and reused every step. Backward only ever fires for train; offline_cache
    # is listed for symmetry but never enters a backward.
    offload = args.train.accelerator.offload_config if args is not None else None
    if offload is None or not offload.enable_activation:
        return None
    _, bwd_context = build_activation_offloading_context(
        offload.enable_activation, args.train.gradient_checkpointing.enable, offload.activation_gpu_limit
    )
    return lambda **kwargs: bwd_context


@register_veomni_context("batch_invariant", phases=("forward", "backward"), modes=("train", "offline_cache"))
def _build_batch_invariant(args, module_trainers):
    # Wired in only when enabled. set_batch_invariant_mode is a generator CM
    # (single-use), so build a FRESH one each step.
    if args is None or not args.train.enable_batch_invariant_mode:
        return None
    return lambda **kwargs: set_batch_invariant_mode(True)


@register_veomni_context("model_reshard", phases=("forward",), modes=("train", "offline_cache"))
def _build_model_reshard(args, module_trainers):
    # Wired in whenever there are module-trainers to cascade to (train / offline
    # cache). The no-op-without-accumulation decision is per step (depends on the
    # runtime num_micro_steps), so it lives in the provider; module_trainers is
    # captured here for the cascade.
    if not module_trainers:
        return None

    def provider(micro_step, num_micro_steps, **kwargs):
        if num_micro_steps <= 1:
            return None
        return _reshard_cm(module_trainers, micro_step, num_micro_steps)

    return provider


@contextmanager
def _reshard_cm(module_trainers, micro_step, num_micro_steps):
    """Cascade the grad-accum reshard intent into every module-trainer.

    The micro-step arithmetic is a *global* grad-accum concept (same for every
    module), so it lives here once: keep params gathered from the first
    micro-step (``reshard=False``) until the last (``reshard=True``), with
    nothing to toggle on intermediate steps. Each ``OmniModuleTrainer`` then
    decides for itself (from its own ``fsdp_config``) whether to apply it.
    """
    if micro_step == 0:
        reshard = False
    elif micro_step == num_micro_steps - 1:
        reshard = True
    else:
        reshard = None  # nothing to toggle on middle steps
    if reshard is not None:
        for module_trainer in module_trainers.values():
            module_trainer.model_reshard(reshard)
    yield


__all__ = [
    "Phase",
    "Mode",
    "ContextProvider",
    "ContextBuilder",
    "VEOMNI_CONTEXT_CATALOG",
    "register_veomni_context",
    "setup_veomni_context",
    "format_active_contexts",
    "veomni_context",
]
