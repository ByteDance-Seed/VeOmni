# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Explicit extension point for precision-preserving rank-local optimizer DCP."""

from contextlib import nullcontext
from dataclasses import dataclass

from torch.optim import Optimizer


_RANK_LOCAL_ADAMW_ADAPTERS = {}


@dataclass(frozen=True)
class RankLocalAdamWAdapter:
    context_factory: object
    algorithm_identity: object
    state_dtype: object
    state_is_offloaded: object


def register_rank_local_adamw(optimizer_type, context_factory, algorithm_identity, state_dtype, state_is_offloaded):
    """Register a trusted exact optimizer type and its sparse checkpoint context.

    The context must expose only actually initialized AdamW master/moment state,
    preserve parameter-group semantics, not mutate live training state while
    serializing, and accept rank-local scalar steps. Registered implementations
    are trusted project code; this is not a sandbox for arbitrary callbacks.
    The identity callback must return immutable values covering every
    non-group setting that can alter updates or restore behavior. The dtype and
    offload callbacks describe each parameter's actual step, master and moment
    state; the step dtype must describe its scalar, not the moment dtype.
    Registration is an internal compatibility contract, not a class-wide opt-in.
    """
    if not isinstance(optimizer_type, type) or not issubclass(optimizer_type, Optimizer):
        raise TypeError("Rank-local AdamW adapter requires a torch Optimizer subclass")
    if not callable(context_factory):
        raise TypeError("Rank-local AdamW adapter context must be callable")
    if not callable(algorithm_identity):
        raise TypeError("Rank-local AdamW adapter algorithm identity must be callable")
    if not callable(state_dtype) or not callable(state_is_offloaded):
        raise TypeError("Rank-local AdamW state dtype and offload policies must be callable")
    if optimizer_type in _RANK_LOCAL_ADAMW_ADAPTERS:
        raise ValueError(f"Optimizer checkpoint adapter already registered: {optimizer_type!r}")
    _RANK_LOCAL_ADAMW_ADAPTERS[optimizer_type] = RankLocalAdamWAdapter(
        context_factory, algorithm_identity, state_dtype, state_is_offloaded
    )


def rank_local_adamw_adapter(optimizer):
    return _RANK_LOCAL_ADAMW_ADAPTERS.get(type(optimizer))


def rank_local_adamw_context(optimizer):
    adapter = rank_local_adamw_adapter(optimizer)
    return adapter.context_factory(optimizer) if adapter is not None else nullcontext()
