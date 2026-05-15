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

"""MoE Router Replay hook.

Provides a registration + helper API that downstream RL training frameworks
(e.g. verl) use to deterministically **record** or **replay** MoE routing
decisions across the log-prob forward and policy-update forward/backward
passes of a single training step. This makes gradients computed across the
two passes consistent with each other (the so-called R2 / R3 routing replay;
see the verl docs for the motivation).

Typical usage, from the RL framework side::

    from veomni.utils.moe_router_replay import set_active_replay
    set_active_replay(my_manager)     # switch router hooks ON
    ...                                # run forward / backward
    set_active_replay(None)            # switch OFF, restoring vanilla behavior

Hook scope
----------
Each supported MoE family's patched ``SparseMoeBlock.forward`` calls
:func:`maybe_replay_indices` immediately after the native router produces
its top-k. The hook is **indices-only**: the manager either stores them
(RECORD), substitutes previously-recorded target indices (REPLAY), or
returns the input unchanged (no manager active). All model-specific
post-topk weight math — ``softmax`` recompute if the native router does
not surface its scoring matrix, ``gather``, ``renorm``, ``scaling``,
dtype cast, etc. — lives in the per-family ``SparseMoeBlock.forward``
patch, not here. That keeps the cross-framework controller
model-agnostic: adding a new MoE family (DeepSeek-V3, GLM-MoE-DSA,
Seed-OSS, ...) touches only the new family's gen_config, never this
module or the downstream manager implementation.

API contract
------------
``maybe_replay_indices(module, routing_scores, top_indices) -> indices``
is the only interaction point. Semantics:

* ``module`` is the router ``nn.Module`` instance (stable ``id()`` keys
  per-layer state in the manager).
* ``routing_scores`` is the post-activation scoring matrix on which
  ``torch.topk`` was drawn, shape ``[num_tokens, num_experts]``. It is
  passed through so managers can do shape/device sanity checks, but
  managers MUST NOT use it to derive replay weights — weight math is
  the caller's responsibility.
* ``top_indices`` is the router's native top-k choice, shape
  ``[num_tokens, top_k]``, dtype int.
* Return value: a ``[num_tokens, top_k]`` int tensor of indices the
  caller should use downstream. In RECORD mode and when no manager is
  active, this equals ``top_indices``; in REPLAY mode it is the target
  recorded on a previous forward (or falls back to ``top_indices`` if
  the manager has no target yet, e.g. during a shape-probe pass).

Numerical guarantee (per-family, enforced by the patched forward)
-----------------------------------------------------------------
When a manager is active, the caller recomputes the expert weights from
the returned indices using the **exact** arithmetic the native router
would apply — same activation, same ``gather``, same ``renorm``, same
cast — so the REPLAY forward is bit-identical to what the native forward
would have produced had ``torch.topk`` picked the target indices. The
RECORD path additionally re-derives the weights from the (unchanged)
native indices, paying one extra ``gather + renorm + cast`` per MoE
layer; the result is bit-equal to the native ``routing_weights`` the
gate returned. On the default training path (no manager installed) the
guarded block is skipped and forward is bit-identical to upstream.

A new family must verify this guarantee when wiring its patch:

1. Read the native ``TopKRouter.forward`` end-to-end.
2. Identify the scoring matrix fed to ``topk`` (post-softmax /
   post-sigmoid / post-bias / whatever the router produces). Pass it as
   ``routing_scores``.
3. Replicate *every* post-topk operation (renorm, scaling factor, dtype
   cast, etc.) in the guarded block after ``maybe_replay_indices``
   returns.

Concurrency
-----------
The active-manager slot is a module-level singleton (mirroring the
``set_active_monitor`` pattern in :mod:`veomni.utils.moe_monitor`). It is
**not thread-safe** and assumes the caller's training loop is
single-threaded per process — which holds for verl's Ray-worker driven
forward/backward loop. Gradient-checkpointing recompute re-enters
``on_router_forward`` in the same thread with the same active-manager
state (RECORD or REPLAY stays set for the whole step), which is the
intended behavior. Do not install a new manager while a forward pass is
in flight.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


__all__ = [
    "SUPPORTED_MOE_MODEL_TYPES",
    "get_active_replay",
    "maybe_replay_indices",
    "set_active_replay",
    "validate_model_for_replay",
]


# Registry of HuggingFace `model_type` strings whose VeOmni MoE
# SparseMoeBlock.forward (or route_tokens_to_experts) has been wired to call
# `maybe_replay_indices`. Extend when wiring replay into a new model family.
SUPPORTED_MOE_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen3_moe",
        "qwen3_5_moe",
    }
)


_active_manager: Any | None = None


def get_active_replay() -> Any | None:
    """Return the currently active router-replay manager, or None if inactive."""
    return _active_manager


def set_active_replay(manager: Any | None) -> None:
    """Install (or clear) a router-replay manager.

    Passing ``None`` disables replay and reverts every router to its native
    top-k selection on the next forward.
    """
    global _active_manager
    _active_manager = manager


def maybe_replay_indices(
    module: nn.Module,
    routing_scores: torch.Tensor,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    """Forward-time hook called by MoE ``SparseMoeBlock`` implementations.

    When no manager is active this returns ``top_indices`` unchanged.
    Otherwise it hands control to the manager, which either stores
    ``top_indices`` (RECORD mode) and returns them, or substitutes a
    previously recorded target tensor (REPLAY mode).

    The manager is duck-typed; it MUST implement::

        on_router_forward(module, routing_scores, top_indices) -> indices

    The caller is responsible for computing replay weights from the
    returned indices using the model-specific post-topk math — see the
    module docstring for the full contract.
    """
    manager = _active_manager
    if manager is None:
        return top_indices
    return manager.on_router_forward(module, routing_scores, top_indices)


def validate_model_for_replay(model: nn.Module) -> None:
    """Raise if ``model`` is not wired for router replay.

    Router replay relies on the model's MoE ``SparseMoeBlock.forward`` (or
    equivalent) calling :func:`maybe_replay_indices`. If a caller enables
    replay on a model whose family has not yet been patched, no router
    ever fires the hook and downstream controller state stays empty —
    later surfacing as a confusing ``"collect called before any router
    fired"`` error far from the root cause.

    This function fails fast at controller install time with a message
    pointing the user at the concrete fix: extend
    :data:`SUPPORTED_MOE_MODEL_TYPES` and patch the corresponding
    ``SparseMoeBlock.forward`` / ``route_tokens_to_experts`` method.

    Args:
        model: The top-level model (usually the HF ``*ForCausalLM``) whose
            ``config.model_type`` identifies the family. If ``model`` is a
            distributed-training wrapper (DDP/FSDP1-style) that hides the
            real model under ``.module``, that fallback is checked too.

    Raises:
        RuntimeError: If ``model.config.model_type`` is missing, or is not
            in :data:`SUPPORTED_MOE_MODEL_TYPES`.
    """
    # Look on the wrapper first, then peel one ``.module`` layer if needed —
    # covers `DistributedDataParallel`, `PeftModel`, and similar wrappers.
    # FSDP2 wraps in-place so the first lookup already succeeds for it.
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is None:
        raise RuntimeError(
            "router replay: cannot determine model.config.model_type. "
            "Router replay requires a model with a recognized HuggingFace config."
        )
    if model_type not in SUPPORTED_MOE_MODEL_TYPES:
        supported = sorted(SUPPORTED_MOE_MODEL_TYPES)
        raise RuntimeError(
            f"router replay is not wired for model_type={model_type!r}. "
            f"Supported model types: {supported}. "
            "To add support, patch the corresponding SparseMoeBlock.forward "
            "(or route_tokens_to_experts) to call "
            "`veomni.utils.moe_router_replay.maybe_replay_indices`, then extend "
            "`SUPPORTED_MOE_MODEL_TYPES` in this file."
        )
