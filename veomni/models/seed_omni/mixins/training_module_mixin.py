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

"""Training-graph hooks shared by every SeedOmni sub-model."""

from __future__ import annotations

from typing import Any, Callable

from torch.nn import Module

from .base_mixin import hook_name, mark_hook


def pre_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for one or more training call-sites."""

    if not contexts:
        raise ValueError("@pre_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        return mark_hook(fn, "_omni_pre_context", tuple(contexts))

    return decorator


def post_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for one or more training call-sites."""

    if not contexts:
        raise ValueError("@post_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        return mark_hook(fn, "_omni_post_context", tuple(contexts))

    return decorator


class TrainingModuleMixin:
    """Training-graph hooks — ``pre_forward`` / ``post_forward`` / ``forward`` / ``dummy_inputs``.

    Hook-name lookup is :func:`~veomni.models.seed_omni.mixins.base_mixin.hook_name`.

    Module-local ``TrainingMixin`` subclasses should define ``__init__`` to set
    training-side runtime caches after ``super().__init__(...)``.
    """

    def pre_forward(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch to the ``@pre_forward(method)``-decorated hook for this call-site."""
        name = hook_name(type(self), "_omni_pre_context", method)
        if name is None:
            return kwargs
        return getattr(self, name)(**kwargs)

    def post_forward(self, method: str, **outputs: Any) -> dict[str, Any]:
        """Dispatch to the ``@post_forward(method)``-decorated hook for this call-site."""
        name = hook_name(type(self), "_omni_post_context", method)
        if name is None:
            return outputs
        return getattr(self, name)(**outputs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Training-graph ``forward`` endpoint.

        Override on the native ``modeling.py`` class when this module appears in
        the training graph. The default skips mixin layers and delegates to the
        first concrete ``forward`` below this mixin in MRO so a stub here does
        not shadow HF-native implementations.
        """
        # Scan only past this mixin: a class above it whose ``forward`` calls
        # ``super().forward()`` lands here and must not be dispatched to again.
        mro = type(self).__mro__
        for base in mro[mro.index(TrainingModuleMixin) + 1 :]:
            if base is Module:
                continue
            impl = base.__dict__.get("forward")
            if impl is None:
                continue
            return impl(self, **kwargs)
        raise NotImplementedError(
            f"{type(self).__name__}.forward(**kwargs) is not implemented. "
            "Override it on the native modeling class if this module appears in the training graph."
        )

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> dict[str, Any]:
        """Zero-tensor placeholders for training-side dummy forward."""
        del batch_size, device, dtype
        return {}

    def on_micro_batches_ready(self, micro_batches: list[dict[str, Any]]) -> None:
        """Called once per optimizer step, with every micro-batch of that step.

        The training loop fetches the whole step before iterating it, so this is
        the only place a module can see quantities that span micro-batches —
        gradient accumulation is expressed as several micro-batches feeding one
        ``backward``, and a loss normalized per *step* (rather than per
        micro-batch) needs the step's total.

        The dLLM loss is normalized per *step*: its sample count spans the whole
        accumulation window and ``dllm_mean_global_loss`` divides by it; a dLLM
        module in this graph reads that contract from this hook. Default: no-op,
        which is right for every module whose loss is a per-micro-batch mean.

        Runs on the live module instance (below the FSDP2 wrap), before the
        first ``pre_forward`` of the step.
        """
        del micro_batches
        return None

    def on_step_context(self, *, seed: int, global_step: int, batch_invariant: bool = False) -> None:
        """Called once per optimizer step, before any micro-batch, with where training is.

        A module whose forward draws randomness per micro-batch needs to derive it
        from something a checkpoint can restore. ``(seed, global_step, micro_step)``
        is that: `_extra_state` persists only the default CPU RNG, so a generator
        carried across steps would rewind to step 0 on resume and replay the first
        steps' draws at step N. This hook hands over the step half of the key and is
        the reset point for the micro-batch counter the forward increments.

        Default: no-op, which is right for every deterministic forward.
        """
        del seed, global_step, batch_invariant
        return None

    #: Whether this module's ``_loss`` is already the step's mean, so the trainer
    #: must not divide the backward by ``num_micro_steps`` again.
    #:
    #: The executor scales every backward by ``1 / num_micro_steps`` so a
    #: per-micro-batch mean accumulates into the step mean. A module that instead
    #: normalizes by a quantity spanning the whole step -- ``dllm_mean_global_loss``
    #: divides by the step's global sample count, and all-reduces it -- already
    #: emits the step's mean from every micro-batch; that division would then scale
    #: its gradient down by ``num_micro_steps``. Such a module sets this to
    #: ``True`` and the trainer leaves its backward unscaled.
    #:
    #: A per-micro-batch mean (``reduce_sequence_parallel_loss``, the token-mean
    #: decode losses) leaves it ``False``: averaging those across the window is
    #: exactly what ``1 / num_micro_steps`` is for.
    loss_is_step_mean: bool = False


__all__ = ["TrainingModuleMixin", "post_forward", "pre_forward"]
