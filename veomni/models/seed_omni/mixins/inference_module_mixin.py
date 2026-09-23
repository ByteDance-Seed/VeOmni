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

"""Generation-FSM hooks shared by every SeedOmni sub-model.

:meth:`InferenceModuleMixin.pre_generate` / :meth:`~InferenceModuleMixin.post_generate`
wrap every graph endpoint in ``OmniModel._run_generation_node``, mirroring what
``pre_forward`` / ``post_forward`` do for the training walk. A module opts in by
decorating a method with ``@pre_generate("<method>")``; a module without the
mixin runs its endpoint bare.

:meth:`~InferenceModuleMixin.reset_global_inference_state` and
:meth:`~InferenceModuleMixin.finalize` are driven by ``OmniModel.reset`` and the
abort-only tail of ``OmniModel.generate``. ``finalize`` runs when the driver
hits ``max_new_tokens`` *before* ``done``, not after a normal FSM completion.

``generate_step`` still has no call-site: both FSM drivers resolve a bare
endpoint to ``generate``, never to ``generate_step``.
"""

from __future__ import annotations

from typing import Any, Callable

from .base_mixin import hook_name, mark_hook


def pre_generate(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for one or more inference call-sites."""

    if not contexts:
        raise ValueError("@pre_generate requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        return mark_hook(fn, "_omni_pre_generate_context", tuple(contexts))

    return decorator


def post_generate(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for one or more inference call-sites."""

    if not contexts:
        raise ValueError("@post_generate requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        return mark_hook(fn, "_omni_post_generate_context", tuple(contexts))

    return decorator


class InferenceModuleMixin:
    """Inference-graph hooks — ``pre_generate`` / ``post_generate`` / ``generate*`` / reset.

    Hook-name lookup is :func:`~veomni.models.seed_omni.mixins.base_mixin.hook_name`.

    Module-local ``InferenceMixin`` subclasses should define ``__init__`` to set
    inference-side runtime caches after ``super().__init__(...)``.
    """

    # Supplied by the host class this mixin is composed onto (the native
    # ``modeling.py`` class, or ``TrainingModuleMixin``); ``generate_step``
    # delegates to it. Annotation only — it must not shadow the real method.
    forward: Callable[..., dict[str, Any]]

    def pre_generate(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch to the ``@pre_generate(method)``-decorated hook for this call-site."""
        name = hook_name(type(self), "_omni_pre_generate_context", method)
        if name is None:
            return kwargs
        return getattr(self, name)(**kwargs)

    def post_generate(self, method: str, **outputs: Any) -> dict[str, Any]:
        """Dispatch to the ``@post_generate(method)``-decorated hook for this call-site."""
        name = hook_name(type(self), "_omni_post_generate_context", method)
        if name is None:
            return outputs
        return getattr(self, name)(**outputs)

    def generate_step(self, **kwargs: Any) -> dict[str, Any]:
        """Single FSM-driven generation step.

        Default: delegate to :meth:`forward`. Override when inference logic
        differs from training.
        """
        return self.forward(**kwargs)

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing generation request."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full request-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Abort-only flush of leftover buffers into a one-shot ``generated`` payload.

        Called when the FSM is cut short (``max_new_tokens``), not when it
        reaches ``done`` — that path already collected a complete span from
        ``generate``. Incomplete image tokens usually cannot be finalized;
        text might still emit.
        """
        del ctx
        return {}


__all__ = ["InferenceModuleMixin", "post_generate", "pre_generate"]
