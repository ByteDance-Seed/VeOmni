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

"""Generation-FSM hooks shared by every SeedOmni V2 sub-model."""

from __future__ import annotations

from typing import Any, Callable


def pre_generate(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for one or more inference call-sites."""

    if not contexts:
        raise ValueError("@pre_generate requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_pre_generate_context = tuple(contexts)
        return fn

    return decorator


def post_generate(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for one or more inference call-sites."""

    if not contexts:
        raise ValueError("@post_generate requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_post_generate_context = tuple(contexts)
        return fn

    return decorator


class InferenceModuleMixin:
    """Inference-graph hooks — ``pre_generate`` / ``post_generate`` / ``generate*`` / reset.

    Hook-name lookup lives on :class:`~veomni.models.seed_omni.mixins.base_mixin.BaseMixin`.

    Module-local ``InferenceMixin`` subclasses should define ``__init__`` to set
    inference-side runtime caches after ``super().__init__(...)``.
    """

    def pre_generate(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch to the ``@pre_generate(method)``-decorated hook for this call-site."""
        name = type(self)._omni_hook_name("_omni_pre_generate_context", method)
        if name is None:
            return kwargs
        return getattr(self, name)(**kwargs)

    def post_generate(self, method: str, **outputs: Any) -> dict[str, Any]:
        """Dispatch to the ``@post_generate(method)``-decorated hook for this call-site."""
        name = type(self)._omni_hook_name("_omni_post_generate_context", method)
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
        """Reset per-turn state inside an ongoing conversation."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full conversation-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Flush module-private generation buffers into a one-shot ``generated`` payload."""
        del ctx
        return {}


__all__ = ["InferenceModuleMixin", "post_generate", "pre_generate"]
