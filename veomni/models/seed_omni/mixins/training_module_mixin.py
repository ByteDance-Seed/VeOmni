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

"""Training-graph hooks shared by every SeedOmni V2 sub-model."""

from __future__ import annotations

from typing import Any, Callable

from torch.nn import Module


def pre_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **pre-hook** for one or more training call-sites."""

    if not contexts:
        raise ValueError("@pre_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_pre_context = tuple(contexts)
        return fn

    return decorator


def post_forward(*contexts: str) -> Callable[[Callable], Callable]:
    """Decorator: register a **post-hook** for one or more training call-sites."""

    if not contexts:
        raise ValueError("@post_forward requires at least one context.")

    def decorator(fn: Callable) -> Callable:
        fn._omni_post_context = tuple(contexts)
        return fn

    return decorator


class TrainingModuleMixin:
    """Training-graph hooks — ``pre_forward`` / ``post_forward`` / ``forward`` / ``dummy_inputs``.

    Hook-name lookup lives on :class:`~veomni.models.seed_omni.mixins.base_mixin.BaseMixin`.

    Module-local ``TrainingMixin`` subclasses should define ``__init__`` to set
    training-side runtime caches after ``super().__init__(...)``.
    """

    def pre_forward(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch to the ``@pre_forward(method)``-decorated hook for this call-site."""
        name = type(self)._omni_hook_name("_omni_pre_context", method)
        if name is None:
            return kwargs
        return getattr(self, name)(**kwargs)

    def post_forward(self, method: str, **outputs: Any) -> dict[str, Any]:
        """Dispatch to the ``@post_forward(method)``-decorated hook for this call-site."""
        name = type(self)._omni_hook_name("_omni_post_context", method)
        if name is None:
            return outputs
        return getattr(self, name)(**outputs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Training-graph ``forward`` endpoint.

        Override on the module ``Accelerated`` class (or native ``modeling.py``)
        when this module appears in the training graph. The default skips mixin
        layers and delegates to the first concrete ``forward`` below this mixin
        in MRO so a stub here does not shadow HF-native implementations.
        """
        for base in type(self).__mro__[1:]:
            if base is TrainingModuleMixin or base is Module:
                continue
            impl = base.__dict__.get("forward")
            if impl is None:
                continue
            return impl(self, **kwargs)
        raise NotImplementedError(
            f"{type(self).__name__}.forward(**kwargs) is not implemented. "
            "Override it on the module Accelerated class if this module appears in the training graph."
        )

    def dummy_inputs(self, *, batch_size: int, device: Any, dtype: Any) -> dict[str, Any]:
        """Zero-tensor placeholders for training-side dummy forward."""
        del batch_size, device, dtype
        return {}


__all__ = ["TrainingModuleMixin", "post_forward", "pre_forward"]
