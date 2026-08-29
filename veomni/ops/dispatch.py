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
"""
OpSlot: a lightweight dispatch point used in generated modeling code.

An ``OpSlot`` is placed at the module level of a generated modeling file.
Inside the model's ``forward`` methods, the pattern is a simple 2-line guard::

    if veomni_moe_experts_forward.use_non_eager_impl:
        return veomni_moe_experts_forward(self, hidden_states, ...)
    # original HF code below, unchanged
"""

from __future__ import annotations

from typing import Any, Callable


class OpsConfigSlot:
    """A module-level config value bound from ``OpsImplementationConfig``."""

    def __init__(self, field_name: str, default: str = "eager"):
        self.field_name = field_name
        self._value = default

    def bind(self, ops_config: Any) -> None:
        self._value = getattr(ops_config, self.field_name)

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"OpsConfigSlot(field_name={self.field_name!r}, value={self._value!r})"


class OpSlot:
    """A named dispatch slot that can be bound to a kernel implementation."""

    def __init__(self, op_name: str, variant: str):
        self.op_name = op_name
        self.variant = variant
        self._kernel: Callable | None = None
        self._impl_name: str | None = None

    @property
    def use_non_eager_impl(self) -> bool:
        """``True`` when a non-eager kernel is bound.

        Named for the guard pattern at call sites: ``if slot.use_non_eager_impl:
        use replacement else fall through to eager HF code``. ``False`` covers
        both "bound to eager" (``KERNEL_REGISTRY.resolve`` returned ``None``)
        and "never bound".
        """
        return self._kernel is not None

    @property
    def use_eager_impl(self) -> bool:
        """``True`` only when the slot was explicitly bound to eager."""
        return self._impl_name == "eager"

    @property
    def is_bound(self) -> bool:
        """``True`` once ``bind()`` has been called."""
        return self._impl_name is not None

    def bound_kernel(self) -> Callable | None:
        """Return the resolved kernel callable, or ``None`` if eager / unbound.

        Use this when a model needs to *cache* the resolved implementation on
        an instance attribute (e.g. ``self.causal_conv1d_fn = slot.bound_kernel()``).
        Storing the OpSlot itself would couple the instance to the module-global
        slot, so a later ``bind()`` from a second model in the same process would
        silently override the first model's kernel.
        """
        return self._kernel

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._kernel is None:
            raise RuntimeError(
                f"OpSlot('{self.op_name}', '{self.variant}') has no kernel bound. "
                "Call .bind() first or check .use_non_eager_impl before calling."
            )
        return self._kernel(*args, **kwargs)

    def __repr__(self) -> str:
        if self._impl_name is None:
            state = "unbound"
        elif self._kernel is None:
            state = "eager"
        else:
            state = f"kernel={self._kernel}"
        return f"OpSlot(op_name={self.op_name!r}, variant={self.variant!r}, {state})"
