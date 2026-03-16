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
At model-build time, ``_bind_veomni_ops`` resolves each slot to a concrete
kernel (or ``None`` for eager) via the global ``KERNEL_REGISTRY``.
Inside the model's ``forward`` methods, the pattern is a simple 2-line guard::

    if veomni_moe_experts_forward.has_kernel:
        return veomni_moe_experts_forward(self, hidden_states, ...)
    # original HF code below, unchanged
"""

from __future__ import annotations

from typing import Any, Callable

from .kernel_registry import KERNEL_REGISTRY


class OpSlot:
    """A named dispatch slot that can be bound to a kernel implementation."""

    def __init__(self, op_name: str, variant: str):
        self.op_name = op_name
        self.variant = variant
        self._kernel: Callable | None = None
        self._bound = False

    def bind(self, impl_name: str) -> None:
        """Resolve *impl_name* via the global registry and bind the result."""
        self._kernel = KERNEL_REGISTRY.resolve(self.op_name, self.variant, impl_name)
        self._bound = True

    @property
    def has_kernel(self) -> bool:
        """``True`` when a non-eager kernel is bound."""
        return self._kernel is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._kernel is None:
            raise RuntimeError(
                f"OpSlot('{self.op_name}', '{self.variant}') has no kernel bound. "
                "Call .bind() first or check .has_kernel before calling."
            )
        return self._kernel(*args, **kwargs)

    def __repr__(self) -> str:
        state = "unbound"
        if self._bound:
            state = f"kernel={self._kernel}" if self._kernel else "eager"
        return f"OpSlot(op_name={self.op_name!r}, variant={self.variant!r}, {state})"
