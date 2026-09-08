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

"""Operation registry.

``OP_REGISTRY`` stores ``OpEntry`` rows keyed by
``(op, variant, impl, device)``. Authors register with the public
triple; ``requirement.device`` (or ``ANY_DEVICE``) fills the fourth key.
``resolve_op`` still takes the triple and adds the current device.
``VeomniOp`` is a local handle that calls ``entry.wrapper``.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable

import torch
from torch import Tensor

from ..utils.device import get_device_type
from .requirement import ANY_DEVICE, KernelRequirement


Output = Tensor | tuple[Tensor, ...]


@dataclass(frozen=True)
class SavedState:
    """Tensors and non-tensor metadata produced by a raw ``forward``.

    ``tensors`` are what autograd ``save_for_backward`` stores. ``metadata``
    holds dims, flags, nested-handle specs, and other non-tensors.
    """

    tensors: tuple[Tensor, ...]
    metadata: Any = None


def _make_autograd_fn(raw_forward: Callable, raw_backward: Callable) -> Callable:
    """Build a modeling wrapper from raw ``forward`` / ``backward``."""

    positional_parameters = tuple(
        parameter
        for parameter in signature(raw_forward).parameters.values()
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )

    class _OpFn(torch.autograd.Function):
        """Generated Function that calls the raw pair and unpacks ``SavedState``."""

        @staticmethod
        def forward(ctx: Any, *args: Any) -> Output:
            """Run raw ``forward`` and stash tensors plus metadata on ``ctx``."""
            *tensors, attrs = args
            output, saved = raw_forward(*tensors, **attrs)
            if not isinstance(saved, SavedState):
                raise TypeError("raw forward must return (output, SavedState)")

            ctx.save_for_backward(*saved.tensors)
            ctx.saved_metadata = saved.metadata
            ctx.n_tensors = len(tensors)

            if not (
                isinstance(output, Tensor)
                or (isinstance(output, tuple) and output and all(isinstance(item, Tensor) for item in output))
            ):
                raise TypeError("raw forward output must be a Tensor or a non-empty tuple of Tensors")

            ctx.n_out = 1 if isinstance(output, Tensor) else len(output)
            return output

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            """Rebuild ``SavedState`` and return grads for the positional tensors."""
            saved = SavedState(ctx.saved_tensors, ctx.saved_metadata)
            grad_output: Output = grad_outputs[0] if ctx.n_out == 1 else grad_outputs
            grads = raw_backward(grad_output, saved)

            if not isinstance(grads, tuple):
                raise TypeError("raw backward must return a tuple of grads matching the positional tensors")
            if len(grads) != ctx.n_tensors:
                raise ValueError(f"raw backward returned {len(grads)} grads, expected {ctx.n_tensors}")
            return (*grads, None)

    def wrapper(*tensors: Tensor | None, **attrs: Any) -> Output:
        """Bind positional tensors, then pack keyword-only attrs for ``apply``."""
        bound_tensors = list(tensors)
        attrs = dict(attrs)
        # Autograd fixes backward arity from the actual ``apply`` arguments,
        # so materialize the raw signature's optional tensor slots as well.
        for parameter in positional_parameters[len(bound_tensors) :]:
            if parameter.name in attrs:
                bound_tensors.append(attrs.pop(parameter.name))
            elif parameter.default is not Parameter.empty:
                bound_tensors.append(parameter.default)
            else:
                break
        return _OpFn.apply(*bound_tensors, attrs)

    return wrapper


@dataclass
class OpEntry:
    """One registered row.

    Either a raw ``forward`` / ``backward`` pair (the wrapper is generated)
    or an opaque ``wrapper``. Hardware ``requirement`` is optional.
    """

    op: str
    variant: str
    impl: str
    forward: Callable | None = None
    backward: Callable | None = None
    wrapper: Callable | None = None
    requirement: KernelRequirement | None = None

    def __post_init__(self) -> None:
        """Validate the raw/wrapper pairing and generate the wrapper if needed."""
        if (self.forward is None) != (self.backward is None):
            raise ValueError("forward and backward must both be set or both be None")
        if self.forward is None and self.wrapper is None:
            raise ValueError("wrapper is required when raw is None")
        if self.forward is not None and self.wrapper is not None:
            raise ValueError("do not pass wrapper with raw math")
        if self.wrapper is None:
            self.wrapper = _make_autograd_fn(self.forward, self.backward)


def _entry_device(entry: OpEntry) -> str:
    """Return the fourth key for ``entry``."""
    if entry.requirement is None:
        return ANY_DEVICE
    return entry.requirement.device


class OpRegistry:
    """Global ``OpEntry`` table keyed by ``(op, variant, impl, device)``."""

    def __init__(self) -> None:
        """Create an empty registry table."""
        self._entries: dict[tuple[str, str, str, str], OpEntry] = {}

    def _requirement_matches(self, entry: OpEntry) -> bool:
        """Return whether ``entry.requirement`` is missing or matches this machine."""
        return entry.requirement is None or entry.requirement.matches()

    def register(self, entry: OpEntry) -> None:
        """Insert ``entry``. Duplicate ``(op, variant, impl, device)`` keys raise."""
        if not isinstance(entry, OpEntry):
            raise TypeError(f"OP_REGISTRY.register expects OpEntry, got {type(entry).__name__}")

        device = _entry_device(entry)
        key = (entry.op, entry.variant, entry.impl, device)
        if key in self._entries:
            raise ValueError(
                f"Duplicate op registration: op={entry.op!r}, "
                f"variant={entry.variant!r}, impl={entry.impl!r}, device={device!r}"
            )
        self._entries[key] = entry

    def resolve(self, op: str, variant: str, impl: str) -> OpEntry:
        """Return the row for ``(op, variant, impl)`` on this device.

        Looks up ``(op, variant, impl, current_device)``, then
        ``ANY_DEVICE``. Unknown triples raise ``KeyError``. A row that
        exists only for other devices, or whose ``requirement`` does not
        match, raises ``RuntimeError``.
        """
        device = get_device_type()
        entry = self._entries.get((op, variant, impl, device))
        if entry is None:
            entry = self._entries.get((op, variant, impl, ANY_DEVICE))
        if entry is None:
            devices = [
                entry_device
                for (entry_op, entry_variant, entry_impl, entry_device) in self._entries
                if entry_op == op and entry_variant == variant and entry_impl == impl
            ]
            if devices:
                raise RuntimeError(
                    f"Op {op!r} variant={variant!r} impl={impl!r} "
                    f"is not registered for device {device!r} (have {sorted(devices)})"
                )
            raise KeyError(f"Unknown op {op!r} variant={variant!r} impl={impl!r}")
        if entry.requirement is not None:
            try:
                entry.requirement.check()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Op {op!r} variant={variant!r} impl={impl!r} requirement is not satisfied: {exc}"
                ) from exc
        return entry

    def list_registered(self, op: str, variant: str) -> list[str]:
        """Return unique registered impl names for ``(op, variant)``."""
        seen: list[str] = []
        for entry_op, entry_variant, impl, _device in self._entries:
            if entry_op == op and entry_variant == variant and impl not in seen:
                seen.append(impl)
        return seen

    def list_available(self, op: str, variant: str) -> list[str]:
        """Return impl names for ``(op, variant)`` that match this machine."""
        device = get_device_type()
        seen: list[str] = []
        for (entry_op, entry_variant, impl, entry_device), entry in self._entries.items():
            if entry_op != op or entry_variant != variant:
                continue
            if entry_device not in (device, ANY_DEVICE):
                continue
            if not self._requirement_matches(entry):
                continue
            if impl not in seen:
                seen.append(impl)
        return seen


OP_REGISTRY = OpRegistry()


def register_op(
    op: str,
    variant: str,
    impl: str,
    forward: Callable | None = None,
    backward: Callable | None = None,
    *,
    wrapper: Callable | None = None,
    requirement: KernelRequirement | None = None,
) -> None:
    """Register one row on ``OP_REGISTRY``.

    Pass a raw pair or an opaque ``wrapper``, not both. ``requirement.device``
    (or ``ANY_DEVICE``) is the fourth key. ``resolve_op`` still takes the
    public triple.
    """
    OP_REGISTRY.register(
        OpEntry(
            op=op,
            variant=variant,
            impl=impl,
            forward=forward,
            backward=backward,
            wrapper=wrapper,
            requirement=requirement,
        )
    )


def resolve_op(op: str, variant: str, impl: str) -> OpEntry:
    """Look up ``(op, variant, impl)`` in ``OP_REGISTRY``."""
    return OP_REGISTRY.resolve(op, variant, impl)


class VeomniOp:
    """Local handle for one ``(op, variant, impl)`` row.

    Interned by triple. Always calls ``entry.wrapper``. Compound Functions
    must use ``resolve_op(...).forward`` / ``.backward``, not this handle.
    """

    _intern: dict[tuple[str, str, str], VeomniOp] = {}

    def __new__(cls, op: str, variant: str, impl: str = "eager"):
        """Return the interned handle for ``(op, variant, impl)``."""
        cached = cls._intern.get((op, variant, impl))
        if cached is not None:
            return cached
        return super().__new__(cls)

    def __init__(self, op: str, variant: str, impl: str = "eager"):
        """Resolve the registry row and intern this handle."""
        if getattr(self, "_entry", None) is not None:
            return
        self.op = op
        self.variant = variant
        self.impl = impl
        self._entry = resolve_op(op, variant, impl)
        type(self)._intern[(op, variant, impl)] = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``entry.wrapper``. Tensors are positional, non-tensors are keywords."""
        if self._entry.wrapper is None:
            raise RuntimeError(f"VeomniOp({self.op!r}, {self.variant!r}, {self.impl!r}) has no wrapper")
        return self._entry.wrapper(*args, **kwargs)

    @property
    def entry(self) -> OpEntry:
        """The resolved ``OpEntry`` for this handle."""
        return self._entry

    def __repr__(self) -> str:
        """Return ``VeomniOp(op=..., variant=..., impl=...)``."""
        return f"VeomniOp(op={self.op!r}, variant={self.variant!r}, impl={self.impl!r})"
