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

"""Kernel registry.

``KERNEL_REGISTRY`` stores ``KernelEntry`` rows keyed by
``(kernel, variant, impl, device)``. Authors register with the public
triple; ``requirement.device`` (or ``ANY_DEVICE``) fills the fourth key.
``resolve_kernel`` still takes the triple and adds the current device.
``VeomniKernel`` is a local handle that calls ``entry.wrapper``.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    class _KernelFn(torch.autograd.Function):
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

    def wrapper(*tensors: Tensor, **attrs: Any) -> Output:
        """Pack keyword attrs as the last ``apply`` argument."""
        return _KernelFn.apply(*tensors, attrs)

    return wrapper


@dataclass
class KernelEntry:
    """One registered row.

    Either a raw ``forward`` / ``backward`` pair (the wrapper is generated)
    or an opaque ``wrapper``. Hardware ``requirement`` is optional.
    """

    kernel: str
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


def _entry_device(entry: KernelEntry) -> str:
    """Return the fourth key for ``entry``."""
    if entry.requirement is None:
        return ANY_DEVICE
    return entry.requirement.device


class KernelRegistry:
    """Global ``KernelEntry`` table keyed by ``(kernel, variant, impl, device)``."""

    def __init__(self) -> None:
        """Create an empty registry table."""
        self._entries: dict[tuple[str, str, str, str], KernelEntry] = {}

    def _requirement_matches(self, entry: KernelEntry) -> bool:
        """Return whether ``entry.requirement`` is missing or matches this machine."""
        return entry.requirement is None or entry.requirement.matches()

    def register(self, entry: KernelEntry) -> None:
        """Insert ``entry``. Duplicate ``(kernel, variant, impl, device)`` keys raise."""
        if not isinstance(entry, KernelEntry):
            raise TypeError(f"KERNEL_REGISTRY.register expects KernelEntry, got {type(entry).__name__}")

        device = _entry_device(entry)
        key = (entry.kernel, entry.variant, entry.impl, device)
        if key in self._entries:
            raise ValueError(
                f"Duplicate kernel registration: kernel={entry.kernel!r}, "
                f"variant={entry.variant!r}, impl={entry.impl!r}, device={device!r}"
            )
        self._entries[key] = entry

    def resolve(self, kernel: str, variant: str, impl: str) -> KernelEntry:
        """Return the row for ``(kernel, variant, impl)`` on this device.

        Looks up ``(kernel, variant, impl, current_device)``, then
        ``ANY_DEVICE``. Unknown triples raise ``KeyError``. A row that
        exists only for other devices, or whose ``requirement`` does not
        match, raises ``RuntimeError``.
        """
        device = get_device_type()
        entry = self._entries.get((kernel, variant, impl, device))
        if entry is None:
            entry = self._entries.get((kernel, variant, impl, ANY_DEVICE))
        if entry is None:
            devices = [
                entry_device
                for (entry_kernel, entry_variant, entry_impl, entry_device) in self._entries
                if entry_kernel == kernel and entry_variant == variant and entry_impl == impl
            ]
            if devices:
                raise RuntimeError(
                    f"Kernel {kernel!r} variant={variant!r} impl={impl!r} "
                    f"is not registered for device {device!r} (have {sorted(devices)})"
                )
            raise KeyError(f"Unknown kernel {kernel!r} variant={variant!r} impl={impl!r}")
        if entry.requirement is not None:
            try:
                entry.requirement.check()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Kernel {kernel!r} variant={variant!r} impl={impl!r} requirement is not satisfied: {exc}"
                ) from exc
        return entry

    def list_registered(self, kernel: str, variant: str) -> list[str]:
        """Return unique registered impl names for ``(kernel, variant)``."""
        seen: list[str] = []
        for entry_kernel, entry_variant, impl, _device in self._entries:
            if entry_kernel == kernel and entry_variant == variant and impl not in seen:
                seen.append(impl)
        return seen

    def list_available(self, kernel: str, variant: str) -> list[str]:
        """Return impl names for ``(kernel, variant)`` that match this machine."""
        device = get_device_type()
        seen: list[str] = []
        for (entry_kernel, entry_variant, impl, entry_device), entry in self._entries.items():
            if entry_kernel != kernel or entry_variant != variant:
                continue
            if entry_device not in (device, ANY_DEVICE):
                continue
            if not self._requirement_matches(entry):
                continue
            if impl not in seen:
                seen.append(impl)
        return seen


KERNEL_REGISTRY = KernelRegistry()


def register_kernel(
    kernel: str,
    variant: str,
    impl: str,
    forward: Callable | None = None,
    backward: Callable | None = None,
    *,
    wrapper: Callable | None = None,
    requirement: KernelRequirement | None = None,
) -> None:
    """Register one row on ``KERNEL_REGISTRY``.

    Pass a raw pair or an opaque ``wrapper``, not both. ``requirement.device``
    (or ``ANY_DEVICE``) is the fourth key. ``resolve_kernel`` still takes the
    public triple.
    """
    KERNEL_REGISTRY.register(
        KernelEntry(
            kernel=kernel,
            variant=variant,
            impl=impl,
            forward=forward,
            backward=backward,
            wrapper=wrapper,
            requirement=requirement,
        )
    )


def resolve_kernel(kernel: str, variant: str, impl: str) -> KernelEntry:
    """Look up ``(kernel, variant, impl)`` in ``KERNEL_REGISTRY``."""
    return KERNEL_REGISTRY.resolve(kernel, variant, impl)


class VeomniKernel:
    """Local handle for one ``(kernel, variant, impl)`` row.

    Interned by triple. Always calls ``entry.wrapper``. Compound Functions
    must use ``resolve_kernel(...).forward`` / ``.backward``, not this handle.
    """

    _intern: dict[tuple[str, str, str], VeomniKernel] = {}

    def __new__(cls, kernel: str, variant: str, impl: str = "eager"):
        """Return the interned handle for ``(kernel, variant, impl)``."""
        cached = cls._intern.get((kernel, variant, impl))
        if cached is not None:
            return cached
        return super().__new__(cls)

    def __init__(self, kernel: str, variant: str, impl: str = "eager"):
        """Resolve the registry row and intern this handle."""
        if getattr(self, "_entry", None) is not None:
            return
        self.kernel = kernel
        self.variant = variant
        self.impl = impl
        self._entry = resolve_kernel(kernel, variant, impl)
        type(self)._intern[(kernel, variant, impl)] = self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``entry.wrapper``. Tensors are positional, non-tensors are keywords."""
        if self._entry.wrapper is None:
            raise RuntimeError(f"VeomniKernel({self.kernel!r}, {self.variant!r}, {self.impl!r}) has no wrapper")
        return self._entry.wrapper(*args, **kwargs)

    @property
    def entry(self) -> KernelEntry:
        """The resolved ``KernelEntry`` for this handle."""
        return self._entry

    def __repr__(self) -> str:
        """Return ``VeomniKernel(kernel=..., variant=..., impl=...)``."""
        return f"VeomniKernel(kernel={self.kernel!r}, variant={self.variant!r}, impl={self.impl!r})"
