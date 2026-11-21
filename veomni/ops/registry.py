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
Hardware ``requirement`` and software ``requires`` decide availability.
``VeomniOp`` is a local handle that calls ``entry.wrapper``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any, Callable

import torch
from torch import Tensor

from ..utils.device import get_device_type
from ..utils.import_utils import is_package_available
from .platform import ANY_DEVICE, KernelRequirement
from .selection import OpsSelection


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

    raw_parameters = signature(raw_forward).parameters
    positional_parameters = tuple(
        parameter
        for parameter in raw_parameters.values()
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    accepts_grad_enabled = "grad_enabled" in raw_parameters

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

            ctx.output_is_tuple = isinstance(output, tuple)
            if torch.compiler.is_compiling():
                # Dynamo's AutogradFunctionApplyVariable appends every intermediate
                # FX tensor node in ``forward`` after the declared outputs. If a
                # trailing no-op op (such as ``.to(x.dtype)`` or ``.contiguous()``)
                # returns the same Tensor object as an earlier intermediate node,
                # C++ autograd overwrites the output's ``output_nr_`` with the
                # intermediate slot and routes ``grad_output`` to an unused slot.
                output = tuple(item.clone() for item in output) if ctx.output_is_tuple else output.clone()
            return output

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            """Rebuild ``SavedState`` and return grads for the positional tensors."""
            saved = SavedState(ctx.saved_tensors, ctx.saved_metadata)
            grad_output: Output = grad_outputs if ctx.output_is_tuple else grad_outputs[0]
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
        # Function.forward always runs under no_grad. Opt-in raw pairs that
        # accept ``grad_enabled`` still see the caller's autograd mode.
        if accepts_grad_enabled:
            attrs.setdefault("grad_enabled", torch.is_grad_enabled())
        # Nothing needs a backward, so skip the autograd node. This is not only
        # cheaper: Dynamo cannot bind ``_OpFn.forward`` in inference mode, where
        # the variadic tensors are followed by a non-tensor argument. Inlining it
        # reports ``args = []`` and fails with "failed to bind arguments when
        # attempting to inline", which breaks every ``torch.compile``d forward
        # that runs under ``no_grad``/``inference_mode``. Calling the raw pair
        # directly keeps that path compilable and produces the same values.
        needs_graph = torch.is_grad_enabled() and any(
            isinstance(tensor, Tensor) and tensor.requires_grad for tensor in bound_tensors
        )
        if not needs_graph:
            return raw_forward(*bound_tensors, **attrs)[0]
        return _OpFn.apply(*bound_tensors, attrs)

    return wrapper


@dataclass
class OpEntry:
    """One registered row.

    Either a raw ``forward`` / ``backward`` pair (the wrapper is generated)
    or an opaque ``wrapper``. ``description`` records stable implementation
    semantics for discovery. Hardware ``requirement`` and software
    ``requires`` are optional and independently participate in resolution.
    ``default`` marks device-specific rows that act as the host default for
    ``(op, variant)`` in registration order.
    """

    op: str
    variant: str
    impl: str
    description: str = field(kw_only=True)
    forward: Callable | None = None
    backward: Callable | None = None
    wrapper: Callable | None = None
    requirement: KernelRequirement | None = None
    requires: tuple[str, ...] = ()
    default: bool = False
    #: Numeric formats this row can round to, as the values its ``dtype`` argument
    #: accepts. Empty for a row that rounds nothing. The format is a row *argument*
    #: rather than part of the name: ``qat`` serves FP8 and FP4 checkpoints, and
    #: ``torchao`` serves whatever torchao can quantize, so naming the row after one
    #: format would either multiply rows or lie.
    dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the raw/wrapper pairing and generate the wrapper if needed."""
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.requires, tuple) or not all(
            isinstance(package, str) and package.strip() for package in self.requires
        ):
            raise TypeError("requires must be a tuple of non-empty import package names")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("requires must not contain duplicate package names")
        if not isinstance(self.default, bool):
            raise TypeError("default must be a bool")
        if not isinstance(self.dtypes, tuple) or not all(
            isinstance(name, str) and name.strip() for name in self.dtypes
        ):
            raise TypeError("dtypes must be a tuple of non-empty format names")
        if self.default and self.requirement is None:
            raise ValueError(
                "default=True requires a device-specific requirement; eager is already the fallback default"
            )
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

    def _requires_satisfied(self, entry: OpEntry) -> bool:
        """Return whether software package requirements for ``entry`` are satisfied."""
        from ..utils import import_utils

        device = _entry_device(entry)
        for package in entry.requires:
            if device == "mlu" and package == "apex":
                if not import_utils.is_apex_mlu_available():
                    return False
            elif device == "mlu" and package == "liger_kernel":
                if not import_utils.is_liger_kernel_mlu_available():
                    return False
            elif not is_package_available(package):
                return False
        return True

    def _entry_is_available(self, entry: OpEntry) -> bool:
        """Return whether both hardware and software requirements are satisfied."""
        hardware_matches = entry.requirement is None or entry.requirement.matches()
        return hardware_matches and self._requires_satisfied(entry)

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

    def _select_entry(self, op: str, variant: str, impl: str, device: str) -> OpEntry | None:
        """Select the device override, falling back to the device-agnostic row."""
        entry = self._entries.get((op, variant, impl, device))
        if entry is not None:
            return entry
        return self._entries.get((op, variant, impl, ANY_DEVICE))

    def resolve(self, op: str, variant: str, impl: str | None = None) -> OpEntry:
        """Return the row for ``(op, variant, impl)`` on this device.

        When ``impl`` is ``None``, selects ``self.default_impl(op, variant)``
        for the active device first. Looks up
        ``(op, variant, impl, current_device)``, then ``ANY_DEVICE``. Unknown
        triples raise ``KeyError``. A row that exists only for other devices,
        whose hardware ``requirement`` does not match, or whose software
        ``requires`` are missing raises ``RuntimeError``.
        """
        device = get_device_type()
        if impl is None:
            impl = self.default_impl(op, variant, device=device)
        entry = self._select_entry(op, variant, impl, device)
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
            raise KeyError(
                f"Unknown op {op!r} variant={variant!r} impl={impl!r} "
                f"(registered impls: {self.list_registered(op, variant)})"
            )
        if entry.requirement is not None:
            try:
                entry.requirement.check()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Op {op!r} variant={variant!r} impl={impl!r} requirement is not satisfied: {exc}"
                ) from exc
        missing_packages = tuple(package for package in entry.requires if not is_package_available(package))
        if missing_packages:
            packages = ", ".join(repr(package) for package in missing_packages)
            raise RuntimeError(
                f"Op {op!r} variant={variant!r} impl={impl!r} requires unavailable package(s): {packages}"
            )
        return entry

    def list_registered(self, op: str, variant: str) -> list[str]:
        """Return unique registered impl names for ``(op, variant)``."""
        seen: list[str] = []
        for entry_op, entry_variant, impl, _device in self._entries:
            if entry_op == op and entry_variant == variant and impl not in seen:
                seen.append(impl)
        return seen

    def all_impls(self, op: str) -> list[str]:
        """Return every registered impl name for ``op``, across variants and devices.

        For the config-time checks that ask what a *family* can express rather than
        what one variant on this host can run: a key written on a family field is
        meaningful if any row of that family takes it.
        """
        seen: list[str] = []
        for entry_op, _variant, impl, _device in self._entries:
            if entry_op == op and impl not in seen:
                seen.append(impl)
        return seen

    def list_available(self, op: str, variant: str) -> list[str]:
        """Return impl names whose hardware and optional packages are available."""
        device = get_device_type()
        seen: list[str] = []
        for (entry_op, entry_variant, impl, entry_device), entry in self._entries.items():
            if entry_op != op or entry_variant != variant:
                continue
            if entry_device not in (device, ANY_DEVICE):
                continue
            if entry is not self._select_entry(op, variant, impl, device):
                continue
            if not self._entry_is_available(entry):
                continue
            if impl not in seen:
                seen.append(impl)
        return seen

    def first_entry(self, op: str, impl: str, *, variant: str | None = None) -> OpEntry | None:
        """Some registered entry for ``(op, impl)`` (optionally matching ``variant``), ignoring device.

        For the checks that must run before hardware resolution: the formats a row
        accepts and the keys it takes do not depend on which device registered it, so
        any entry answers. Returns ``None`` when no device registers the name, which
        the caller leaves to ``resolve`` to report.
        """
        for (entry_op, entry_variant, entry_impl, _device), entry in self._entries.items():
            if entry_op == op and entry_impl == impl and (variant is None or entry_variant == variant):
                return entry
        return None

    def row_dtypes(self, op: str, impl: str, *, variant: str | None = None) -> tuple[str, ...] | None:
        """The formats ``impl`` of ``op`` can round to, or ``None`` if unregistered.

        An empty tuple is a valid answer: an ``EXACT`` row rounds nothing, so there is
        no format to choose. ``None`` means the row does not exist at all.
        """
        entry = self.first_entry(op, impl, variant=variant)
        return None if entry is None else entry.dtypes

    def row_arguments(self, op: str, impl: str, *, variant: str | None = None) -> dict[str, Any] | None:
        """The keyword arguments a selection may set on ``impl`` of ``op``, name to default.

        Read from the raw ``forward``, which carries the real signature; the
        generated ``wrapper`` is ``(*tensors, **attrs)`` and would accept anything.

        Only keyword-only parameters *with a default* are listed. A parameter with
        no default is one the row cannot run without, so the modeling always passes
        it and it describes the shape of the computation -- ``num_experts`` is the
        expert count of the module, not a preference. Letting a config set it would
        let a YAML typo launch the kernel over the wrong number of experts. It stays
        call-only, which is also why the third precedence step does not reach it.

        When ``variant`` is given, inspects only that variant; otherwise aggregates
        across variants so family-wide config checks see every row's arguments.
        Returns ``None`` when ``impl`` is not registered for ``op`` on any device.
        """
        found = False
        collected: dict[str, Any] = {}
        for (entry_op, entry_variant, entry_impl, _device), entry in self._entries.items():
            if entry_op != op or entry_impl != impl:
                continue
            if variant is not None and entry_variant != variant:
                continue
            fn = entry.forward or entry.wrapper
            if fn is None:
                continue
            found = True
            # ``KEYWORD_ONLY`` only: the tensor operands are positional, so a
            # selection never supplies them and listing them would suggest it could.
            for name, parameter in signature(fn).parameters.items():
                if parameter.kind is Parameter.KEYWORD_ONLY and parameter.default is not Parameter.empty:
                    collected.setdefault(name, parameter.default)
            if variant is not None:
                break
        return collected if found else None

    def list_entries(self, op: str, variant: str) -> list[OpEntry]:
        """Return every registered row for ``(op, variant)``.

        Device-specific rows are kept distinct, so the same implementation
        name can appear more than once with different requirements.
        """
        return [
            entry
            for (entry_op, entry_variant, _impl, _device), entry in self._entries.items()
            if entry_op == op and entry_variant == variant
        ]

    def default_impl(self, op: str, variant: str, *, device: str | None = None) -> str:
        """Return the default impl name for ``(op, variant)`` on ``device``.

        Checks ``default=True`` rows for ``device`` in registration order,
        returning the first whose hardware and software requirements are
        satisfied. Returns ``eager`` when no default candidate is available,
        independent of registration order. Explicit implementations are
        validated by ``resolve`` and never fall back.
        """
        if device is None:
            device = get_device_type()
        entries = self.list_entries(op, variant)
        if not entries:
            raise KeyError(f"Unknown op {op!r} variant={variant!r}")

        for entry in entries:
            if _entry_device(entry) == device and entry.default and self._entry_is_available(entry):
                return entry.impl
        return "eager"


OP_REGISTRY = OpRegistry()


def register_op(
    op: str,
    variant: str,
    impl: str,
    forward: Callable | None = None,
    backward: Callable | None = None,
    *,
    description: str,
    wrapper: Callable | None = None,
    requirement: KernelRequirement | None = None,
    requires: tuple[str, ...] = (),
    default: bool = False,
    dtypes: tuple[str, ...] = (),
) -> None:
    """Register one row on ``OP_REGISTRY``.

    Pass a raw pair or an opaque ``wrapper``, not both. ``description`` is
    required discovery metadata and should describe stable implementation
    semantics rather than duplicating hardware constraints.
    ``requirement.device`` (or ``ANY_DEVICE``) is the fourth key.
    ``requires`` lists import package names checked without importing them.
    ``default=True`` marks a device-specific row as a host default candidate
    in registration order. ``resolve_op`` resolves ``impl=None`` through
    ``default_impl(op, variant)``.
    ``dtypes`` lists the formats the row's ``dtype`` argument accepts. The format is
    an argument, not part of the name: ``qat`` serves an FP8 and an FP4 checkpoint,
    and ``torchao`` serves whatever torchao can quantize.
    """
    OP_REGISTRY.register(
        OpEntry(
            op=op,
            variant=variant,
            impl=impl,
            description=description,
            forward=forward,
            backward=backward,
            wrapper=wrapper,
            requirement=requirement,
            requires=requires,
            default=default,
            dtypes=dtypes,
        )
    )


def resolve_op(op: str, variant: str, impl: str | None = None) -> OpEntry:
    """Look up ``(op, variant, impl)`` in ``OP_REGISTRY``."""
    return OP_REGISTRY.resolve(op, variant, impl)


def _kwargs_key(kwargs: Mapping[str, Any]) -> tuple:
    """Hashable form of a row-argument mapping, for the intern key."""
    return tuple(sorted(kwargs.items()))


def _rebuild_op(op: str, variant: str, impl: str, kwargs: Mapping[str, Any]) -> VeomniOp:
    """Reconstruct an interned handle from its pickle payload."""
    return VeomniOp(op, variant, impl).with_kwargs(kwargs)


def _filter_kwargs(entry: OpEntry, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``kwargs`` this row can honour, dropping the rest.

    A row argument says how to round: ``dtype``, ``block_size``, the checkpoint's
    ``expert_dtype``. A family field names one row for every site of a family, so
    the enclosing block's ``kwargs`` describe *that* row; a site that overrides the
    name to a row with a different signature is not asking for them. Passing a key
    the row does not take would either raise on a legitimate config or -- worse,
    for a row whose signature has a ``**attrs`` tail -- be dropped at the call.

    Dropping is only safe because the config layer has already refused a key that
    is meaningless for *every* row of the family (see
    ``OpsImplementationConfig._validate_row_arguments``): a typo like ``block_siz``
    is caught there, against the whole family, before any site is resolved. What
    reaches here and is dropped is inherited, not requested at this site.

    A parameter the row *requires* is never passed from a config either; it
    describes the shape of the computation, so the modeling supplies it at the call.

    A ``dtype`` the row cannot round to is **not** dropped, even when the key itself
    is accepted: that is a config asking for a format this row does not implement,
    and the run would quietly use another one. ``emulated`` is the same kind of
    answer -- a row that offers only one mode cannot honour a request for the other,
    so it is dropped, but a non-bool is a mistake in any row that takes it.
    """
    if not kwargs:
        return {}
    accepted = OP_REGISTRY.row_arguments(entry.op, entry.impl, variant=entry.variant)
    if accepted is None:
        # Unregistered for every device; ``resolve`` reports that, not this.
        return dict(kwargs)
    if "dtype" in kwargs and entry.dtypes:
        _check_dtype(entry, kwargs["dtype"])
    kept = {key: value for key, value in kwargs.items() if key in accepted}
    if "emulated" in kept and kept["emulated"] is not None and type(kept["emulated"]) is not bool:
        raise ValueError(
            f"{entry.op}/{entry.variant}/{entry.impl}: emulated must be a bool, got {kept['emulated']!r}."
        )
    if "recipe" in kept and kept["recipe"] is not None and entry.impl == "torchao":
        from .torchao import validate_float8_recipe

        validate_float8_recipe(kept["recipe"])
    return kept


def _check_dtype(entry: OpEntry, dtype: Any) -> None:
    """Refuse a format the row does not round to."""
    if not entry.dtypes:
        return
    if dtype not in entry.dtypes:
        raise ValueError(
            f"{entry.op}/{entry.variant}/{entry.impl} cannot round to {dtype!r}; it supports "
            f"{sorted(entry.dtypes)}. The format is a row argument, not part of the name, so a row that "
            "handles several formats states which ones here."
        )


class VeomniOp:
    """Local handle for one ``(op, variant, impl)`` row plus its row arguments.

    Interned by the resolved triple *and* the row arguments: two sites that
    resolve to the same row but pass different ``dtype`` must not share a handle,
    or one site's rounding would leak into the other. Sites with identical
    arguments still share one object, which is the common case -- a family with no
    overrides produces one handle no matter how many layers bind it.

    Compound Functions must use ``resolve_op(...).forward`` / ``.backward``, not
    this handle.
    """

    _intern: dict[tuple[str, str, str, tuple], VeomniOp] = {}

    def __new__(
        cls,
        op: str,
        variant: str,
        impl: str | OpsSelection | None = None,
        *,
        site: str | None = None,
    ):
        """Return the interned handle for the resolved row and its arguments."""
        kwargs: dict[str, Any] = {}
        if isinstance(impl, OpsSelection):
            kwargs = impl.resolve_kwargs(site)
            impl = impl.resolve_impl(site)
        # Drop arguments the row cannot honour *before* resolving it, so a key the
        # site does not use never reaches a row that would reject it, and so the
        # intern key is built from the arguments the row actually runs with.
        # Filtering by name rather than by resolved entry keeps it working on a host
        # where the row does not resolve: what a row accepts is a property of the
        # row, not of which device registered it.
        if kwargs and impl is not None:
            offered = OP_REGISTRY.first_entry(op, impl, variant=variant)
            if offered is not None:
                kwargs = _filter_kwargs(offered, kwargs)
        # The un-resolved name is checked first, exactly as before this change. A
        # handle whose row was resolved under a different device view is reused
        # instead of re-resolved: a test that mocks the host as NPU and then as CPU
        # must get its earlier handle back, not a failure from a device where the
        # row does not exist.
        if impl is not None:
            cached = cls._intern.get((op, variant, impl, _kwargs_key(kwargs)))
            if cached is not None:
                return cached
        entry = resolve_op(op, variant, impl)
        if kwargs and impl is None:
            kwargs = _filter_kwargs(entry, kwargs)
        key = (op, variant, entry.impl, _kwargs_key(kwargs))
        cached = cls._intern.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        instance.op = op
        instance.variant = variant
        instance.impl = entry.impl
        instance.kwargs = kwargs
        instance._entry = entry
        cls._intern[key] = instance
        return instance

    def __init__(
        self,
        op: str,
        variant: str,
        impl: str | OpsSelection | None = None,
        *,
        site: str | None = None,
    ):
        """Resolve the registry row and intern this handle."""
        return None

    def __copy__(self):
        """Interned handles are identity-copied."""
        return self

    def __deepcopy__(self, memo):
        """Interned handles are shared across deepcopy, including nn.Module clones."""
        memo[id(self)] = self
        return self

    def __reduce__(self):
        """Rebuild through the interned row and its arguments rather than a bare ``__new__``."""
        return (_rebuild_op, (self.op, self.variant, self.impl, self.kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``entry.wrapper``. Tensors are positional, non-tensors are keywords.

        The handle's row arguments win over the call's. A row argument has three
        sources, and each may correct the one before it: the row's own signature
        default, the value the modeling passes from the checkpoint, and the value the
        config sets. The checkpoint records what the weights were *stored* as
        (``expert_dtype`` is ``"fp4"`` in a V4-Flash checkpoint), while the config
        records what this run is *asked* to train at -- so the config has to be the
        one that survives, or a YAML that says ``dtype: fp8`` over an FP4 checkpoint
        would be silently ignored at every call site.

        A call site therefore passes what the module knows and the handle corrects it.
        Sites that the config says nothing about keep the modeling's value, because
        then ``self.kwargs`` simply has no entry for that key.
        """
        if self._entry.wrapper is None:
            raise RuntimeError(f"VeomniOp({self.op!r}, {self.variant!r}, {self.impl!r}) has no wrapper")
        if not self.kwargs:
            return self._entry.wrapper(*args, **kwargs)
        return self._entry.wrapper(*args, **{**kwargs, **self.kwargs})

    def with_kwargs(self, kwargs: Mapping[str, Any]) -> VeomniOp:
        """Return the interned handle for this row with ``kwargs`` as its arguments."""
        kwargs = _filter_kwargs(self._entry, kwargs)
        key = (self.op, self.variant, self.impl, _kwargs_key(kwargs))
        cached = self._intern.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(VeomniOp)
        instance.op = self.op
        instance.variant = self.variant
        instance.impl = self.impl
        instance.kwargs = dict(kwargs)
        instance._entry = self._entry
        self._intern[key] = instance
        return instance

    @property
    def entry(self) -> OpEntry:
        """The resolved ``OpEntry`` for this handle."""
        return self._entry

    @property
    def rounds(self) -> bool:
        """Whether this row rounds the operands it is handed.

        A row declares ``dtypes`` exactly when it rounds to one of them, so that
        tuple is the registry-visible answer. The per-family ``QUANTIZING_IMPLS``
        sets say the same thing for the families that own one; a guard that has a
        handle should read this, because it is derived rather than hand-listed.
        """
        return bool(self._entry.dtypes)

    def __repr__(self) -> str:
        """Return ``VeomniOp(op=..., variant=..., impl=...)``."""
        if self.kwargs:
            return f"VeomniOp(op={self.op!r}, variant={self.variant!r}, impl={self.impl!r}, kwargs={self.kwargs!r})"
        return f"VeomniOp(op={self.op!r}, variant={self.variant!r}, impl={self.impl!r})"
