# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Per-site op selection.

One config field names a *family* ("the projections", "the routed experts"). A
model may want different treatment for different sites of that family -- the
first ten layers in the model dtype, the routed experts quantized but the shared
expert not. :class:`OpsSelection` expresses that: a default row, an optional
low-precision block, and regex overrides keyed by site.

The site string is chosen by the model, because only it knows what its sites are
called: ``layers.3.q_a_proj``, ``layers.12.experts``, ``layers.0.window_kv``.
Overrides are matched against it with :func:`re.fullmatch`, in order, so a later
rule wins over an earlier one that matched the same site.

Resolution happens once, when the model builds its ``VeomniOp`` handle, not on
every call: the handle needs a concrete impl to resolve against the registry, and
it stays interned by ``(op, variant, impl)`` afterwards. Passing ``site=None``
skips override matching entirely, which is what a model that never configured
overrides gets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


__all__ = [
    "LowPrecisionConfig",
    "OpsSelection",
    "SiteOverride",
    "low_precision_impl",
]


#: ``(dtype, qat, backend)`` → the row that implements it. Only combinations a
#: registered row actually serves appear here; anything else raises rather than
#: falling back to the family's reference row, because a config that asks for
#: real NVFP4 and silently trains FP8 is the failure this table exists to prevent.
#:
#: ``qat`` selects fake quantization (training computes in the model dtype on
#: rounded operands, so a deployment kernel's numerics are reproduced) against
#: ``False`` for the vendor's real low-precision GEMM.
_IMPL_BY_TRIPLE: dict[tuple[str, bool, str], str] = {
    # TileLang blockwise FP8, fake quantized. Both the dense and the MoE rows.
    ("fp8", True, "tilelang"): "qat",
    # torchao NVFP4, fake quantized.
    ("nvfp4", True, "torchao"): "torchao_nvfp4",
    # DeepGEMM blockwise FP8 routed experts, real GEMM.
    ("fp8", False, "deepgemm"): "deepgemm_fp8",
}

#: Backends whose rows exist only as fake quantization today. Named here so the
#: error for ``("nvfp4", False, "torchao")`` says *what* is missing rather than
#: listing the whole table.
_QAT_ONLY_BACKENDS = frozenset({"tilelang"})


@dataclass
class LowPrecisionConfig:
    """The low-precision half of a selection: precision, mode and backend.

    ``dtype``, ``qat`` and ``backend`` are orthogonal up to what has been
    registered: ``torchao`` is expected to serve both ``nvfp4`` and ``mxfp4``,
    ``tilelang`` only fake FP8, ``deepgemm`` only real FP8. :func:`low_precision_impl`
    is the single place that knows which triples exist.
    """

    #: ``None`` means "inherit", which only an override may do. A top-level block
    #: with ``None`` behaves as ``False``.
    enable: bool | None = None
    dtype: str | None = None
    qat: bool | None = None
    backend: str | None = None
    #: Tile side length. ``None`` keeps the row's own default, which is what the
    #: deployment recipe and the checkpoint usually agree on.
    block_size: int | None = None

    def __post_init__(self) -> None:
        if self.enable is not None and type(self.enable) is not bool:
            raise ValueError(f"low_precision.enable must be a bool, got {self.enable!r}.")
        if self.qat is not None and type(self.qat) is not bool:
            raise ValueError(f"low_precision.qat must be a bool, got {self.qat!r}.")
        if self.block_size is not None and (type(self.block_size) is not int or self.block_size <= 0):
            raise ValueError(f"low_precision.block_size must be a positive int, got {self.block_size!r}.")
        if not self.enable:
            return
        if self.dtype is None and self.backend is None:
            raise ValueError(
                "low_precision.enable is set but neither dtype nor backend is; nothing would be quantized."
            )
        if self.backend is None:
            raise ValueError("low_precision.enable requires a backend.")

    @classmethod
    def from_dict(cls, raw: Any) -> LowPrecisionConfig:
        """Build from a YAML mapping, rejecting unknown keys."""
        if not isinstance(raw, dict):
            raise TypeError(f"low_precision must be a mapping, got {type(raw).__name__}.")
        known = {"enable", "dtype", "qat", "backend", "block_size"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown low_precision keys: {sorted(unknown)}. Known: {sorted(known)}.")
        return cls(**raw)


@dataclass
class SiteOverride:
    """One ``site`` pattern and the selection it replaces.

    Only the fields written here change; everything else is inherited from the
    enclosing :class:`OpsSelection`. A rule that sets ``low_precision.enable:
    false`` therefore falls back to the enclosing ``name``, which is how "keep
    these layers in the model dtype" is spelled.
    """

    site: str
    name: str | None = None
    low_precision: LowPrecisionConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("override site must be a nonempty string.")
        try:
            self._pattern = re.compile(self.site)
        except re.error as error:
            raise ValueError(f"override site {self.site!r} is not a valid regex: {error}") from error

    def matches(self, site: str) -> bool:
        """Whether ``site`` is covered by this rule."""
        return self._pattern.fullmatch(site) is not None

    @classmethod
    def from_dict(cls, raw: Any) -> SiteOverride:
        """Build from a YAML mapping, rejecting unknown keys."""
        if not isinstance(raw, dict):
            raise TypeError(f"an override must be a mapping, got {type(raw).__name__}.")
        known = {"site", "name", "low_precision"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown override keys: {sorted(unknown)}. Known: {sorted(known)}.")
        if "site" not in raw:
            raise ValueError("an override requires a site pattern.")
        low_precision = raw.get("low_precision")
        return cls(
            site=raw["site"],
            name=raw.get("name"),
            low_precision=None if low_precision is None else LowPrecisionConfig.from_dict(low_precision),
        )


@dataclass
class OpsSelection:
    """One config field: a reference row, an optional low-precision block, and site overrides."""

    name: str | None = None
    low_precision: LowPrecisionConfig | None = None
    overrides: list[SiteOverride] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"name must be a nonempty string or None, got {self.name!r}.")
        if self.low_precision is not None and not isinstance(self.low_precision, LowPrecisionConfig):
            raise TypeError("low_precision must be a LowPrecisionConfig.")

    @property
    def is_plain(self) -> bool:
        """Whether this selection is just a row name, i.e. the pre-override shape."""
        return self.low_precision is None and not self.overrides

    def effective(self, site: str | None) -> OpsSelection:
        """This selection with every override matching ``site`` applied, in order.

        Later rules win over earlier ones, so a narrow rule goes after a broad one.
        """
        if not self.overrides or site is None:
            return self
        return self._apply(rule for rule in self.overrides if rule.matches(site))

    def _apply(self, rules) -> OpsSelection:
        """Merge ``rules`` over this selection, in order."""
        name, low_precision = self.name, self.low_precision
        for rule in rules:
            if rule.name is not None:
                name = rule.name
            if rule.low_precision is None:
                continue
            current = low_precision
            override = rule.low_precision
            low_precision = LowPrecisionConfig(
                enable=override.enable if override.enable is not None else (current.enable if current else None),
                dtype=override.dtype if override.dtype is not None else (current.dtype if current else None),
                qat=override.qat if override.qat is not None else (current.qat if current else None),
                backend=override.backend if override.backend is not None else (current.backend if current else None),
                block_size=override.block_size
                if override.block_size is not None
                else (current.block_size if current else None),
            )
        return OpsSelection(name=name, low_precision=low_precision)

    def resolve_impl(self, site: str | None) -> str | None:
        """The row this selection names at ``site``, or ``None`` for the host default.

        ``None`` is not a failure: it means "let ``OP_REGISTRY.default_impl`` pick",
        which is exactly what an unset field does today.
        """
        effective = self.effective(site)
        if effective.low_precision is not None and effective.low_precision.enable:
            return low_precision_impl(effective.low_precision)
        return effective.name

    def all_impls(self) -> list[str | None]:
        """Every row this selection can resolve to, default and overrides alike.

        Config-time validation has no site to match, so it has to clear every row
        the field could produce on this host.
        """
        seen: list[str | None] = []

        def add(selection: OpsSelection) -> None:
            impl = selection.resolve_impl(None)
            if impl not in seen:
                seen.append(impl)

        add(self)
        for rule in self.overrides:
            add(self._apply([rule]))
        return seen

    @classmethod
    def from_config_value(cls, raw: Any) -> OpsSelection:
        """Accept either a plain row name or the mapping form.

        The mapping form adds ``low_precision`` and ``overrides``; the bare string
        stays valid so a config that only names a row keeps working.
        """
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, OpsSelection):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"ops selection must be a string or mapping, got {type(raw).__name__}.")
        known = {"name", "low_precision", "overrides"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown ops selection keys: {sorted(unknown)}. Known: {sorted(known)}.")
        low_precision = raw.get("low_precision")
        return cls(
            name=raw.get("name"),
            low_precision=None if low_precision is None else LowPrecisionConfig.from_dict(low_precision),
            overrides=[SiteOverride.from_dict(item) for item in raw.get("overrides") or ()],
        )


def low_precision_impl(config: LowPrecisionConfig) -> str:
    """The registered row for ``config``'s ``(dtype, qat, backend)``.

    Raises:
        ValueError: when no row serves the triple. Falling back to the reference
            row would train something other than what the config asked for, which
            is the one outcome this function exists to make impossible.
    """
    backend = config.backend
    dtype = config.dtype
    # An inherited-away ``None`` means "not set": a block that enables the feature
    # without saying ``qat`` gets the fake-quantization reading of its backend.
    qat = bool(config.qat)
    if backend is None:
        raise ValueError("low_precision.backend is required when low_precision.enable is set.")
    candidates = [d for d in (dtype, "fp8", "nvfp4", "mxfp4") if d is not None]
    for candidate in dict.fromkeys(candidates):
        key = (candidate, qat, backend)
        if key in _IMPL_BY_TRIPLE:
            return _IMPL_BY_TRIPLE[key]
    if dtype is None:
        # A backend-only selection is only unambiguous when it serves one row.
        served = [
            impl
            for (_, row_qat, served_backend), impl in _IMPL_BY_TRIPLE.items()
            if served_backend == backend and row_qat == qat
        ]
        if len(served) == 1:
            return served[0]
        raise ValueError(
            f"low_precision.backend={backend!r} with qat={qat} serves more than one dtype; "
            f"set low_precision.dtype to one of {sorted({d for (d, _q, b) in _IMPL_BY_TRIPLE if b == backend})}."
        )
    hint = ""
    if backend in _QAT_ONLY_BACKENDS and not qat:
        hint = f" {backend!r} registers only fake-quantization rows today, so qat must be true."
    elif (dtype, not qat, backend) in _IMPL_BY_TRIPLE:
        hint = f" Did you mean qat={not qat}?"
    raise ValueError(
        f"No registered row for dtype={dtype!r}, qat={qat}, backend={backend!r}. "
        f"Registered combinations: {sorted(_IMPL_BY_TRIPLE)}.{hint}"
    )
