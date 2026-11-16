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
model may want different rows at different sites of that family -- the first ten
layers in the model dtype, the routed experts quantized while the shared expert is
not. :class:`OpsSelection` expresses that: a default row, its row arguments, and
regex overrides keyed by site.

Both halves are just the row name. ``qat``, ``torchao_nvfp4`` and ``deepgemm_fp8``
are rows, so "which precision", "fake quantization or the vendor GEMM" and "which
package provides it" are already encoded in the name the registry resolves; what a
row still needs to know per call is its *arguments* -- ``dtype``, ``block_size`` --
and those travel with the selection.

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
    "OpsSelection",
    "SiteOverride",
]


@dataclass
class SiteOverride:
    """One ``site`` pattern and what it changes.

    Everything is optional and inherited: a rule that only sets ``kwargs`` keeps
    the enclosing ``name``, and a rule that only sets ``name`` keeps the enclosing
    ``kwargs``. A rule that sets ``name: eager`` is how "keep these layers in the
    model dtype" is spelled -- there is no separate "disable" flag to keep in sync
    with the row name.
    """

    site: str
    name: str | None = None
    #: Merged over the enclosing ``kwargs``. ``None`` means "not specified" and
    #: inherits; a mapping replaces only the keys it names. To drop inherited
    #: arguments -- which is what switching to a row with a different signature
    #: needs -- write ``kwargs: {}`` together with ``replace_kwargs: true``.
    kwargs: dict[str, Any] | None = None
    replace_kwargs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("override site must be a nonempty string.")
        try:
            self._pattern = re.compile(self.site)
        except re.error as error:
            raise ValueError(f"override site {self.site!r} is not a valid regex: {error}") from error
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"override name must be a nonempty string, got {self.name!r}.")
        if self.kwargs is not None and not isinstance(self.kwargs, dict):
            raise TypeError(f"override kwargs must be a mapping, got {type(self.kwargs).__name__}.")
        if type(self.replace_kwargs) is not bool:
            raise ValueError(f"override replace_kwargs must be a bool, got {self.replace_kwargs!r}.")
        if self.replace_kwargs and self.kwargs is None:
            raise ValueError("replace_kwargs requires kwargs (use {} to drop every inherited argument).")

    def matches(self, site: str) -> bool:
        """Whether ``site`` is covered by this rule."""
        return self._pattern.fullmatch(site) is not None

    @classmethod
    def from_dict(cls, raw: Any) -> SiteOverride:
        """Build from a YAML mapping, rejecting unknown keys."""
        if not isinstance(raw, dict):
            raise TypeError(f"an override must be a mapping, got {type(raw).__name__}.")
        known = {"site", "name", "kwargs", "replace_kwargs"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown override keys: {sorted(unknown)}. Known: {sorted(known)}.")
        if "site" not in raw:
            raise ValueError("an override requires a site pattern.")
        return cls(
            site=raw["site"],
            name=raw.get("name"),
            kwargs=raw.get("kwargs"),
            replace_kwargs=raw.get("replace_kwargs", False),
        )


@dataclass
class OpsSelection:
    """One config field: a row name, its row arguments, and site overrides.

    ``name is None`` means the host default, which is exactly what an unset field
    does today. ``kwargs`` is passed to whichever row ends up running; a row that
    does not accept a key raises, so a typo cannot silently drop a setting.
    """

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    overrides: list[SiteOverride] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"name must be a nonempty string or None, got {self.name!r}.")
        if not isinstance(self.kwargs, dict):
            raise TypeError(f"kwargs must be a mapping, got {type(self.kwargs).__name__}.")

    def effective(self, site: str | None) -> OpsSelection:
        """This selection with every override matching ``site`` applied, in order.

        Later rules win over earlier ones, so a narrow rule goes after a broad one.
        """
        if not self.overrides or site is None:
            return self
        return self._apply(rule for rule in self.overrides if rule.matches(site))

    def _apply(self, rules) -> OpsSelection:
        """Merge ``rules`` over this selection, in order."""
        name, kwargs = self.name, dict(self.kwargs)
        for rule in rules:
            if rule.name is not None:
                name = rule.name
            if rule.replace_kwargs:
                kwargs = dict(rule.kwargs or {})
            elif rule.kwargs:
                kwargs.update(rule.kwargs)
        return OpsSelection(name=name, kwargs=kwargs)

    def resolve_impl(self, site: str | None) -> str | None:
        """The row this selection names at ``site``, or ``None`` for the host default."""
        return self.effective(site).name

    def resolve_kwargs(self, site: str | None) -> dict[str, Any]:
        """The row arguments in force at ``site``."""
        return dict(self.effective(site).kwargs)

    def all_impls(self) -> list[str | None]:
        """Every row this selection can resolve to, default and overrides alike.

        Config-time validation has no site to match, so it has to clear every row
        the field could produce on this host.
        """
        seen: list[str | None] = []

        def add(name: str | None) -> None:
            if name not in seen:
                seen.append(name)

        add(self.name)
        for rule in self.overrides:
            add(self._apply([rule]).name)
        return seen

    def all_kwargs_keys(self) -> set[str]:
        """Every row argument this selection can pass, default and overrides alike."""
        keys = set(self.kwargs)
        for rule in self.overrides:
            keys.update(rule.kwargs or {})
        return keys

    @property
    def is_plain(self) -> bool:
        """Whether this selection is just a row name, i.e. the pre-override shape."""
        return not self.kwargs and not self.overrides

    @classmethod
    def from_config_value(cls, raw: Any) -> OpsSelection:
        """Accept either a plain row name or the mapping form.

        The mapping form adds ``kwargs`` and ``overrides``; the bare string stays
        valid so a config that only names a row keeps working.
        """
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, OpsSelection):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"ops selection must be a string or mapping, got {type(raw).__name__}.")
        known = {"name", "kwargs", "overrides"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown ops selection keys: {sorted(unknown)}. Known: {sorted(known)}.")
        return cls(
            name=raw.get("name"),
            kwargs=raw.get("kwargs") or {},
            overrides=[SiteOverride.from_dict(item) for item in raw.get("overrides") or ()],
        )
