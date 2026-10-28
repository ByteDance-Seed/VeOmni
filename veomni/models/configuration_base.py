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

"""Base HuggingFace ``PretrainedConfig`` with VeOmni ``ops_config`` support."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


_orig_post_init = PretrainedConfig.__post_init__
_orig_from_dict = PretrainedConfig.from_dict.__func__
_orig_to_diff_dict = PretrainedConfig.to_diff_dict

_DEFER_OPS_VALIDATION: ContextVar[bool] = ContextVar("_DEFER_OPS_VALIDATION", default=False)


@contextmanager
def _defer_ops_validation():
    """Defer hardware and cross-field ``OpsImplementationConfig`` validation until overrides are applied."""
    token = _DEFER_OPS_VALIDATION.set(True)
    try:
        yield
    finally:
        _DEFER_OPS_VALIDATION.reset(token)


def _prune_subconfig_ops_diff(config: PretrainedConfig, diff: dict[str, Any]) -> None:
    if not diff.get("ops_implementation"):
        diff.pop("ops_implementation", None)
    parent_ops = getattr(config, "ops_implementation", None) or {}
    for key in config.sub_configs:
        sub_cfg = getattr(config, key, None)
        sub_dict = diff.get(key)
        if not isinstance(sub_cfg, PretrainedConfig) or not isinstance(sub_dict, dict):
            continue
        _prune_subconfig_ops_diff(sub_cfg, sub_dict)
        if "ops_implementation" in sub_dict:
            sub_ops = sub_dict["ops_implementation"]
            if isinstance(sub_ops, Mapping):
                sub_diff = {k: v for k, v in sub_ops.items() if parent_ops.get(k) != v}
                if sub_diff:
                    sub_dict["ops_implementation"] = sub_diff
                else:
                    sub_dict.pop("ops_implementation", None)


def _snapshot_subconfigs(config: PretrainedConfig) -> dict[int, tuple[PretrainedConfig, dict[str, Any], str | None]]:
    snapshots: dict[int, tuple[PretrainedConfig, dict[str, Any], str | None]] = {}
    for key in config.sub_configs:
        sub = getattr(config, key, None)
        if not isinstance(sub, PretrainedConfig) or id(sub) in snapshots:
            continue
        snapshots[id(sub)] = (
            sub,
            dict(getattr(sub, "ops_implementation", None) or {}),
            getattr(sub, "_attn_implementation_internal", None),
        )
        snapshots.update(_snapshot_subconfigs(sub))
    return snapshots


def _collect_explicit_attn_map_ids(config: PretrainedConfig, attn_value: Any) -> set[int]:
    """Return ``id(sub)`` for every descendant explicitly targeted by a nested ``attn_implementation`` mapping."""
    if attn_value is None:
        return set()
    if not isinstance(attn_value, Mapping):
        ids = {id(config)}
        for key in config.sub_configs:
            sub = getattr(config, key, None)
            if isinstance(sub, PretrainedConfig):
                ids.update(_collect_explicit_attn_map_ids(sub, attn_value))
        return ids

    ids: set[int] = set()
    if attn_value.get("") is not None:
        ids.add(id(config))
    for key in config.sub_configs:
        if key not in attn_value:
            continue
        sub = getattr(config, key, None)
        if isinstance(sub, PretrainedConfig):
            ids.update(_collect_explicit_attn_map_ids(sub, attn_value[key]))
    return ids


def _veomni_post_init(self: PretrainedConfig, **kwargs: Any) -> None:
    from ..arguments import OpsImplementationConfig

    sub_snapshots = _snapshot_subconfigs(self)
    ops_implementation = kwargs.pop("ops_implementation", getattr(self, "ops_implementation", None))
    explicit_attn = kwargs.get("attn_implementation")

    self.ops_implementation = {}
    _orig_post_init(self, **kwargs)

    explicit_map_ids = (
        _collect_explicit_attn_map_ids(self, explicit_attn) if isinstance(explicit_attn, Mapping) else set()
    )
    has_ops_attn = isinstance(ops_implementation, (OpsImplementationConfig, SimpleNamespace)) or (
        isinstance(ops_implementation, Mapping) and ops_implementation.get("attn_implementation") is not None
    )
    for sub_id, (sub, sub_ops, sub_attn) in sub_snapshots.items():
        if sub_id in explicit_map_ids:
            override_attn = getattr(sub, "_attn_implementation_internal", None)
            restored_ops = dict(sub_ops)
            if override_attn is not None:
                restored_ops["attn_implementation"] = override_attn
            sub.ops_implementation = restored_ops
            continue
        sub.ops_implementation = dict(sub_ops)
        if sub_attn is not None:
            sub._attn_implementation_internal = sub_attn
        elif not isinstance(explicit_attn, str) or has_ops_attn:
            sub._attn_implementation_internal = None

    self._set_ops_implementation(ops_implementation, _is_post_init=True)


@property
def _veomni_attn_implementation(self: PretrainedConfig) -> str | None:
    return getattr(self, "_attn_implementation_internal", None)


@_veomni_attn_implementation.setter
def _veomni_attn_implementation(self: PretrainedConfig, value: str | dict[str, Any] | None) -> None:
    current_attn = getattr(self, "_attn_implementation_internal", None)
    attn_implementation = value if not isinstance(value, Mapping) else value.get("", current_attn)
    self._attn_implementation_internal = attn_implementation

    ops_dict = getattr(self, "ops_implementation", None)
    if isinstance(ops_dict, dict) and "attn_implementation" in ops_dict and attn_implementation is not None:
        ops_dict["attn_implementation"] = attn_implementation

    if value is None:
        return

    for subconfig_key in self.sub_configs:
        subconfig = getattr(self, subconfig_key, None)
        if subconfig is None:
            continue
        if isinstance(value, Mapping):
            if subconfig_key in value:
                subconfig._attn_implementation = value[subconfig_key]
        else:
            subconfig._attn_implementation = value


class VeOmniPretrainedConfig(PretrainedConfig):
    """Base ``PretrainedConfig`` carrying per-instance ``ops_implementation`` and ``ops_config``.

    ``self.ops_implementation`` is the single source of truth (a sparse dict of
    op selections serialized to ``config.json``). :attr:`ops_config` projects it
    to an :class:`~veomni.arguments.OpsImplementationConfig` via
    :meth:`~veomni.arguments.OpsImplementationConfig.from_dict` with
    ``validate=False`` for fast repeated reads during model construction.
    """

    def __init__(
        self,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | Any | None = None,
        **kwargs: Any,
    ):
        self.ops_implementation: dict[str, Any] = {}
        super().__init__(ops_implementation=ops_implementation, **kwargs)

    @property
    def ops_config(self) -> OpsImplementationConfig:
        """Resolved :class:`~veomni.arguments.OpsImplementationConfig` for this config."""
        from ..arguments import OpsImplementationConfig

        ops_dict = dict(getattr(self, "ops_implementation", None) or {})
        if "attn_implementation" not in ops_dict:
            hf_attn = getattr(self, "_attn_implementation_internal", None)
            if hf_attn is not None:
                ops_dict["attn_implementation"] = hf_attn
        return OpsImplementationConfig.from_dict(ops_dict, validate=False)

    def _set_ops_implementation(
        self,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | Any | None,
        *,
        _is_post_init: bool = False,
        _validate_tree: bool = False,
    ) -> None:
        from ..arguments import OpsImplementationConfig

        defer_validation = _DEFER_OPS_VALIDATION.get()
        replace_all = isinstance(ops_implementation, (OpsImplementationConfig, SimpleNamespace))
        raw_override: dict[str, Any] = {}
        override_for_children: dict[str, Any] = {}

        if isinstance(ops_implementation, OpsImplementationConfig):
            new_ops = ops_implementation.to_dict()
        elif isinstance(ops_implementation, SimpleNamespace):
            new_ops = {k: v for k, v in vars(ops_implementation).items() if v is not None}
        elif ops_implementation is None or isinstance(ops_implementation, Mapping):
            raw_override = dict(ops_implementation or {})
            if _is_post_init:
                merged = {k: v for k, v in raw_override.items() if v is not None}
            else:
                own = dict(getattr(self, "ops_implementation", None) or {})
                merged = {**own, **raw_override}
            if defer_validation:
                new_ops = {k: v for k, v in merged.items() if v is not None}
                override_for_children = dict(raw_override)
            else:
                new_ops = OpsImplementationConfig.validate_and_extract_dict(merged)
                override_for_children = {k: new_ops.get(k) for k in raw_override}
        else:
            raise ValueError(
                f"ops_implementation must be a dict or OpsImplementationConfig, got {type(ops_implementation).__name__}."
            )

        tree_snapshots = None if _is_post_init else _snapshot_subconfigs(self)
        prev_self_ops = dict(getattr(self, "ops_implementation", None) or {})
        prev_self_attn = getattr(self, "_attn_implementation_internal", None)

        self.ops_implementation = new_ops
        if _is_post_init or replace_all or _validate_tree:
            if "attn_implementation" in self.ops_implementation:
                self._attn_implementation_internal = self.ops_implementation["attn_implementation"]
        elif "attn_implementation" in raw_override and "attn_implementation" in self.ops_implementation:
            self._attn_implementation_internal = self.ops_implementation["attn_implementation"]

        try:
            for key in self.sub_configs:
                subconfig = getattr(self, key, None)
                if not isinstance(subconfig, PretrainedConfig):
                    continue
                if replace_all:
                    subconfig._set_ops_implementation(ops_implementation, _validate_tree=_validate_tree)
                elif _is_post_init:
                    sub_own = dict(getattr(subconfig, "ops_implementation", None) or {})
                    sub_attn = getattr(subconfig, "_attn_implementation_internal", None)
                    if "attn_implementation" not in sub_own and sub_attn is not None:
                        sub_own["attn_implementation"] = sub_attn
                    sub_merged = {**self.ops_implementation, **sub_own}
                    subconfig._set_ops_implementation(sub_merged, _is_post_init=True)
                elif override_for_children or _validate_tree:
                    subconfig._set_ops_implementation(override_for_children, _validate_tree=_validate_tree)
        except Exception:
            if tree_snapshots is not None:
                self.ops_implementation = prev_self_ops
                self._attn_implementation_internal = prev_self_attn
                for sub, sub_ops, sub_attn in tree_snapshots.values():
                    sub.ops_implementation = dict(sub_ops)
                    sub._attn_implementation_internal = sub_attn
            raise

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        ops_implementation = kwargs.pop("ops_implementation", None)
        has_explicit_attn = "attn_implementation" in kwargs and kwargs["attn_implementation"] is not None
        explicit_attn = kwargs.get("attn_implementation") if has_explicit_attn else None
        with _defer_ops_validation():
            result = _orig_from_dict(cls, config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if explicit_attn is not None:
            config._attn_implementation = explicit_attn
        if ops_implementation is not None:
            config._set_ops_implementation(ops_implementation, _validate_tree=True)
        elif not _DEFER_OPS_VALIDATION.get():
            config._set_ops_implementation({}, _validate_tree=True)
        return result

    def to_diff_dict(self) -> dict[str, Any]:
        diff = _orig_to_diff_dict(self)
        _prune_subconfig_ops_diff(self, diff)
        return diff


def apply_pretrained_config_ops_patch() -> None:
    """Attach ``ops_config`` and ``ops_implementation`` lifecycle hooks to ``PretrainedConfig``."""
    if getattr(PretrainedConfig, "_veomni_ops_config_patched", False):
        return
    PretrainedConfig.__post_init__ = _veomni_post_init  # type: ignore[method-assign]
    PretrainedConfig._attn_implementation = _veomni_attn_implementation  # type: ignore[assignment]
    PretrainedConfig.ops_config = VeOmniPretrainedConfig.ops_config  # type: ignore[attr-defined]
    PretrainedConfig._set_ops_implementation = VeOmniPretrainedConfig._set_ops_implementation  # type: ignore[attr-defined]
    PretrainedConfig.from_dict = classmethod(VeOmniPretrainedConfig.from_dict.__func__)  # type: ignore[assignment]
    PretrainedConfig.to_diff_dict = VeOmniPretrainedConfig.to_diff_dict  # type: ignore[method-assign]
    PretrainedConfig._veomni_ops_config_patched = True  # type: ignore[attr-defined]


apply_pretrained_config_ops_patch()


__all__ = [
    "VeOmniPretrainedConfig",
    "apply_pretrained_config_ops_patch",
]
