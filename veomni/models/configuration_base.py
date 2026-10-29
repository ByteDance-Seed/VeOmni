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
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


_DEFER_OPS_VALIDATION: ContextVar[bool] = ContextVar("_DEFER_OPS_VALIDATION", default=False)


@contextmanager
def _defer_ops_validation():
    """Validate a loaded config only after all caller overrides have been applied."""
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


class VeOmniPretrainedConfig(PretrainedConfig):
    """Config with instance-local, sparse ``ops_implementation`` as its only ops state.

    Native HF config classes opt in through :func:`get_veomni_config_class`.
    HF classes and instances outside VeOmni entry points are never patched.
    """

    def __post_init__(self, **kwargs: Any) -> None:
        ops = kwargs.pop("ops_implementation", getattr(self, "ops_implementation", None))
        attention = kwargs.pop("attn_implementation", None)
        self.ops_implementation = {}
        selections = {key: _capture_selections(getattr(self, key, None)) for key in self.sub_configs}
        # Concrete HF post-init constructs sub-configs before calling its base.
        # Suppress HF's unconditional None attention broadcast during that phase.
        self._initializing_ops = True
        try:
            super().__post_init__(**kwargs)
        finally:
            del self._initializing_ops
        for key in self.sub_configs:
            child = getattr(self, key, None)
            if isinstance(child, PretrainedConfig):
                as_veomni_config(child, validate=False)
                _restore_selections(child, selections[key])
        if isinstance(attention, str):
            self.ops_implementation["attn_implementation"] = _normalize_attention(attention)
        self._set_ops_implementation(ops, _is_post_init=True)
        if isinstance(attention, Mapping):
            self._attn_implementation = attention

    @property
    def ops_config(self) -> OpsImplementationConfig:
        """Resolve defaults without repeating hardware validation during model construction."""
        from ..arguments import OpsImplementationConfig

        return OpsImplementationConfig.from_dict(self.ops_implementation, validate=False)

    @property
    def _attn_implementation(self) -> str | None:
        return self.ops_config.attn_implementation

    @_attn_implementation.setter
    def _attn_implementation(self, value: str | Mapping[str, Any] | None) -> None:
        if getattr(self, "_initializing_ops", False):
            return
        updates = []

        def collect(config, selection):
            own = selection.get("") if isinstance(selection, Mapping) else selection
            if own is not None:
                updates.append((config, _normalize_attention(own)))
            elif selection is None:
                updates.append((config, None))
            for key in config.sub_configs:
                child = getattr(config, key, None)
                if isinstance(child, VeOmniPretrainedConfig):
                    if not isinstance(selection, Mapping):
                        collect(child, selection)
                    elif key in selection:
                        collect(child, selection[key])

        collect(self, value)
        for config, selection in updates:
            if selection is None:
                config.ops_implementation.pop("attn_implementation", None)
            else:
                config.ops_implementation["attn_implementation"] = selection

    @property
    def _attn_implementation_internal(self) -> str | None:
        """Alias used by HF model initialization; no separate private state exists."""
        return self._attn_implementation

    @_attn_implementation_internal.setter
    def _attn_implementation_internal(self, value: str | None) -> None:
        if value is None:
            self.ops_implementation.pop("attn_implementation", None)
        elif value != self._attn_implementation:
            # HF reassigns the resolved default during model init. Keep that
            # implicit unless its check actually selects a different backend.
            self.ops_implementation["attn_implementation"] = _normalize_attention(value)

    def _set_ops_implementation(
        self,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | Any | None,
        *,
        _is_post_init: bool = False,
        _validate_tree: bool = False,
    ) -> None:
        from ..arguments import OpsImplementationConfig

        replace = isinstance(ops_implementation, (OpsImplementationConfig, SimpleNamespace))
        if isinstance(ops_implementation, OpsImplementationConfig):
            override = ops_implementation.to_dict()
        elif isinstance(ops_implementation, SimpleNamespace):
            override = vars(ops_implementation)
        elif ops_implementation is None or isinstance(ops_implementation, Mapping):
            override = dict(ops_implementation or {})
        else:
            raise ValueError("ops_implementation must be a dict or OpsImplementationConfig.")
        validate = not _DEFER_OPS_VALIDATION.get() and not isinstance(ops_implementation, SimpleNamespace)
        pending = []

        def collect(config, incoming):
            merged = dict(incoming) if replace else {**config.ops_implementation, **incoming}
            proposed = (
                OpsImplementationConfig.validate_and_extract_dict(merged)
                if validate
                else {k: v for k, v in merged.items() if v is not None}
            )
            pending.append((config, proposed))
            for key in config.sub_configs:
                child = getattr(config, key, None)
                if not isinstance(child, VeOmniPretrainedConfig):
                    continue
                if _is_post_init and not replace:
                    collect(child, {**proposed, **child.ops_implementation})
                elif replace or incoming or _validate_tree:
                    collect(child, incoming)

        # Validate the whole proposed tree before modifying any instance.
        collect(self, override)
        for config, proposed in pending:
            config.ops_implementation = proposed

    def _apply_ops_overrides(
        self,
        *,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | Any | None = None,
        attn_implementation: str | dict[str, Any] | None = None,
    ) -> None:
        """Apply load-time overrides atomically; explicit attention ranks highest."""
        import copy

        candidate = copy.deepcopy(self)
        with _defer_ops_validation():
            if ops_implementation is not None:
                candidate._set_ops_implementation(ops_implementation)
            if attn_implementation is not None:
                candidate._attn_implementation = attn_implementation
        if not _DEFER_OPS_VALIDATION.get() and not isinstance(ops_implementation, SimpleNamespace):
            candidate._set_ops_implementation({}, _validate_tree=True)
        # Preserve HF's explicit attention spelling after validation.
        if attn_implementation is not None:
            candidate._attn_implementation = attn_implementation

        def apply(config, source):
            config.ops_implementation = dict(source.ops_implementation)
            for key in config.sub_configs:
                child, proposed = getattr(config, key, None), getattr(source, key, None)
                if isinstance(child, VeOmniPretrainedConfig) and isinstance(proposed, VeOmniPretrainedConfig):
                    apply(child, proposed)

        apply(self, candidate)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        ops = kwargs.pop("ops_implementation", None)
        attention = kwargs.pop("attn_implementation", None)
        with _defer_ops_validation():
            result = super().from_dict(dict(config_dict), **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config._apply_ops_overrides(ops_implementation=ops, attn_implementation=attention)
        return result

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        """Export remote config code from its native schema rather than the adapter."""
        if type(self).__dict__.get("_hf_config_class") is not None and self._auto_class is not None:
            return as_hf_config(self).save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        return super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

    def to_diff_dict(self) -> dict[str, Any]:
        diff = super().to_diff_dict()
        _prune_subconfig_ops_diff(self, diff)
        return diff


def _ops_dict(value):
    from ..arguments import OpsImplementationConfig

    if isinstance(value, OpsImplementationConfig):
        return value.to_dict()
    if isinstance(value, SimpleNamespace):
        return {key: selection for key, selection in vars(value).items() if selection is not None}
    return dict(value or {})


def _capture_selections(source):
    """Retain explicit selections before native composite post-init broadcasts None."""
    if isinstance(source, Mapping):
        return {key: _capture_selections(value) for key, value in source.items()}
    if isinstance(source, PretrainedConfig):
        return {
            "ops_implementation": _ops_dict(getattr(source, "ops_implementation", None)),
            "attn_implementation": vars(source).get("_attn_implementation_internal"),
            **{key: _capture_selections(getattr(source, key, None)) for key in source.sub_configs},
        }
    return source


def _restore_selections(config, saved):
    if not isinstance(saved, Mapping):
        return
    own = _ops_dict(saved.get("ops_implementation"))
    attention = saved.get("attn_implementation")
    if isinstance(attention, str):
        own.setdefault("attn_implementation", _normalize_attention(attention))
    config.ops_implementation.update(own)
    for key in config.sub_configs:
        child = getattr(config, key, None)
        if isinstance(child, VeOmniPretrainedConfig):
            _restore_selections(child, saved.get(key))
    if isinstance(attention, Mapping):
        config._attn_implementation = attention


def _normalize_attention(value: str) -> str:
    from ..arguments import OpsImplementationConfig

    return OpsImplementationConfig.normalize_hub_attention_backend(value)


@cache
def get_veomni_config_class(config_class: type[PretrainedConfig]) -> type[VeOmniPretrainedConfig]:
    """Return a cached VeOmni subclass while retaining the native HF config contract."""
    if issubclass(config_class, VeOmniPretrainedConfig):
        return config_class
    if config_class is PretrainedConfig:
        return VeOmniPretrainedConfig
    return type(
        config_class.__name__,
        (VeOmniPretrainedConfig, config_class),
        {
            "__module__": __name__,
            "__init__": config_class.__init__,
            "__reduce__": _reduce_config,
            "_hf_config_class": config_class,
            "sub_configs": {
                key: get_veomni_config_class(value)
                if isinstance(value, type) and issubclass(value, PretrainedConfig)
                else value
                for key, value in config_class.sub_configs.items()
            },
        },
    )


def _reduce_config(config):
    return _restore_config, (type(config).__dict__.get("_hf_config_class", type(config)), vars(config))


def _restore_config(config_class, state):
    config = object.__new__(get_veomni_config_class(config_class))
    vars(config).update(state)
    return config


def as_veomni_config(config: PretrainedConfig, *, validate: bool = True) -> VeOmniPretrainedConfig:
    """Opt a native config tree into VeOmni in place, preserving caller references."""
    native = not isinstance(config, VeOmniPretrainedConfig)
    if native:
        own = _ops_dict(getattr(config, "ops_implementation", None))
        attention = vars(config).pop("_attn_implementation_internal", None)
        if attention is not None:
            own.setdefault("attn_implementation", attention)
        config.__class__ = get_veomni_config_class(type(config))
        config.ops_implementation = own
    for key in config.sub_configs:
        child = getattr(config, key, None)
        if isinstance(child, PretrainedConfig):
            as_veomni_config(child, validate=False)
    if native:
        with _defer_ops_validation():
            config._set_ops_implementation(None, _is_post_init=True)
    if validate:
        config._set_ops_implementation({}, _validate_tree=True)
    return config


__all__ = ["VeOmniPretrainedConfig", "as_veomni_config", "get_veomni_config_class"]


def as_hf_config(config: PretrainedConfig) -> PretrainedConfig:
    """Copy a VeOmni adapter tree back to native types for HF AutoModel dispatch."""
    import copy

    result = copy.deepcopy(config)

    def restore(node):
        native_class = type(node).__dict__.get("_hf_config_class")
        attention = node._attn_implementation
        if native_class is not None:
            node.__class__ = native_class
            node._attn_implementation_internal = attention
        for key in node.sub_configs:
            child = getattr(node, key, None)
            if isinstance(child, PretrainedConfig):
                restore(child)

    restore(result)
    return result
