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
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


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
        # A missing child key means its device default, not the parent's choice.
        resets = {
            key: None
            for key, value in parent_ops.items()
            if value is not None and key not in sub_cfg.ops_implementation
        }
        if resets:
            sub_dict.setdefault("ops_implementation", {}).update(resets)


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
        self._set_ops_implementation(ops, _is_post_init=True, _validate=False)
        if isinstance(attention, Mapping):
            self._attn_implementation = attention

    @property
    def ops_config(self) -> OpsImplementationConfig:
        """Return the normally constructed ops instance, refreshing it after dict updates."""
        cached = vars(self).get("_ops_config_cache")
        if cached is None or cached[0] != self.ops_implementation:
            selections, resolved = _resolve_ops(self.ops_implementation)
            self.ops_implementation.update(selections)
            self._ops_config_cache = (dict(self.ops_implementation), resolved)
        return self._ops_config_cache[1]

    @property
    def _attn_implementation(self) -> str | None:
        from ..arguments import OpsImplementationConfig

        selected = self.ops_implementation.get("attn_implementation")
        if selected is not None:
            return selected
        default = OpsImplementationConfig.__dataclass_fields__["attn_implementation"].default
        return OpsImplementationConfig.normalize_attention_backend(default)

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
        _validate: bool = True,
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
        pending = []

        def collect(config, incoming):
            merged = dict(incoming) if replace and not _is_post_init else {**config.ops_implementation, **incoming}
            proposed = dict(merged)
            if proposed.get("attn_implementation") is not None:
                proposed["attn_implementation"] = OpsImplementationConfig.normalize_attention_backend(
                    proposed["attn_implementation"]
                )
            pending.append((config, proposed))
            for key in config.sub_configs:
                child = getattr(config, key, None)
                if not isinstance(child, VeOmniPretrainedConfig):
                    continue
                if _is_post_init:
                    collect(child, {**proposed, **child.ops_implementation})
                elif replace or incoming or _validate_tree:
                    collect(child, incoming)

        # Validate the whole proposed tree before modifying any instance.
        collect(self, override)
        resolved = (
            [_resolve_ops(proposed) for _, proposed in pending] if _validate else [(p, None) for _, p in pending]
        )
        for (config, _), (proposed, ops_config) in zip(pending, resolved):
            config.ops_implementation = proposed
            config.__dict__.pop("_ops_config_cache", None)
            if ops_config is not None:
                config._ops_config_cache = (dict(proposed), ops_config)

    def _apply_ops_overrides(
        self,
        *,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | Any | None = None,
        attn_implementation: str | dict[str, Any] | None = None,
        _validate: bool = True,
    ) -> None:
        """Apply load-time overrides atomically; explicit attention ranks highest."""
        import copy

        candidate = copy.deepcopy(self)
        if ops_implementation is not None:
            candidate._set_ops_implementation(ops_implementation, _validate=False)
        if attn_implementation is not None:
            candidate._attn_implementation = attn_implementation
        if _validate:
            candidate._set_ops_implementation({}, _validate_tree=True)
        # Preserve HF's explicit attention spelling after validation.
        if attn_implementation is not None:
            candidate._attn_implementation = attn_implementation

        def apply(config, source):
            config.ops_implementation = dict(source.ops_implementation)
            config.__dict__.pop("_ops_config_cache", None)
            if "_ops_config_cache" in vars(source):
                resolved = source._ops_config_cache[1]
                resolved.attn_implementation = source._attn_implementation
                config._ops_config_cache = (dict(config.ops_implementation), resolved)
            for key in config.sub_configs:
                child, proposed = getattr(config, key, None), getattr(source, key, None)
                if isinstance(child, VeOmniPretrainedConfig) and isinstance(proposed, VeOmniPretrainedConfig):
                    apply(child, proposed)

        apply(self, candidate)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        validate = kwargs.pop("_validate_ops", True)
        ops = kwargs.pop("ops_implementation", None)
        attention = kwargs.pop("attn_implementation", None)
        result = super().from_dict(dict(config_dict), **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        as_veomni_config(config, validate=False)
        config._set_ops_implementation(None, _is_post_init=True, _validate=False)
        config._apply_ops_overrides(ops_implementation=ops, attn_implementation=attention, _validate=validate)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize selections and schema fields without the resolved runtime cache."""
        output = super().to_dict()
        output.pop("_ops_config_cache", None)
        return output

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


def _resolve_ops(selections):
    """Validate final selections through the standard ops dataclass constructor."""
    from ..arguments import OpsImplementationConfig

    resolved = OpsImplementationConfig(**{key: value for key, value in selections.items() if value is not None})
    # Explicit HF attention spellings retain their original dispatch semantics.
    if selections.get("attn_implementation") is not None:
        resolved.attn_implementation = OpsImplementationConfig.normalize_attention_backend(
            selections["attn_implementation"], veomni_names=False
        )
    return {key: getattr(resolved, key) if value is not None else None for key, value in selections.items()}, resolved


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

    return OpsImplementationConfig.normalize_attention_backend(value, veomni_names=False)


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
        config._set_ops_implementation(None, _is_post_init=True, _validate=False)
    if validate:
        _ = config.ops_config
        for key in config.sub_configs:
            child = getattr(config, key, None)
            if isinstance(child, VeOmniPretrainedConfig):
                as_veomni_config(child)
    return config


__all__ = ["VeOmniPretrainedConfig", "as_veomni_config", "get_veomni_config_class"]


def as_hf_config(config: PretrainedConfig) -> PretrainedConfig:
    """Copy a VeOmni adapter tree back to native types for HF AutoModel dispatch."""
    import copy

    result = copy.deepcopy(config)

    def restore(node):
        native_class = type(node).__dict__.get("_hf_config_class")
        attention = node._attn_implementation
        node.__dict__.pop("_ops_config_cache", None)
        if native_class is not None:
            node.__class__ = native_class
            node._attn_implementation_internal = attention
        for key in node.sub_configs:
            child = getattr(node, key, None)
            if isinstance(child, PretrainedConfig):
                restore(child)

    restore(result)
    return result
