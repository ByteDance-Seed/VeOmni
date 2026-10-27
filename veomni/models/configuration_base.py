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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


_orig_post_init = PretrainedConfig.__post_init__
_orig_from_dict = PretrainedConfig.from_dict.__func__
_orig_to_diff_dict = PretrainedConfig.to_diff_dict


def _veomni_post_init(self: PretrainedConfig, **kwargs: Any) -> None:
    sub_snapshots = {
        key: (
            dict(getattr(sub, "ops_implementation", None) or {}),
            getattr(sub, "_attn_implementation_internal", None),
        )
        for key in self.sub_configs
        if isinstance(sub := getattr(self, key, None), PretrainedConfig)
    }
    ops_implementation = kwargs.pop("ops_implementation", getattr(self, "ops_implementation", None))
    explicit_attn = kwargs.get("attn_implementation")

    self.ops_implementation = {}
    _orig_post_init(self, **kwargs)

    for key, (sub_ops, sub_attn) in sub_snapshots.items():
        sub = getattr(self, key, None)
        if not isinstance(sub, PretrainedConfig):
            continue
        sub.ops_implementation = dict(sub_ops)
        if isinstance(explicit_attn, Mapping) and key in explicit_attn:
            continue
        if sub_attn is not None:
            sub._attn_implementation_internal = sub_attn

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
    ) -> None:
        from ..arguments import OpsImplementationConfig

        replace_all = isinstance(ops_implementation, (OpsImplementationConfig, SimpleNamespace))
        if isinstance(ops_implementation, OpsImplementationConfig):
            self.ops_implementation = ops_implementation.to_dict()
        elif isinstance(ops_implementation, SimpleNamespace):
            self.ops_implementation = {k: v for k, v in vars(ops_implementation).items() if v is not None}
        elif ops_implementation is None or isinstance(ops_implementation, Mapping):
            self.ops_implementation = OpsImplementationConfig.validate_and_extract_dict(ops_implementation)
        else:
            raise ValueError(
                f"ops_implementation must be a dict or OpsImplementationConfig, got {type(ops_implementation).__name__}."
            )

        if "attn_implementation" in self.ops_implementation:
            self._attn_implementation_internal = self.ops_implementation["attn_implementation"]

        for key in self.sub_configs:
            subconfig = getattr(self, key, None)
            if not isinstance(subconfig, PretrainedConfig):
                continue
            if replace_all:
                subconfig._set_ops_implementation(ops_implementation)
            elif _is_post_init:
                sub_own = dict(getattr(subconfig, "ops_implementation", None) or {})
                sub_attn = getattr(subconfig, "_attn_implementation_internal", None)
                merged = {**self.ops_implementation, **sub_own}
                subconfig._set_ops_implementation(merged)
                if "attn_implementation" not in sub_own and sub_attn is not None:
                    subconfig._attn_implementation_internal = sub_attn
            elif self.ops_implementation:
                sub_own = dict(getattr(subconfig, "ops_implementation", None) or {})
                subconfig._set_ops_implementation({**sub_own, **self.ops_implementation})

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        ops_implementation = kwargs.pop("ops_implementation", None)
        result = _orig_from_dict(cls, config_dict, **kwargs)
        if ops_implementation is not None:
            config = result[0] if isinstance(result, tuple) else result
            if isinstance(ops_implementation, Mapping):
                own = dict(getattr(config, "ops_implementation", None) or {})
                config._set_ops_implementation({**own, **dict(ops_implementation)})
            else:
                config._set_ops_implementation(ops_implementation)
        return result

    def to_diff_dict(self) -> dict[str, Any]:
        diff = _orig_to_diff_dict(self)
        if not diff.get("ops_implementation"):
            diff.pop("ops_implementation", None)
        parent_ops = getattr(self, "ops_implementation", None) or {}
        for key in self.sub_configs:
            sub_dict = diff.get(key)
            if isinstance(sub_dict, dict) and "ops_implementation" in sub_dict:
                sub_ops = sub_dict["ops_implementation"]
                if isinstance(sub_ops, Mapping):
                    sub_diff = {k: v for k, v in sub_ops.items() if parent_ops.get(k) != v}
                    if sub_diff:
                        sub_dict["ops_implementation"] = sub_diff
                    else:
                        sub_dict.pop("ops_implementation", None)
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
