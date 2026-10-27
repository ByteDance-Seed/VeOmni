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
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


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

    def __post_init__(self, **kwargs: Any) -> None:
        ops_implementation = kwargs.pop("ops_implementation", getattr(self, "ops_implementation", None))
        self.ops_implementation = {}
        super().__post_init__(**kwargs)
        self._set_ops_implementation(ops_implementation)

    @property
    def ops_config(self) -> OpsImplementationConfig:
        """Resolved :class:`~veomni.arguments.OpsImplementationConfig` for this config."""
        from ..arguments import OpsImplementationConfig

        return OpsImplementationConfig.from_dict(getattr(self, "ops_implementation", None), validate=False)

    def _set_ops_implementation(
        self,
        ops_implementation: dict[str, Any] | OpsImplementationConfig | None,
    ) -> None:
        from ..arguments import OpsImplementationConfig

        if isinstance(ops_implementation, OpsImplementationConfig):
            self.ops_implementation = ops_implementation.to_dict()
        elif ops_implementation is None or isinstance(ops_implementation, Mapping):
            self.ops_implementation = OpsImplementationConfig.validate_and_extract_dict(ops_implementation)
        else:
            raise ValueError(
                f"ops_implementation must be a dict or OpsImplementationConfig, got {type(ops_implementation).__name__}."
            )

        for key in self.sub_configs:
            if (subconfig := getattr(self, key, None)) is not None:
                subconfig._set_ops_implementation(self.ops_implementation)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        ops_implementation = kwargs.pop("ops_implementation", None)
        result = super().from_dict(config_dict, **kwargs)
        if ops_implementation is not None:
            config = result[0] if isinstance(result, tuple) else result
            if isinstance(ops_implementation, Mapping):
                own = dict(getattr(config, "ops_implementation", None) or {})
                config._set_ops_implementation({**own, **dict(ops_implementation)})
            else:
                config._set_ops_implementation(ops_implementation)
        return result

    def to_diff_dict(self) -> dict[str, Any]:
        diff = super().to_diff_dict()
        if not diff.get("ops_implementation"):
            diff.pop("ops_implementation", None)
        return diff


_orig_remove_keys_not_serialized = PretrainedConfig._remove_keys_not_serialized


def _veomni_remove_keys_not_serialized(self: PretrainedConfig, d: dict[str, Any]) -> None:
    for key in self.sub_configs:
        sub_dict = d.get(key)
        if isinstance(sub_dict, dict):
            sub_dict.pop("ops_implementation", None)
    _orig_remove_keys_not_serialized(self, d)


def apply_pretrained_config_ops_patch() -> None:
    """Attach ``ops_config`` and ``_set_ops_implementation`` to ``PretrainedConfig``."""
    if getattr(PretrainedConfig, "_veomni_ops_config_patched", False):
        return
    PretrainedConfig.ops_config = VeOmniPretrainedConfig.ops_config  # type: ignore[attr-defined]
    PretrainedConfig._set_ops_implementation = VeOmniPretrainedConfig._set_ops_implementation  # type: ignore[attr-defined]
    PretrainedConfig._remove_keys_not_serialized = _veomni_remove_keys_not_serialized  # type: ignore[method-assign]
    PretrainedConfig._veomni_ops_config_patched = True  # type: ignore[attr-defined]


apply_pretrained_config_ops_patch()


__all__ = [
    "VeOmniPretrainedConfig",
    "apply_pretrained_config_ops_patch",
]
