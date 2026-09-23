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

"""HF-native base config for every SeedOmni sub-module.

A module's own ``config.json`` is a subclass instance: its hyperparameters plus
the kernels it was exported with. ``OmniConfig._module_configs[name]`` is that
object. ``OmniConfig._module_entries[name]`` is the checkpoint dict that says
where the module lives and what the composed model overwrites. Loading merges
the file's kernels, the composed model's default, and that entry onto this
config.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig


class OmniModuleConfig(PretrainedConfig):
    """Base for every ``modules/<family>/<sub>/configuration.py`` config class.

    Subclasses add the module's own hyperparameters and a ``model_type``. The
    fields here are what an omni checkpoint records about a module: where it
    lives, and the kernels / config / processor overwrites applied when the
    composed model loads it.
    """

    def __init__(
        self,
        model_path: str | None = None,
        ops_implementation: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        processor_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Relative (resolved against the omni checkpoint root) or absolute (an
        # external checkpoint). ``None`` on a module's own config.json, where
        # the file's own directory is the answer.
        self.model_path = model_path
        # Sparse: only what this location explicitly selected. A composed load
        # merges the file, the model's default, and the module entry per field.
        self.ops_implementation = dict(ops_implementation or {})
        # Overrides the composed model applies to the module: hyperparameters
        # for the model, kwargs for the preprocessor.
        self.model_config = dict(model_config or {})
        self.processor_config = dict(processor_config or {})
        super().__init__(**kwargs)

    def to_diff_dict(self) -> dict[str, Any]:
        """Drop the composed-model fields this config never set.

        HF keeps any key ``PretrainedConfig`` itself does not define, even when
        the value still equals this class' default. Without this, every
        module's own ``config.json`` grows four empty fields — and three of
        them (``model_path``, ``model_config``, ``processor_config``) are
        things only a composed model says about a module.
        """
        diff = super().to_diff_dict()
        for field in ("model_path", "ops_implementation", "model_config", "processor_config"):
            if not diff.get(field):
                diff.pop(field, None)
        return diff

    @staticmethod
    def resolve_path(
        checkpoint_root: str | os.PathLike | None,
        name: str,
        model_path: str | None = None,
    ) -> str:
        """On-disk directory to load module ``name`` from.

        An absolute ``model_path`` wins over ``<root>/<name>``, present or not:
        a module pointed at another checkpoint must keep loading from there
        even when a same-named directory happens to exist under the root.
        """
        path = model_path or name
        if os.path.isabs(path):
            return path
        if checkpoint_root is None:
            return path
        return os.path.join(str(checkpoint_root), path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> OmniModuleConfig:
        """Load this module's ``config.json``, then apply composed-model overwrites.

        Called on the base class, the concrete config comes from
        ``OMNI_MODEL_REGISTRY`` (the same gate as every other omni load).

        ``model_config`` and ``processor_config`` are the module entry's
        overwrites. ``ops_implementation`` is that entry's kernels.
        ``base_ops_implementation`` is the composed model's default, passed in
        separately because an entry does not carry it. Per field, lowest to
        highest: what this file loaded, then ``base_ops_implementation``, then
        the entry's ``ops_implementation``.
        """
        model_config = kwargs.pop("model_config", None)
        processor_config = kwargs.pop("processor_config", None)
        entry_ops = kwargs.pop("ops_implementation", None)
        base_ops = kwargs.pop("base_ops_implementation", None)

        if cls is OmniModuleConfig:
            from . import OMNI_MODEL_REGISTRY, read_model_type

            cls = OMNI_MODEL_REGISTRY[read_model_type(str(pretrained_model_name_or_path))]().config_class

        hf_config = super().from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        if not isinstance(hf_config, OmniModuleConfig):
            raise TypeError(
                f"Module config loaded from {pretrained_model_name_or_path} is a "
                f"{type(hf_config).__name__}, not a OmniModuleConfig."
            )
        hf_config._apply_composed_overwrites(
            model_config=model_config,
            processor_config=processor_config,
            ops_implementation=entry_ops,
            base_ops_implementation=base_ops,
        )
        return hf_config

    def _apply_composed_overwrites(
        self,
        *,
        model_config: dict[str, Any] | None,
        processor_config: dict[str, Any] | None,
        ops_implementation: dict[str, Any] | None,
        base_ops_implementation: dict[str, Any] | None,
    ) -> None:
        """Write a composed model's overwrites onto this module config."""

        def as_dict(field: str, value: Any) -> dict[str, Any] | None:
            if value is None:
                return None
            if not isinstance(value, dict):
                raise ValueError(f"{field} must be a dict, got {type(value).__name__}.")
            return value

        model_config = as_dict("model_config", model_config)
        processor_config = as_dict("processor_config", processor_config)
        entry_ops = as_dict("ops_implementation", ops_implementation)
        base_ops = as_dict("base_ops_implementation", base_ops_implementation)
        if model_config:
            self.update(deepcopy(model_config))
        if processor_config:
            current = dict(getattr(self, "processor_config", None) or {})
            self.processor_config = {**current, **deepcopy(processor_config)}
        if base_ops or entry_ops:
            own = dict(getattr(self, "ops_implementation", None) or {})
            self.ops_implementation = {**own, **(base_ops or {}), **(entry_ops or {})}
        attn_implementation = dict(self.ops_implementation or {}).get("attn_implementation")
        if attn_implementation is not None:
            # HF selects the attention class from this attribute, not from ``ops_implementation``.
            self._attn_implementation = attn_implementation


__all__ = [
    "OmniModuleConfig",
]
