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

"""HF-native per-module config for a composed :class:`OmniConfig`.

Family ``configuration.py`` files keep their own ``PretrainedConfig`` subclasses
(the hydrated ``config.json`` next to a module's weights). This module owns the
*slot* in ``OmniConfig.modules[name]``: descriptor dict, hydrated family config,
subfolder / ``model_path`` resolution, export slimming, and the flatten back
onto launcher fields.

The VeOmni runtime counterpart is per-module launcher ``ModelArguments``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from transformers import PretrainedConfig


def safe_checkpoint_subfolder(name: str) -> str:
    """Return ``name`` if it is a single relative path component, else raise.

    Joined onto ``save_directory`` by omni checkpoint writers. Absolute paths,
    ``.`` / ``..``, and any separator would let a module key write outside the
    checkpoint root.
    """
    if not name or name in {".", ".."}:
        raise ValueError(f"Module name {name!r} is not a safe checkpoint subfolder.")
    if os.path.isabs(name):
        raise ValueError(
            f"Module name {name!r} is an absolute path; checkpoint subfolders must be "
            "a single relative path component."
        )
    if os.path.sep in name or (os.path.altsep is not None and os.path.altsep in name):
        raise ValueError(f"Module name {name!r} is not a safe checkpoint subfolder; use a single path component.")
    return name


class OmniModuleConfig:
    """HF-native view of one ``OmniConfig.modules[name]`` entry.

    ``entry`` is a descriptor dict (``subfolder`` / ``model`` / ``processor_config``),
    a hydrated family :class:`~transformers.PretrainedConfig`, or a path string.
    """

    def __init__(self, name: str, entry: Any = None):
        self.name = name
        self.entry = {} if entry is None else entry

    @property
    def checkpoint_subfolder(self) -> str:
        """Relative subfolder under an omni checkpoint root for this module."""
        return safe_checkpoint_subfolder(self.name)

    @property
    def subfolder(self) -> str:
        """On-disk path segment (relative subfolder, or an absolute load path)."""
        entry = self.entry
        if isinstance(entry, PretrainedConfig):
            return self.checkpoint_subfolder
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            model_block = entry.get("model")
            if isinstance(model_block, dict):
                path = model_block.get("model_path") or model_block.get("weights_path")
                if path:
                    return path
            subfolder = entry.get("subfolder")
            if subfolder:
                return str(subfolder)
        return self.name

    def model_config_overrides(self) -> dict[str, Any]:
        """Per-module ``from_pretrained`` overrides stored on the omni entry."""
        entry = self.entry
        if isinstance(entry, PretrainedConfig) or not isinstance(entry, dict):
            return {}
        model_block = entry.get("model")
        if not isinstance(model_block, dict):
            return {}
        overrides = model_block.get("model_config")
        return dict(overrides or {})

    def processor_config(self) -> dict[str, Any]:
        """Per-module preprocessor ``from_pretrained`` kwargs."""
        entry = self.entry
        if not isinstance(entry, dict):
            return {}
        return dict(entry.get("processor_config") or {})

    def resolve_path(self, checkpoint_root: str | os.PathLike | None) -> str:
        """Resolve the on-disk path for this module under ``checkpoint_root``."""
        subfolder = self.subfolder
        if os.path.isabs(subfolder):
            return subfolder
        if checkpoint_root is None:
            return subfolder
        return os.path.join(str(checkpoint_root), subfolder)

    def to_export_dict(self) -> dict[str, Any]:
        """Slim descriptor for HF ``config.json`` (subfolder + optional config overrides)."""
        slim: dict[str, Any] = {"subfolder": self.checkpoint_subfolder}
        model_config = self.model_config_overrides()
        if model_config:
            slim["model"] = {"model_config": model_config}
        processor_config = self.processor_config()
        if processor_config:
            slim["processor_config"] = deepcopy(processor_config)
        return slim

    def as_runtime_fields(self) -> dict[str, Any]:
        """Flatten a checkpoint-shaped descriptor onto launcher ``ModelArguments`` keys."""
        if not isinstance(self.entry, dict):
            raise TypeError(f"Module '{self.name}' must be a mapping to flatten onto runtime fields.")
        cfg = deepcopy(self.entry)
        cfg.pop("subfolder", None)
        model_block = cfg.pop("model", None)
        if isinstance(model_block, dict):
            for key, value in model_block.items():
                if key not in cfg:
                    cfg[key] = value
        return cfg

    def hydrate(self, checkpoint_root: str | os.PathLike) -> Any:
        """Load this module's ``config.json`` via the omni module registry when it is in-root."""
        if isinstance(self.entry, PretrainedConfig):
            return self.entry

        from . import OMNI_MODEL_REGISTRY, read_hf_model_type

        root = str(checkpoint_root)
        overrides = self.model_config_overrides()
        subfolder = self.checkpoint_subfolder
        # Hydrate from ``root/<subfolder>`` only. An entry whose weights live
        # outside the root has no ``config.json`` here, stays a descriptor,
        # and :meth:`resolve_path` keeps its own path.
        module_dir = os.path.join(root, subfolder)
        if not os.path.isfile(os.path.join(module_dir, "config.json")):
            return self.entry if self.entry is not None else {"subfolder": subfolder}
        model_type = read_hf_model_type(module_dir)
        hf_config = OMNI_MODEL_REGISTRY[model_type]().config_class.from_pretrained(module_dir)
        if overrides:
            hf_config.update(deepcopy(overrides))
        return hf_config

    @classmethod
    def from_runtime(
        cls,
        name: str,
        *,
        model_path: str | None = None,
        model_config: dict[str, Any] | None = None,
        processor_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an ``OmniConfig.modules`` descriptor dict from runtime fields.

        ``model_path`` is carried through explicitly (not just ``subfolder:
        name``): by the time this runs, the launcher has already resolved it
        to an absolute path — usually ``<checkpoint_root>/<name>``, but a
        module override may point at a wholly different checkpoint.
        """
        model_block: dict[str, Any] = {}
        if model_path:
            model_block["model_path"] = model_path
        if model_config:
            model_block["model_config"] = deepcopy(model_config)
        entry: dict[str, Any] = {
            "subfolder": name,
            "processor_config": deepcopy(processor_config or {}),
        }
        if model_block:
            entry["model"] = model_block
        return entry


__all__ = ["OmniModuleConfig", "safe_checkpoint_subfolder"]
