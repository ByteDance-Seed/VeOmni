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

"""OmniModel argument parsing — :func:`parse_omni_args` + its helpers.

This is the OmniModel-specific counterpart to the generic
:func:`veomni.arguments.parser.parse_args`.  It reuses the generic argparse /
merge primitives from :mod:`veomni.arguments.parser` and layers on two
behaviours the omni launchers need:

* **Schema-less CLI flags** — arbitrary ``--a.b.c value`` flags not declared on
  the dataclass tree are coerced and deep-merged (so free-form ``dict`` fields
  like ``infer.generation_kwargs`` and pre-loaded module blocks grow keys from
  the CLI).
* **Path-field pre-loading** — a field that is normally a *path* (e.g.
  ``model.modules``) is loaded into a dict before CLI overrides are applied, so
  ``--model.modules.<name>.* value`` merges into the loaded blocks.
"""

import argparse
from typing import Any, Dict, Type

import yaml

from .parser import T, _add_arguments_recursive, _deep_update, _instantiate_recursive


def _coerce_cli_value(raw: str) -> Any:
    """Coerce a raw CLI string into a Python scalar.

    Unknown (schema-less) CLI flags carry no dataclass type, so we lean on
    ``yaml.safe_load`` for the same coercion YAML would apply (``5.0`` -> float,
    ``true`` -> bool, ``null`` -> None, ``eager`` -> str, ``[1, 2]`` -> list).
    Falls back to the raw string if parsing fails.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _parse_unknown_cli(unknown: list, sink: Dict[str, Any]) -> None:
    """Fold leftover ``--a.b.c value`` / ``--a.b.c=value`` flags into ``sink``.

    These are flags not declared on the dataclass tree (e.g. a free-form
    ``--infer.generation_kwargs.guidance_scale 5.0`` or a per-module override
    ``--model.modules.janus_siglip.accelerator.fsdp_config.fsdp_mode eager``).
    The dotted key becomes a nested dict; ``_deep_update`` then merges it onto
    the YAML config and ``_instantiate_recursive`` keeps anything that lands in
    a free-form ``dict`` field (or a pre-loaded module dict).
    """
    i = 0
    n = len(unknown)
    while i < n:
        token = unknown[i]
        if not token.startswith("--"):
            i += 1
            continue
        body = token[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
            i += 1
        else:
            key = body
            # A following non-flag token is the value; otherwise treat as a
            # boolean switch (``--flag`` -> True).
            if i + 1 < n and not unknown[i + 1].startswith("--"):
                raw = unknown[i + 1]
                i += 2
            else:
                raw = "true"
                i += 1

        value = _coerce_cli_value(raw)
        keys = key.split(".")
        current = sink
        for k in keys[:-1]:
            nxt = current.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                current[k] = nxt
            current = nxt
        current[keys[-1]] = value


def _preload_path_field(config: Dict[str, Any], path_keys) -> None:
    """Replace a string YAML-path field with its loaded contents, in place.

    Lets a field that is normally a *path* (e.g. ``model.modules``) accept CLI
    sub-key overrides: once the file is loaded into a dict, a later
    ``--model.modules.<name>.* value`` deep-merges into it instead of clobbering
    the path string.  No-op when the field is absent or already a dict/list.
    """
    current = config
    for k in path_keys[:-1]:
        nxt = current.get(k) if isinstance(current, dict) else None
        if not isinstance(nxt, dict):
            return
        current = nxt
    last = path_keys[-1]
    value = current.get(last) if isinstance(current, dict) else None
    if not isinstance(value, str) or not value.endswith((".yaml", ".yml")):
        return
    with open(value) as f:
        loaded = yaml.safe_load(f)
    if loaded is not None:
        current[last] = loaded


def parse_omni_args(
    root_class: Type[T],
    *,
    preload_path_fields: tuple = (),
    return_config_path: bool = False,
):
    """Like :func:`veomni.arguments.parser.parse_args`, but tolerant of
    **arbitrary** CLI overrides.

    Adds two behaviours on top of ``parse_args``:

    1. **Schema-less flags** — any ``--a.b.c value`` not declared on the
       dataclass tree is parsed (value coerced via YAML) and merged into the
       config.  This lets free-form ``dict`` fields (e.g.
       ``infer.generation_kwargs``) grow new keys from the CLI.
    2. **Path-field pre-loading** — for each dotted key in
       ``preload_path_fields`` (e.g. ``"model.modules"``), a string YAML path in
       the config is loaded into a dict *before* CLI overrides are applied, so
       ``--model.modules.<name>.* value`` deep-merges into the loaded module
       blocks instead of replacing the path string.

    CLI precedence over YAML is preserved (CLI overrides win via
    ``_deep_update``).
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("config_file", nargs="?", help="Path to YAML config file")
    _add_arguments_recursive(parser, root_class)
    args, unknown = parser.parse_known_args()

    final_config: Dict[str, Any] = {}
    config_path: Any = getattr(args, "config_file", None) or None
    if config_path and (config_path.endswith(".yaml") or config_path.endswith(".yml")):
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config = yaml_config

    # Pre-load path fields (e.g. model.modules) so sub-key CLI overrides merge.
    for dotted in preload_path_fields:
        _preload_path_field(final_config, dotted.split("."))

    cli_config: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "config_file":
            continue
        keys = key.split(".")
        current_level = cli_config
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value

    # Schema-less / nested overrides not declared on the dataclass tree.
    _parse_unknown_cli(unknown, cli_config)

    final_config = _deep_update(final_config, cli_config)

    instance = _instantiate_recursive(root_class, final_config)
    if return_config_path:
        return instance, config_path
    return instance


__all__ = ["parse_omni_args"]
