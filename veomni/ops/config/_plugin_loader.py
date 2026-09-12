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

"""External ops-backend plugin loader.

Third-party kernel packages can add **opt-in** backends to VeOmni's
config-driven kernel selection without modifying VeOmni, by declaring an
entry point in the ``veomni.ops_backends`` group::

    [project.entry-points."veomni.ops_backends"]
    my_kernels = "my_kernels.veomni_plugin"

The referenced module must be pure data (no heavy imports, no side effects)
and expose two attributes:

* ``VEOMNI_PLUGIN_API_VERSION`` — protocol version; must exactly equal the
  version supported here, otherwise the plugin is rejected.
* ``VEOMNI_OPS_BACKENDS`` — declaration payload::

      VEOMNI_OPS_BACKENDS = {
          "ops": {  # merged into OpSpec.backends (config-field dispatch)
              "rms_norm": {
                  "my_backend": {"entry": "my_kernels:MyRMSNorm",
                                  "requires": ["my_kernels"]},
              },
          },
          "kernels": [  # merged into KERNEL_REGISTRY (OpSlot dispatch)
              {"name": "my_backend", "op_name": "rms_norm", "variant": "standard",
               "factory": "my_kernels:rms_norm", "device_type": "gpu",
               "description": "..."},
          ],
      }

Properties guaranteed by the loader:

* **Opt-in only.** The payload schema has no default/fallback keys; plugins
  can make a backend name resolvable, never change any op's default.
* **Atomic per plugin.** The whole payload is validated before anything is
  registered; a plugin with any invalid entry registers nothing.
* **Never fatal.** Any plugin failure (import error, bad payload, name
  conflict) degrades to a warning and the plugin is skipped — exactly as if
  it were not installed.
* **One-shot.** Discovery runs once per process at ``veomni.ops`` import
  time; ``VEOMNI_OPS_PLUGINS=0`` disables the mechanism entirely.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from ...utils import logging
from ...utils.env import get_env
from ..kernel_registry import KERNEL_REGISTRY, HardwareRequirement, KernelSpec
from .registry import BackendSpec, OpScope, _import_entry, extend_op_backends, get_op


logger = logging.get_logger(__name__)

PLUGIN_GROUP = "veomni.ops_backends"
SUPPORTED_PLUGIN_API_VERSION = 1

_PAYLOAD_KEYS = frozenset({"ops", "kernels"})
_BACKEND_KEYS = frozenset(
    {"entry", "requires", "side_effect", "replace_forward", "entry_is_factory", "target_override"}
)
_PER_MODEL_ONLY_KEYS = ("replace_forward", "entry_is_factory", "target_override")
_KERNEL_REQUIRED_KEYS = ("name", "op_name", "variant", "factory", "device_type")
_KERNEL_KEYS = frozenset(_KERNEL_REQUIRED_KEYS + ("min_compute_capability", "max_compute_capability", "description"))
_DEVICE_TYPES = frozenset({"gpu", "npu", "any"})


class PluginRejected(Exception):
    """A plugin declaration is invalid; the whole plugin is skipped."""


# ---------------------------------------------------------------------------
# Validation / translation (pure; no registry mutation)
# ---------------------------------------------------------------------------


def _check_entry_string(value: Any, what: str) -> None:
    if not isinstance(value, str):
        raise PluginRejected(f"{what} must be a 'module:attr' string, got {type(value).__name__}.")
    module, sep, attr = value.partition(":")
    if not (sep and module and attr):
        raise PluginRejected(f"{what} must be of the form 'module:attr', got {value!r}.")


def _build_backend_spec(op_name: str, backend_name: str, desc: Any) -> BackendSpec:
    if not isinstance(desc, dict):
        raise PluginRejected(f"ops.{op_name}.{backend_name} must be a mapping, got {type(desc).__name__}.")
    unknown = desc.keys() - _BACKEND_KEYS
    if unknown:
        raise PluginRejected(f"ops.{op_name}.{backend_name} has unknown key(s) {sorted(unknown)}.")
    if "entry" not in desc:
        raise PluginRejected(f"ops.{op_name}.{backend_name} is missing required key 'entry'.")
    _check_entry_string(desc["entry"], f"ops.{op_name}.{backend_name}.entry")

    requires = desc.get("requires", ())
    if not isinstance(requires, (list, tuple)) or not all(isinstance(p, str) for p in requires):
        raise PluginRejected(f"ops.{op_name}.{backend_name}.requires must be a list of package names.")

    side_effect = desc.get("side_effect")
    if side_effect is not None:
        _check_entry_string(side_effect, f"ops.{op_name}.{backend_name}.side_effect")
    replace_forward = desc.get("replace_forward", False)
    entry_is_factory = desc.get("entry_is_factory", False)
    if not isinstance(replace_forward, bool) or not isinstance(entry_is_factory, bool):
        raise PluginRejected(f"ops.{op_name}.{backend_name}.replace_forward/entry_is_factory must be booleans.")
    target_override = desc.get("target_override")
    if target_override is not None and not isinstance(target_override, str):
        raise PluginRejected(f"ops.{op_name}.{backend_name}.target_override must be a string.")

    return BackendSpec(
        entry=desc["entry"],
        requires=tuple(requires),
        side_effect=side_effect,
        replace_forward=replace_forward,
        entry_is_factory=entry_is_factory,
        target_override=target_override,
    )


def _check_scope_compatibility(op_name: str, backend_name: str, desc: dict[str, Any], scope: OpScope) -> None:
    """Reject keys the dispatch engine would silently ignore for this scope."""
    if scope is OpScope.GLOBAL:
        used = [key for key in _PER_MODEL_ONLY_KEYS if desc.get(key)]
        if used:
            raise PluginRejected(f"ops.{op_name}.{backend_name}: key(s) {used} only apply to per-model ops.")
    elif desc.get("side_effect"):
        raise PluginRejected(f"ops.{op_name}.{backend_name}: 'side_effect' only applies to global ops.")


def _build_kernel_spec(desc: Any) -> KernelSpec:
    if not isinstance(desc, dict):
        raise PluginRejected(f"kernels[] entries must be mappings, got {type(desc).__name__}.")
    unknown = desc.keys() - _KERNEL_KEYS
    if unknown:
        raise PluginRejected(f"kernels[] entry has unknown key(s) {sorted(unknown)}.")
    for key in _KERNEL_REQUIRED_KEYS:
        if key not in desc:
            raise PluginRejected(f"kernels[] entry is missing required key {key!r}.")
        if not isinstance(desc[key], str):
            raise PluginRejected(f"kernels[].{key} must be a string.")
    _check_entry_string(desc["factory"], "kernels[].factory")

    device_type = desc["device_type"]
    if device_type not in _DEVICE_TYPES:
        raise PluginRejected(f"kernels[].device_type must be one of {sorted(_DEVICE_TYPES)}, got {device_type!r}.")
    min_cc = desc.get("min_compute_capability")
    max_cc = desc.get("max_compute_capability")
    for cc, key in ((min_cc, "min_compute_capability"), (max_cc, "max_compute_capability")):
        if cc is not None and not isinstance(cc, int):
            raise PluginRejected(f"kernels[].{key} must be an integer.")
    hardware = HardwareRequirement(
        device_type=device_type, min_compute_capability=min_cc, max_compute_capability=max_cc
    )
    factory_entry = desc["factory"]
    return KernelSpec(
        name=desc["name"],
        op_name=desc["op_name"],
        variant=desc["variant"],
        factory=lambda: _import_entry(factory_entry),  # lazy: resolved on first use
        hardware=hardware,
        description=desc.get("description", ""),
    )


def _plan_from_module(module: Any) -> dict[str, Any]:
    """Validate a declaration module and build the registration plan.

    Pure function: raises :class:`PluginRejected` on any problem without
    touching registry state.
    """
    api_version = getattr(module, "VEOMNI_PLUGIN_API_VERSION", None)
    if api_version != SUPPORTED_PLUGIN_API_VERSION:
        raise PluginRejected(
            f"VEOMNI_PLUGIN_API_VERSION is {api_version!r}; this VeOmni supports exactly "
            f"{SUPPORTED_PLUGIN_API_VERSION}."
        )
    payload = getattr(module, "VEOMNI_OPS_BACKENDS", None)
    if payload is None:
        raise PluginRejected("module does not define VEOMNI_OPS_BACKENDS.")
    if not isinstance(payload, dict):
        raise PluginRejected(f"VEOMNI_OPS_BACKENDS must be a mapping, got {type(payload).__name__}.")
    unknown = payload.keys() - _PAYLOAD_KEYS
    if unknown:
        raise PluginRejected(
            f"VEOMNI_OPS_BACKENDS has unknown key(s) {sorted(unknown)}; plugins can only declare "
            "'ops' and 'kernels' (never defaults or fallbacks)."
        )

    ops_plan: dict[str, dict[str, BackendSpec]] = {}
    for op_name, backends in payload.get("ops", {}).items():
        try:
            op = get_op(op_name)
        except KeyError as e:
            raise PluginRejected(f"unknown op {op_name!r}.") from e
        if not isinstance(backends, dict):
            raise PluginRejected(f"ops.{op_name} must map backend names to declarations.")
        overlap = op.backends.keys() & backends.keys()
        if overlap:
            raise PluginRejected(f"backend name(s) {sorted(overlap)} already registered for op {op_name!r}.")
        specs: dict[str, BackendSpec] = {}
        for backend_name, desc in backends.items():
            _check_scope_compatibility(op_name, backend_name, desc, op.scope)
            specs[backend_name] = _build_backend_spec(op_name, backend_name, desc)
        ops_plan[op_name] = specs

    kernels_payload = payload.get("kernels", [])
    if not isinstance(kernels_payload, list):
        raise PluginRejected("'kernels' must be a list of declarations.")
    kernels_plan: list[KernelSpec] = []
    for desc in kernels_payload:
        spec = _build_kernel_spec(desc)
        if spec.name in KERNEL_REGISTRY.list_available(spec.op_name, spec.variant):
            raise PluginRejected(
                f"kernel name {spec.name!r} is already registered for op={spec.op_name!r}, variant={spec.variant!r}."
            )
        kernels_plan.append(spec)

    return {"ops": ops_plan, "kernels": kernels_plan}


def _summarize(plan: dict[str, Any]) -> str:
    parts = [f"{op_name}: {', '.join(sorted(specs))}" for op_name, specs in plan["ops"].items()]
    parts += [f"kernel {s.op_name}/{s.variant}: {s.name}" for s in plan["kernels"]]
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Registration + discovery
# ---------------------------------------------------------------------------


def _register_plan(plan: dict[str, Any]) -> None:
    for op_name, specs in plan["ops"].items():
        extend_op_backends(op_name, specs)
    for spec in plan["kernels"]:
        KERNEL_REGISTRY.register(spec)


def _load_plugins(eps: Iterable[EntryPoint]) -> tuple[str, ...]:
    if get_env("VEOMNI_OPS_PLUGINS") == "0":
        return ()
    loaded: list[str] = []
    for ep in sorted(eps, key=lambda e: e.name):
        try:
            module = ep.load()
            plan = _plan_from_module(module)
            _register_plan(plan)
        except Exception as e:  # noqa: BLE001 -- a broken plugin must never break VeOmni startup
            logger.warning_rank0(f"Ops backend plugin {ep.name!r} skipped: {e}")
            continue
        logger.info_rank0(f"Loaded ops backend plugin {ep.name!r}: {_summarize(plan)}")
        loaded.append(ep.name)
    return tuple(loaded)


@lru_cache(maxsize=1)
def _load_discovered_plugins() -> tuple[str, ...]:
    return _load_plugins(entry_points(group=PLUGIN_GROUP))


def load_ops_backend_plugins(eps: Iterable[EntryPoint] | None = None) -> tuple[str, ...]:
    """Discover and register external ops-backend plugins (opt-in additions).

    Called once at ``veomni.ops`` import time, after all built-in ops are
    registered; the discovery result is cached for the process. Returns the
    names of successfully loaded plugins. ``eps`` lets tests inject explicit
    entry points instead of scanning the environment.
    """
    if eps is None:
        return _load_discovered_plugins()
    return _load_plugins(eps)
