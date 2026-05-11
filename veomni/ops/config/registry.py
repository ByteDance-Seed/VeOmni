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

"""Kernel registry and dispatch engine.

Each op (cross-entropy loss, RMSNorm, RoPE, ...) registers an ``OpSpec`` with:
- ``config_field``: the matching field name on ``OpsImplementationConfig``
- ``scope``: ``GLOBAL`` (function-pointer slot) or ``PER_MODEL`` (setattr via
  ``device_patch.py``)
- a mapping of backend name (``"eager"`` / ``"liger_kernel"`` / ``"npu"`` /
  ``"triton"``) to ``BackendSpec``.

GLOBAL ops are resolved by ``apply_global_ops()``: the selected backend's
``entry`` is lazily imported and assigned to ``global_slot``; an optional
``side_effect`` is then invoked (e.g. installing ``LOSS_MAPPING["ForCausalLM"]
= chunk_loss_function`` for the NPU chunked-loss backend).

PER-MODEL ops are resolved inside each model's ``device_patch.py`` via
``apply_per_model_patches(hf_module, model_name, targets={op: attr})``. The
engine looks up each op in the registry, resolves the selected backend, and
replaces ``hf_module.<attr>`` (or ``.forward``) accordingly.  Model-specific
overrides can be supplied via ``extra_backends``; a ``custom_patches``
callback remains as an escape hatch for truly one-off behaviour.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import TYPE_CHECKING, Callable

from ...utils import logging
from .singleton import get_ops_config


if TYPE_CHECKING:
    from ...arguments.arguments_types import OpsImplementationConfig

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class OpScope(str, Enum):
    GLOBAL = "global"
    PER_MODEL = "per_model"


@dataclass(frozen=True)
class BackendSpec:
    """One concrete backend implementation for an op.

    Attributes:
        entry: ``"module:attr"`` - lazily imported to resolve the replacement.
        requires: Package names that must be available, checked before
            resolution. Supported values: ``"liger_kernel"``, ``"torch_npu"``.
        side_effect: GLOBAL ops only. ``"module:callable"`` invoked after
            ``entry`` is bound to ``global_slot`` (e.g. installing additional
            ``LOSS_MAPPING`` entries).
        replace_forward: PER_MODEL ops only. If ``True``, replace
            ``hf_module.<target>.forward`` rather than ``hf_module.<target>``.
        entry_is_factory: PER_MODEL ops only. If ``True``, ``entry`` is a
            zero-arg callable that returns the actual replacement (used e.g.
            by DeepSeek V3's deterministic RoPE factory).
        target_override: PER_MODEL ops only. If set, patch this attribute on
            ``hf_module`` instead of the per-model default (used when a
            backend targets a different symbol than the default backends,
            e.g. DeepSeek V3's Triton RoPE targets ``DeepseekV3RotaryEmbedding``
            while the default ``npu``/``liger_kernel`` backends target
            ``apply_rotary_pos_emb``).
    """

    entry: str
    requires: tuple[str, ...] = ()
    side_effect: str | None = None
    replace_forward: bool = False
    entry_is_factory: bool = False
    target_override: str | None = None


@dataclass(frozen=True)
class OpSpec:
    """Declarative spec for one op.

    Attributes:
        name: Machine identifier, e.g. ``"rms_norm"``.
        config_field: Matching field name on ``OpsImplementationConfig``.
        label: Human-readable label used in log lines, e.g. ``"RMSNorm"``.
        scope: ``GLOBAL`` or ``PER_MODEL``.
        default: Default backend name; must equal the dataclass default.
        backends: Mapping from backend name to ``BackendSpec``.
        global_slot: GLOBAL ops only. ``"module:attr"`` holding the function
            pointer; the engine performs ``setattr(module, attr, entry)``.
    """

    name: str
    config_field: str
    label: str
    scope: OpScope
    default: str
    backends: dict[str, BackendSpec] = field(default_factory=dict)
    global_slot: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_OPS_REGISTRY: dict[str, OpSpec] = {}


def register_op(spec: OpSpec) -> None:
    """Register an ``OpSpec``. Re-registration with an identical spec is a
    no-op (safe under repeated ``importlib.reload`` during tests); a
    re-registration with a different spec raises ``ValueError``."""
    existing = _OPS_REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(f"Op {spec.name!r} is already registered with a different spec.")
    _OPS_REGISTRY[spec.name] = spec


def get_op(name: str) -> OpSpec:
    """Return the ``OpSpec`` for ``name``, raising ``KeyError`` if unknown."""
    return _OPS_REGISTRY[name]


def list_ops(scope: OpScope | None = None) -> list[OpSpec]:
    """Return all registered ops, optionally filtered by ``scope``."""
    ops = list(_OPS_REGISTRY.values())
    if scope is not None:
        ops = [op for op in ops if op.scope == scope]
    return ops


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _import_entry(entry: str) -> object:
    """Resolve ``"module:attr"`` to the underlying object via lazy import."""
    module_path, _, attr = entry.partition(":")
    if not attr:
        raise ValueError(f"Backend entry {entry!r} must be of the form 'module:attr'.")
    return getattr(importlib.import_module(module_path), attr)


def _check_requires(requires: tuple[str, ...]) -> None:
    """Validate that the listed packages are importable."""
    from ...utils.import_utils import is_liger_kernel_available, is_package_available, is_torch_npu_available

    for pkg in requires:
        if pkg == "liger_kernel":
            if not is_liger_kernel_available():
                raise RuntimeError("liger_kernel backend requested but liger-kernel is not installed.")
        elif pkg == "torch_npu":
            if not is_torch_npu_available():
                raise RuntimeError("npu backend requested but torch_npu is not installed.")
        elif pkg == "triton":
            if not is_package_available("triton"):
                raise RuntimeError(
                    "triton backend requested but the 'triton' package is not installed "
                    "(or 'triton-ascend' on NPU). Install it or set the field to 'eager'."
                )
        else:
            raise ValueError(f"Unsupported 'requires' token: {pkg!r}")


# ---------------------------------------------------------------------------
# GLOBAL dispatch
# ---------------------------------------------------------------------------


def apply_global_ops(ops_config: OpsImplementationConfig) -> list[str]:
    """Resolve every GLOBAL op and bind its selected backend.

    Returns the list of human-readable ``"<label> (<backend>)"`` entries that
    were bound (useful for logging).
    """
    applied: list[str] = []
    for op in list_ops(OpScope.GLOBAL):
        if op.global_slot is None:
            raise ValueError(f"GLOBAL op {op.name!r} must define global_slot.")
        value = getattr(ops_config, op.config_field)
        backend = op.backends.get(value)
        if backend is None:
            # Unknown value may be a user-registered custom backend.  Skip.
            continue
        _check_requires(backend.requires)

        entry_obj = _import_entry(backend.entry)

        slot_module, _, slot_attr = op.global_slot.partition(":")
        setattr(importlib.import_module(slot_module), slot_attr, entry_obj)

        if backend.side_effect is not None:
            side_effect_fn = _import_entry(backend.side_effect)
            side_effect_fn()

        applied.append(f"{op.label} ({value})")
    return applied


# ---------------------------------------------------------------------------
# PER-MODEL dispatch
# ---------------------------------------------------------------------------


def _resolve_backend(
    op: OpSpec,
    value: str,
    op_overrides: dict[str, BackendSpec | None],
) -> BackendSpec | None:
    """Pick the BackendSpec for *value*, preferring the per-model override.

    Returns ``None`` when there's nothing to bind: ``value == "eager"`` (keep
    HF default), the model explicitly opted out (``op_overrides[value] = None``),
    or *value* is unknown. The caller distinguishes these cases.
    """
    if value in op_overrides:
        return op_overrides[value]
    return op.backends.get(value)


def _raise_no_backend(
    model_name: str,
    op: OpSpec,
    value: str,
    op_overrides: dict[str, BackendSpec | None],
) -> None:
    """Raise on a non-eager value with no resolvable backend.

    Distinguishes "explicitly disabled" (``op_overrides[value] is None``) from
    "unknown backend" so the user knows whether to switch to eager or fix a typo.
    """
    explicitly_disabled = value in op_overrides and op_overrides[value] is None
    disabled_names = {k for k, v in op_overrides.items() if v is None}
    available = sorted(((op.backends.keys() | op_overrides.keys()) - disabled_names) | {"eager"})
    reason = (
        f"is explicitly disabled for {model_name} (no kernel matching this backend's signature)"
        if explicitly_disabled
        else f"is not a registered backend for {model_name}"
    )
    raise ValueError(
        f"{op.config_field}={value!r} {reason}. "
        f"Set {op.config_field} to one of {available}. "
        f"Use 'eager' for the HuggingFace reference."
    )


def _patch_target(hf_module: ModuleType, target_attr: str, backend: BackendSpec) -> None:
    """Bind *backend* onto ``hf_module.<target_attr>``."""
    entry_obj = _import_entry(backend.entry)
    if backend.entry_is_factory:
        entry_obj = entry_obj()
    effective_target = backend.target_override or target_attr
    if backend.replace_forward:
        getattr(hf_module, effective_target).forward = entry_obj
    else:
        setattr(hf_module, effective_target, entry_obj)


def apply_per_model_patches(
    hf_module: ModuleType,
    model_name: str,
    targets: dict[str, str],
    *,
    extra_backends: dict[str, dict[str, BackendSpec | None]] | None = None,
    custom_patches: Callable[[OpsImplementationConfig, list[str]], None] | None = None,
) -> None:
    """Patch ``hf_module`` based on the current ``OpsImplementationConfig``.

    Args:
        hf_module: HuggingFace modeling module whose attributes get replaced.
        model_name: Display name for log lines and error messages.
        targets: ``{op_name: hf_module_attr}`` — e.g. ``{"rms_norm": "LlamaRMSNorm"}``.
        extra_backends: Per-model overrides ``{op_name: {backend_name: spec}}``.
            ``BackendSpec`` adds/replaces a backend; ``None`` is an **explicit
            opt-out** for cases where the registry default doesn't fit the
            model's target signature (e.g. Wan ``rope_apply(x, **kw)`` vs.
            Liger ``(q, k, cos, sin)``). Users then get a clean "explicitly
            disabled" error instead of a runtime crash.
        custom_patches: ``(ops_config, applied_list) -> None`` escape hatch for
            one-off behavior that doesn't fit ``BackendSpec``.
    """
    ops_config = get_ops_config()
    if ops_config is None:
        return

    extra_backends = extra_backends or {}
    applied: list[str] = []

    for op_name, target_attr in targets.items():
        try:
            op = get_op(op_name)
        except KeyError as e:
            raise KeyError(f"Unknown op {op_name!r} referenced by {model_name} device_patch.py.") from e
        if op.scope != OpScope.PER_MODEL:
            raise ValueError(f"{model_name}: op {op_name!r} is {op.scope.value}, not per_model.")

        value = getattr(ops_config, op.config_field)
        op_overrides = extra_backends.get(op_name, {})
        backend = _resolve_backend(op, value, op_overrides)

        if backend is None:
            # ``eager`` is the only intentional no-backend case (HF default
            # stays). Anything else is a misconfiguration — raise instead of
            # silently downgrading, otherwise users think they got the fast
            # kernel they asked for.
            if value == "eager":
                continue
            _raise_no_backend(model_name, op, value, op_overrides)

        _check_requires(backend.requires)
        _patch_target(hf_module, target_attr, backend)
        applied.append(f"{op.label} ({value})")

    if custom_patches is not None:
        custom_patches(ops_config, applied)

    if applied:
        logger.info_rank0(f"Apply ops patches to {model_name}: {', '.join(applied)}.")
