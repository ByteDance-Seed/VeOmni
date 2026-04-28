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

"""Auto-resolution policy for ``OpsImplementationConfig`` fields.

Each kernel's ``__init__.py`` registers an :class:`OpPolicy` next to its
``register_op`` / ``KERNEL_REGISTRY.register`` calls. ``OpsImplementationConfig.
__post_init__`` walks the registered policies once and rewrites:

- ``"auto"`` -> the per-device backend declared in :attr:`OpPolicy.auto_backends`,
  falling back to ``"eager"`` (with a warning) when the chosen backend's
  software / hardware requirements are not met.
- legacy aliases (e.g. ``moe_implementation="fused"``) -> their replacement,
  with a deprecation warning.

Keeping the policy table separate from both ``_OPS_REGISTRY`` (per-model patches)
and ``KERNEL_REGISTRY`` (OpSlot dispatch) lets ops that live in only one of them
(``cross_entropy_loss``, ``moe_experts``) participate in auto-resolution without
forcing a fake entry into the other registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpPolicy:
    """Auto-resolution + legacy-alias policy for one ``*_implementation`` field.

    Attributes:
        config_field: Matching field name on ``OpsImplementationConfig``,
            e.g. ``"rms_norm_implementation"``.
        auto_backends: Mapping from device type (``"gpu"`` / ``"npu"``) to the
            backend name to pick when the field's value is ``"auto"``. A device
            with no entry resolves to ``"eager"``.
        legacy_aliases: Mapping from a deprecated value to its replacement
            (e.g. ``{"fused": "auto"}``). Applied before the ``"auto"`` lookup
            and emits a deprecation warning.
        label: Human-readable label used in log lines (e.g. ``"MoE"``).
    """

    config_field: str
    auto_backends: dict[str, str]
    legacy_aliases: dict[str, str] = field(default_factory=dict)
    label: str = ""


_OP_POLICIES: dict[str, OpPolicy] = {}


def register_op_policy(policy: OpPolicy) -> None:
    """Register an :class:`OpPolicy`. Re-registration with an identical policy
    is a no-op (safe under repeated imports during tests); a re-registration
    with a different policy raises ``ValueError``."""
    existing = _OP_POLICIES.get(policy.config_field)
    if existing is not None and existing != policy:
        raise ValueError(f"OpPolicy for {policy.config_field!r} is already registered with a different value.")
    _OP_POLICIES[policy.config_field] = policy


def get_op_policy(config_field: str) -> OpPolicy | None:
    """Return the :class:`OpPolicy` registered for *config_field*, or ``None``."""
    return _OP_POLICIES.get(config_field)


def list_op_policies() -> list[OpPolicy]:
    """Return all registered :class:`OpPolicy` entries."""
    return list(_OP_POLICIES.values())
