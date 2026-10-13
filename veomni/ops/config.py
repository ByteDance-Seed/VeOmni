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

"""Installed operation-implementation config.

One process-global object. Modeling reads ``get_ops_config`` during construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils import logging


if TYPE_CHECKING:
    from ..arguments.arguments_types import OpsImplementationConfig


logger = logging.get_logger(__name__)

_ops_config: OpsImplementationConfig | None = None


def set_ops_config(config: Any) -> None:
    """Install the process-global operation-implementation config."""
    global _ops_config
    _ops_config = config


def get_ops_config() -> Any:
    """Return the installed operation-implementation config, or ``None``."""
    return _ops_config


def resolve_op_impl(field: str, *, variant: str | None = None, eager_as: str | None = None) -> str:
    """Return the impl name on the installed ops config, or ``eager``.

    ``variant`` applies the variant-scoped deprecated aliases for fields shared
    by several variants, such as ``rms_norm_implementation='npu'`` on the
    ``unweighted`` variant. ``eager_as`` remaps ``eager`` for modeling without a
    local ``eager_attention_forward``, which the ``attention/standard/eager``
    row requires. Missing config is eager so unit tests can construct a module
    without ``set_ops_config``.
    """
    cfg = get_ops_config()
    impl = "eager" if cfg is None else getattr(cfg, field, "eager")
    if variant is not None:
        from ..arguments.arguments_types import _DEPRECATED_VARIANT_IMPLEMENTATION_ALIASES

        alias = _DEPRECATED_VARIANT_IMPLEMENTATION_ALIASES.get(field, {}).get(variant, {}).get(impl)
        if alias is not None:
            logger.info_once(f"{field}={impl!r} has no third-party {variant!r} kernel; {variant!r} uses {alias!r}.")
            impl = alias
    if eager_as is not None and impl == "eager":
        return eager_as
    return impl


def resolve_qat_impl() -> str:
    """Return the active model-level quantization recipe, or ``none``.

    This is not an op-registry impl. Missing config is ``none`` (off), not
    ``eager``.
    """
    cfg = get_ops_config()
    return "none" if cfg is None else getattr(cfg, "qat_implementation", "none")
