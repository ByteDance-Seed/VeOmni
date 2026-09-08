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

"""VeOmni operation registry and built-in implementations.

``compound`` holds nested-handle helpers. Importing this package registers
ops whose concrete implementations live under ``kernels``: ``rms_norm``, ``rope``, ``rope_vision``,
``async_ulysses_*``, ``dsa_attention`` / ``dsa_indexer``, ``swiglu_mlp``,
``moe_experts``, ``loss`` (LB + CE), ``gated_delta_rule``, and
``attention``. Process-wide integrations live in ``install`` and the opt-in
ATen overrides live in ``batch_invariant``.
"""

from . import kernels as _op_families  # noqa: F401
from .install import apply_ops_patch
from .registry import OP_REGISTRY, VeomniOp, register_op, resolve_op


apply_ops_patch()


__all__ = [
    "OP_REGISTRY",
    "VeomniOp",
    "apply_ops_patch",
    "register_op",
    "resolve_op",
]
