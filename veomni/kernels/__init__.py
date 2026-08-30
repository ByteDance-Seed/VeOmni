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

"""VeOmni kernel registry and the families registered on import.

``compound`` holds nested-handle helpers. Importing this package registers
``rms_norm``, ``rope``, ``rope_vision``, ``async_ulysses_*``, and ``swiglu_mlp``.
"""

from . import async_ulysses as _async_ulysses  # noqa: F401
from . import rms_norm as _rms_norm  # noqa: F401
from . import rope as _rope  # noqa: F401
from . import rope_vision as _rope_vision  # noqa: F401
from . import swiglu_mlp as _swiglu_mlp  # noqa: F401
from .registry import KERNEL_REGISTRY, VeomniKernel, register_kernel, resolve_kernel


__all__ = [
    "KERNEL_REGISTRY",
    "VeomniKernel",
    "register_kernel",
    "resolve_kernel",
]
