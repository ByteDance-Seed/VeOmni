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

"""Cache for Triton JIT factory functions.

Not a kernel registry. A factory is a no-arg function that imports Triton,
defines one ``@triton.jit`` kernel, and returns it. ``functools.cache``
memoizes that factory so the kernel object is built once. Compilation still
happens on the first launch with a concrete shape, not when the factory is
first called. Launch stays ``factory()[grid](...)``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import TypeVar


F = TypeVar("F", bound=Callable[[], Callable])


def cached_triton_kernel(factory: F) -> F:
    """Run ``factory`` once; later calls reuse the same JIT kernel object.

    The factory return is cached. Triton compiles on first launch.
    """
    return cache(factory)  # type: ignore[return-value]
