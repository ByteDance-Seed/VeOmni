# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Private replicated-dummy sequence-parallel scope.

Only ``Qwen3_5VisionModel.dummy_forward`` may activate this sentinel. Flash
treats the active scope as a replicated/local path. Flex fail-closes. Ordinary
callers cannot turn the bypass on through public arguments or kwargs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_PUBLIC_BYPASS_ERROR = (
    "skip_sequence_parallel is not a public argument; the replicated dummy path is a private scoped context."
)
_FORGED_SCOPE_ERROR = "forged activation of the replicated dummy sequence-parallel scope"
_FLEX_SCOPE_ERROR = (
    "FlexAttention does not support the replicated dummy sequence-parallel bypass; "
    "the private dummy scope is Flash-only because a global BlockMask is layout-unsafe."
)

_ACTIVE: ContextVar[bool] = ContextVar("veomni_replicated_dummy_sp", default=False)


class _ReplicatedDummySPToken:
    __slots__ = ()


_DUMMY_SP_TOKEN = _ReplicatedDummySPToken()


def is_replicated_dummy_sequence_parallel() -> bool:
    return bool(_ACTIVE.get())


def reject_public_sequence_parallel_bypass(kwargs: dict) -> None:
    if "skip_sequence_parallel" in kwargs:
        raise TypeError(_PUBLIC_BYPASS_ERROR)


@contextmanager
def _replicated_dummy_sequence_parallel(token: object) -> Iterator[None]:
    """Private scoped sentinel. Auto-restores on exit, including exceptions."""
    if token is not _DUMMY_SP_TOKEN:
        raise RuntimeError(_FORGED_SCOPE_ERROR)
    reset_token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(reset_token)
