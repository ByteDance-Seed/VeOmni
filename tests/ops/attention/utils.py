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

"""Shared helpers for attention operation tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class UlyssesHelperRecorder:
    """Record calls to the standard attention Ulysses helper contract."""

    query_head_count: int = 4
    local_query_head_count: int = 2
    local_key_value_head_count: int = 1
    calls: list[tuple[object, ...]] = field(default_factory=list, init=False)

    def prepare(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        group: object,
        ulysses_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Record and emulate ``prepare_ulysses_qkv``."""
        self.calls.append(("prepare", query, key, value, group, ulysses_size))
        return (
            query[:, :, : self.local_query_head_count],
            key[:, :, : self.local_key_value_head_count],
            value[:, :, : self.local_key_value_head_count],
            self.query_head_count,
        )

    def slice_auxiliary(
        self,
        auxiliary: Tensor,
        *,
        query_head_count: int,
        local_query_head_count: int,
        group: object,
    ) -> Tensor:
        """Record and emulate ``slice_ulysses_head_auxiliary``."""
        self.calls.append(("slice", auxiliary, query_head_count, local_query_head_count, group))
        return auxiliary[:local_query_head_count]

    def restore(self, output: Tensor, *, group: object) -> Tensor:
        """Record an identity ``restore_ulysses_output`` call."""
        self.calls.append(("restore", output, group))
        return output
