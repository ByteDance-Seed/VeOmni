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

"""GraphProfiler — request-local execution path and optional node timing.

This is deliberately graph-owned, not a module mixin: the graph knows state,
node, transition, and endpoint identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

import torch

from ....utils.device import get_torch_device


class GraphProfiler:
    """Collect graph execution path lines and optional per-node timing."""

    def __init__(
        self,
        *,
        enable_wall_time: bool = False,
        enable_cuda_events: bool = False,
        enable_memory: bool = False,
    ) -> None:
        self.enable_wall_time = enable_wall_time
        self.enable_cuda_events = enable_cuda_events
        self.enable_memory = enable_memory
        self._lines: list[str] = []
        if self.enable_memory:
            self._reset_peak_memory_stats()

    @contextmanager
    def node(self, line: str) -> Iterator[None]:
        start_wall = perf_counter() if self.enable_wall_time else None
        start_event = end_event = None
        if self.enable_cuda_events and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        try:
            yield
        finally:
            suffixes: list[str] = []
            if start_wall is not None:
                suffixes.append(f"wall_ms={(perf_counter() - start_wall) * 1000:.3f}")
            if start_event is not None and end_event is not None:
                end_event.record()
                end_event.synchronize()
                suffixes.append(f"cuda_ms={start_event.elapsed_time(end_event):.3f}")
            if self.enable_memory:
                suffixes.extend(self._memory_suffixes())
            self._lines.append(line + (f" | {' | '.join(suffixes)}" if suffixes else ""))

    def record(self, line: str) -> None:
        self._lines.append(line)

    def save_records(self) -> list[str]:
        return list(self._lines)

    def _reset_peak_memory_stats(self) -> None:
        device = get_torch_device()
        reset = getattr(device, "reset_peak_memory_stats", None)
        if reset is not None:
            reset()

    def _memory_suffixes(self) -> list[str]:
        device = get_torch_device()
        max_allocated = getattr(device, "max_memory_allocated", None)
        max_reserved = getattr(device, "max_memory_reserved", None)
        if max_allocated is None or max_reserved is None:
            return []
        return [
            f"peak_allocated_gb={max_allocated() / (1024**3):.3f}",
            f"peak_reserved_gb={max_reserved() / (1024**3):.3f}",
        ]


__all__ = ["GraphProfiler"]
