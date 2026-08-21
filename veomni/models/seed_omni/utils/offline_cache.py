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

from __future__ import annotations

import os
import pickle
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset

from ....distributed.parallel_state import get_parallel_state
from ....utils import logging
from .conversation import ConversationItem


logger = logging.get_logger(__name__)


class SeedOmniOfflineCacheWriter:
    """Write cached SeedOmni conversation-carrier samples as parquet shards."""

    def __init__(self, save_path: str, *, max_rows_per_shard: int = 1000) -> None:
        parallel_state = get_parallel_state()
        self.rank = parallel_state.dp_rank if parallel_state.dp_rank >= 0 else int(os.getenv("RANK", 0))
        self.world_size = parallel_state.dp_size if parallel_state.dp_size > 0 else int(os.getenv("WORLD_SIZE", 1))
        self.save_path = save_path
        self.max_rows_per_shard = max_rows_per_shard
        self.shard_index = 0
        self.buffer: list[dict[str, bytes]] = []
        self.rows_written = 0
        os.makedirs(save_path, exist_ok=True)
        logger.info_rank0(f"SeedOmni offline cache writer saving parquet shards under {save_path}.")

    @staticmethod
    def _cpu_recursive(value: Any) -> Any:
        if isinstance(value, ConversationItem):
            return ConversationItem(
                type=value.type,
                value=SeedOmniOfflineCacheWriter._cpu_recursive(value.value),
                role=value.role,
                source=value.source,
                meta=SeedOmniOfflineCacheWriter._cpu_recursive(value.meta),
            )
        if isinstance(value, dict):
            return {k: SeedOmniOfflineCacheWriter._cpu_recursive(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SeedOmniOfflineCacheWriter._cpu_recursive(v) for v in value]
        if isinstance(value, tuple):
            return tuple(SeedOmniOfflineCacheWriter._cpu_recursive(v) for v in value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def save_conversation_list(self, conversation_list: list[list[ConversationItem]]) -> None:
        for sample in conversation_list:
            persisted = [self._cpu_recursive(item) for item in sample]
            if not persisted:
                continue

            self.buffer.append({"conversation_list": pickle.dumps(persisted)})
            self.rows_written += 1
            if len(self.buffer) >= self.max_rows_per_shard:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        dataset = Dataset.from_list(self.buffer)
        global_shard_index = self.shard_index * self.world_size + self.rank
        path = os.path.join(self.save_path, f"shard_{global_shard_index:06d}.parquet")
        dataset.to_parquet(path)
        logger.info(f"Rank {self.rank} wrote {len(self.buffer)} cached SeedOmni sample(s) to {path}.")
        self.buffer = []
        self.shard_index += 1

    def finalize(self) -> None:
        self.flush()
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()

        if not distributed or self.rank == 0:
            self._compact_shard_names()

        if distributed:
            dist.barrier()

    def _compact_shard_names(self) -> None:
        shard_names = sorted(
            name for name in os.listdir(self.save_path) if name.startswith("shard_") and name.endswith(".parquet")
        )
        for next_index, name in enumerate(shard_names):
            target_name = f"shard_{next_index:06d}.parquet"
            if name == target_name:
                continue
            os.replace(os.path.join(self.save_path, name), os.path.join(self.save_path, target_name))


__all__ = ["SeedOmniOfflineCacheWriter"]
