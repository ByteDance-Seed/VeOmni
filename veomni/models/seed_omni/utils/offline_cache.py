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


_COMPLETE_MARKER = ".offline_cache_complete"


def _first_rank_of_its_dp_rank(dp_ranks: list[int], rank: int) -> bool:
    """Whether global ``rank`` is the lowest of the ranks sharing its ``dp_rank``."""
    return dp_ranks.index(dp_ranks[rank]) == rank


def _any_rank(flag: bool) -> bool:
    if dist.is_available() and dist.is_initialized():
        flags = [None] * dist.get_world_size()
        dist.all_gather_object(flags, flag)
        return any(flags)
    return flag


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class SeedOmniOfflineCacheWriter:
    """Write cached SeedOmni conversation-carrier samples as parquet shards.

    Resumable alongside the dataloader cursor: :meth:`state_dict` flushes the
    buffer before reporting the shard position, so the rows every saved cursor
    covers are on disk, and :meth:`load_state_dict` + :meth:`open` reopen the
    directory at that position. Call :meth:`open` once, after any
    :meth:`load_state_dict` and before the first sample.
    """

    def __init__(self, save_path: str, *, max_rows_per_shard: int = 1000) -> None:
        parallel_state = get_parallel_state()
        self.rank = parallel_state.dp_rank if parallel_state.dp_rank >= 0 else int(os.getenv("RANK", 0))
        self.world_size = parallel_state.dp_size if parallel_state.dp_size > 0 else int(os.getenv("WORLD_SIZE", 1))
        # SP / CP / TP ranks of one data-parallel group hold the same batch and
        # would write the same shard paths, so only the first of them writes.
        self.writes = True
        if dist.is_available() and dist.is_initialized():
            dp_ranks = [None] * dist.get_world_size()
            dist.all_gather_object(dp_ranks, self.rank)
            self.writes = _first_rank_of_its_dp_rank(dp_ranks, dist.get_rank())
        self.save_path = save_path
        self.max_rows_per_shard = max_rows_per_shard
        self.shard_index = 0
        self.buffer: list[dict[str, bytes]] = []
        self.rows_written = 0
        self._resume_state: dict[str, int] | None = None
        self._opened = False
        os.makedirs(save_path, exist_ok=True)
        logger.info_rank0(f"SeedOmni offline cache writer saving parquet shards under {save_path}.")

    def _shard_names(self) -> list[str]:
        return sorted(
            name for name in os.listdir(self.save_path) if name.startswith("shard_") and name.endswith(".parquet")
        )

    def state_dict(self) -> dict[str, int]:
        """Flush, then report the shard position, so the cursor saved with it covers exactly the rows on disk."""
        self.flush()
        return {"shard_index": self.shard_index, "rows_written": self.rows_written, "world_size": self.world_size}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if self._opened:
            raise RuntimeError("load_state_dict must come before SeedOmniOfflineCacheWriter.open().")
        self._resume_state = dict(state)

    def open(self) -> None:
        """Check the directory against this run, collectively, before any rank flushes."""
        if self._resume_state is None:
            self._reject_existing_shards()
        else:
            self._resume_from(self._resume_state)
        self._opened = True

    def _reject_existing_shards(self) -> None:
        """Refuse to start a fresh cache in a directory that already holds shards.

        Shard names are a pure function of rank and write order, so a second
        run over the same directory overwrites the low shards and leaves the
        high ones behind. :meth:`_compact_shard_names` then renumbers whatever
        it finds into one contiguous range, quietly presenting the leftovers of
        the earlier run as part of this cache.

        Every rank checks the directory, then the results are all-gathered so a
        non-empty directory fails on every rank together. A barrier after a
        clean check keeps a fast rank from flushing its first shard while a
        slow rank is still checking.
        """
        if _any_rank(bool(self._shard_names())):
            raise FileExistsError(
                f"SeedOmni offline cache directory '{self.save_path}' already contains "
                f"shard parquet file(s). Writing here would overwrite some and silently "
                f"keep the rest. Point `save_path` at a new directory, delete the existing "
                f"shards first, or resume the run that wrote them from its checkpoint."
            )
        _barrier()

    def _resume_from(self, state: dict[str, int]) -> None:
        """Reopen at the saved shard position and drop this rank's shards written after it.

        Those shards hold steps the restored cursor replays, so keeping them
        would cache those samples twice. Shard names are only meaningful before
        :meth:`finalize` renumbers them and only for the dp size that wrote
        them, so either mismatch fails on every rank.
        """
        if _any_rank(os.path.exists(os.path.join(self.save_path, _COMPLETE_MARKER))):
            raise FileExistsError(
                f"SeedOmni offline cache directory '{self.save_path}' was already finalized; "
                f"its shards were renumbered and cannot be resumed into."
            )
        if _any_rank(state["world_size"] != self.world_size):
            raise ValueError(
                f"Cannot resume the SeedOmni offline cache under '{self.save_path}': it was written "
                f"with data-parallel size {state['world_size']}, this run has {self.world_size}."
            )
        self.shard_index = state["shard_index"]
        self.rows_written = state["rows_written"]
        if self.writes:
            for name in self._shard_names():
                global_index = int(name[len("shard_") : -len(".parquet")])
                if global_index % self.world_size == self.rank and global_index // self.world_size >= self.shard_index:
                    os.remove(os.path.join(self.save_path, name))
                    logger.info(f"Rank {self.rank} dropped {name}, written after the restored checkpoint.")
        _barrier()

    @staticmethod
    def _cpu_recursive(value: Any) -> Any:
        if isinstance(value, ConversationItem):
            return ConversationItem(
                type=value.type,
                value=SeedOmniOfflineCacheWriter._cpu_recursive(value.value),
                role=value.role,
                is_dummy=value.is_dummy,
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
        if not self.writes:
            return
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
        if not self._opened:
            raise RuntimeError("SeedOmniOfflineCacheWriter.open() must run before the first shard is written.")

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

        if not distributed or dist.get_rank() == 0:
            self._compact_shard_names()
            with open(os.path.join(self.save_path, _COMPLETE_MARKER), "w"):
                pass

        if distributed:
            dist.barrier()

    def _compact_shard_names(self) -> None:
        for next_index, name in enumerate(self._shard_names()):
            target_name = f"shard_{next_index:06d}.parquet"
            if name == target_name:
                continue
            os.replace(os.path.join(self.save_path, name), os.path.join(self.save_path, target_name))


__all__ = ["SeedOmniOfflineCacheWriter"]
