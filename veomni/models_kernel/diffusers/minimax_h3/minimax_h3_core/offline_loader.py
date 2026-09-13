"""MiniMax H3 offline data loader.

Scans directory for VeOmni offline_embedding parquet shards and yields
per-row dicts (pickle-bytes columns). The minimax_h3_offline data_transform
restores each row to the per-sample dict that process_condition consumes.
"""

from __future__ import annotations

import os
import random

import torch
from torch.utils.data import Dataset, IterableDataset

from veomni.data.dataset import DATASET_REGISTRY, IterativeDataset


def _search_parquet_files(base_path: str) -> list[str]:
    """Recursively collect .parquet shard paths."""
    cached = []
    for entry in os.listdir(base_path):
        full = os.path.join(base_path, entry)
        if os.path.isdir(full):
            cached.extend(_search_parquet_files(full))
        elif entry.endswith(".parquet"):
            cached.append(full)
    return sorted(cached)


class _ParquetIterableDataset(IterableDataset):
    """Iterable dataset over .parquet shards; each yielded item is one row dict.

    Checkpoint resume: state_dict records the per-worker position (repeat,
    shard index, row index just past the last yielded row); load_state_dict
    restores it so iteration continues after that row. Resume assumes the
    same num_workers / base seed, otherwise the per-worker shard slices and
    shuffle order change and the saved position is meaningless. Shuffle order
    is re-derived per repeat pass (seed = epoch seed + repeat index), so a
    resume inside a repeat lands on the same order it was recorded in. DP
    split is row-level: each rank keeps rows at (global row position in the
    pass order) % dp_size == dp_rank, so every rank yields data even when
    shard count < dp_size. Resume also assumes the same dp_rank/dp_size.
    """

    def __init__(
        self, file_paths: list[str], shuffle: bool, seed: int, repeat: int = 1, dp_rank: int = 0, dp_size: int = 1
    ):
        self._paths = file_paths
        self._shuffle = shuffle
        self._base_seed = seed  # configured seed; set_epoch derives epoch seeds from it
        self._seed = seed
        self._repeat = repeat
        # Row-level DP split: every rank iterates every shard and keeps only
        # rows at (global row position) % dp_size == dp_rank, so shard count
        # no longer bounds dp_size (total row count does).
        self._dp_rank = dp_rank
        self._dp_size = dp_size
        self._row_counts = {}  # path -> num_rows (parquet metadata, cached)
        if dp_size > 1:
            total_rows = sum(self._row_count(p) for p in file_paths)
            if total_rows < dp_size:
                raise ValueError(
                    f"Only {total_rows} rows across {len(file_paths)} shards for "
                    f"dp_size={dp_size}: every rank would get zero samples. "
                    "Re-shard the data or run offline embedding on more ranks."
                )
        self._pos = None  # {worker_key: {"rep": int, "path_idx": int, "row_idx": int}}, set while iterating
        self._resume = None  # same schema, restored via load_state_dict

    def _worker_key(self, wid):
        return wid if wid is not None else 0

    def _row_count(self, path: str) -> int:
        if path not in self._row_counts:
            import pyarrow.parquet as pq

            self._row_counts[path] = pq.ParquetFile(path).metadata.num_rows
        return self._row_counts[path]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = 0 if worker_info is None else worker_info.id
        nw = 1 if worker_info is None else worker_info.num_workers

        resume = None if self._resume is None else self._resume.get(self._worker_key(wid))
        self._pos = {self._worker_key(wid): {"rep": 0, "path_idx": 0, "row_idx": 0}}

        import pandas as pd

        for rep in range(resume["rep"] if resume is not None else 0, self._repeat):
            # Each repeat pass shuffles its own copy with a per-repeat seed:
            # deterministic across restarts (same base seed + rep), and the
            # saved {rep, path_idx} resume position still lands on the same
            # shard order for the repeat it was recorded in. All ranks and
            # workers derive the same pass order from the same seed.
            rep_paths = list(self._paths)
            if self._shuffle:
                random.Random(self._seed + rep).shuffle(rep_paths)
            # Global row offset of each file in the pass order; the DP filter
            # below is positional: (offset + row index) % dp_size == dp_rank.
            offset_by_path = {}
            off = 0
            for p in rep_paths:
                offset_by_path[p] = off
                off += self._row_count(p)
            # Worker round-robin over the pass order, so every worker receives
            # shards even when num_workers > len(self._paths).
            worker_paths = rep_paths[wid::nw]
            for pi, p in enumerate(worker_paths):
                if resume is not None and rep == resume["rep"] and pi < resume["path_idx"]:
                    continue
                df = pd.read_parquet(p)
                base = offset_by_path[p]
                for ri, row in enumerate(df.to_dict(orient="records")):
                    if (
                        resume is not None
                        and rep == resume["rep"]
                        and pi == resume["path_idx"]
                        and ri < resume["row_idx"]
                    ):
                        continue
                    if (base + ri) % self._dp_size != self._dp_rank:
                        continue
                    # Record the next position BEFORE yielding: the generator
                    # suspends at the yield, so a post-yield write would only
                    # run on resume and a checkpoint taken mid-iteration would
                    # record the row it just yielded (re-yielding it on resume).
                    self._pos[self._worker_key(wid)] = {"rep": rep, "path_idx": pi, "row_idx": ri + 1}
                    yield row

    def set_epoch(self, epoch: int):
        # Keep the configured base seed and derive the epoch seed additively,
        # instead of replacing it: two configs with different base seeds stay
        # distinct at the same epoch.
        self._seed = self._base_seed + epoch

    def state_dict(self):
        # Empty before the first sample is consumed; checkpointing then
        # resumes from the beginning.
        return {"worker_positions": self._pos if self._pos is not None else {}}

    def load_state_dict(self, state_dict):
        self._resume = dict(state_dict.get("worker_positions", {}))


@DATASET_REGISTRY.register("minimax_h3_online")
def build_minimax_h3_online_dataset(
    train_path: str,
    transform=None,
    source_name: str = None,
    **kwargs,
) -> Dataset:
    """Build mapping csv dataset for online raw-data embedding (stage1).

    Reuses the generic mapping builder (load_dataset csv). The minimax_h3_online
    data_transform loads video/audio raw data for condition_model.get_condition().
    """
    from veomni.data.dataset import build_mapping_dataset

    return build_mapping_dataset(train_path, transform=transform, source_name=source_name)


@DATASET_REGISTRY.register("minimax_h3_offline")
def build_minimax_h3_offline_dataset(
    train_path: str,
    seed: int = 42,
    shuffle: bool = True,
    transform=None,
    **kwargs,
) -> IterableDataset:
    """Build IterableDataset from VeOmni offline_embedding parquet shards.

    Args:
        train_path: Root directory containing .parquet shards (flat or nested).
        seed: Shuffle seed.
        shuffle: Shuffle file order.
        transform: Optional data_transform callable applied to each sample.

    Returns:
        IterableDataset yielding transformed dicts.
    """
    parquet_files = _search_parquet_files(train_path)
    if not parquet_files:
        raise ValueError(f"No .parquet files found under {train_path}")

    from veomni.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info_rank0(f"Minimax H3 offline: found {len(parquet_files)} .parquet files")

    from veomni.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    dp_rank = parallel_state.dp_rank
    dp_size = parallel_state.dp_size

    # Repeat the whole pass repeat times so each DP rank has enough iterations.
    # Controlled by data.mm_configs.repeat in YAML.
    mm_configs = kwargs.get("mm_configs", {}) or {}
    repeat = int(mm_configs.get("repeat", 1))

    # Row-level DP split: every rank iterates every shard and keeps only the
    # rows at (global row position) % dp_size == dp_rank, so shard count no
    # longer bounds dp_size (total row count does). Checkpoint resume records
    # per-worker positions, unaffected by the filter.
    raw_dataset = _ParquetIterableDataset(
        file_paths=parquet_files, shuffle=shuffle, seed=seed, repeat=repeat, dp_rank=dp_rank, dp_size=dp_size
    )
    return IterativeDataset(raw_dataset, transform=transform)
