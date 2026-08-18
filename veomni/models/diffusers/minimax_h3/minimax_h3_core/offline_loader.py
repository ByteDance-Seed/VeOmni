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
    """Iterable dataset over .parquet shards; each yielded item is one row dict."""

    def __init__(self, file_paths: list[str], shuffle: bool, seed: int, repeat: int = 1):
        self._paths = file_paths
        self._shuffle = shuffle
        self._seed = seed
        self._repeat = repeat

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            paths = list(self._paths)
        else:
            per_worker = len(self._paths) // worker_info.num_workers
            wid = worker_info.id
            start = wid * per_worker
            end = start + per_worker if wid < worker_info.num_workers - 1 else len(self._paths)
            paths = list(self._paths[start:end])

        if self._shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(paths)

        import pandas as pd

        for _ in range(self._repeat):
            for p in paths:
                df = pd.read_parquet(p)
                yield from df.to_dict(orient="records")

    def set_epoch(self, epoch: int):
        self._seed = epoch

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


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

    # Repeat each file repeat times so each DP rank has enough iterations.
    # Controlled by data.mm_configs.repeat in YAML.
    mm_configs = kwargs.get("mm_configs", {}) or {}
    repeat = int(mm_configs.get("repeat", 1))

    # Round-robin distribution: rank i gets files[i], files[i+dp_size], ...
    rank_files = parquet_files[dp_rank::dp_size]
    if not rank_files:
        logger.warning(
            f"Rank {dp_rank} has no files after sharding "
            f"({len(parquet_files)} files / {dp_size} ranks); using all files."
        )
        rank_files = list(parquet_files)

    raw_dataset = _ParquetIterableDataset(file_paths=rank_files, shuffle=shuffle, seed=seed, repeat=repeat)
    return IterativeDataset(raw_dataset, transform=transform)
