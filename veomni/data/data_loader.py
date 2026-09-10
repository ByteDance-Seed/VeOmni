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


import math
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import get_device_type
from ..utils.registry import Registry
from .data_collator import (
    MainCollator,
    MakeMicroBatchCollator,
    NoopDataCollator,
    UnpackDataCollator,
)
from .dataset import (
    DynamicBatchingSizeDataset,
    _MapStyleSamplerWrapper,
    get_length_by_attention_mask_fn,
    get_length_fn_by_count_mode,
)
from .dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


DATALOADER_REGISTRY = Registry("dataloader")
logger = logging.get_logger(__name__)


def build_dataloader(dataloader_type: str, **kwargs):
    return DATALOADER_REGISTRY[dataloader_type](**kwargs)


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class ExactDistributedBatchSampler(Sampler[Sequence[int]]):
    """Partition map-style evaluation data without padding or materializing indices.

    Every rank yields the same number of batches so distributed forwards stay
    collective-safe. Across all ranks, every dataset index is visited exactly
    once. Configurations that cannot satisfy both properties without an empty
    batch are rejected instead of duplicating samples.
    """

    def __init__(self, dataset_size: int, batch_size: int, num_replicas: int, rank: int) -> None:
        if dataset_size <= 0:
            raise ValueError(f"Validation dataset must be non-empty, got {dataset_size} samples.")
        if batch_size <= 0:
            raise ValueError(f"Validation batch size must be positive, got {batch_size}.")
        if num_replicas <= 0:
            raise ValueError(f"Number of replicas must be positive, got {num_replicas}.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Rank must be in [0, {num_replicas}), got {rank}.")

        num_steps = math.ceil(dataset_size / (num_replicas * batch_size))
        num_batch_slots = num_steps * num_replicas
        if num_batch_slots > dataset_size:
            raise ValueError(
                "Validation dataset cannot be partitioned into non-empty batches with equal steps per rank: "
                f"dataset_size={dataset_size}, batch_size={batch_size}, num_replicas={num_replicas}. "
                "Increase the validation dataset or batch size."
            )

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_steps = num_steps
        self._num_batch_slots = num_batch_slots
        self._base_batch_size, self._remainder = divmod(dataset_size, num_batch_slots)

    def __iter__(self) -> Iterator[Sequence[int]]:
        for step in range(self.num_steps):
            slot = step * self.num_replicas + self.rank
            size = self._base_batch_size + int(slot < self._remainder)
            start = slot * self._base_batch_size + min(slot, self._remainder)
            yield range(start, start + size)

    def __len__(self) -> int:
        return self.num_steps

    def set_epoch(self, _epoch: int) -> None:
        """Keep the exact source order across validation epochs."""


def _build_worker_init_fn(worker_num_threads: int) -> Callable[[int], None]:
    def worker_init_fn(_worker_id: int) -> None:
        torch.set_num_threads(worker_num_threads)

    return worker_init_fn


@DATALOADER_REGISTRY.register("native")
def build_native_dataloader(
    dataset: "Dataset",
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    bsz_warmup_ratio: float = 0.02,
    bsz_warmup_init_mbtoken: int = 200,
    dyn_bsz: bool = True,
    dyn_bsz_runtime: Literal["main", "worker"] = "main",
    dyn_bsz_count_mode: Literal["total", "effective"] = "total",
    dyn_bsz_physical_overflow_ratio: float = 1.5,
    dyn_bsz_dataset_save_by_idx: bool = False,  # Whether to save dynamic-batching buffers by index for worker-side checkpoint/resume.
    dyn_bsz_buffer_size: int = 200,
    num_workers: int = 8,
    worker_num_threads: Optional[int] = None,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    in_order: bool = True,
    shuffle: bool = True,
    seed: int = 0,
    collate_fn: Optional[Callable] = None,
    build_collate_fn: bool = True,
    collate_fn_kwargs: Optional[Dict[str, Any]] = None,
    multiprocessing_context=None,
    save_steps: int = 0,
    batch_sampler: Optional[Sampler[Sequence[int]]] = None,
    generator: Optional[torch.Generator] = None,
) -> "DistributedDataloader":
    """Build the native training dataloader.

    Args:
        dyn_bsz_runtime: Which process dynamic batching runs in. ``"main"`` keeps the
            legacy main-process ``DynamicBatchSizeDataLoader`` path, while ``"worker"``
            batches inside each DataLoader worker via ``DynamicBatchingSizeDataset`` so
            worker state can participate in ``StatefulDataLoader`` checkpoint/resume.

            Data format by stage when ``dyn_bsz=True``:

            ``dyn_bsz_runtime="main"``

                dataset
                  │  yields: ``list[dict]``
                  ▼
                DataLoader(batch_size=1, collate_fn=UnpackDataCollator)
                  │  yields: ``list[dict]``
                  ▼
                DynamicBatchSizeDataLoader / TextBatchingStrategy
                  │  flatten each upstream item: ``list[dict]`` -> ``dict``
                  │  internal buffer entry: ``dict``
                  │  micro batch from strategy: ``list[dict]``
                  ▼
                trainer step input
                     ``list[list[dict]]``
                     (outer list = micro batches in one optimizer step,
                      inner list = samples in one micro batch)

            ``dyn_bsz_runtime="worker"``

                dataset
                  │  yields: ``list[dict]``
                  ▼
                DynamicBatchingSizeDataset (inside each worker)
                  │  flatten each upstream item: ``list[dict]`` -> ``dict``
                  │  internal buffer entry: ``dict``
                  │  micro batch before collate: ``list[dict]``
                  ▼
                StatefulDataLoader(batch_size=num_micro_batch, collate_fn=NoopDataCollator)
                  │ ``list[list[dict]]``
                  ▼
                trainer step input
                  │ ``list[list[dict]]``

        multiprocessing_context: Optional worker start method override.
            Use ``"spawn"`` when worker-side code must be pickle-safe and should not
            inherit parent-process state; keep ``"fork"`` for the legacy Linux behavior.
            Example: ``multiprocessing_context="spawn"``.
        batch_sampler: Optional custom map-style batch sampler. It is mutually
            exclusive with dynamic batching and replaces the native distributed
            sampler, batch size, and ``drop_last`` configuration.
        generator: Optional dedicated RNG used by the underlying dataloader.
    """
    if collate_fn_kwargs is None:
        collate_fn_kwargs = {}
    parallel_state = get_parallel_state()

    if batch_sampler is not None and dyn_bsz:
        raise ValueError("A custom batch_sampler is supported only when dyn_bsz=False.")

    if collate_fn is None:
        if build_collate_fn:
            collate_fn = MainCollator(**collate_fn_kwargs)
        else:
            collate_fn = NoopDataCollator()

    num_micro_batch = global_batch_size // (
        micro_batch_size * parallel_state.dp_size
    )  # num_micro_batch = num accumulation steps

    if dyn_bsz:
        batching_token_len = micro_batch_size * max_seq_len
        bsz_warmup_steps = int(train_steps * bsz_warmup_ratio)

        logger.info_rank0(
            f"Use dynamic_batching -->\n"
            f"micro_batch_size: {micro_batch_size}, max_seq_len: {max_seq_len}, "
            f"batching_token_len = micro_batch_size * max_seq_len = {batching_token_len}.\n"
            f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}, "
            f"global_batch_size: {global_batch_size}, micro_batch_size: {micro_batch_size}, "
            f"num_micro_batch: {num_micro_batch}.\n"
            f"train_steps: {train_steps}, bsz_warmup_steps: {bsz_warmup_steps}, "
            f"bsz_warmup_init_mbtoken: {bsz_warmup_init_mbtoken}."
        )
        dyn_bsz_collate_fn = collate_fn
        dyn_bsz_length_fn = get_length_fn_by_count_mode(dyn_bsz_count_mode)
        if dyn_bsz_count_mode == "effective":
            if dyn_bsz_physical_overflow_ratio < 1.0:
                raise ValueError(
                    f"dyn_bsz_physical_overflow_ratio must be >= 1.0, got {dyn_bsz_physical_overflow_ratio}."
                )
            physical_token_cap = math.ceil(batching_token_len * dyn_bsz_physical_overflow_ratio)
            dyn_bsz_physical_length_fn = get_length_by_attention_mask_fn
        else:
            physical_token_cap = None
            dyn_bsz_physical_length_fn = None
        if dyn_bsz_runtime == "main":
            batching_strategy = TextBatchingStrategy(
                token_micro_bsz=batching_token_len,
                buffer_size=dyn_bsz_buffer_size,
                bsz_warmup_steps=bsz_warmup_steps,
                bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
                get_length_fn=dyn_bsz_length_fn,
                physical_token_cap=physical_token_cap,
                get_physical_length_fn=dyn_bsz_physical_length_fn,
            )

            collate_fn = UnpackDataCollator()
        else:
            if not isinstance(dataset, IterableDataset):
                # Map-style datasets lose their DistributedSampler once wrapped into the
                # (iterable) DynamicBatchingSizeDataset. Adapt them to a per-rank,
                # per-worker iterable with sampler-equivalent index assignment.
                dataset = _MapStyleSamplerWrapper(
                    dataset,
                    num_replicas=parallel_state.dp_size,
                    rank=parallel_state.dp_rank,
                    shuffle=shuffle,
                    seed=seed,
                )
            dataset = DynamicBatchingSizeDataset(
                dataset=dataset,
                micro_batch_seq_length=batching_token_len,
                ready_for_micro_batch_threshold=dyn_bsz_buffer_size,
                get_length_fn=dyn_bsz_length_fn,
                physical_token_cap=physical_token_cap,
                get_physical_length_fn=dyn_bsz_physical_length_fn,
                dynamic_batching_collate_fn=dyn_bsz_collate_fn,
                save_by_idx=dyn_bsz_dataset_save_by_idx,
            )
            collate_fn = NoopDataCollator()
    else:
        logger.info_rank0(
            f"Use fixed_sample_batching -->\n"
            f"fixed_sample_num in one batch = micro_batch_size: {micro_batch_size}.\n"
            f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}, "
            f"global_batch_size: {global_batch_size}, micro_batch_size: {micro_batch_size}, "
            f"num_micro_batch: {num_micro_batch}.\n"
            f"train_steps: {train_steps}."
        )
        collate_fn = MakeMicroBatchCollator(num_micro_batch=num_micro_batch, internal_data_collator=collate_fn)

    sampler = None
    if batch_sampler is None and not isinstance(dataset, IterableDataset):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=parallel_state.dp_size,
            rank=parallel_state.dp_rank,
            shuffle=shuffle,
            seed=seed,
        )

    worker_init_fn = _build_worker_init_fn(worker_num_threads) if worker_num_threads is not None else None
    if not in_order and num_workers > 0:
        logger.warning_rank0(
            "data.dataloader.in_order=False can improve throughput for uneven worker loads, "
            "but StatefulDataLoader does not guarantee exact checkpoint/resume ordering in this mode."
        )
    # Snapshot is only consumed at save; widen to save_steps in worker mode (1:1 next/step), else keep the every-step default so resume sees a fresh snapshot.
    if save_steps and save_steps > 0 and not (dyn_bsz and dyn_bsz_runtime == "main"):
        snapshot_every_n_steps = save_steps
    else:
        snapshot_every_n_steps = 1
    dataloader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        pin_memory_device=get_device_type(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
        in_order=in_order,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )
    if batch_sampler is None:
        dataloader_kwargs.update(
            batch_size=dataloader_batch_size,
            sampler=sampler,
            drop_last=drop_last,
        )
    else:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    dataloader = DistributedDataloader(dataset, **dataloader_kwargs)

    if dyn_bsz and dyn_bsz_runtime == "main":
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )

    return dataloader
