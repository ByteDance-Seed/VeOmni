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
from typing import Any, Callable, Dict, Literal, Optional

import torch
from torch.utils.data import Dataset, IterableDataset
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
    WeightedMultiSourceDataset,
    _MapStyleSamplerWrapper,
    get_length_fn,
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

    def __init__(self, *args, resume_replay_timeout_seconds=7200, **kwargs):
        if type(resume_replay_timeout_seconds) is not int or resume_replay_timeout_seconds <= 0:
            raise ValueError("resume_replay_timeout_seconds must be a positive integer")
        super().__init__(*args, **kwargs)
        self.resume_replay_timeout_seconds = resume_replay_timeout_seconds
        self._preloaded_resume_iterator = None
        dataset = self.dataset
        while isinstance(dataset, DynamicBatchingSizeDataset):
            dataset = dataset.dataset
        self._weighted_resume_dataset = dataset if isinstance(dataset, WeightedMultiSourceDataset) else None
        if self.persistent_workers and self.num_workers and isinstance(dataset, WeightedMultiSourceDataset):
            dataset.enable_persistent_epoch()

    def validate_weighted_resume_state(self, state, expected_epoch=None):
        """Preflight all worker source states before installing any loader state."""
        weighted = self._weighted_resume_dataset
        if weighted is None:
            return None
        if self.num_workers and self.in_order is not True:
            raise ValueError("exact multi-source resume requires in-order data workers")
        if self.generator is not None:
            raise ValueError("exact multi-source resume does not support a custom loader generator")
        if weighted is not None:
            if not isinstance(state, dict):
                raise ValueError("multi-source checkpoint loader state must be a dict")
            if state.get("_veomni_weighted_geometry") != self._weighted_resume_geometry():
                raise ValueError("multi-source checkpoint loader batch or worker geometry differs")
            if self.num_workers:
                snapshot = state.get("_snapshot", {})
                if not isinstance(snapshot, dict):
                    raise ValueError("multi-source checkpoint snapshot is missing")
                main = snapshot.get("_main_snapshot", {})
                workers = snapshot.get("_worker_snapshots", {})
                if (
                    not isinstance(main, dict)
                    or not isinstance(workers, dict)
                    or main.get("_num_workers") != self.num_workers
                    or len(workers) != self.num_workers
                ):
                    raise ValueError("multi-source checkpoint worker geometry differs")
                candidates = []
                for worker_id in range(self.num_workers):
                    worker = workers.get(f"worker_{worker_id}")
                    if not isinstance(worker, dict) or worker.get("worker_id") != worker_id:
                        raise ValueError("multi-source checkpoint worker state is missing")
                    candidates.append((worker.get("dataset_state"), (worker_id, self.num_workers)))
            else:
                candidates = [(state.get("dataset_state"), (None, 0))]
            states = []
            for source_state, geometry in candidates:
                wrapped = self.dataset
                while isinstance(wrapped, DynamicBatchingSizeDataset):
                    if not isinstance(source_state, dict):
                        raise ValueError("multi-source checkpoint dataset state is missing")
                    source_state = source_state.get("dynamic_batch_upstream_dataset_state")
                    wrapped = wrapped.dataset
                weighted.validate_exact_resume_state(source_state, worker_geometry=geometry)
                states.append(source_state)
            epochs = {entry["runtime"].get("epoch", weighted._pending_epoch) for entry in states}
            if len(epochs) != 1 or None in epochs:
                raise ValueError("multi-source checkpoint worker epochs differ or are missing")
            epoch = epochs.pop()
            if expected_epoch is not None and epoch != expected_epoch:
                raise ValueError("multi-source checkpoint epoch differs from the saved trainer position")
            return epoch

    def _weighted_resume_geometry(self):
        import multiprocessing

        from .dynamic_batching import _resume_callable_identity

        context = self.multiprocessing_context
        start_method = None
        if self.num_workers:
            start_method = (context or multiprocessing.get_context()).get_start_method()
        return {
            "version": 1,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "in_order": self.in_order,
            "prefetch_factor": self.prefetch_factor,
            "snapshot_every_n_steps": self.snapshot_every_n_steps,
            "multiprocessing_context": start_method,
            "collate_fn": _resume_callable_identity(self.collate_fn),
            "worker_init_fn": _resume_callable_identity(self.worker_init_fn),
        }

    def state_dict(self):
        state = super().state_dict()
        if self._weighted_resume_dataset is not None:
            state = dict(state)
            state["_veomni_weighted_geometry"] = self._weighted_resume_geometry()
        return state

    def load_state_dict(self, state):
        epoch = self.validate_weighted_resume_state(state)
        if epoch is not None and self._weighted_resume_dataset._pending_epoch is None:
            self._weighted_resume_dataset.prepare_resume_epoch(epoch)
        previous = self._preloaded_resume_iterator
        if previous is not None and hasattr(previous, "_shutdown_workers"):
            previous._shutdown_workers()
        self._preloaded_resume_iterator = None
        if epoch is not None:
            state = dict(state)
            state.pop("_veomni_weighted_geometry")
        result = super().load_state_dict(state)
        if epoch is not None:
            # Worker state (including replay) is loaded during iterator startup.
            # Keep this iterator so the trainer consumes exactly this cursor.
            timeout = self.timeout
            if self.num_workers and (timeout == 0 or timeout > self.resume_replay_timeout_seconds):
                self.timeout = self.resume_replay_timeout_seconds
            try:
                iterator = super().__iter__()
                iterator._timeout = timeout
                self._preloaded_resume_iterator = iterator
            finally:
                self.timeout = timeout
        return result

    def __iter__(self):
        if self._preloaded_resume_iterator is not None:
            iterator = self._preloaded_resume_iterator
            self._preloaded_resume_iterator = None
            return iterator
        return super().__iter__()

    def prime_persistent_epoch_for_resume(self):
        """Start new-epoch workers before restoring the trainer's CPU RNG."""
        if (
            not self.persistent_workers
            or not self.num_workers
            or self._weighted_resume_dataset is None
            or self._iterator is not None
            or self._preloaded_resume_iterator is not None
        ):
            raise ValueError("cannot prime this loader for a fresh persistent-worker epoch")
        timeout = self.timeout
        if timeout == 0 or timeout > self.resume_replay_timeout_seconds:
            self.timeout = self.resume_replay_timeout_seconds
        try:
            iterator = super().__iter__()
            iterator._timeout = timeout
            self._preloaded_resume_iterator = iterator
        finally:
            self.timeout = timeout

    def prepare_resume_epoch(self, epoch: int) -> None:
        prepare = getattr(self.dataset, "prepare_resume_epoch", None)
        if callable(prepare):
            prepare(epoch)

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


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
    _dpo_no_synthetic_padding: bool = False,
    resume_replay_timeout_seconds: int = 7200,
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
    """
    if collate_fn_kwargs is None:
        collate_fn_kwargs = {}
    parallel_state = get_parallel_state()
    if (
        dyn_bsz
        and dyn_bsz_runtime == "main"
        and not drop_last
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and not _dpo_no_synthetic_padding
    ):
        raise ValueError(
            "Distributed main-process dynamic batching requires drop_last=True until tail steps are aligned"
        )

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
            dyn_bsz_physical_length_fn = get_length_fn
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
    if not isinstance(dataset, IterableDataset):
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
    dataloader = DistributedDataloader(
        dataset,
        resume_replay_timeout_seconds=resume_replay_timeout_seconds,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        pin_memory_device=get_device_type(),
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
        in_order=in_order,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )

    if dyn_bsz and dyn_bsz_runtime == "main":
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )
        if _dpo_no_synthetic_padding:
            dataloader._dpo_disallow_implicit_restart = True

    return dataloader
