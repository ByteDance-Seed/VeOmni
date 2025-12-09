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


from typing import TYPE_CHECKING, Callable, List, Optional, Union

from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from veomni.utils.device import get_device_type

from ...distributed.parallel_state import get_parallel_state
from ...utils import logging
from ..data_collator import (
    CollatePipeline,
    DataCollatorWithPadding,
    MakeMicroBatchCollator,
)
from ..data_loader import DistributedDataloader


if TYPE_CHECKING:
    from torch.utils.data import Dataset


logger = logging.get_logger(__name__)


def build_dit_dataloader(
    dataset: "Dataset",
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    train_steps: int,
    collate_fn: Optional[Union[Callable, List[Callable]]] = None,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
) -> "DistributedDataloader":
    parallel_state = get_parallel_state()
    num_micro_batch = global_batch_size // (
        micro_batch_size * parallel_state.dp_size
    )  # num_micro_batch = num accumulation steps

    logger.info_rank0(
        f"train_steps: {train_steps},"
        f"num_micro_batch: {num_micro_batch}, "
        f"micro_batch_size: {micro_batch_size}, global_batch_size: {global_batch_size}, "
        f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}."
    )

    if collate_fn is None:
        collate_fn_list = []
        collate_fn_list.append(DataCollatorWithPadding())

        collate_fn = CollatePipeline(collate_fn_list)

    if isinstance(collate_fn, list):
        collate_fn = CollatePipeline(collate_fn)

    collate_fn = MakeMicroBatchCollator(num_micro_batch=num_micro_batch, internal_data_collator=collate_fn)

    sampler = None
    if not isinstance(dataset, IterableDataset):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=parallel_state.dp_size,
            rank=parallel_state.dp_rank,
            shuffle=True,
            seed=seed,
        )

    dataloader = DistributedDataloader(
        dataset,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        pin_memory_device=get_device_type(),
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )

    return dataloader
