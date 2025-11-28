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
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.device import get_device_type
from ..utils.registry import Registry
from .batching_strategy import TextBatchingStrategy
from .data_collator import (
    CollatePipeline,
    DataCollatorWithPacking,
    DataCollatorWithPadding,
    DataCollatorWithPositionIDs,
    MakeMicroBatchCollator,
    TextSequenceShardCollator,
    UnpackDataCollator,
)
from .dynamic_batching import DynamicBatchSizeDataLoader


if TYPE_CHECKING:
    from torch.utils.data import Dataset

DATALOADER_REGISTRY = Registry("dataloader")
logger = logging.get_logger(__name__)


def build_dataloader(dataloader_type: str, **kwargs):
    return DATALOADER_REGISTRY[dataloader_type](**kwargs)


class DistributedDataloader(StatefulDataLoader):
    dataset: "Dataset"
    sampler: "StatefulDistributedSampler"

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


@DATALOADER_REGISTRY.register("native")
def build_native_dataloader(
    dataset: "Dataset",
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    rmpad: bool = True,
    rmpad_with_pos_ids: bool = False,
    bsz_warmup_ratio: float = 0.02,
    bsz_warmup_init_mbtoken: int = 200,
    dyn_bsz_buffer_size: int = 500,
    dyn_bsz_margin: int = 0,
    collate_fn: Optional[Union[Callable, List[Callable]]] = None,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
) -> "DistributedDataloader":
    parallel_state = get_parallel_state()
    token_micro_bsz = micro_batch_size * max_seq_len
    num_micro_batch = global_batch_size // (
        micro_batch_size * parallel_state.dp_size
    )  # num_micro_batch = num accumulation steps
    bsz_warmup_steps = int(train_steps * bsz_warmup_ratio)
    use_rmpad = rmpad or rmpad_with_pos_ids
    logger.info_rank0(
        f"train_steps: {train_steps}, max_seq_len: {max_seq_len}, use_rmpad: {use_rmpad}, "
        f"bsz_warmup_steps: {bsz_warmup_steps}, bsz_warmup_init_mbtoken: {bsz_warmup_init_mbtoken}, "
        f"token_micro_bsz: {token_micro_bsz}, num_micro_batch: {num_micro_batch}, "
        f"micro_batch_size: {micro_batch_size}, global_batch_size: {global_batch_size}, "
        f"dp_size: {parallel_state.dp_size}, sp_size: {parallel_state.sp_size}."
    )

    if collate_fn is None:
        collate_fn_list = []
        if rmpad_with_pos_ids:
            collate_fn_list.append(DataCollatorWithPositionIDs())
        elif rmpad:
            collate_fn_list.append(DataCollatorWithPacking())
        else:
            collate_fn_list.append(DataCollatorWithPadding())

        if parallel_state.sp_enabled:
            collate_fn_list.append(TextSequenceShardCollator(rmpad=rmpad, rmpad_with_pos_ids=rmpad_with_pos_ids))

        collate_fn = CollatePipeline(collate_fn_list)

    if isinstance(collate_fn, list):
        collate_fn = CollatePipeline(collate_fn)

    if use_rmpad:
        batching_strategy = TextBatchingStrategy(
            token_micro_bsz=token_micro_bsz - dyn_bsz_margin * max_seq_len,
            buffer_size=dyn_bsz_buffer_size,
            bsz_warmup_steps=bsz_warmup_steps if bsz_warmup_steps else -1,
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        )
        dyn_bsz_collate_fn = collate_fn
        collate_fn = UnpackDataCollator()
    else:
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
    if use_rmpad:
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )

    return dataloader
