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


from typing import Any, Dict, Optional

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
from .dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


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
    bsz_warmup_ratio: float = 0.02,
    bsz_warmup_init_mbtoken: int = 200,
    dyn_bsz: bool = True,
    dyn_bsz_buffer_size: int = 500,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    build_collate_fn: bool = True,
    collate_fn_kwargs: Optional[Dict[str, Any]] = {},
) -> "DistributedDataloader":
    parallel_state = get_parallel_state()

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
            f"train_steps: {train_steps}, bsz_warmup_steps: {bsz_warmup_steps},"
            f"bsz_warmup_init_mbtoken: {bsz_warmup_init_mbtoken}."
        )

        batching_strategy = TextBatchingStrategy(
            token_micro_bsz=batching_token_len,
            buffer_size=dyn_bsz_buffer_size,
            bsz_warmup_steps=bsz_warmup_steps,
            bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        )
        dyn_bsz_collate_fn = collate_fn
        collate_fn = UnpackDataCollator()
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
    if dyn_bsz:
        dataloader = DynamicBatchSizeDataLoader(
            dataloader,
            batching_strategy=batching_strategy,
            collate_fn=dyn_bsz_collate_fn,
            num_micro_batch=num_micro_batch,
            length=train_steps,
            drop_last=drop_last,
        )

    return dataloader
