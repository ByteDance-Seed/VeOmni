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


import copy
import sys
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, Optional

from ..utils import logging


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from .batching_strategy import BaseBatchingStrategy


class DynamicBatchSizeDataLoader:
    """Dynamic batch DataLoader.

    Args:
        dataloader: torch DataLoader
        batching_strategy: dynamic batch strategy
        collate_fn: DataLoader collate_fn, collate data after get data from batching_strategy
        num_micro_batch: num_micro_batch, if num_micro_batch == 1, return micro_batch for gradient accumulation
        length: length of dataloader, if length == -1, length = sys.maxsize, default len(dataloader)
        drop_last: if True, drop last batch if batch size < num_micro_batch

    """

    def __init__(
        self,
        dataloader: Any,
        batching_strategy: "BaseBatchingStrategy",
        collate_fn: Optional[Callable] = None,
        num_micro_batch: int = 1,
        length: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.batching_strategy = batching_strategy
        self.num_micro_batch = num_micro_batch
        self.dataloader_item_buffer = deque()
        self.item_buffer = deque()
        self.step = 0
        self._collate_fn = collate_fn
        self._dataloader = dataloader
        self._drop_last = drop_last
        self._data_iter: Iterator
        self._resume = False
        self._batch_data_iter: Generator

        if length > 0:
            self._length = length
        elif length == -1:
            self._length = sys.maxsize
        else:
            self._length = len(self._dataloader)

    def __len__(self):
        if self._length:
            return self._length
        else:
            raise RuntimeError("length must set at init. before call len()")

    def __iter__(self) -> Iterator:
        if not self._resume:
            self.step = 0
            self._data_iter = iter(self._dataloader)
            self._batch_data_iter = self.batch_data_generator()
        self._resume = False
        return self

    def __next__(self):
        return next(self._batch_data_iter)

    def batch_data_generator(self):
        batch = []

        while True:
            if self._length and self.step >= self._length:
                return

            if self.batching_strategy.is_full_filled():
                micro_batch = self.batching_strategy.get_micro_batch(self.step)
                if self._collate_fn:
                    micro_batch = self._collate_fn(micro_batch)
                batch.append(micro_batch)
                if len(batch) == self.num_micro_batch:
                    yield batch
                    self.step += 1
                    batch = []

            try:
                processing_item = next(self._data_iter)
            except Exception as e:
                if isinstance(e, StopIteration):
                    if self.step < self._length:
                        # call iter until reach length
                        self._data_iter = iter(self._dataloader)
                        processing_item = next(self._data_iter)
                    elif not self._drop_last and not self.batching_strategy.empty():
                        while not self.batching_strategy.empty():
                            micro_batch = self.batching_strategy.get_micro_batch(self.step)
                            if self._collate_fn:
                                micro_batch = self._collate_fn(micro_batch)
                            batch.append(micro_batch)
                            if len(batch) == self.num_micro_batch:
                                yield batch
                                self.step += 1
                                batch = []

                        while len(batch) < self.num_micro_batch:
                            padding_batch = copy.deepcopy(micro_batch)
                            padding_batch["padding_flag"] = True
                            batch.append(padding_batch)
                        yield batch
                        self.step += 1
                        return
                    else:
                        return
                else:
                    logger.error(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise

            # put processing_item to buffer
            if isinstance(processing_item, dict):
                processing_item = [processing_item]

            for item in processing_item:
                self.batching_strategy.put_item(item)

    def state_dict(self):
        # save state
        state = self.__dict__.copy()
        # remove internal fields
        for k in list(state.keys()):
            if k.startswith("_"):
                del state[k]

        # save dataloader state
        if hasattr(self._dataloader, "state_dict"):
            state["dataloader_state"] = self._dataloader.state_dict()
        elif hasattr(self._dataloader, "__getstate__"):
            state["dataloader_state"] = self._dataloader.__getstate__()

        if hasattr(self.batching_strategy, "state_dict"):
            state["batching_strategy_state"] = self.batching_strategy.state_dict()  # type: ignore
            del state["batching_strategy"]

        return copy.deepcopy(state)

    def load_state_dict(self, state: Dict[str, Any]):
        if state["num_micro_batch"] != self.num_micro_batch:
            logger.warning(
                f"num_micro_batch changed: [ {state['num_micro_batch']} -> {self.num_micro_batch} ], will clear prefetch buffer"
            )
            del state["num_micro_batch"]
        self.__dict__.update(state)
        self._resume = True

        if hasattr(self._dataloader, "load_state_dict"):
            self._dataloader.load_state_dict(state["dataloader_state"])
        elif hasattr(self._dataloader, "__getstate__"):
            self._dataloader.__setstate__(state["dataloader_state"])

        if "batching_strategy_state" in state:
            self.batching_strategy.load_state_dict(  # type: ignore
                state["batching_strategy_state"]
            )
            del state["batching_strategy_state"]

        self._data_iter = iter(self._dataloader)
        self._batch_data_iter = self.batch_data_generator()

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._dataloader, "set_epoch"):
            self._dataloader.set_epoch(epoch)
