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
from typing import Any, Callable, Dict, Generator, Iterator, Optional

from ..utils import logging


logger = logging.get_logger(__name__)


# TODO: add state dict for buffer to resume training.
class DynBszBuffer:
    """
    A buffer to store samples for dynamic batch size.

    Args:
        get_length_fn: optional callable returning the per-sample token count used to
            decide when a micro batch is ready. Defaults to ``attention_mask.sum()``
            (i.e. total tokens). Pass a callable counting only ``labels != IGNORE_INDEX``
            to balance by effective (loss-contributing) tokens.
        get_physical_length_fn: optional callable returning the physical per-sample
            token count used as a hard cap while selecting a micro batch.
    """

    def __init__(
        self,
        get_length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        get_physical_length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
    ):
        self._buffer = []
        self._buffer_sample_lens = []
        self._buffer_physical_lens = []
        self.del_idxs = []
        self.cur_idx = 0
        self.all_token_cnt = 0
        self.all_physical_token_cnt = 0
        self._get_length_fn = get_length_fn
        self._get_physical_length_fn = get_physical_length_fn

    def append(self, item: Dict[str, Any]):
        """
        Append a sample to the buffer.
        Args:
            item: a sample to append to the buffer.
                The sample should be a dict containing an ``attention_mask`` tensor
                whose ``.sum()`` gives the number of valid tokens for batching.
        """
        self._buffer.append(item)
        if self._get_length_fn is not None:
            length = self._get_length_fn(item)
        else:
            if "attention_mask" not in item:
                raise KeyError("Expected 'attention_mask' in item")
            length = int(item["attention_mask"].sum())
        if self._get_physical_length_fn is not None:
            physical_length = self._get_physical_length_fn(item)
        else:
            physical_length = length
        self._buffer_sample_lens.append(length)
        self._buffer_physical_lens.append(physical_length)
        self.all_token_cnt += length
        self.all_physical_token_cnt += physical_length

    def get_samples(self, n_token_per_iter: int, force: bool = True, physical_token_cap: Optional[int] = None):
        """
        get samples from the buffer.
        Args:
            n_token_per_iter: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.
            This can emit one sample that exceeds ``physical_token_cap`` by itself,
            but the cap still prevents adding more samples to that micro batch.
        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        cum_physical_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < n_token_per_iter:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            physical_seq_len = self._buffer_physical_lens[self.cur_idx]
            fits_effective_budget = seq_len <= n_token_per_iter - cum_seq_len
            fits_physical_cap = physical_token_cap is None or (
                physical_seq_len <= physical_token_cap - cum_physical_len
            )
            first_forced_sample = force is True and cum_seq_len == 0
            if self.cur_idx not in self.del_idxs and (
                first_forced_sample or (fits_effective_budget and fits_physical_cap)
            ):
                cum_seq_len += seq_len
                cum_physical_len += physical_seq_len
                samples.append(self._buffer[self.cur_idx])
                self.del_idxs.append(self.cur_idx)
            self.cur_idx += 1
        assert len(samples) > 0
        return samples

    def __len__(self):
        return len(self._buffer)

    def flush(self):
        """ "
        Flush the buffer.
        """
        self.cur_idx = 0
        self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in self.del_idxs])
        self.all_physical_token_cnt -= sum([self._buffer_physical_lens[idx] for idx in self.del_idxs])
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
        ]
        self._buffer_physical_lens = [
            self._buffer_physical_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
        ]
        self.del_idxs = []

    def merge(self, buffer_to_merge: "DynBszBuffer"):
        """ "
        Merge the buffer with another buffer.
        Args:
            buffer_to_merge: the buffer to merge.
        """
        self.flush()
        buffer_to_merge.flush()
        for item in buffer_to_merge._buffer:
            self.append(item)


class BaseBatchingStrategy:
    """
    Base class for batching strategy.
    """

    def is_ready_for_micro_batch(self) -> bool:
        raise NotImplementedError("should implement `is_ready_for_micro_batch`")

    def put_item(self, item: Dict[str, Any]):
        raise NotImplementedError("should implement `put_item`")

    def get_micro_batch(self, step: int) -> Any:
        raise NotImplementedError("should implement `get_micro_batch` ")

    def empty(self) -> bool:
        raise NotImplementedError("should implement `empty`")


class TextBatchingStrategy(BaseBatchingStrategy):
    """ "
    Batching strategy for text data.
    Args:
        token_micro_bsz: the number of tokens to get for each request.
        bsz_warmup_steps: the number of steps to warm up the batch size.
        bsz_warmup_init_mbtoken: the initial number of tokens to get for each request.
        buffer_size: the size of the buffer.
        get_length_fn: optional per-sample length callable; see ``DynBszBuffer``.
        physical_token_cap: optional hard cap on the total physical tokens selected
            into each micro batch.
        get_physical_length_fn: optional per-sample physical length callable; see
            ``DynBszBuffer``.
    """

    def __init__(
        self,
        token_micro_bsz,
        buffer_size: int = 500,
        bsz_warmup_steps: int = 0,
        bsz_warmup_init_mbtoken: int = 200,
        get_length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
        physical_token_cap: Optional[int] = None,
        get_physical_length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
    ) -> None:
        super().__init__()
        self._step = 0
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken
        if bsz_warmup_steps > 0:
            assert self.bsz_warmup_init_mbtoken > 0

        self.buffer_size = buffer_size  # minimum samples in buffer
        self.physical_token_cap = physical_token_cap
        self.get_physical_length_fn = get_physical_length_fn
        self.buffer = DynBszBuffer(get_length_fn=get_length_fn, get_physical_length_fn=get_physical_length_fn)

    def is_ready_for_micro_batch(self) -> bool:
        effective_ready = len(self.buffer) >= self.buffer_size and self.buffer.all_token_cnt >= self.token_micro_bsz
        physical_ready = (
            len(self.buffer) >= self.buffer_size
            and self.physical_token_cap is not None
            and self.buffer.all_physical_token_cnt >= self.physical_token_cap
        )
        return effective_ready or physical_ready

    def put_item(self, item: Dict[str, Any]):
        if item["input_ids"].shape[-1] <= 1:
            print("WARNING: EMPTY STRING.")
            return
        self.buffer.append(item)

    def get_cur_token_micro_bsz(self):
        warmup = self.bsz_warmup_steps > 0 and self._step <= self.bsz_warmup_steps
        if warmup:
            return (
                self.token_micro_bsz - self.bsz_warmup_init_mbtoken
            ) * self._step // self.bsz_warmup_steps + self.bsz_warmup_init_mbtoken
        else:
            return self.token_micro_bsz

    def get_micro_batch(self, step) -> Any:
        """
        Get a micro batch from the buffer according to the current step.
        Args:
            step: the current step.
        Returns:
            data: a list of samples.
        """

        self._step = step
        cur_token_micro_bsz = self.get_cur_token_micro_bsz()
        samples = self.buffer.get_samples(cur_token_micro_bsz, physical_token_cap=self.physical_token_cap)
        self.buffer.flush()  # remove the selected samples.
        return samples

    def empty(self) -> bool:
        return len(self.buffer) == 0


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

            if self.batching_strategy.is_ready_for_micro_batch():
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
