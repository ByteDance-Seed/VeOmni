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

from torch.utils.data import IterableDataset

from veomni.data.data_collator import DataCollatorWithPositionIDs

from ..utils import logging


logger = logging.get_logger(__name__)

class DynamicBatchingSizeDataset(IterableDataset):
    """Dynamic batching dataset that yields micro batches based on token count.

    Unlike DynamicBatchSizeDataLoader which constructs micro batches from items fetched
    from a dataloader in the main process, DynamicBatchingSizeDataset provides a
    dataset-like interface to yield micro batches, which can be applied to multi-worker
    dataloaders.

    This dataset buffers samples and creates batches when the total token count reaches
    the specified threshold, enabling efficient variable-length sequence batching.

    Attributes:
        dataset: The underlying iterable dataset to batch.
        dynamic_batching_collate_fn: Function to collate samples into a batch.
        ready_for_micro_batch_threshold: Minimum number of samples in buffer before batching.
        micro_batch_seq_length: Target total token count per batch.
        get_length_fn: Function to get the length (token count) of a sample.
        save_by_idx: Whether to save sample indices for checkpoint resumption.
        force_generate_long_sequence: If True, when a sample is longer than micro_batch_seq_length, force to generate a micro batch with this sample.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        micro_batch_seq_length: int,
        ready_for_micro_batch_threshold: int,
        dynamic_batching_collate_fn: Optional[Callable] = DataCollatorWithPositionIDs(),
        save_by_idx: bool = False,
        get_length_fn: Optional[Callable] = len,
        force_generate_long_sequence: bool = True,
    ) -> None:
        """Initialize the DynamicBatchingSizeDataset.

        Args:
            dataset: The underlying iterable dataset to batch from.
            micro_batch_seq_length: Target total token count per micro batch.
            ready_for_micro_batch_threshold: Minimum number of samples required in
                buffer before attempting to create a batch.
            dynamic_batching_collate_fn: Callable to collate samples into a batch.
                Defaults to DataCollatorWithPositionIDs().
            save_by_idx: If True, saves sample indices for checkpoint resumption.
                Requires dataset to have get_item method and output_refetch_idx attribute.
            get_length_fn: Function to compute the length (token count) of a sample.
                Defaults to len.
            force_generate_long_sequence: If True, when a sample is longer than micro_batch_seq_length, force to generate a micro batch with this sample.

        Raises:
            ValueError: If save_by_idx is True but dataset lacks required methods.
        """
        self.dataset = dataset
        self.dynamic_batching_collate_fn = dynamic_batching_collate_fn
        self.ready_for_micro_batch_threshold = ready_for_micro_batch_threshold
        self.micro_batch_seq_length = micro_batch_seq_length
        self.get_length_fn = get_length_fn
        self.save_by_idx = save_by_idx
        self.force_generate_long_sequence = force_generate_long_sequence

        if self.save_by_idx and not (hasattr(self.dataset, "get_item") and hasattr(self.dataset, "output_refetch_idx")):
            raise ValueError("save_by_idx is True, but dataset does not have get_item method or output_refetch_idx attribute to resume samples in buffers based on idx")
        self.dataset.output_refetch_idx = self.save_by_idx

        self._buffer = []
        self._buffer_token_count = 0
        self._data_iter = None

    def __iter__(self):
        """Iterate over the dataset and yield dynamically batched micro batches.

        Buffers samples from the underlying dataset and yields micro batches when
        the buffer contains enough samples and tokens. Each yielded batch is collated
        using the dynamic_batching_collate_fn.

        Yields:
            Collated micro batch when buffer conditions are met.

        Raises:
            Exception: Re-raises any exception other than StopIteration encountered
                during iteration.
        """
        if not self._data_iter:
            self._data_iter = iter(self.dataset)

        while True:
            try:
                item = next(self._data_iter)

                if self.save_by_idx:
                    item, refetch_idx = item[0], item[1]

                length = self.get_length_fn(item)
                if length > self.micro_batch_seq_length and not self.force_generate_long_sequence:
                    logger.warning(f"Sample length {length} exceeds micro batch seq length {self.micro_batch_seq_length}, skipping. If you want to force generate a micro batch with this sample, enable force_generate_long_sequence.")
                    continue

                if self.save_by_idx:
                    self._buffer.append((item, length, refetch_idx))
                else:
                    self._buffer.append((item, length))

                self._buffer_token_count += self._buffer[-1][1]

                if len(self._buffer) >= self.ready_for_micro_batch_threshold and self._buffer_token_count >= self.micro_batch_seq_length:
                    micro_batch = self._get_micro_batch()
                    micro_batch = self.dynamic_batching_collate_fn(micro_batch)
                    if micro_batch is not None:
                        yield micro_batch
                    else:
                        logging.warn('dynamic_batching_collate_fn returned None, skip this micro_batch')

            except Exception as e:
                if isinstance(e, StopIteration):
                    while len(self._buffer) > 0:
                        micro_batch = self._get_micro_batch()
                        micro_batch = self.dynamic_batching_collate_fn(micro_batch)
                        if micro_batch is not None:
                            yield micro_batch
                        else:
                            logging.warn('dynamic_batching_collate_fn returned None, skip this micro_batch')
                    return
                else:
                    logger.error(
                        f'DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}')
                    raise

    def _get_micro_batch(self):
        """Construct a micro batch from buffered samples.

        Selects samples from the buffer to create a batch with total token count
        not exceeding micro_batch_seq_length. Samples that don't fit are kept in
        the buffer for the next batch.

        Returns:
            list: A list of samples forming the micro batch.

        Raises:
            AssertionError: If no samples could be selected for the batch.
        """
        micro_batch = []
        seq_length = 0
        selected_indices = []

        for idx, item in enumerate(self._buffer):
            sample, length = item[0], item[1]

            if length + seq_length > self.micro_batch_seq_length and not (seq_length == 0 and self.force_generate_long_sequence):
                continue

            micro_batch.append(sample)
            seq_length += length
            self._buffer_token_count -= length
            selected_indices.append(idx)

            if seq_length >= self.micro_batch_seq_length:
                break

        # Remove selected items from buffer (iterate backwards to maintain indices)
        for idx in reversed(selected_indices):
            del self._buffer[idx]

        assert len(micro_batch) > 0
        return micro_batch


    def state_dict(self):
        """Get the state dictionary for checkpointing.

        Saves the current buffer state and token count. If save_by_idx is True,
        only saves sample indices; otherwise saves the full buffer contents.
        Also saves the upstream dataset state if available.

        Returns:
            dict: State dictionary containing:
                - save_by_idx: Whether indices are saved instead of samples.
                - buffer_token_count: Total token count in the buffer.
                - buffer: Buffered samples or their indices.
                - dynamic_batch_upstream_dataset_state: Upstream dataset state (if available).
        """
        state = {
            "save_by_idx": self.save_by_idx,
            "buffer_token_count": self._buffer_token_count,
        }
        if self.save_by_idx:
            state['buffer'] = [item[2] for item in self._buffer]
        else:
            state['buffer'] = self._buffer

        if hasattr(self.dataset, "state_dict"):
            state['dynamic_batch_upstream_dataset_state'] = self.dataset.state_dict()

        return state
    
    def load_state_dict(self, state_dict):
        """Load state from a checkpoint.

        Restores the buffer and token count from a saved state. Handles both
        index-based and full-sample buffer restoration based on the saved state.
        Also restores the upstream dataset state if available.

        Args:
            state_dict: State dictionary from a previous checkpoint, containing:
                - save_by_idx: Whether the saved buffer contains indices.
                - buffer: Saved buffer (samples or indices).
                - buffer_token_count: Saved token count.
                - dynamic_batch_upstream_dataset_state: Upstream dataset state (optional).

        Raises:
            AssertionError: If restored buffer_token_count doesn't match actual token count.
        """
        # prev_save_by_idx does not have to be equal to self.save_by_idx, however, we still need to resume the buffer according to it.
        prev_save_by_idx = state_dict["save_by_idx"]
        if prev_save_by_idx:
            self._buffer = []
            for idx in state_dict["buffer"]:
                item = self.dataset.get_item(idx)
                length = self.get_length_fn(item)
                if self.save_by_idx:
                    self._buffer.append((item, length, idx))
                else:
                    self._buffer.append((item, length))
        else:
            self._buffer = state_dict["buffer"]
            if self.save_by_idx and self._buffer:
                raise ValueError("save_by_idx is True, but previous buffer does not contain indices")

        self._buffer_token_count = state_dict["buffer_token_count"]
        # Verify buffer_token_count matches the sum of token lengths
        assert self._buffer_token_count == sum([item[1] for item in self._buffer]), (
            "buffer_token_count does not match the sum of token lengths in buffer"
        )
        assert self._buffer_token_count == sum(self.get_length_fn(item[0]) for item in self._buffer), (
            "buffer_token_count does not match the sum of lengths computed from samples in buffer"
        )
        del state_dict["buffer"]

        if 'dynamic_batch_upstream_dataset_state' in state_dict:
            self.dataset.load_state_dict(state_dict['dynamic_batch_upstream_dataset_state'])

    def set_epoch(self, epoch: int):
        """Set the epoch for the upstream dataset.

        Passes the epoch to the upstream dataset if it supports set_epoch.
        Has no direct effect on dynamic batching itself.

        Args:
            epoch: The epoch number to set.
        """
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


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
