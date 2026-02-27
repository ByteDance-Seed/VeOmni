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

from torch.utils.data import IterableDataset

from ..utils import logging


logger = logging.get_logger(__name__)


# TODO: add state dict for buffer to resume training.
class DynBszBuffer:
    """
    A buffer to store samples for dynamic batch size.
    """

    def __init__(self):
        self._buffer = []
        self._buffer_sample_lens = []
        self.del_idxs = []
        self.cur_idx = 0
        self.all_token_cnt = 0

    def append(self, item: Dict[str, Any]):
        """
        Append a sample to the buffer.
        Args:
            item: a sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        self._buffer_sample_lens.append(item["attention_mask"].sum())
        self.all_token_cnt += self._buffer_sample_lens[-1]

    def get_samples(self, n_token_per_iter: int, force: bool = True):
        """
        get samples from the buffer.
        Args:
            n_token_per_iter: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.
        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < n_token_per_iter:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            if self.cur_idx not in self.del_idxs and (
                (force is True and cum_seq_len == 0) or (seq_len <= n_token_per_iter - cum_seq_len)
            ):
                cum_seq_len += seq_len
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
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
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
    """

    def __init__(
        self,
        token_micro_bsz,
        buffer_size: int = 500,
        bsz_warmup_steps: int = 0,
        bsz_warmup_init_mbtoken: int = 200,
    ) -> None:
        super().__init__()
        self._step = 0
        self.token_micro_bsz = token_micro_bsz
        self.bsz_warmup_steps = bsz_warmup_steps
        self.bsz_warmup_init_mbtoken = bsz_warmup_init_mbtoken
        if bsz_warmup_steps > 0:
            assert self.bsz_warmup_init_mbtoken > 0

        self.buffer_size = buffer_size  # minimum samples in buffer
        self.buffer = DynBszBuffer()

    def is_ready_for_micro_batch(self) -> bool:
        return len(self.buffer) >= self.buffer_size and self.buffer.all_token_cnt >= self.token_micro_bsz

    def put_item(self, item: Dict[str, Any]):
        if len(item["input_ids"]) == 1:
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
        samples = self.buffer.get_samples(cur_token_micro_bsz)
        self.buffer.flush()  # remove the selected samples.
        return samples

    def empty(self) -> bool:
        return len(self.buffer) == 0


class DynamicBatchingSizeDataset(IterableDataset):
    """Dynamic batching dataset that yields micro batches based on token count.

    Unlike ``DynamicBatchSizeDataLoader``, which constructs micro batches in the
    main process after fetching from a plain DataLoader, ``DynamicBatchingSizeDataset``
    performs batching inside each DataLoader worker process.
    It is also compatible with ``StatefulDataLoader``'s per-worker ``state_dict()`` /
    ``load_state_dict()`` mechanism, enabling exact checkpoint / resume for dynamic-batching workloads.

    Internally each worker maintains a sample buffer.  A micro batch is emitted once
    the buffer holds at least ``ready_for_micro_batch_threshold`` samples **and** their
    combined token count reaches ``micro_batch_seq_length``.  When the upstream dataset
    is exhausted, remaining buffer contents are drained and emitted as final batches
    regardless of the threshold.

    Attributes:
        dataset: The upstream iterable dataset to read samples from.
        ready_for_micro_batch_threshold: Minimum number of samples that must be in the
            buffer before a microbatch can be formed.
        micro_batch_seq_length: Target total token count per micro batch (soft upper
            bound; may be exceeded by a single overlong sample when
            ``force_generate_long_sequence`` is True).
        get_length_fn: Function that returns the token count of a single sample.
        save_by_idx: Whether to checkpoint the buffer as sample indices (smaller checkpoint size)
            rather than full sample tensors.
        force_generate_long_sequence: If True, a sample whose length alone exceeds
            ``micro_batch_seq_length`` is emitted as a single-sample batch instead of
            being silently discarded. This is not supported yet.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        micro_batch_seq_length: int,
        ready_for_micro_batch_threshold: int,
        dynamic_batching_collate_fn: Callable,
        save_by_idx: bool = True,
        get_length_fn: Optional[Callable] = len,
        force_generate_long_sequence: bool = False,
    ) -> None:
        """Initialize the DynamicBatchingSizeDataset.

        Args:
            dataset: The underlying iterable dataset to batch from.
            micro_batch_seq_length: Target total token count per micro batch.
            ready_for_micro_batch_threshold: Minimum number of samples required in
                buffer before attempting to create a batch.
            save_by_idx: If True, saves sample indices for checkpoint resumption.
                Requires dataset to have get_item method and output_refetch_idx attribute.
            get_length_fn: Function to compute the length (token count) of a sample.
                Defaults to len.
            force_generate_long_sequence: If True, a sample whose length alone exceeds
                ``micro_batch_seq_length`` is emitted as a single-sample batch instead of
                being silently discarded. This is not supported yet.

        Raises:
            ValueError: If ``save_by_idx`` is True but ``dataset`` does not expose the
                ``get_item()`` method and ``output_refetch_idx`` attribute required to
                reconstruct the buffer from indices on resume.
        """
        self.dataset = dataset
        self.dynamic_batching_collate_fn = dynamic_batching_collate_fn
        self.ready_for_micro_batch_threshold = ready_for_micro_batch_threshold
        self.micro_batch_seq_length = micro_batch_seq_length
        self.get_length_fn = get_length_fn
        self.save_by_idx = save_by_idx
        self._data_iter = None

        if force_generate_long_sequence:
            raise ValueError("force_generate_long_sequence is not supported yet.")

        self.force_generate_long_sequence = force_generate_long_sequence

        if self.save_by_idx and not (
            hasattr(self.dataset, "get_item") and hasattr(self.dataset, "output_refetch_idx")
        ):
            raise ValueError(
                "save_by_idx is True, but dataset does not have get_item method or output_refetch_idx attribute to resume samples in buffers based on idx"
            )
        self.dataset.output_refetch_idx = self.save_by_idx

        self._buffer = []
        self._buffer_of_refetch_idx = []
        self._buffer_token_count = 0

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
        self._data_iter = iter(self.dataset)

        while True:
            try:
                if (
                    len(self._buffer) >= self.ready_for_micro_batch_threshold
                    and self._buffer_token_count >= self.micro_batch_seq_length
                ):
                    micro_batch = self._get_micro_batch()
                    micro_batch = self.dynamic_batching_collate_fn(micro_batch)
                    if micro_batch is not None:
                        yield micro_batch
                    else:
                        logging.warn("dynamic_batching_collate_fn returned None, skip this micro_batch")

                item = next(self._data_iter)
                if self.save_by_idx:
                    item, refetch_idx = item[0], item[1]

                length = self.get_length_fn(item)
                if length > self.micro_batch_seq_length and not self.force_generate_long_sequence:
                    # TODO: record the count of discarded long examples for monitoring
                    logger.warning(
                        f"Sample length {length} exceeds micro batch seq length {self.micro_batch_seq_length}, skipping. If you want to force generate a micro batch with this sample, enable force_generate_long_sequence."
                    )
                    continue

                self._buffer.append((item, length))
                if self.save_by_idx:
                    self._buffer_of_refetch_idx.append(refetch_idx)

                self._buffer_token_count += self._buffer[-1][1]

            except Exception as e:
                if isinstance(e, StopIteration):
                    while len(self._buffer) > 0:
                        micro_batch = self._get_micro_batch()
                        micro_batch = self.dynamic_batching_collate_fn(micro_batch)
                        if micro_batch is not None:
                            yield micro_batch
                        else:
                            logging.warn("dynamic_batching_collate_fn returned None, skip this micro_batch")
                    return
                else:
                    logger.error(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise

    def _get_micro_batch(self):
        """Construct a micro batch from buffered samples using a greedy first-fit strategy.

        Iterates the buffer in order and greedily adds each sample whose length fits
        within the remaining token budget (``micro_batch_seq_length - seq_length``).
        Samples that do not fit are left in the buffer for subsequent batches.

        Special case: when the buffer's first sample alone exceeds
        ``micro_batch_seq_length`` and ``force_generate_long_sequence`` is True, that
        sample is taken unconditionally (``seq_length == 0`` guard) so that the dataset
        never stalls on an overlong sequence.

        Returns:
            list: Non-empty list of samples forming the micro batch.

        Raises:
            AssertionError: If no sample could be selected (should never happen under
                normal operation).
        """
        micro_batch = []
        seq_length = 0
        indices_to_remove_from_buffer = []

        for idx, item in enumerate(self._buffer):
            sample, length = item[0], item[1]

            if length + seq_length > self.micro_batch_seq_length:
                if seq_length > 0:
                    continue
                elif not self.force_generate_long_sequence:
                    # Usually it is impossible to reach this branch because too long samples would not be added to the buffer if force_generate_long_sequence is False.
                    continue

            micro_batch.append(sample)
            seq_length += length
            self._buffer_token_count -= length
            indices_to_remove_from_buffer.append(idx)

            if seq_length >= self.micro_batch_seq_length:
                break

        # Remove selected items from buffer (iterate backwards to maintain indices)
        for idx in reversed(indices_to_remove_from_buffer):
            del self._buffer[idx]
            if self.save_by_idx:
                del self._buffer_of_refetch_idx[idx]

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
            # Make sure we store an integer instead of any tensor
            "buffer_token_count": int(self._buffer_token_count),
        }

        # the state_dict might be called frequently with StatefulDataloaders(see more details of snapshot_every_n_steps)
        # so we try to not include extra calculations here.
        if self.save_by_idx:
            state["buffer"] = copy.deepcopy(self._buffer_of_refetch_idx)
        else:
            # deepcopy buffer so that it can be transfered through multiple processes
            state["buffer"] = copy.deepcopy(self._buffer)

        if hasattr(self.dataset, "state_dict"):
            state["dynamic_batch_upstream_dataset_state"] = self.dataset.state_dict()

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
            AssertionError: If the restored ``buffer_token_count`` does not match the
                sum of token lengths recomputed from the reconstructed buffer.
            ValueError: If ``save_by_idx`` is True on the current instance but the
                checkpoint buffer holds some full samples instead of indices (incompatible
                checkpoint format).
        """
        # prev_save_by_idx does not have to be equal to self.save_by_idx, however, we still need to resume the buffer according to it.
        prev_save_by_idx = state_dict["save_by_idx"]
        if prev_save_by_idx:
            self._buffer = []
            self._buffer_of_refetch_idx = []
            for idx in state_dict["buffer"]:
                item = self.dataset.get_item(idx)
                length = self.get_length_fn(item)
                self._buffer.append((item, length))
                if self.save_by_idx:
                    self._buffer_of_refetch_idx.append(idx)
        else:
            self._buffer = state_dict["buffer"]
            if self.save_by_idx and len(self._buffer) > 0:
                raise ValueError("save_by_idx is True, but previous buffer contains valid samples instead of indices")
            self._buffer_of_refetch_idx = []

        self._buffer_token_count = state_dict["buffer_token_count"]
        # Verify buffer_token_count matches the sum of token lengths
        assert self._buffer_token_count == sum([item[1] for item in self._buffer]), (
            "buffer_token_count does not match the sum of token lengths in buffer"
        )
        assert self._buffer_token_count == sum(self.get_length_fn(item[0]) for item in self._buffer), (
            "buffer_token_count does not match the sum of lengths computed from samples in buffer"
        )
        del state_dict["buffer"]

        if "dynamic_batch_upstream_dataset_state" in state_dict:
            self.dataset.load_state_dict(state_dict["dynamic_batch_upstream_dataset_state"])

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
