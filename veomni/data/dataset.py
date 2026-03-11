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
import os
import random
import traceback
from functools import partial
from typing import Callable, Dict, List, Literal, Optional

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import interleave_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, IterableDataset

from ..utils.registry import Registry


try:
    from hdfs_io import isdir, listdir
except ImportError:
    from ..utils.hdfs_io import isdir, listdir

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.dist_utils import main_process_first
from ..utils.multisource_utils import parse_multisource_config


logger = logging.get_logger(__name__)

DATASET_REGISTRY = Registry("Dataset")


def build_dataset(dataset_name: str, **kwargs) -> "Dataset":
    return DATASET_REGISTRY[dataset_name](**kwargs)


class MappingDataset(Dataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform
        self.indices = list(range(len(self._data)))
        self.data_len = len(self.indices)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if index >= len(self.indices):
            random.shuffle(self.indices)
            index = index % len(self.indices)
        mapped_idx = self.indices[index]
        if self._transform is not None:
            return self._transform(self._data[mapped_idx])
        else:
            return self._data[mapped_idx]


class IterativeDataset(IterableDataset):
    def __init__(self, data: "HFIterableDataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                yield self._transform(sample)
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


class InterleavedIterableDataset(IterativeDataset):
    def __init__(self, data: "HFIterableDataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                ds_idx = sample["ds_idx"]
                transformed_sample = self._transform(sample)
                if isinstance(transformed_sample, List):
                    for idx in range(len(transformed_sample)):
                        transformed_sample[idx]["ds_idx"] = ds_idx
                    yield transformed_sample
                else:
                    transformed_sample["ds_idx"] = ds_idx
                    yield transformed_sample
            else:
                yield sample


class InterleavedMappingDataset(MappingDataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        super().__init__(data, transform)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if index >= len(self.indices):
            random.shuffle(self.indices)
            index = index % len(self.indices)
        mapped_idx = self.indices[index]
        if self._transform is not None:
            sample = self._data[mapped_idx]
            ds_idx = sample["ds_idx"]
            transformed_sample = self._transform(sample)
            if isinstance(transformed_sample, List):
                for idx in range(len(transformed_sample)):
                    transformed_sample[idx]["ds_idx"] = ds_idx
            else:
                transformed_sample["ds_idx"] = ds_idx
            return transformed_sample
        else:
            return self._data[mapped_idx]


class EnergonDataset(IterativeDataset):
    """
    A specialized wrapper for Megatron-Energon datasets that provides:
    - Automatic WorkerConfig management
    - TextSample to dict conversion
    - Native state management using save_state/restore_state
    - Epoch-based state reset

    Args:
        data (Dataset): underlying Megatron-Energon dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self):
        """Get the length of the dataset."""
        if hasattr(self._data, "__len__"):
            return len(self._data)

    def __iter__(self):
        """Iterate over the dataset with WorkerConfig management and TextSample conversion."""
        # For Megatron-Energon datasets, we need to set up the WorkerConfig properly
        if hasattr(self._data, "worker_config"):
            try:
                from megatron.energon import WorkerConfig

                # Ensure active_worker_config is None before activation
                WorkerConfig.active_worker_config = None
                # Activate the worker config
                self._data.worker_config.worker_activate(sample_index=0)
                logger.debug("Activated WorkerConfig for Megatron-Energon dataset")
            except Exception as e:
                logger.warning(f"Failed to activate WorkerConfig: {e}")

        try:
            for sample in self._data:
                # Convert Megatron-Energon TextSample to dict for compatibility
                if hasattr(sample, "__dict__") and not isinstance(sample, dict):
                    # Convert TextSample or similar objects to dict
                    sample_dict = {}
                    for key, value in sample.__dict__.items():
                        if not key.startswith("_"):  # Skip private attributes
                            sample_dict[key] = value

                    # Handle special case for TextSample
                    if hasattr(sample, "text"):
                        sample_dict["text"] = sample.text

                    sample = sample_dict

                if self._transform is not None:
                    yield self._transform(sample)
                else:
                    yield sample
        finally:
            # Clean up WorkerConfig
            if hasattr(self._data, "worker_config"):
                try:
                    self._data.worker_config.worker_deactivate()
                    logger.debug("Deactivated WorkerConfig for Megatron-Energon dataset")
                except Exception as e:
                    logger.warning(f"Failed to deactivate WorkerConfig: {e}")

    def load_state_dict(self, state_dict):
        """Load the state of the dataset from checkpointing."""
        if hasattr(self._data, "restore_state"):
            # Use Megatron-Energon's native restore_state method
            try:
                self._data.restore_state(state_dict["dataset"])
            except Exception as e:
                logger.warning(f"Failed to restore state using restore_state: {e}")
        elif hasattr(self._data, "load_state_dict"):
            # Fallback to load_state_dict if available
            self._data.load_state_dict(state_dict["dataset"])
        else:
            logger.warning(f"Dataset {type(self._data).__name__} does not support state restoration")

    def state_dict(self):
        """Get the state of the dataset for checkpointing."""
        if hasattr(self._data, "save_state"):
            # Use Megatron-Energon's native save_state method
            try:
                state = self._data.save_state()
                return {"dataset": state}
            except Exception as e:
                logger.warning(f"Failed to save state using save_state: {e}")
                return {"dataset": {}}
        elif hasattr(self._data, "state_dict"):
            # Fallback to state_dict if available
            return {"dataset": self._data.state_dict()}
        else:
            # Return empty state dict for datasets that don't support state management
            return {"dataset": {}}

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        if hasattr(self._data, "set_epoch"):
            self._data.set_epoch(epoch)
        elif hasattr(self._data, "reset_state_deep"):
            # For Megatron-Energon datasets, reset state when epoch changes
            try:
                self._data.reset_state_deep()
                logger.debug(f"Reset state for epoch {epoch}")
            except Exception as e:
                logger.warning(f"Failed to reset state for epoch {epoch}: {e}")
        else:
            logger.debug(f"Dataset {type(self._data).__name__} does not support set_epoch or state reset")


def get_length_by_attention_mask_fn(sample):
    return int(sample["attention_mask"].sum())


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
        get_length_fn: Optional[Callable] = get_length_by_attention_mask_fn,
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

        if force_generate_long_sequence:
            raise ValueError("force_generate_long_sequence is not supported yet.")
        self.force_generate_long_sequence = force_generate_long_sequence

        self._buffer = []
        self._buffer_of_refetch_idx = []
        self._buffer_token_count = 0

        self._just_resumed = False  # Flag to indicate if the dataset has just been resumed from a checkpoint, used to skip buffer checks on the first iteration after resume.

    @property
    def save_by_idx(self) -> bool:
        return self._save_by_idx

    @save_by_idx.setter
    def save_by_idx(self, value: bool) -> None:
        if value and not (hasattr(self.dataset, "get_item") and hasattr(self.dataset, "output_refetch_idx")):
            raise ValueError(
                "save_by_idx is True, but dataset does not have get_item method or output_refetch_idx attribute to resume samples in buffers based on idx"
            )
        self._save_by_idx = value
        if hasattr(self.dataset, "output_refetch_idx"):
            self.dataset.output_refetch_idx = value

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

        if not self._just_resumed:
            # Clear buffer state on new iteration unless we just resumed from a checkpoint,
            # in which case we want to keep the buffer contents.
            self._buffer = []
            self._buffer_of_refetch_idx = []
            self._buffer_token_count = 0
        else:
            self._just_resumed = False

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

                samples_to_add = []
                if type(item) is list:
                    samples_to_add = item
                else:
                    samples_to_add = [item]
                for item in samples_to_add:
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

        self._just_resumed = True

    def set_epoch(self, epoch: int):
        """Set the epoch for the upstream dataset.

        Passes the epoch to the upstream dataset if it supports set_epoch.
        Has no direct effect on dynamic batching itself.

        Args:
            epoch: The epoch number to set.
        """
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


def get_data_files(train_path):
    data_files = []
    data_paths = train_path.split(",")
    for data_path in data_paths:
        if data_path.startswith("hdfs://"):
            if not isdir(data_path):
                raise FileNotFoundError(f"Dataset {data_path} not exists.")

            for filename in listdir(data_path):
                from ..utils.helper import get_cache_dir

                data_files.append(hf_hub_download(data_path, os.path.split(filename)[-1], cache_dir=get_cache_dir()))

        elif os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in sorted(os.listdir(data_path))])
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    return data_files, file_extenstion


@DATASET_REGISTRY.register("mapping")
def build_mapping_dataset(
    train_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    source_name: Optional[str] = None,
    **kwargs,
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        train_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        source_name (Optional[str]): source name
    Returns:
        Dataset: mapping dataset
    """
    logger.info_rank0("Start building mapping dataset")
    data_files, file_extenstion = get_data_files(train_path)
    with main_process_first():
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    if transform:
        transform = partial(transform, source_name=source_name)
    return MappingDataset(data=dataset, transform=transform)


@DATASET_REGISTRY.register("iterable")
def build_iterable_dataset(
    train_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    source_name: Optional[str] = None,
    split_by_node: bool = True,
    **kwargs,
) -> "IterableDataset":
    """
    Build iterative dataset.
    Args:
        train_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
        source_name (Optional[str]): source name
    Returns:
        IterableDataset: iterative dataset
    """
    logger.info_rank0("Start building iterative dataset")
    data_files, file_extenstion = get_data_files(train_path)
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    if split_by_node:
        parallel_state = get_parallel_state()
        dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    if transform:
        transform = partial(transform, source_name=source_name)
    return IterativeDataset(dataset, transform=transform)


@DATASET_REGISTRY.register("interleave")
def build_interleave_dataset(
    train_path: str,
    datasets_type: str = "mapping",
    namespace: Literal["train", "test"] = "train",
    transform: Optional[Callable] = None,
    seed: int = 42,
    **kwargs,
):
    """
    Build interleave dataset.
    Args:
        train_path (str): data path
        datasets_type (str): datasets type
        namespace (Literal["train", "test"]): dataset namespace
        transform (Optional[Callable]): transform function
        seed (int): random seed
    Returns:
        InterleavedIterableDataset: interleaved iterable dataset
        or
        InterleavedMappingDataset: interleaved mapping dataset
    """
    logger.info_rank0("Start building interleave dataset")
    multisource_config = parse_multisource_config(train_path)
    logger.info_rank0(f"multisource_config: {multisource_config}")
    sources = multisource_config["sources"]
    schedule = multisource_config["schedule"]
    source_names = multisource_config["names"]

    if len(schedule) > 1 or schedule[0]["schedule_type"] != "const":
        logger.info_rank0("Interleaved dataset only supports const schedule type.")

    weights = schedule[0]["weights"]

    datasets = []
    if datasets_type == "iterable":
        logger.info_rank0("Start building iterable multisource dataset")

        def add_ds_idx_to_iterable(dataset, ds_idx, source_name):
            def trans_example(example):
                return {**example, "ds_idx": ds_idx, "source_name": source_name}

            return dataset.map(trans_example)

        for idx, source in enumerate(sources):
            dataset = build_iterable_dataset(source, namespace=namespace, seed=seed, split_by_node=False)
            ds = dataset._data
            ds = add_ds_idx_to_iterable(ds, idx, source_names[idx])
            datasets.append(ds)

        interleave_dataset = interleave_datasets(datasets=datasets, probabilities=weights, seed=seed)
        # split dataset by node
        parallel_state = get_parallel_state()
        interleave_dataset = split_dataset_by_node(interleave_dataset, parallel_state.dp_rank, parallel_state.dp_size)

        interleave_dataset = InterleavedIterableDataset(
            interleave_dataset,
            transform=transform,
        )
    elif datasets_type == "mapping":
        logger.info_rank0("Start building mapping multisource dataset")

        for idx, source in enumerate(sources):
            dataset = build_mapping_dataset(source, namespace=namespace)
            ds = dataset._data
            ds = ds.add_column("ds_idx", [idx] * len(ds))
            ds = ds.add_column("source_name", [source_names[idx]] * len(ds))
            datasets.append(ds)
        interleave_dataset = InterleavedMappingDataset(
            interleave_datasets(datasets=datasets, probabilities=weights, seed=seed),
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported datasets_type: {datasets_type}")

    return interleave_dataset


@DATASET_REGISTRY.register("energon")
def build_energon_dataset(
    train_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    max_samples_per_sequence: Optional[int] = None,
    virtual_epoch_length: Optional[int] = 0,
    shuffle_buffer_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    **kwargs,
) -> "Dataset":
    """
    Build Megatron-Energon native dataset using the official get_train_dataset function.

    This is the recommended way to use Megatron-Energon datasets as it provides:
    - Automatic length calculation based on virtual_epoch_length
    - Built-in field mapping (txt -> text)
    - Professional streaming dataset support
    - Built-in error handling and performance optimizations

    Args:
        train_path (str): Path to the energon dataset directory
        transform (Optional[Callable]): Transform function to apply to samples
        namespace (Literal["train", "test"]): Dataset namespace (not used for energon)
        max_samples_per_sequence (Optional[int]): Maximum samples per sequence
        virtual_epoch_length (Optional[int]): Virtual epoch length for length calculation
        shuffle_buffer_size (Optional[int]): Shuffle buffer size
        num_workers (Optional[int]): Number of workers (if None, will be auto-detected)

    Returns:
        Dataset: Megatron-Energon native dataset
    """
    try:
        from megatron.energon import WorkerConfig, get_train_dataset
    except ImportError:
        raise ImportError("Megatron-Energon is not installed. Please install it with: pip install megatron-energon")

    logger.info_rank0(f"Start building Megatron-Energon native dataset from {train_path}")
    # Get parallel state for distributed training
    parallel_state = get_parallel_state()

    # Auto-detect number of workers if not provided
    if num_workers is None:
        # Try to get from environment or use a reasonable default
        num_workers = int(os.environ.get("TORCH_DATA_WORKERS", "1"))

    # Create base WorkerConfig
    base_worker_config = WorkerConfig(
        rank=parallel_state.dp_rank, world_size=parallel_state.dp_size, num_workers=num_workers
    )

    # Wrap it with our compatible version
    worker_config = base_worker_config

    logger.info(f"Created WorkerConfig: rank={parallel_state.dp_rank}, world_size={parallel_state.dp_size}")

    if virtual_epoch_length is None:
        # Estimate based on data path - look for .nv-meta/info.json
        try:
            meta_path = os.path.join(train_path, ".nv-meta", "info.json")
            if os.path.exists(meta_path):
                import json

                with open(meta_path) as f:
                    info = json.load(f)
                    if "splits" in info and "train" in info["splits"]:
                        virtual_epoch_length = info["splits"]["train"].get("num_samples", 1000000)
                    else:
                        virtual_epoch_length = 0
        except Exception as e:
            logger.warning(f"Could not determine virtual_epoch_length from metadata: {e}")
        if virtual_epoch_length is None:
            virtual_epoch_length = 0  # Fallback
    logger.info(f"  - max_samples_per_sequence: {max_samples_per_sequence}")
    logger.info(f"  - virtual_epoch_length: {virtual_epoch_length}")
    logger.info(f"  - shuffle_buffer_size: {shuffle_buffer_size}")

    # Get the dataset using Megatron-Energon's official function
    dataset = get_train_dataset(
        path=train_path,
        split_part=namespace,
        worker_config=worker_config,
        batch_size=None,  # No batching at dataset level
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples_per_sequence=max_samples_per_sequence,
        virtual_epoch_length=virtual_epoch_length,
        repeat=True,  # Always repeat for training
    )

    logger.info(f"Dataset type: {type(dataset)} Dataset length: {len(dataset)}")

    # Wrap in our EnergonDataset for Megatron-Energon specific functionality
    return EnergonDataset(dataset, transform)
