import math
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset, IterableDataset

from veomni.utils import helper
from veomni.utils.device import get_device_type
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)


class DummyMappingDataset(Dataset):
    """Mapping-style dataset that generates deterministic dummy samples by index.

    * Sample at 0-based index ``i`` contains **i + 1** tokens, each with value
      ``i + 1``.  For example index 0 → ``[1]``, index 4 → ``[5, 5, 5, 5, 5]``.
    """

    def __init__(self, size: int = 100):
        """
        Args:
            size: Total number of samples in the dataset.
        """
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Return the dummy sample at position *idx*.

        Args:
            idx: 0-based integer index into the dataset.

        Returns:
            dict with keys:

            * ``"input_ids"`` – 1-D ``LongTensor`` of length ``idx + 1``, filled
              with the scalar value ``idx + 1``.
            * ``"attention_mask"`` – all-ones tensor of the same shape.
            * ``"labels"`` – clone of ``input_ids``.

        Raises:
            IndexError: If ``idx`` is outside ``[0, size)``.
        """
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range [0, {self.size})")

        # Follow the same generation pattern: index + 1
        index = idx + 1
        input_ids = torch.tensor([index] * index, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}


class DummyIterableDataset(IterableDataset):
    """Iterable wrapper around ``DummyMappingDataset`` with built-in sharding and optional shuffle.

    Designed to tested with ``DynamicBatchingSizeDataset`` and ``StatefulDataLoader`` checkpointing:

    * **Sharding** – samples are distributed across distributed ranks *and* DataLoader
      workers using a round-robin interleave strategy (rank-major, then worker-minor),
      so each dataloader worker on each rank sees a disjoint, deterministic subset of the data.
    * **Shuffle** – when ``shuffle=True``, a fixed ``torch.randperm`` generated from
      ``seed`` at construction time is used so that the shuffled order is reproducible
      and consistent across checkpoint / resume cycles.
    * **Index output** – when ``output_refetch_idx`` is set to ``True`` (by
      ``DynamicBatchingSizeDataset`` when ``save_by_idx=True``), each ``__iter__``
      yield is a ``(sample_dict, original_index)`` tuple instead of a bare dict,
      allowing the consumer to store the indices instead of the full samples when saving checkpoints,
      and reconstruct the buffer from indices on resume.
    * **State dict** – ``state_dict()`` / ``load_state_dict()`` persist
      ``_current_idx`` so that ``StatefulDataLoader`` can snapshot and restore the
      exact position of the iterator.
    """

    def __init__(self, mapping_dataset: DummyMappingDataset, shuffle: bool = False, seed: int = 42):
        """
        Args:
            mapping_dataset: The upstream ``DummyMappingDataset`` to read from.
            shuffle: Whether to shuffle the reading order.  Shuffling is performed
                once at construction time using ``seed`` so that it is stable across
                distributed workers.
            seed: Random seed used to generate the permutation when ``shuffle=True``.
        """
        self.mapping_dataset = mapping_dataset
        self.shuffle = shuffle
        self.seed = seed
        self.output_refetch_idx = False  # Will be set by DynamicBatchingSizeDataset if needed
        self._current_idx = 0  # Track current position in iteration

        # Generate index permutation at initialization if shuffle is enabled
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self.indices = torch.randperm(len(self.mapping_dataset), generator=generator).tolist()
        else:
            self.indices = list(range(len(self.mapping_dataset)))

    def __iter__(self):
        """Iterate through the dataset in order or shuffled order with rank and worker sharding.

        Sharding strategy:
        - First shard by rank (for distributed training)
        - Then shard by worker (for multi-worker DataLoader)
        - Each rank+worker combination gets a unique subset of data

        Example with 2 ranks, 2 workers, 8 samples:
        - Rank 0, Worker 0: indices 0, 4
        - Rank 0, Worker 1: indices 1, 5
        - Rank 1, Worker 0: indices 2, 6
        - Rank 1, Worker 1: indices 3, 7
        """
        import torch.distributed as dist

        # Get worker info for multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()

        # Get distributed info
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Calculate which indices this rank+worker should process
        if worker_info is not None:
            # Multi-worker case: shard by rank first, then by worker
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        total_workers = world_size * num_workers
        start_idx = self._current_idx if self._current_idx > 0 else rank * num_workers + worker_id

        for i in range(start_idx, len(self.indices), total_workers):
            idx = self.indices[i]
            self._current_idx = i + total_workers
            if self.output_refetch_idx:
                yield (self.mapping_dataset[idx], idx)
            else:
                yield self.mapping_dataset[idx]

    def get_item(self, idx):
        """Fetch a single sample by its original dataset index.

        Used by ``DynamicBatchingSizeDataset.load_state_dict()`` to reconstruct
        buffer contents when ``save_by_idx=True``: the saved indices are passed
        back here one-by-one to rebuild the exact pre-checkpoint buffer.

        Args:
            idx: 0-based integer index into the underlying ``DummyMappingDataset``.

        Returns:
            Sample as returned by ``DummyMappingDataset.__getitem__``.
        """
        return self.mapping_dataset[idx]

    def state_dict(self):
        """Save the current iteration state."""
        return {
            "current_idx": self._current_idx,
        }

    def load_state_dict(self, state_dict):
        """Restore the iteration state."""
        self._current_idx = state_dict["current_idx"]


class DummyDataset:
    def __init__(self, size=100, num_shard=2, dataset_name: str = "test_dataset") -> None:
        self.size = size
        self.num_shard = num_shard

        self.save_path = get_cache_dir(f"./{dataset_name}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.build_dummy_dataset()

        if dist.is_initialized():
            dist.barrier()

    def generate_data(self, index_list: List):
        for index in index_list:
            input_ids = [index + 1] * (index + 1)
            yield {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": input_ids}

    def build_dummy_dataset(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        batch_len = math.ceil(self.size / self.num_shard)
        print(f"Total length: {self.size}, batch length: {batch_len}")

        index = 0
        for i in range(0, self.size, batch_len):
            print(f"Generating {index}th parquet file")
            ds = HuggingFaceDataset.from_generator(
                self.generate_data,
                gen_kwargs={"index_list": list(range(i + 1, i + batch_len + 1))},
                keep_in_memory=True,
                num_proc=1,
            )
            ds.to_parquet(os.path.join(self.save_path, f"{index}.parquet"))
            index += 1

    def clean_cache(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if os.path.exists(self.save_path):
                os.system(f"rm -rf {self.save_path}")

    def __del__(self):
        self.clean_cache()


def process_dummy_example(
    example: Dict[str, Any],
    max_seq_len: int,
    rmpad_with_pos_ids: bool = False,
    source_name: str = None,
) -> List[Dict[str, "torch.Tensor"]]:
    tokenized_example = {}
    for k, v in example.items():
        if k == "ds_idx" or k == "source_name":
            continue
        else:
            tokenized_example[k] = torch.tensor(v[:max_seq_len], dtype=torch.long)
    if rmpad_with_pos_ids:  # precompute position_ids
        tokenized_example["position_ids"] = torch.arange(0, len(tokenized_example["input_ids"]), dtype=torch.long)
    return [tokenized_example]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Linear(1, 1)


def compare_items(item, rank, group_size, group):
    item = item.to(get_device_type())
    item_list = [torch.empty_like(item) for _ in range(group_size)]

    dist.all_gather(item_list, item, group=group)

    for i in range(0, group_size):
        if not torch.equal(item, item_list[i]):
            logger.info(f"[rank{rank}]: group_rank {i} item is not equal to item {rank}")
            return False

    return True


def compare_global_batch(global_batch_list, global_batch_resume_list):
    for global_batch, global_batch_resume in zip(global_batch_list, global_batch_resume_list):
        for micro_batch, micro_batch_resume in zip(global_batch, global_batch_resume):
            for key in micro_batch.keys():
                if torch.is_tensor(micro_batch[key]):
                    assert torch.all(micro_batch[key] == micro_batch_resume[key])


def compare_metrics(metrics, metrics_resume):
    assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]
