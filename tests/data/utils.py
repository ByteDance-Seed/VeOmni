import math
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset, IterableDataset
from transformers import PretrainedConfig

from veomni.distributed.parallel_state import _init_parallel_state
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.trainer.callbacks import GlobalStateCallback, TrainerState
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)

TEST_RESUME_STATE_KEY = "test_resume_position"


def mock_empty_cache() -> None:
    """Patch target for tests that run on CPU but call empty_cache."""
    pass


def setup_test_distributed(args):
    """Initialize a minimal distributed runtime for data tests.

    Returns the device and the ``ParallelState``.
    """
    device_type = get_device_type()
    if device_type != "cpu":
        device_str = f"{device_type}:{args.train.local_rank}"
        get_torch_device().set_device(device_str)
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )

    parallel_state = _init_parallel_state(
        dp_size=args.model.accelerator.dp_size,
        dp_replicate_size=args.model.accelerator.dp_replicate_size,
        dp_shard_size=args.model.accelerator.dp_shard_size,
        tp_size=args.model.accelerator.tp_size,
        pp_size=args.model.accelerator.pp_size,
        cp_size=args.model.accelerator.cp_size,
        ulysses_size=args.model.accelerator.ulysses_size,
        extra_parallel_sizes=args.model.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.model.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.model.accelerator.extra_parallel_names,
        dp_mode=args.model.accelerator.fsdp_config.fsdp_mode,
    )
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    return device, parallel_state


class StepAwareTestGlobalStateCallback(GlobalStateCallback):
    """Test-only global-state callback that preserves per-epoch step position."""

    resume_state_key = TEST_RESUME_STATE_KEY

    def state_dict(self, state: TrainerState):
        curr_step = getattr(state, "curr_step", None)
        if curr_step is None:
            raise AttributeError("StepAwareTestGlobalStateCallback requires TrainerState.curr_step in tests")

        global_state = super().state_dict(state)
        global_state[self.resume_state_key] = {"epoch": state.epoch, "curr_step": curr_step}
        return global_state

    def _restore_position(self, global_state):
        """Resume from the exact cursor rather than rederiving it from ``global_step``.

        The base callback splits ``global_step`` by ``args.train_steps``, which
        is only the true per-epoch step count when batches are fixed-size. Under
        ``dyn_bsz`` it is a token-budget estimate, so the derived epoch/step pair
        drifts and the resumed run consumes a different slice of the data.
        """
        resume_state = global_state.get(self.resume_state_key)
        if resume_state is None:
            super()._restore_position(global_state)
            return
        self.trainer.start_epoch = resume_state["epoch"]
        self.trainer.start_step = resume_state["curr_step"] + 1


class StepAwareResumeGlobalStateCallback(StepAwareTestGlobalStateCallback):
    """Shared global-state callback for step-aware resume tests."""

    def on_step_end(self, state: TrainerState, **kwargs):
        # logger.error(f"[END][rank{self.trainer.args.train.global_rank}][epoch{state.epoch}][step{state.curr_step}][global_step{state.global_step}] metrics {getattr(getattr(self.trainer, 'step_env_metrics', None), 'consume_tokens(M)', None)}")
        if (
            not getattr(self.trainer, "is_resume_train", False)
            and state.epoch == self.trainer.save_epoch
            and state.curr_step == self.trainer.save_step
        ):
            # logger.error(f"save checkpoint {state.global_step} {state.epoch} {state.curr_step} {self.trainer.environ_meter.state_dict()}")
            self.save_global_state(state)
            self.trainer.resume_dcp_path = os.path.join(
                self.trainer.args.train.checkpoint.save_path, f"global_step_{state.global_step}"
            )
            self.trainer.args.train.checkpoint.load_path = self.trainer.resume_dcp_path

    def on_epoch_end(self, state: TrainerState, **kwargs):
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass


class ShardedIterableDataset(IterableDataset):
    """Deterministic iterable dataset with rank/worker sharding and optional shuffle.

    Designed to tested with ``DynamicBatchingSizeDataset`` and ``StatefulDataLoader`` checkpointing:

    * **Deterministic sample generation** – sample ``i`` uses token value ``i + 1``.
      Its length defaults to ``i + 1``, or cycles through ``sample_length_range``
      when configured.
    * **Sharding** – samples are distributed across distributed ranks *and* DataLoader
      workers using a round-robin interleave strategy (rank-major, then worker-minor),
      so each dataloader worker on each rank sees a disjoint, deterministic subset of the data.
    * **Shuffle** – when ``shuffle=True``, a fixed ``torch.randperm`` generated from
      ``seed`` at construction time is used so that the shuffled order is reproducible
      and consistent across checkpoint / resume cycles.
    * **Index output** – when ``output_index_for_resume`` is set to ``True`` (by
      ``DynamicBatchingSizeDataset`` when ``save_by_idx=True``), each ``__iter__``
      yield is a ``(sample_dict, original_index)`` tuple instead of a bare dict,
      allowing the consumer to store the indices instead of the full samples when saving checkpoints,
      and reconstruct the buffer from indices on resume.
    * **State dict** – ``state_dict()`` / ``load_state_dict()`` persist
      ``_current_idx`` so that ``StatefulDataLoader`` can snapshot and restore the
      exact position of the iterator.
    """

    def __init__(
        self,
        size: int = 100,
        shuffle: bool = False,
        seed: int = 42,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None,
        sample_length_range: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            size: Total number of samples in the dataset.
            shuffle: Whether to shuffle the reading order.  Shuffling is performed
                once at construction time using ``seed`` so that it is stable across
                distributed workers.
            seed: Random seed used to generate the permutation when ``shuffle=True``.
            transform: Optional transform applied in ``__getitem__`` / ``get_item``.
                It may return either one sample dict or ``list[dict]``.
            sample_length_range: If set, sample lengths cycle through this inclusive
                range instead of growing with the sample index.
        """
        self.size = size
        self.shuffle = shuffle
        self.seed = seed
        self.transform = transform
        self.sample_length_range = sample_length_range
        self.output_index_for_resume = False  # Will be set by DynamicBatchingSizeDataset if needed
        self._current_idx = -1  # Track current position in iteration
        self._just_resumed = False

        # Generate index permutation at initialization if shuffle is enabled
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self.indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            self.indices = list(range(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Return the transformed dummy sample at position *idx*."""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range [0, {self.size})")

        index = idx + 1
        if self.sample_length_range is None:
            sample_length = index
        else:
            min_length, max_length = self.sample_length_range
            sample_length = min_length + idx % (max_length - min_length + 1)
        input_ids = torch.tensor([index] * sample_length, dtype=torch.long)
        sample = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}
        return self.transform(sample) if self.transform is not None else sample

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
        if not self._just_resumed or self._current_idx < 0:
            self._current_idx = rank * num_workers + worker_id
        else:
            self._just_resumed = False

        for i in range(self._current_idx, len(self.indices), total_workers):
            idx = self.indices[i]
            self._current_idx = i + total_workers
            if self.output_index_for_resume:
                yield (self[idx], idx)
            else:
                yield self[idx]

    def get_item(self, idx):
        """Fetch a single sample by its original dataset index.

        Used by ``DynamicBatchingSizeDataset.load_state_dict()`` to reconstruct
        buffer contents when ``save_by_idx=True``: the saved indices are passed
        back here one-by-one to rebuild the exact pre-checkpoint buffer.

        Args:
            idx: 0-based integer index into the dataset.

        Returns:
            Sample or ``list[dict]`` as returned by ``ShardedIterableDataset.__getitem__``.
        """
        return self[idx]

    def state_dict(self):
        """Save the current iteration state."""
        return {
            "current_idx": self._current_idx,
        }

    def load_state_dict(self, state_dict):
        """Restore the iteration state."""
        self._current_idx = state_dict["current_idx"]
        self._just_resumed = True


class ShardedMappingDataset(Dataset):
    """Map-style sibling of :class:`ShardedIterableDataset`.

    Reuses :class:`ShardedIterableDataset`'s deterministic per-index sample
    generation but is a plain map-style ``Dataset`` consumed purely by ``__len__`` /
    ``__getitem__`` -- the form ``_MapStyleSamplerWrapper`` expects.
    """

    def __init__(
        self,
        size: int = 100,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None,
        sample_length_range: Optional[tuple[int, int]] = None,
    ):
        self.size = size
        self.transform = transform
        self.sample_length_range = sample_length_range

    __len__ = ShardedIterableDataset.__len__
    __getitem__ = ShardedIterableDataset.__getitem__


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
    source_name: str = None,
) -> List[Dict[str, "torch.Tensor"]]:
    tokenized_example = {}
    for k, v in example.items():
        if k == "ds_idx" or k == "source_name":
            continue
        else:
            tokenized_example[k] = torch.tensor(v[:max_seq_len], dtype=torch.long)
    tokenized_example["id"] = torch.tensor(tokenized_example["input_ids"][0].item(), dtype=torch.long)
    return [tokenized_example]


class FakeModel(nn.Module):
    _no_split_modules = ["ffn"]

    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Linear(1, 1)


class FakeModelRuntime(VeOmniModelRuntime):
    """A runtime that installs a toy module instead of loading a real checkpoint.

    Data-pipeline tests only need something the trainer can hold and step; going
    through ``build_foundation_model`` would make them slow and network-bound.
    """

    def build_model(self) -> None:
        self.model = FakeModel().to(get_device_type())
        self.model_config = PretrainedConfig()


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
    for global_batch, global_batch_resume in zip(global_batch_list, global_batch_resume_list, strict=True):
        for micro_batch, micro_batch_resume in zip(global_batch, global_batch_resume, strict=True):
            for key in micro_batch.keys():
                if torch.is_tensor(micro_batch[key]):
                    assert torch.all(micro_batch[key] == micro_batch_resume[key]), (
                        f"rank {dist.get_rank()} key {key} is not equal in micro_batch and micro_batch_resume"
                    )


def compare_metrics(metrics, metrics_resume):
    if (
        metrics is not None
        and metrics_resume is not None
        and "consume_tokens(M)" in metrics
        and "consume_tokens(M)" in metrics_resume
    ):
        assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]
