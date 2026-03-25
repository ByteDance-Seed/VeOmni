import math
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import IterableDataset

from veomni.distributed.parallel_state import init_parallel_state
from veomni.trainer.callbacks import CheckpointerCallback, TrainerState
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.helper import get_cache_dir


logger = helper.create_logger(__name__)

TEST_RESUME_STATE_KEY = "test_resume_position"


def mock_empty_cache() -> None:
    """Patch target for tests that run on CPU but call empty_cache."""
    pass


def setup_test_distributed(args) -> torch.device:
    """Initialize a minimal distributed runtime for data tests."""
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

    init_parallel_state(
        dp_size=args.train.accelerator.dp_size,
        dp_replicate_size=args.train.accelerator.dp_replicate_size,
        dp_shard_size=args.train.accelerator.dp_shard_size,
        tp_size=args.train.accelerator.tp_size,
        pp_size=args.train.accelerator.pp_size,
        cp_size=args.train.accelerator.cp_size,
        ulysses_size=args.train.accelerator.ulysses_size,
        extra_parallel_sizes=args.train.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.train.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.train.accelerator.extra_parallel_names,
        dp_mode=args.train.accelerator.fsdp_config.fsdp_mode,
    )
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    return device


class StepAwareTestCheckpointerCallback(CheckpointerCallback):
    """Test-only checkpoint callback that preserves per-epoch step position."""

    resume_state_key = TEST_RESUME_STATE_KEY

    def _load_checkpoint(self):
        args = self.trainer.args
        if args.train.checkpoint.load_path is None:
            return

        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:
            self.trainer.checkpointer.save_future.result()

        self.trainer.checkpointer.load(args.train.checkpoint.load_path, state)

        extra_state = state["extra_state"]
        self.trainer.state.global_step = extra_state["global_step"]

        resume_state = extra_state.get(self.resume_state_key)
        if resume_state is not None:
            self.trainer.start_epoch = resume_state["epoch"]
            self.trainer.start_step = resume_state["curr_step"] + 1
        else:
            self.trainer.start_epoch = self.trainer.state.global_step // args.train_steps
            self.trainer.start_step = self.trainer.state.global_step % args.train_steps

        self.trainer.lr_scheduler.load_state_dict(extra_state["lr_scheduler"])

        if self.trainer.train_dataloader is not None and extra_state.get("train_dataloader") is not None:
            self.trainer.train_dataloader.load_state_dict(extra_state["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(extra_state["environ_meter"])
        torch.set_rng_state(extra_state["torch_rng_state"])
        if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
            iter(self.trainer.train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        args = self.trainer.args
        curr_step = getattr(state, "curr_step", None)
        if curr_step is None:
            raise AttributeError("StepAwareTestCheckpointerCallback requires TrainerState.curr_step in tests")

        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        ckpt_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {
                "global_step": state.global_step,
                self.resume_state_key: {
                    "epoch": state.epoch,
                    "curr_step": curr_step,
                },
                "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
                "train_dataloader": (
                    self.trainer.train_dataloader.state_dict() if self.trainer.train_dataloader is not None else None
                ),
                "environ_meter": self.trainer.environ_meter.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        self.trainer.checkpointer.save(save_checkpoint_path, ckpt_state, save_async=args.train.checkpoint.save_async)

        helper.empty_cache()
        dist.barrier()

        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


class StepAwareResumeCheckpointerCallback(StepAwareTestCheckpointerCallback):
    """Shared checkpoint callback for step-aware resume tests."""

    def on_step_end(self, state: TrainerState, **kwargs):
        # logger.error(f"[END][rank{self.trainer.args.train.global_rank}][epoch{state.epoch}][step{state.curr_step}][global_step{state.global_step}] metrics {getattr(getattr(self.trainer, 'step_env_metrics', None), 'consume_tokens(M)', None)}")
        if (
            not getattr(self.trainer, "is_resume_train", False)
            and state.epoch == self.trainer.save_epoch
            and state.curr_step == self.trainer.save_step
        ):
            # logger.error(f"save checkpoint {state.global_step} {state.epoch} {state.curr_step} {self.trainer.environ_meter.state_dict()}")
            self._save_checkpoint(state)
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

    * **Deterministic sample generation** – sample at 0-based index ``i`` contains
      **i + 1** tokens, each with value ``i + 1``.
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

    def __init__(self, size: int = 100, shuffle: bool = False, seed: int = 42):
        """
        Args:
            size: Total number of samples in the dataset.
            shuffle: Whether to shuffle the reading order.  Shuffling is performed
                once at construction time using ``seed`` so that it is stable across
                distributed workers.
            seed: Random seed used to generate the permutation when ``shuffle=True``.
        """
        self.size = size
        self.shuffle = shuffle
        self.seed = seed
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
        """Return the dummy sample at position *idx*."""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range [0, {self.size})")

        index = idx + 1
        input_ids = torch.tensor([index] * index, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}

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
            Sample as returned by ``ShardedIterableDataset.__getitem__``.
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
                    assert torch.all(micro_batch[key] == micro_batch_resume[key])


def compare_metrics(metrics, metrics_resume):
    if (
        metrics is not None
        and metrics_resume is not None
        and "consume_tokens(M)" in metrics
        and "consume_tokens(M)" in metrics_resume
    ):
        assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]
