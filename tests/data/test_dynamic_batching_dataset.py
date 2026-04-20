"""Tests for DynamicBatchingSizeDataset functionality.

This module tests the ``DynamicBatchingSizeDataset`` class using ``ShardedIterableDataset``.
It validates that ``DynamicBatchingSizeDataset`` can properly:

1. Batch samples based on token count (``micro_batch_seq_length``).
2. Handle buffer management with ``ready_for_micro_batch_threshold``.
3. Work with both shuffled and non-shuffled iterable datasets.
4. Drain remaining buffer contents after the upstream dataset is exhausted.
5. Reject invalid construction arguments (``save_by_idx`` without ``get_item``).
6. Save and restore buffer state for exact checkpoint / resume in distributed
   environments, both by storing full samples and by storing only indices.

The test suite includes:

    Unit tests (run without distributed setup, CPU-compatible):
        - ``test_dynamic_batching_basic`` – core batching logic and expected batch
          contents for shuffled and non-shuffled data.
        - ``test_last_batch_on_dataset_end`` – remaining buffer items are yielded
          after upstream exhaustion.
        - ``test_dynamic_batching_without_get_item`` – ``ValueError`` is raised when
          ``save_by_idx=True`` but the dataset lacks ``get_item``.

    End-to-end distributed tests (require ``torchrun`` with 2 processes):
        - ``test_dynamic_batching_dataset_distributed`` – parametrised over
          ``shuffle × save_by_idx × multi_sample_per_iteration`` (5 combinations), verifying that resumed
          batches are byte-for-byte identical to the original run.
"""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, List
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from tools.launch_utils import find_free_port
from torch.utils.data import IterableDataset
from transformers import PretrainedConfig
from utils import (
    FakeModel,
    ShardedIterableDataset,
    StepAwareResumeCheckpointerCallback,
    compare_global_batch,
    compare_items,
    compare_metrics,
    mock_empty_cache,
    setup_test_distributed,
)

from veomni.arguments import VeOmniArguments, parse_args
from veomni.data import build_dataloader
from veomni.data.data_collator import MainCollator
from veomni.data.dataset import DynamicBatchingSizeDataset
from veomni.distributed.parallel_state import get_parallel_state
from veomni.trainer.base import BaseTrainer
from veomni.trainer.callbacks import Callback, EnvironMeterCallback, TrainerState
from veomni.utils import helper
from veomni.utils.device import get_device_type


logger = helper.create_logger(__name__)


MICRO_BATCH_SEQ_LENGTH = 32  # Max tokens per batch
READY_FOR_MICRO_BATCH_THRESHOLD = 10  # Minimum samples in buffer before batching
DATASET_SIZE = 50


def get_length_fn(item):
    return item["attention_mask"].sum()


def single_sample_transform(sample: Dict[str, torch.Tensor]):
    return [sample]


def multi_sample_transform(sample: Dict[str, torch.Tensor]):
    secondary_sample = copy.deepcopy(sample)
    secondary_sample["input_ids"] = secondary_sample["input_ids"] + 1000
    secondary_sample["labels"] = secondary_sample["labels"] + 1000
    return [sample, secondary_sample]


# Fixtures
@pytest.fixture
def setup_dynamic_batching_dataset():
    """Fixture to create DynamicBatchingSizeDataset with standard configuration.

    Returns:
        A tuple of (iterable_dataset, dynamic_ds)
    """
    iterable_dataset = ShardedIterableDataset(size=DATASET_SIZE, shuffle=False)

    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_dataset,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=MainCollator(),
        get_length_fn=get_length_fn,
    )

    return iterable_dataset, dynamic_ds


# Unit tests (can run without distributed setup)
@pytest.mark.parametrize(
    "shuffle,seed",
    [
        (False, 42),
        (True, 42),
    ],
)
def test_dynamic_batching_basic(shuffle, seed):
    """Unit test for DynamicBatchingSizeDataset basic functionality.

    Tests the core dynamic batching logic without requiring distributed setup:
    - Creates batches based on token count threshold
    - Properly buffers samples before batching
    - Collates samples using DataCollatorWithPositionIDs
    - Yields batches with reasonable token counts

    This test can run on CPU and does not require GPUs.

    Args:
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.
    """
    # Create a simple dataset
    iterable_ds = ShardedIterableDataset(size=DATASET_SIZE, shuffle=shuffle, seed=seed)

    # Create data collator
    collator = MainCollator()

    # Create dynamic batching dataset
    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_ds,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=collator,
        get_length_fn=get_length_fn,
    )

    # Iterate and check batches
    batch_count = 0
    # the total length of each batch cannot be greater than MICRO_BATCH_SEQ_LENGTH(32)
    # Expected input_ids for shuffle=False
    expected_input_ids_no_shuffle = [
        [
            i for i in range(1, 8) for _ in range(i)
        ],  # [1, 2,2, 3,3,3, 4,4,4,4, 5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7,7] = 28 tokens
        [i for i in range(8, 11) for _ in range(i)],  # [8]*8 + [9]*9 + [10]*10 = 27 tokens
        [i for i in range(11, 13) for _ in range(i)],  # [11]*11 + [12]*12 = 23 tokens
        [i for i in range(13, 15) for _ in range(i)],  # [13]*13 + [14]*14 = 27 tokens
        [i for i in range(15, 17) for _ in range(i)],  # [15]*15 + [16]*16 = 31 tokens
    ]

    # Expected input_ids for shuffle=True (seed=42)
    # Samples longer than MICRO_BATCH_SEQ_LENGTH (32) are skipped (force_generate_long_sequence=False).
    expected_input_ids_shuffle = [
        [18] * 18 + [1] + [9] * 9 + [4] * 4,  # 32 tokens
        [31] * 31,  # 31 tokens
        [21] * 21 + [3] * 3 + [2] * 2,  # 26 tokens
        [26] * 26 + [6] * 6,  # 32 tokens
        [19] * 19 + [7] * 7,  # 26 tokens
    ]

    expected_input_ids = expected_input_ids_shuffle if shuffle else expected_input_ids_no_shuffle

    for batch in dynamic_ds:
        batch_count += 1
        # Each batch should be a dict (after collation)
        assert isinstance(batch, dict)
        assert "attention_mask" in batch
        assert batch["attention_mask"].sum() > 0

        # Calculate total tokens in batch
        total_tokens = batch["attention_mask"].sum()

        # Check expected input_ids
        assert batch["input_ids"].tolist()[0] == expected_input_ids[batch_count - 1], (
            f"Batch {batch_count} input_ids mismatch. Expected: {expected_input_ids[batch_count - 1]}, Got: {batch['input_ids'].tolist()[0]}"
        )

        # check buffer size
        buffer_length = len(dynamic_ds._buffer)
        assert buffer_length <= READY_FOR_MICRO_BATCH_THRESHOLD, (
            f"Buffer has {buffer_length} samples, exceeds ready_for_micro_batch_threshold {READY_FOR_MICRO_BATCH_THRESHOLD}"
        )

        # check if any remaining item in buffer can be added to the batch
        if buffer_length > 0:
            for item in dynamic_ds._buffer:
                assert item[1] + total_tokens > MICRO_BATCH_SEQ_LENGTH, (
                    f"Buffer item {item[0]} has {item[1]} tokens, it can still fit into the batch"
                )

        if batch_count >= 5:  # Just test a few batches
            break

    assert batch_count > 0, "Should produce at least one batch"
    logger.info(f"test_dynamic_batching_basic (shuffle={shuffle}) passed! Produced {batch_count} batches")


def test_last_batch_on_dataset_end(setup_dynamic_batching_dataset):
    """Test that remaining buffer items are yielded when dataset ends.

    This test verifies that when the upstream dataset ends (StopIteration),
    DynamicBatchingSizeDataset will continue to yield batches from the remaining
    buffer items until the buffer is empty, even if they don't meet the normal
    threshold conditions.
    """
    iterable_dataset, dynamic_ds = setup_dynamic_batching_dataset

    iterator = iter(dynamic_ds)
    batch_count = 0
    found_last_batch_scenario = False

    while True:
        # Check upstream dataset state before getting next batch
        upstream_exhausted = iterable_dataset._current_idx >= len(iterable_dataset.indices)
        buffer_size = len(dynamic_ds._buffer)
        buffer_tokens = dynamic_ds._buffer_token_count

        # Check if buffer meets normal threshold conditions
        buffer_meets_threshold = (
            buffer_size >= READY_FOR_MICRO_BATCH_THRESHOLD and buffer_tokens >= MICRO_BATCH_SEQ_LENGTH
        )

        if upstream_exhausted and not buffer_meets_threshold and buffer_size > 0:
            # Try to get a batch - should succeed even though buffer doesn't meet threshold
            try:
                batch = next(iterator)
                batch_count += 1
                total_tokens = batch["attention_mask"].sum()
                if total_tokens > 0:
                    found_last_batch_scenario = True
            except StopIteration as e:
                raise AssertionError("Expected to get a batch from remaining buffer, but got StopIteration") from e
        else:
            # Normal batch retrieval
            try:
                batch = next(iterator)
                batch_count += 1
            except StopIteration:
                break

    assert found_last_batch_scenario, (
        "Did not find the scenario where upstream is exhausted but buffer doesn't meet threshold"
    )


def test_dynamic_batching_without_get_item():
    """Test DynamicBatchingSizeDataset initialization without get_item provided.

    Tests that DynamicBatchingSizeDataset cannot be initialized with save_by_idx=True
    when the dataset doesn't have get_item method.
    """

    class IterableDatasetWithoutGetItem(IterableDataset):
        def __iter__(self):
            for i in range(10):
                yield {"input_ids": [i] * i, "attention_mask": [1] * i}

    iterable_dataset = IterableDatasetWithoutGetItem()

    # Test with save_by_idx=True (should raise ValueError)
    with pytest.raises(ValueError, match="save_by_idx is True, but dataset does not have get_item method"):
        _ = DynamicBatchingSizeDataset(
            dataset=iterable_dataset,
            micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
            ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
            dynamic_batching_collate_fn=MainCollator(),
            get_length_fn=get_length_fn,
            save_by_idx=True,
        )


@pytest.mark.parametrize("save_by_idx", [False, True])
def test_save_load_state_dict(save_by_idx):
    """Unit test for DynamicBatchingSizeDataset state_dict and load_state_dict.

    Iterates 2 batches, saves the dataset state, then verifies that a fresh
    dataset restored from that state produces identical subsequent batches.

    Args:
        save_by_idx: Whether to save the buffer by index (True) or by full
            sample tensors (False).
    """
    iterable_ds = ShardedIterableDataset(size=DATASET_SIZE, shuffle=False)

    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_ds,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=MainCollator(),
        get_length_fn=get_length_fn,
        save_by_idx=save_by_idx,
    )

    iterator = iter(dynamic_ds)

    # Consume 2 batches before saving state
    for _ in range(2):
        next(iterator)

    state = dynamic_ds.state_dict()

    # Collect remaining batches from the original iterator
    batches_original = list(iterator)

    # Restore state into a fresh dataset instance
    iterable_ds2 = ShardedIterableDataset(size=DATASET_SIZE, shuffle=False)

    dynamic_ds2 = DynamicBatchingSizeDataset(
        dataset=iterable_ds2,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=MainCollator(),
        get_length_fn=get_length_fn,
        save_by_idx=save_by_idx,
    )
    dynamic_ds2.load_state_dict(state)

    batches_resumed = list(dynamic_ds2)

    assert len(batches_original) == len(batches_resumed), (
        f"Batch count mismatch after resume: original={len(batches_original)}, resumed={len(batches_resumed)}"
    )
    for i, (orig, resumed) in enumerate(zip(batches_original, batches_resumed)):
        for key in orig:
            if torch.is_tensor(orig[key]):
                assert torch.all(orig[key] == resumed[key]), f"Batch {i} key '{key}' mismatch after resume"

    logger.info(f"test_save_load_state_dict (save_by_idx={save_by_idx}) passed!")


@pytest.mark.parametrize(
    "shuffle,save_by_idx,multi_sample_per_iteration",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_dynamic_batching_dataset_distributed(shuffle, save_by_idx, multi_sample_per_iteration):
    """Test DynamicBatchingSizeDataset in distributed setting.

    Runs _main_distributed_test() by torchrun with or without data shuffling
    and with or without save_by_idx for checkpoint buffer saving.

    Args:
        shuffle: Whether to enable data shuffling.
        save_by_idx: Whether to save buffer by index for checkpointing.
        multi_sample_per_iteration: Whether one dataset iteration emits two samples.

    Raises:
        subprocess.CalledProcessError: If the distributed test fails.
    """
    command = build_command(
        shuffle=shuffle,
        save_by_idx=save_by_idx,
        multi_sample_per_iteration=multi_sample_per_iteration,
    )
    # Pass current environment to subprocess to inherit virtual environment
    result = subprocess.run(command, check=True, env=os.environ.copy())
    assert result.returncode == 0


def build_command(shuffle=True, save_by_idx=True, multi_sample_per_iteration=False):
    """Build torchrun command for distributed testing.

    Constructs a command to launch the test script with torchrun for
    distributed execution with 2 processes.

    Args:
        shuffle: Whether to enable data shuffling.
        save_by_idx: Whether to save buffer by index for checkpointing.
        multi_sample_per_iteration: Whether one dataset iteration emits two samples.

    Returns:
        list: Command arguments for subprocess.run().
    """
    port = find_free_port()

    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"--master_port={port}",
        "tests/data/test_dynamic_batching_dataset.py",
        "--model.config_path=test",
        "--data.train_path=None",
        "--data.train_size=2000",
        f"--data.dyn_bsz_buffer_size={READY_FOR_MICRO_BATCH_THRESHOLD}",
        "--data.max_seq_len=16",
        "--data.dataloader.num_workers=2",
        "--data.dataloader.drop_last=false",
        "--train.micro_batch_size=2",
        f"--shuffle={str(shuffle).lower()}",
        "--train.global_batch_size=16",
        "--train.accelerator.fsdp_config.fsdp_mode=ddp",
        "--train.checkpoint.manager=dcp",
        "--train.checkpoint.output_dir=.tests/cache",
        "--train.dyn_bsz=true",
        "--train.dyn_bsz_runtime=worker",
        f"--save_by_idx={str(save_by_idx).lower()}",
        f"--multi_sample_per_iteration={str(multi_sample_per_iteration).lower()}",
        "--train.seed=42",
    ]
    return command


class TrainerTest(BaseTrainer):
    gt_data_list: List[List[Dict[str, Any]]] = []
    pred_data_list: List[List[Dict[str, Any]]] = []
    golden_env_metrics: Dict[str, Any] = {}
    resume_dcp_path: str

    save_epoch, save_step = 1, 0
    is_resume_train: bool = False

    def __init__(
        self,
        args: VeOmniArguments,
        shuffle: bool,
        save_by_idx: bool,
        multi_sample_per_iteration: bool,
    ):
        self.shuffle = shuffle
        self.save_by_idx = save_by_idx
        self.multi_sample_per_iteration = multi_sample_per_iteration
        super().__init__(args)

    def _setup(self):
        self.device = setup_test_distributed(self.args)

    def _freeze_model_module(self):
        pass

    def _build_model(self):
        self.model = FakeModel().to(get_device_type())
        self.model_config = PretrainedConfig()

    def _build_model_assets(self):
        self.model_assets = [self.model_config]

    def _build_data_transform(self):
        pass

    def _build_dataset(self):
        args = self.args
        transform = multi_sample_transform if self.multi_sample_per_iteration else single_sample_transform
        self.train_dataset = ShardedIterableDataset(
            size=DATASET_SIZE,
            shuffle=self.shuffle,
            seed=args.train.seed,
            transform=transform,
        )
        effective_dataset_size = DATASET_SIZE * (2 if self.multi_sample_per_iteration else 1)
        args.compute_train_steps(effective_dataset_size)
        args.train.num_train_epochs = 2
        self.train_steps = args.train_steps

    def _build_dataloader(self):
        args = self.args
        dataloader_kwargs = asdict(args.data.dataloader)
        dataloader_type = dataloader_kwargs.pop("type")
        self.train_dataloader = build_dataloader(
            dataloader_type=dataloader_type,
            dataset=self.train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train_steps,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz=args.train.dyn_bsz,
            dyn_bsz_runtime=args.train.dyn_bsz_runtime,
            dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
            dyn_bsz_dataset_save_by_idx=self.save_by_idx,
            seed=args.train.seed,
            collate_fn=self.collate_fn,
            **dataloader_kwargs,
        )

    def _build_parallelized_model(self):
        self.model.train()

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.train.optimizer.lr)

    def _build_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def _build_training_context(self):
        self.model_fwd_context = nullcontext()
        self.model_bwd_context = nullcontext()

    def _init_callbacks(self):
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.checkpointer_callback = StepAwareResumeCheckpointerCallback(self)
        self.check_callback = CheckCallback(self)
        self.state = TrainerState()

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.check_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.check_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.state.curr_step = self.start_step - 1
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.check_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):
        self.environ_meter_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.check_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.check_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs) -> None:
        try:
            self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        except AttributeError as e:
            # Skip metrics on CPU (torch.cpu has no attribute 'get_device_name')
            logger.warning(f"[rank{self.args.train.global_rank}] Skipping metrics: {e}")
            self.step_env_metrics = {}
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.check_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        self.state.global_step += 1
        self.state.curr_step += 1
        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        self.on_step_end(loss=0.0, loss_dict={}, grad_norm=0.0)

    def resume_train(self):
        self.is_resume_train = True
        super().train()

    def destroy_distributed(self):
        if self.is_resume_train:
            super().destroy_distributed()


class CheckCallback(Callback):
    trainer: TrainerTest

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        # micro_batch_output = [list(set(d['input_ids'].tolist()[0])) for d in micro_batches]
        # logger.error(f"[BEGIN][rank{self.trainer.args.train.global_rank}][epoch{state.epoch}][step{state.curr_step}][global_step{state.global_step}] metrics {  getattr(getattr(self.trainer, 'step_env_metrics', None), 'consume_tokens(M)', None)} micro_batches: {micro_batch_output}")
        if state.global_step == 1:
            helper.print_example(example=micro_batches[0], rank=self.trainer.args.train.local_rank)
            assert 1 < len(micro_batches) <= 4, f"Unexpected micro batch count: {len(micro_batches)}"
            if get_parallel_state().sp_enabled:
                assert (
                    micro_batches[0]["input_ids"].shape[-1] * get_parallel_state().sp_size
                    == micro_batches[0]["attention_mask"].shape[-1]
                )
                assert compare_items(
                    micro_batches[0]["attention_mask"],
                    rank=get_parallel_state().sp_rank,
                    group_size=get_parallel_state().sp_size,
                    group=get_parallel_state().sp_group,
                )

        if state.epoch == self.trainer.save_epoch and state.curr_step > self.trainer.save_step:
            if not self.trainer.is_resume_train:
                self.trainer.gt_data_list.append(micro_batches)
            else:
                self.trainer.pred_data_list.append(micro_batches)

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.is_resume_train:
            assert len(self.trainer.gt_data_list) == len(self.trainer.pred_data_list), (
                f"Batch count mismatch: gt={len(self.trainer.gt_data_list)}, pred={len(self.trainer.pred_data_list)}"
            )

            for i, (gt_batch, pred_batch) in enumerate(zip(self.trainer.gt_data_list, self.trainer.pred_data_list)):
                assert len(gt_batch) == len(pred_batch), f"Micro batch count mismatch at batch {i}"

            compare_global_batch(self.trainer.gt_data_list, self.trainer.pred_data_list)
            compare_metrics(self.trainer.step_env_metrics, self.trainer.golden_env_metrics)

            logger.info(f"[rank{self.trainer.args.train.global_rank}] ✅ All batches matched successfully!")
        else:
            self.trainer.golden_env_metrics = copy.deepcopy(self.trainer.step_env_metrics)


def _main_distributed_test():
    """Entry point for the distributed test launched by ``torchrun``.

    It wraps ``_run_distributed_test()` and in the testing it is supposed to be
    triggered by test_dynamic_batching_dataset_distributed().
    """
    # Patch both the source function and DCP's imported alias to avoid CPU AttributeError.
    with (
        patch("veomni.utils.device.empty_cache", mock_empty_cache),
    ):
        _parser = argparse.ArgumentParser()
        _parser.add_argument("--shuffle", type=lambda x: x.lower() == "true", default=True)
        _parser.add_argument("--save_by_idx", type=lambda x: x.lower() == "true", default=True)
        _parser.add_argument("--multi_sample_per_iteration", type=lambda x: x.lower() == "true", default=False)
        test_args, remaining_argv = _parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv

        args = parse_args(VeOmniArguments)
        trainer = TrainerTest(
            args,
            shuffle=test_args.shuffle,
            save_by_idx=test_args.save_by_idx,
            multi_sample_per_iteration=test_args.multi_sample_per_iteration,
        )
        trainer.train()
        assert trainer.args.train.checkpoint.load_path is not None
        trainer.resume_train()


if __name__ == "__main__":
    _main_distributed_test()
