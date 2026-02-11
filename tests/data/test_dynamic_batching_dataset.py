"""Tests for DynamicBatchingSizeDataset functionality.

This module tests the DynamicBatchingSizeDataset class using DummyIterableDataset
and DummyMappingDataset. It validates that DynamicBatchingSizeDataset can properly:

1. Batch samples based on token count (micro_batch_seq_length)
2. Handle buffer management with ready_for_micro_batch_threshold
3. Work with both shuffled and non-shuffled iterable datasets
4. Support state_dict save/load for checkpointing in distributed environments

The test suite includes:
    - Unit tests that can run without distributed setup:
        - test_dynamic_batching_basic
    - End-to-end tests that require multi-GPU distributed environments:
        - test_dynamic_batching_dataset_shuffled
        - test_dynamic_batching_dataset_no_shuffle
"""

import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import IterableDataset

from utils import DummyIterableDataset, DummyMappingDataset


# Patch empty_cache to avoid AttributeError on CPU
def _mock_empty_cache():
    """Mock empty_cache that does nothing on CPU."""
    pass

# Import logger first for unit tests
from veomni.utils import helper
logger = helper.create_logger(__name__)

import torch.distributed as dist
from transformers import PretrainedConfig
from utils import FakeModel

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.checkpoint import build_checkpointer
from veomni.data.data_collator import DataCollatorWithPositionIDs
from veomni.data.dynamic_batching import DynamicBatchingSizeDataset
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

# Import find_free_port from tests/tools/launch_utils.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.launch_utils import find_free_port

MICRO_BATCH_SEQ_LENGTH = 32  # Max tokens per batch
READY_FOR_MICRO_BATCH_THRESHOLD = 10  # Minimum samples in buffer before batching
DATASET_SIZE = 50


def get_length_fn(item):
    return item["attention_mask"].sum()


# Fixtures
@pytest.fixture
def setup_dynamic_batching_dataset():
    """Fixture to create DynamicBatchingSizeDataset with standard configuration.

    Returns:
        A tuple of (mapping_dataset, iterable_dataset, dynamic_ds)
    """
    mapping_dataset = DummyMappingDataset(size=DATASET_SIZE)
    iterable_dataset = DummyIterableDataset(mapping_dataset, shuffle=False)

    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_dataset,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=DataCollatorWithPositionIDs(),
        get_length_fn=get_length_fn,
    )

    return mapping_dataset, iterable_dataset, dynamic_ds


# Unit tests (can run without distributed setup)
@pytest.mark.parametrize("shuffle,seed", [
    (False, 42),
    (True, 42),
])
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
    mapping_ds = DummyMappingDataset(size=DATASET_SIZE)
    iterable_ds = DummyIterableDataset(mapping_ds, shuffle=shuffle, seed=seed)

    # Create data collator
    collator = DataCollatorWithPositionIDs()

    # Create dynamic batching dataset
    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_ds,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=collator,
        save_by_idx=False,
        get_length_fn=get_length_fn,
    )

    # Iterate and check batches
    batch_count = 0
    # the total length of each batch cannot be greater than MICRO_BATCH_SEQ_LENGTH(32)
    # Expected input_ids for shuffle=False
    expected_input_ids_no_shuffle = [
        [i for i in range(1, 8) for _ in range(i)],  # [1, 2,2, 3,3,3, 4,4,4,4, 5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7,7] = 28 tokens
        [i for i in range(8, 11) for _ in range(i)],  # [8]*8 + [9]*9 + [10]*10 = 27 tokens
        [i for i in range(11, 13) for _ in range(i)],  # [11]*11 + [12]*12 = 23 tokens
        [i for i in range(13, 15) for _ in range(i)],  # [13]*13 + [14]*14 = 27 tokens
        [i for i in range(15, 17) for _ in range(i)],  # [15]*15 + [16]*16 = 31 tokens
    ]

    # Expected input_ids for shuffle=True (seed=42)
    # Note: force_generate_long_sequence=True allows batches to exceed micro_batch_seq_length
    # when the buffer is empty and only has one long sample
    expected_input_ids_shuffle = [
        [43]*43,  # 43 tokens (exceeds 32 due to force_generate_long_sequence)
        [18]*18 + [1] + [9]*9,  # 28 tokens
        [31]*31,  # 31 tokens
        [35]*35,  # 35 tokens (exceeds 32 due to force_generate_long_sequence)
        [21]*21 + [4]*4 + [3]*3 + [2]*2,  # 30 tokens
    ]

    expected_input_ids = expected_input_ids_shuffle if shuffle else expected_input_ids_no_shuffle

    for batch in dynamic_ds:
        batch_count += 1
        # Each batch should be a dict (after collation)
        assert isinstance(batch, dict)
        assert "attention_mask" in batch
        assert batch["attention_mask"].sum() > 0

        # Calculate total tokens in batch
        # Note: With force_generate_long_sequence=True, batches can exceed micro_batch_seq_length
        # when the buffer only has one long sample
        total_tokens = batch["attention_mask"].sum()

        # Check expected input_ids
        assert batch["input_ids"].tolist()[0] == expected_input_ids[batch_count - 1], \
            f"Batch {batch_count} input_ids mismatch. Expected: {expected_input_ids[batch_count - 1]}, Got: {batch['input_ids'].tolist()[0]}"

        # check buffer size
        buffer_length = len(dynamic_ds._buffer)
        assert buffer_length <= READY_FOR_MICRO_BATCH_THRESHOLD, f"Buffer has {buffer_length} samples, exceeds ready_for_micro_batch_threshold {READY_FOR_MICRO_BATCH_THRESHOLD}"


        # check if any remaining item in buffer can be added to the batch
        if buffer_length > 0:
            for item in dynamic_ds._buffer:
                assert item[1] + total_tokens > MICRO_BATCH_SEQ_LENGTH, f"Buffer item {item[0]} has {item[1]} tokens, it can still fit into the batch"

        if batch_count >= 5:  # Just test a few batches
            break

    assert batch_count > 0, "Should produce at least one batch"
    logger.info(f"test_dynamic_batching_basic (shuffle={shuffle}) passed! Produced {batch_count} batches")


def test_force_long_sequence():
    """Test that force_generate_long_sequence allows batches exceeding micro_batch_seq_length.

    When force_generate_long_sequence=True and the buffer only has one long sample,
    the batch should be allowed to exceed micro_batch_seq_length.
    """

    mapping_dataset = DummyMappingDataset(size=DATASET_SIZE)
    iterable_dataset = DummyIterableDataset(mapping_dataset, shuffle=False)

    # Test with force_generate_long_sequence=True (default)
    dynamic_ds = DynamicBatchingSizeDataset(
        dataset=iterable_dataset,
        micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
        ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
        dynamic_batching_collate_fn=DataCollatorWithPositionIDs(),
        get_length_fn=get_length_fn,  # Use the global get_length_fn that works with dict
        force_generate_long_sequence=True,
    )

    # Iterate through batches to find one that exceeds micro_batch_seq_length
    found_long_batch_value = None

    batch_idx = 0
    for batch in dynamic_ds:
        total_tokens = batch["attention_mask"].sum().item()
        input_ids = batch["input_ids"].tolist()[0]
        unique_values = set(input_ids)
        is_single_sample = len(unique_values) == 1

        if total_tokens > MICRO_BATCH_SEQ_LENGTH and is_single_sample:
            sample_value = unique_values.pop()
            found_long_batch_value = sample_value
            break

        batch_idx += 1

    # Verify the found batch
    assert found_long_batch_value == MICRO_BATCH_SEQ_LENGTH + 1, f"Expected long batch to contain value {MICRO_BATCH_SEQ_LENGTH + 1}, got value={found_long_batch_value}"


def test_last_batch_on_dataset_end(setup_dynamic_batching_dataset):
    """Test that remaining buffer items are yielded when dataset ends.

    This test verifies that when the upstream dataset ends (StopIteration),
    DynamicBatchingSizeDataset will continue to yield batches from the remaining
    buffer items until the buffer is empty, even if they don't meet the normal
    threshold conditions.
    """
    mapping_dataset, iterable_dataset, dynamic_ds = setup_dynamic_batching_dataset

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
            buffer_size >= READY_FOR_MICRO_BATCH_THRESHOLD and
            buffer_tokens >= MICRO_BATCH_SEQ_LENGTH
        )

        if upstream_exhausted and not buffer_meets_threshold and buffer_size > 0:
            # Try to get a batch - should succeed even though buffer doesn't meet threshold
            try:
                batch = next(iterator)
                batch_count += 1
                total_tokens = batch["attention_mask"].sum()
                if total_tokens > 0:
                    found_last_batch_scenario = True
            except StopIteration:
                assert False, "Expected to get a batch from remaining buffer, but got StopIteration"
        else:
            # Normal batch retrieval
            try:
                batch = next(iterator)
                batch_count += 1
            except StopIteration:
                break

    assert found_last_batch_scenario, \
        "Did not find the scenario where upstream is exhausted but buffer doesn't meet threshold"

def test_dynamic_batching_without_get_item():
    """Test DynamicBatchingSizeDataset initialization without get_item povided.

    Tests that DynamicBatchingSizeDataset cannot be initialized with save_by_idx=True
    when the dataset doesn't have get_item method.
    """
    class DummyIterableDatasetWithoutGetItem(IterableDataset):
        def __iter__(self):
            for i in range(10):
                yield {"input_ids": [i] * i, "attention_mask": [1] * i}
    iterable_dataset = DummyIterableDatasetWithoutGetItem()

    # Test with save_by_idx=True (should raise ValueError)
    with pytest.raises(ValueError, match="save_by_idx is True, but dataset does not have get_item method"):
        dynamic_ds = DynamicBatchingSizeDataset(
                dataset=iterable_dataset,
                micro_batch_seq_length=MICRO_BATCH_SEQ_LENGTH,
                ready_for_micro_batch_threshold=READY_FOR_MICRO_BATCH_THRESHOLD,
                dynamic_batching_collate_fn=DataCollatorWithPositionIDs(),
                get_length_fn=get_length_fn,
                save_by_idx=True,
            )



@pytest.mark.parametrize("shuffle", [False, True])
def test_dynamic_batching_dataset_distributed(shuffle):
    """Test DynamicBatchingSizeDataset in distributed setting.

    Runs main_distributed_test() by torchrun with or without data shuffling.

    Args:
        shuffle: Whether to enable data shuffling.

    Raises:
        subprocess.CalledProcessError: If the distributed test fails.
    """
    command = build_command(shuffle=shuffle)
    # Pass current environment to subprocess to inherit virtual environment
    result = subprocess.run(command, check=True, env=os.environ.copy())
    assert result.returncode == 0

def build_command(shuffle=True):
    """Build torchrun command for distributed testing.

    Constructs a command to launch the test script with torchrun for
    distributed execution with 2 processes.

    Args:
        shuffle: Whether to enable data shuffling.

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
        "--data.train_size=1000",
        "--data.max_seq_len=16",
        f"--data.shuffle={str(shuffle).lower()}",
        "--train.global_batch_size=8",
        "--train.micro_batch_size=2",
        "--train.data_parallel_mode=ddp",
        "--train.ckpt_manager=dcp",
        "--train.output_dir=.tests/cache",
        "--train.rmpad=false",
        "--train.rmpad_with_pos_ids=true",
        "--train.seed=42",
    ]
    return command


@dataclass
class Arguments:
    """
    Container for training arguments.

    Attributes:
        model: Model configuration arguments.
        data: Data loading and processing arguments.
        train: Training loop and optimization arguments.
    """
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main_distributed_test():
    """
    Tests:
    - Dynamic batching with shuffled iterable dataset
    - Checkpoint save/load with buffer state
    - Multi-process distributed training
    """
    # Patch empty_cache to avoid AttributeError on CPU
    with patch('veomni.utils.device.empty_cache', _mock_empty_cache):
        _run_distributed_test()


def _run_distributed_test():
    """Internal function that runs the actual distributed test."""
    args = parse_args(Arguments)
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")

    # Use gloo backend for CPU, otherwise use get_dist_comm_backend()
    try:
        backend = get_dist_comm_backend()
    except RuntimeError:
        # Fallback to gloo for CPU
        backend = "gloo"

    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    # Create DummyMappingDataset and DummyIterableDataset
    mapping_dataset = DummyMappingDataset(size=DATASET_SIZE)
    # Use shuffle from args.data if available, otherwise default to True
    shuffle = getattr(args.data, 'shuffle', True)
    iterable_dataset = DummyIterableDataset(mapping_dataset, shuffle=shuffle, seed=args.train.seed)

    # Create DynamicBatchingSizeDataset
    micro_batch_seq_length = args.train.micro_batch_size * args.data.max_seq_len
    dataloader_batch_size = args.train.global_batch_size // (args.train.micro_batch_size * get_parallel_state().dp_size)

    dynamic_dataset = DynamicBatchingSizeDataset(
        dataset=iterable_dataset,
        micro_batch_seq_length=micro_batch_seq_length,
        ready_for_micro_batch_threshold=10,  # Minimum samples in buffer before batching
        dynamic_batching_collate_fn=DataCollatorWithPositionIDs(),
        save_by_idx=True,
        get_length_fn=get_length_fn,
    )

    config = PretrainedConfig()
    environ_meter = helper.EnvironMeter(
        config=config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    batches_before_save = []
    batches_after_save = []
    max_steps = 10
    global_step = 0
    save_step = 5

    fake_model = FakeModel().to(get_device_type())

    # First pass: collect batches
    logger.info(f"[rank{rank}] Starting first pass to collect batches")
    data_iterator = iter(dynamic_dataset)
    start_time = time.time()
    
    for step in range(max_steps):
        global_step += 1
        try:
            micro_batch = next(data_iterator)
        except StopIteration:
            logger.info(f"[rank{rank}] Iterator finished at step {step}")
            break

        if global_step == 1:
            helper.print_example(example=micro_batch, rank=args.train.local_rank)
            logger.info(f"[rank{rank}] First batch keys: {micro_batch.keys()}")
            logger.info(f"[rank{rank}] First batch input_ids shape: {micro_batch['input_ids'].shape}")

        if global_step > save_step:
            batches_after_save.append(micro_batch)
        else:
            batches_before_save.append(micro_batch)

        environ_meter.add(micro_batch)
        delta_time = time.time() - start_time
        try:
            metrics = environ_meter.step(delta_time, global_step=global_step)
        except AttributeError as e:
            # Skip metrics on CPU (torch.cpu has no attribute 'get_device_name')
            logger.warning(f"[rank{rank}] Skipping metrics: {e}")
            metrics = {}

        if global_step == save_step:
            state = {
                "model": fake_model,
                "extra_state": {
                    "global_step": global_step,
                    "dynamic_dataset": dynamic_dataset.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                },
            }
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()

    logger.info(f"[rank{rank}] Collected {len(batches_before_save)} batches before save_step and {len(batches_after_save)} batches after save_step")

    # Verify batches against expected values
    if shuffle:
        # Expected values for shuffle=True, seed=42
        expected_batches_before_save = [
            [43]*43,  # Step 1, 43 tokens
            [18]*18 + [1] + [9]*9,  # Step 2, 28 tokens
            [31]*31,  # Step 3, 31 tokens
            [35]*35,  # Step 4, 35 tokens
            [21]*21 + [4]*4 + [3]*3 + [2]*2,  # Step 5, 30 tokens
        ]
        expected_batches_after_save = [
            [26]*26,  # Step 6, 26 tokens
            [37]*37,  # Step 7, 37 tokens
            [19]*19 + [7]*7,  # Step 8, 26 tokens
            [29]*29,  # Step 9, 29 tokens
            [40]*40,  # Step 10, 40 tokens
        ]
    else:
        # Expected values for shuffle=False, seed=42
        expected_batches_before_save = [
            [1] + [2]*2 + [3]*3 + [4]*4 + [5]*5 + [6]*6 + [7]*7,  # Step 1, 28 tokens
            [8]*8 + [9]*9 + [10]*10,  # Step 2, 27 tokens
            [11]*11 + [12]*12,  # Step 3, 23 tokens
            [13]*13 + [14]*14,  # Step 4, 27 tokens
            [15]*15 + [16]*16,  # Step 5, 31 tokens
        ]
        expected_batches_after_save = [
            [17]*17,  # Step 6, 17 tokens
            [18]*18,  # Step 7, 18 tokens
            [19]*19,  # Step 8, 19 tokens
            [20]*20,  # Step 9, 20 tokens
            [21]*21,  # Step 10, 21 tokens
        ]

    # Verify batches_before_save
    assert len(batches_before_save) == len(expected_batches_before_save), \
        f"[rank{rank}] batches_before_save count mismatch: {len(batches_before_save)} vs {len(expected_batches_before_save)}"

    for i, (batch, expected_input_ids) in enumerate(zip(batches_before_save, expected_batches_before_save)):
        actual_input_ids = batch["input_ids"].tolist()[0]
        assert actual_input_ids == expected_input_ids, \
            f"[rank{rank}] Batch {i+1} (before save) input_ids mismatch.\nExpected: {expected_input_ids}\nGot: {actual_input_ids}"

    logger.info(f"[rank{rank}] ✅ All batches_before_save matched expected values!")

    # Verify batches_after_save
    assert len(batches_after_save) == len(expected_batches_after_save), \
        f"[rank{rank}] batches_after_save count mismatch: {len(batches_after_save)} vs {len(expected_batches_after_save)}"

    for i, (batch, expected_input_ids) in enumerate(zip(batches_after_save, expected_batches_after_save)):
        actual_input_ids = batch["input_ids"].tolist()[0]
        assert actual_input_ids == expected_input_ids, \
            f"[rank{rank}] Batch {i+save_step+1} (after save) input_ids mismatch.\nExpected: {expected_input_ids}\nGot: {actual_input_ids}"

    logger.info(f"[rank{rank}] ✅ All batches_after_save matched expected values!")

    # Resume from checkpoint
    logger.info(f"[rank{rank}] Loading checkpoint from {save_checkpoint_path}")

    # Recreate the datasets for resume
    mapping_dataset_resume = DummyMappingDataset(size=DATASET_SIZE)
    iterable_dataset_resume = DummyIterableDataset(mapping_dataset_resume, shuffle=shuffle, seed=args.train.seed)

    dynamic_dataset_resume = DynamicBatchingSizeDataset(
        dataset=iterable_dataset_resume,
        micro_batch_seq_length=micro_batch_seq_length,
        ready_for_micro_batch_threshold=10,
        dynamic_batching_collate_fn=DataCollatorWithPositionIDs(),
        save_by_idx=True,
        get_length_fn=get_length_fn,
    )

    state = {"model": fake_model, "extra_state": {}}
    Checkpointer.load(save_checkpoint_path, state)
    dynamic_dataset_resume.load_state_dict(state["extra_state"]["dynamic_dataset"])
    environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
    global_step = state["extra_state"]["global_step"]

    # Second pass: verify batches match
    second_pass_batches_after_resume = []
    data_iterator_resume = iter(dynamic_dataset_resume)

    for step in range(global_step, max_steps):
        global_step += 1
        try:
            micro_batch = next(data_iterator_resume)
        except StopIteration:
            logger.info(f"[rank{rank}] Resume iterator finished at step {step}")
            break

        if global_step > save_step:
            second_pass_batches_after_resume.append(micro_batch)

        start_time = time.time()
        environ_meter.add(micro_batch)
        delta_time = time.time() - start_time
        try:
            metrics_resume = environ_meter.step(delta_time, global_step=global_step)
        except AttributeError as e:
            # Skip metrics on CPU (torch.cpu has no attribute 'get_device_name')
            logger.warning(f"[rank{rank}] Skipping metrics: {e}")
            metrics_resume = {}

    logger.info(f"[rank{rank}] Collected {len(second_pass_batches_after_resume)} batches after resume")

    # Compare batches
    assert len(batches_after_save) == len(second_pass_batches_after_resume), f"Batch count mismatch: {len(batches_after_save)} vs {len(second_pass_batches_after_resume)}"

    for i, (ground_truth_batch, resume_batch) in enumerate(zip(batches_after_save, second_pass_batches_after_resume)):
        for key in ground_truth_batch.keys():
            if torch.is_tensor(ground_truth_batch[key]):
                assert torch.all(ground_truth_batch[key] == resume_batch[key]), \
                    f"[rank{rank}] Batch {i} key {key} mismatch"

    logger.info(f"[rank{rank}] All batches matched successfully!")

    # Compare metrics (only if available, may be empty on CPU)
    if "consume_tokens(M)" in metrics and "consume_tokens(M)" in metrics_resume:
        assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]

    if dist.is_initialized():
        dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main_distributed_test()


