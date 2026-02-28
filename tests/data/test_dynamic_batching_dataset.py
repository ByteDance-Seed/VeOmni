"""Tests for DynamicBatchingSizeDataset functionality.

This module tests the ``DynamicBatchingSizeDataset`` class using ``DummyIterableDataset``.
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
          ``shuffle × save_by_idx`` (4 combinations), verifying that resumed
          batches are byte-for-byte identical to the original run.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.distributed as dist
from tools.launch_utils import find_free_port
from torch.utils.data import IterableDataset
from transformers import PretrainedConfig
from utils import DummyIterableDataset, DummyMappingDataset, FakeModel

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args
from veomni.checkpoint import build_checkpointer
from veomni.data import build_dataloader
from veomni.data.data_collator import MainCollator
from veomni.data.dynamic_batching import DynamicBatchingSizeDataset
from veomni.distributed.parallel_state import init_parallel_state
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


logger = helper.create_logger(__name__)


# Patch empty_cache to avoid AttributeError on CPU
def _mock_empty_cache():
    """Mock empty_cache that does nothing on CPU."""
    pass


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
        dynamic_batching_collate_fn=MainCollator(),
        get_length_fn=get_length_fn,
    )

    return mapping_dataset, iterable_dataset, dynamic_ds


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
    mapping_ds = DummyMappingDataset(size=DATASET_SIZE)
    iterable_ds = DummyIterableDataset(mapping_ds, shuffle=shuffle, seed=seed)

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
            except StopIteration:
                assert False, "Expected to get a batch from remaining buffer, but got StopIteration"
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

    class DummyIterableDatasetWithoutGetItem(IterableDataset):
        def __iter__(self):
            for i in range(10):
                yield {"input_ids": [i] * i, "attention_mask": [1] * i}

    iterable_dataset = DummyIterableDatasetWithoutGetItem()

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
    mapping_ds = DummyMappingDataset(size=DATASET_SIZE)
    iterable_ds = DummyIterableDataset(mapping_ds, shuffle=False)

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
    mapping_ds2 = DummyMappingDataset(size=DATASET_SIZE)
    iterable_ds2 = DummyIterableDataset(mapping_ds2, shuffle=False)

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
    "shuffle,save_by_idx",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_dynamic_batching_dataset_distributed(shuffle, save_by_idx):
    """Test DynamicBatchingSizeDataset in distributed setting.

    Runs _main_distributed_test() by torchrun with or without data shuffling
    and with or without save_by_idx for checkpoint buffer saving.

    Args:
        shuffle: Whether to enable data shuffling.
        save_by_idx: Whether to save buffer by index for checkpointing.

    Raises:
        subprocess.CalledProcessError: If the distributed test fails.
    """
    command = build_command(shuffle=shuffle, save_by_idx=save_by_idx)
    # Pass current environment to subprocess to inherit virtual environment
    result = subprocess.run(command, check=True, env=os.environ.copy())
    assert result.returncode == 0


def build_command(shuffle=True, save_by_idx=True):
    """Build torchrun command for distributed testing.

    Constructs a command to launch the test script with torchrun for
    distributed execution with 2 processes.

    Args:
        shuffle: Whether to enable data shuffling.
        save_by_idx: Whether to save buffer by index for checkpointing.

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
        "--data.max_seq_len=16",
        "--train.micro_batch_size=2",
        f"--shuffle={str(shuffle).lower()}",
        "--train.global_batch_size=16",
        "--train.data_parallel_mode=ddp",
        "--train.ckpt_manager=dcp",
        "--train.output_dir=.tests/cache",
        "--train.dyn_bsz=true",
        "--dyn_bsz_in_dataloader=false",
        f"--save_by_idx={str(save_by_idx).lower()}",
        "--train.seed=42",
    ]
    return command


@dataclass
class Arguments(VeOmniArguments):
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


def _main_distributed_test():
    """Entry point for the distributed test launched by ``torchrun``.

    It wraps ``_run_distributed_test()` and in the testing it is supposed to be
    triggered by test_dynamic_batching_dataset_distributed().
    """
    # Patch empty_cache to avoid AttributeError on CPU
    with patch("veomni.utils.device.empty_cache", _mock_empty_cache):
        _run_distributed_test()


def _run_distributed_test():
    """Run a full checkpoint-resume cycle and assert batch reproducibility.

    Procedure
    ---------
    1. **Parse CLI flags**
    2. **Initialise torch distributed state**
    3. **Build a StatefulDataLoader** wrapping ``DummyIterableDataset`` →
       ``DynamicBatchingSizeDataset`` with ``num_workers=2``.
    4. **First pass (2 epochs)** – iterate the dataloader for both epochs.  Batches
       before the designated save point (``epoch=1, step=2``) are discarded; batches
       *after* that point are stored in ``batches_after_save_step`` as ground truth.
       At the save point a checkpoint is written via ``Checkpointer.save()``,
       capturing model weights, ``dataloader.state_dict()``, and
       ``environ_meter.state_dict()``.
    5. **Load checkpoint** – ``Checkpointer.load()`` restores all state; the
       dataloader, dataset and environ-meter are restored through ``load_state_dict()``.
    6. **Second pass (resume)** – iterate from the saved epoch / step through the
       end of both epochs, collecting resumed batches in ``batch_after_resume``.
    7. **Assert equality** – verify that ``batches_after_save_step`` and
       ``batch_after_resume`` have the same length and that every tensor in every
       micro-batch is identical element-wise.
    """
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--shuffle", type=lambda x: x.lower() == "true", default=True)
    _parser.add_argument("--save_by_idx", type=lambda x: x.lower() == "true", default=True)
    _parser.add_argument("--dyn_bsz_in_dataloader", type=lambda x: x.lower() == "true", default=True)
    test_args, remaining_argv = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

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
    iterable_dataset = DummyIterableDataset(mapping_dataset, shuffle=test_args.shuffle, seed=args.train.seed)

    # Compute train_steps based on dataset size
    dataset_length = len(mapping_dataset)
    args.compute_train_steps(dataset_length)
    train_steps = args.train_steps

    dataloader = build_dataloader(
        dataloader_type="native",
        dataset=iterable_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=train_steps,
        dyn_bsz=args.train.dyn_bsz,
        dyn_bsz_in_dataloader=test_args.dyn_bsz_in_dataloader,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        dyn_bsz_buffer_size=READY_FOR_MICRO_BATCH_THRESHOLD,
        dyn_bsz_dataset_save_by_idx=test_args.save_by_idx,
        num_workers=2,
        drop_last=False,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    config = PretrainedConfig()
    environ_meter = helper.EnvironMeter(
        config=config,
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    batches_after_save_step = []
    epoch_num = 2  # Run 2 epochs
    start_epoch, start_step, global_step = 0, 0, 0
    save_epoch, save_step = 1, 0

    fake_model = FakeModel().to(get_device_type())

    # First pass: run 2 epochs and collect batches after save_step
    logger.info(
        f"[rank{rank}] Starting first pass: running {epoch_num} epochs, train_steps={train_steps}, save_step={save_step}"
    )
    for epoch in range(start_epoch, epoch_num):
        dataloader.set_epoch(epoch)
        data_iterator = iter(dataloader)
        start_time = time.time()

        for local_step in range(start_step, train_steps):
            global_step += 1
            try:
                micro_batches = next(data_iterator)
            except StopIteration:
                logger.info(
                    f"[rank{rank}] epoch:{epoch} Dataloader finished at global_step={global_step - 1}, local_step={local_step}"
                )
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            # Print batch info for debugging
            """
            logger.error(f"[rank{rank}] epoch:{epoch} step:{local_step} global_step:{global_step} num_micro_batches:{len(micro_batches)} dataset_iter: {dataloader.dataset._data_iter}")
            for micro_idx, micro_batch in enumerate(micro_batches):
                # Extract sample indices from input_ids (each sample has all same values)
                input_ids = micro_batch["input_ids"].squeeze(0)  # Remove batch dim
                input_ids = set(input_ids.tolist())
                logger.error(f"[rank{rank}] epoch:{epoch} step:{local_step} global_step:{global_step} micro_batch[{micro_idx}]: {input_ids}")
            """

            if epoch > save_epoch or (epoch == save_epoch and local_step > save_step):
                batches_after_save_step.append(micro_batches)

            for _, micro_batch in enumerate(micro_batches):
                environ_meter.add(micro_batch)

            delta_time = time.time() - start_time
            try:
                metrics = environ_meter.step(delta_time, global_step=global_step)
            except AttributeError as e:
                # Skip metrics on CPU (torch.cpu has no attribute 'get_device_name')
                logger.warning(f"[rank{rank}] Skipping metrics: {e}")
                metrics = {}

            if epoch == save_epoch and local_step == save_step:
                state = {
                    "model": fake_model,
                    "extra_state": {
                        "curr_epoch": epoch,
                        "curr_step": local_step,
                        "train_dataloader": dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                    },
                }
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()

        start_step = 0  # Reset for next epoch

    logger.info(f"[rank{rank}] Collected {len(batches_after_save_step)} batches after save_step in first pass")

    # Resume from checkpoint
    logger.info(f"[rank{rank}] Loading checkpoint from {save_checkpoint_path}")
    state = {"model": fake_model, "extra_state": {}}
    Checkpointer.load(save_checkpoint_path, state)
    dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
    environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
    start_epoch = state["extra_state"]["curr_epoch"]
    assert start_epoch == 1
    start_step = state["extra_state"]["curr_step"] + 1
    assert start_step == 1
    dl_state = state["extra_state"]["train_dataloader"]
    logger.error(f"[rank{rank}] Loaded dataloader state: {dl_state}")

    # Second pass: resume and collect batches
    batch_after_resume = []
    logger.info(f"[rank{rank}] Resuming from epoch {start_epoch}, step {start_step}")

    for epoch in range(start_epoch, epoch_num):
        dataloader.set_epoch(epoch)
        data_iter = iter(dataloader)

        for local_step in range(start_step, train_steps):
            global_step += 1
            try:
                micro_batches = next(data_iter)
            except StopIteration:
                logger.info(f"[rank{rank}] epoch:{epoch} step:{local_step} Dataloader finished on resume")
                break

            if epoch > save_epoch or (epoch == save_epoch and local_step > save_step):
                batch_after_resume.append(micro_batches)

            for _, micro_batch in enumerate(micro_batches):
                environ_meter.add(micro_batch)

            delta_time = time.time() - start_time
            try:
                metrics_resume = environ_meter.step(delta_time, global_step=global_step)
            except AttributeError as e:
                # Skip metrics on CPU
                logger.warning(f"[rank{rank}] Skipping metrics: {e}")
                metrics_resume = {}

        start_step = 0  # Reset for next epoch

    logger.info(f"[rank{rank}] Collected {len(batch_after_resume)} batches after save_step in second pass")

    # Compare batches
    assert len(batches_after_save_step) == len(batch_after_resume), (
        f"[rank{rank}] Batch count mismatch: {len(batches_after_save_step)} vs {len(batch_after_resume)}"
    )

    for i, (gt_batches, pred_batches) in enumerate(zip(batches_after_save_step, batch_after_resume)):
        assert len(gt_batches) == len(pred_batches), (
            f"[rank{rank}] Micro batch count mismatch at step {i}: {len(gt_batches)} vs {len(pred_batches)}"
        )

        for j, (gt_batch, pred_batch) in enumerate(zip(gt_batches, pred_batches)):
            for key in gt_batch.keys():
                if torch.is_tensor(gt_batch[key]):
                    assert torch.all(gt_batch[key] == pred_batch[key]), (
                        f"[rank{rank}] Batch {i} micro_batch {j} key {key} mismatch"
                    )

    if (
        metrics is not None
        and metrics_resume is not None
        and "consume_tokens(M)" in metrics
        and "consume_tokens(M)" in metrics_resume
    ):
        assert metrics["consume_tokens(M)"] == metrics_resume["consume_tokens(M)"]

    logger.info(f"[rank{rank}] ✅ All batches matched successfully!")

    if dist.is_initialized():
        dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    _main_distributed_test()
