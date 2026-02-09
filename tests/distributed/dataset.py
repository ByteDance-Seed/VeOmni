"""Deterministic datasets and dataloader for FSDP2 integration tests.

Provides simple, debuggable dataloaders using parquet files with explicit
index calculations. Removes complex production logic (seeds, shuffling, sampling).
"""

from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizer

from veomni.data.data_collator import MainCollator
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.constants import IGNORE_INDEX


# =============================================================================
# Deterministic Datasets
# =============================================================================


class SimpleDeterministicDataset(Dataset):
    """Simple deterministic dataset for baseline (single GPU) testing.

    Args:
        parquet_path: Path to parquet file
        text_key: Column name containing text data
    """

    def __init__(self, parquet_path: str, text_key: str = "qwen3_truncated_text"):
        self.df = pd.read_parquet(parquet_path)
        self.text_key = text_key

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range [0, {len(self.df)})")
        return {"text": self.df.iloc[idx][self.text_key]}


class DistributedDeterministicDataset(Dataset):
    """Deterministic dataset for distributed (DP/DP+SP) testing.

    Calculates parquet row indices per DP rank:
        row = step * global_batch_size + micro_batch * (micro_batch_size * dp_size)
              + dp_rank * micro_batch_size + sample_idx

    Args:
        parquet_path: Path to parquet file
        dp_rank: Data parallel rank (0 to dp_size-1)
        dp_size: Number of data parallel ranks
        global_batch_size: Total samples per global batch across all ranks
        micro_batch_size: Samples per micro-batch per rank
        num_train_steps: Number of training steps
        text_key: Column name containing text data
    """

    def __init__(
        self,
        parquet_path: str,
        dp_rank: int,
        dp_size: int,
        global_batch_size: int,
        micro_batch_size: int,
        num_train_steps: int,
        text_key: str = "qwen3_truncated_text",
    ):
        self.df = pd.read_parquet(parquet_path)
        self.text_key = text_key
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_train_steps = num_train_steps

        # Calculate gradient accumulation steps
        self.grad_acc_steps = global_batch_size // (micro_batch_size * dp_size)

        if self.grad_acc_steps < 1:
            raise ValueError(
                f"Invalid configuration: grad_acc_steps = {global_batch_size} // "
                f"({micro_batch_size} * {dp_size}) = {self.grad_acc_steps} < 1"
            )

        # Total samples this rank will see across all steps
        self.total_samples = num_train_steps * self.grad_acc_steps * micro_batch_size

        # Validate we have enough data
        total_global_samples = num_train_steps * global_batch_size
        if total_global_samples > len(self.df):
            raise ValueError(
                f"Not enough data: need {total_global_samples} samples "
                f"({num_train_steps} steps * {global_batch_size} batch_size) "
                f"but parquet has only {len(self.df)} rows"
            )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, local_idx: int) -> dict:
        if local_idx < 0 or local_idx >= self.total_samples:
            raise IndexError(f"Index {local_idx} out of range [0, {self.total_samples})")

        # Decompose local_idx into (global_batch_step, micro_batch_idx, within_micro_idx)
        global_batch_step = local_idx // (self.grad_acc_steps * self.micro_batch_size)
        within_step_idx = local_idx % (self.grad_acc_steps * self.micro_batch_size)
        micro_batch_idx = within_step_idx // self.micro_batch_size
        within_micro_idx = within_step_idx % self.micro_batch_size

        # Calculate global parquet row index
        base_idx = global_batch_step * self.global_batch_size
        micro_offset = micro_batch_idx * (self.micro_batch_size * self.dp_size)
        rank_offset = self.dp_rank * self.micro_batch_size
        row_idx = base_idx + micro_offset + rank_offset + within_micro_idx

        if row_idx >= len(self.df):
            raise IndexError(
                f"Calculated row_idx {row_idx} exceeds dataset size {len(self.df)}. "
                f"local_idx={local_idx}, global_step={global_batch_step}, "
                f"micro_batch={micro_batch_idx}, within_micro={within_micro_idx}, "
                f"dp_rank={self.dp_rank}"
            )

        return {"text": self.df.iloc[row_idx][self.text_key]}


class TransformDataset(Dataset):
    """Wrapper that applies transform to base dataset."""

    def __init__(self, base_dataset: Dataset, transform: callable):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx]
        return self.transform(sample)


# =============================================================================
# Collators
# =============================================================================


class PaddingCollator:
    """Simple padding collator for baseline (single-GPU) testing.

    Pads input_ids and attention_mask with 0, labels with IGNORE_INDEX.
    """

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = defaultdict(list)
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            if key in ("input_ids", "attention_mask", "position_ids"):
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ("labels",):
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)

        return dict(batch)


# =============================================================================
# Transform and Dataloader
# =============================================================================


def create_tokenize_transform(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    text_key: str = "text",
) -> callable:
    """Create tokenization transform function."""

    def transform(sample: dict, source_name: Optional[str] = None) -> dict[str, torch.Tensor]:
        text = sample.get("text", "")
        encoded = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return transform


def build_test_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_seq_len: int,
    text_key: str = "qwen3_truncated_text",
    data_format: str = "padded_bsh",
    is_distributed: bool = False,
    num_workers: int = 0,
    num_train_steps: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> DataLoader:
    """Build deterministic dataloader for integration testing.

    Modes:
    - Baseline (is_distributed=False): Sequential access with padding collator
    - Distributed (is_distributed=True): DP rank-based partitioning with MainCollator

    Args:
        data_path: Path to parquet file
        tokenizer: Tokenizer instance
        batch_size: Batch size per GPU (micro_batch_size in distributed training)
        max_seq_len: Maximum sequence length
        text_key: Key for text field in parquet
        data_format: "padded_bsh" or "rmpad_with_pos_ids"
        is_distributed: Whether running in distributed mode
        num_workers: Number of dataloader workers (default 0 for deterministic ordering)
        num_train_steps: Number of training steps (required for distributed mode)
        global_batch_size: Global batch size (required for distributed mode)

    Returns:
        DataLoader instance with appropriate collators
    """
    transform = create_tokenize_transform(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_key=text_key,
    )

    if is_distributed:
        if global_batch_size is None or num_train_steps is None:
            raise ValueError("global_batch_size and num_train_steps required for distributed mode")

        ps = get_parallel_state()
        dp_rank = ps.dp_rank
        dp_size = ps.dp_size

        print(
            f"[Rank {dp_rank}] Creating DistributedDeterministicDataset: "
            f"dp_rank={dp_rank}, dp_size={dp_size}, "
            f"global_batch_size={global_batch_size}, micro_batch_size={batch_size}, "
            f"num_train_steps={num_train_steps}"
        )

        base_dataset = DistributedDeterministicDataset(
            parquet_path=data_path,
            dp_rank=dp_rank,
            dp_size=dp_size,
            global_batch_size=global_batch_size,
            micro_batch_size=batch_size,
            num_train_steps=num_train_steps,
            text_key=text_key,
        )
        sampler = None
        shuffle = False
    else:
        print(f"Creating SimpleDeterministicDataset with batch_size={batch_size}")

        base_dataset = SimpleDeterministicDataset(
            parquet_path=data_path,
            text_key=text_key,
        )

        if num_train_steps is not None:
            total_samples = num_train_steps * batch_size
            total_samples = min(total_samples, len(base_dataset))
            print(f"Limiting baseline dataset to {total_samples} samples ({num_train_steps} steps)")
            base_dataset = Subset(base_dataset, range(total_samples))

        sampler = None
        shuffle = False

    dataset = TransformDataset(base_dataset, transform=transform)

    # Build collator
    if data_format == "padded_bsh":
        collate_fn = PaddingCollator()
    elif data_format == "rmpad_with_pos_ids":
        # Use VeOmni's MainCollator which internally chains:
        # PrecomputePositionIDsCollator -> PackingCollator -> SequenceParallelCollator (if SP)
        collate_fn = MainCollator()
    else:
        raise ValueError(f"Unsupported data_format: {data_format}. Use 'padded_bsh' or 'rmpad_with_pos_ids'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader
