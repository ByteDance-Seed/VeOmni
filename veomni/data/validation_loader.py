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

"""Validation dataloader builder.

Reuses the existing :func:`veomni.data.build_dataloader` infrastructure so
that validation data benefits from the same ``StatefulDistributedSampler``
and ``DistributedDataloader`` used for training.
"""

from typing import TYPE_CHECKING, Optional

from ..utils import logging
from .data_loader import DistributedDataloader, build_dataloader
from .data_transform import build_data_transform
from .dataset import build_dataset
from .data_collator import MainCollator

if TYPE_CHECKING:
    from ..arguments import VeOmniArguments

logger = logging.get_logger(__name__)


def build_validation_dataloader(
    args: "VeOmniArguments",
    tokenizer,
    chat_template=None,
) -> Optional[DistributedDataloader]:
    """Build a validation dataloader from ``args.data.eval_path``.

    Reuses the same dataset / transform / collator / dataloader stack as
    training, but with ``shuffle=False`` and ``drop_last=False`` so every
    validation sample is seen exactly once per epoch.

    Args:
        args: Global :class:`VeOmniArguments`.
        tokenizer: Tokenizer instance from the trainer.
        chat_template: Optional chat template for conversation data.

    Returns:
        A :class:`DistributedDataloader` for validation, or ``None`` if
        ``args.data.eval_path`` is not set.
    """
    if args.data.eval_path is None:
        logger.warning_rank0(
            "data.eval_path is not set; skipping validation dataloader build. "
            "Set data.eval_path in your config to enable training-time validation."
        )
        return None

    logger.info_rank0(f"Building validation dataloader from {args.data.eval_path}")

    # Build data transform (same as training)
    data_transform = build_data_transform(
        args.data.data_type,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
    )

    # Build validation dataset
    val_dataset = build_dataset(
        dataset_name=args.data.datasets_type,
        transform=data_transform,
        seed=args.train.seed,
        train_path=args.data.eval_path,
        data_type=args.data.data_type,
        max_seq_len=args.data.max_seq_len,
        text_keys=args.data.text_keys,
        chat_template=args.data.chat_template,
        source_name=args.data.source_name,
        datasets_type=args.data.datasets_type,
        multisource_datasets_type=args.data.multisource_datasets_type,
        train_size=1.0,
        train_sample=0,
        silent_exception=args.data.silent_exception,
    )

    # Build collator (same as training)
    seq_classification = args.data.data_type == "classification"
    collate_fn = MainCollator(
        pad_to_length=args.train.pad_to_length,
        seq_classification=seq_classification,
    )

    # Build dataloader with validation-friendly defaults
    # Use a fixed batch size (no dynamic batching) for deterministic eval
    val_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader.type,
        dataset=val_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.micro_batch_size * args.train.accelerator.dp_size,
        dataloader_batch_size=args.train.micro_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=0,  # Not used for validation
        dyn_bsz=False,  # Fixed batch size for deterministic evaluation
        collate_fn=collate_fn,
        num_workers=args.data.dataloader.num_workers,
        prefetch_factor=args.data.dataloader.prefetch_factor,
        persistent_workers=args.data.dataloader.persistent_workers,
        drop_last=False,  # Keep all validation samples
        pin_memory=args.data.dataloader.pin_memory,
        seed=args.train.seed,
        shuffle=False,  # Deterministic order for validation
        save_steps=0,
    )

    return val_dataloader
