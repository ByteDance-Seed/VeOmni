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

from contextlib import contextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from ..data import build_dataloader, build_dataset
from ..data.data_loader import ExactDistributedBatchSampler
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import logging
from ..utils.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from .base import BaseTrainer


logger = logging.get_logger(__name__)


@contextmanager
def _preserve_module_training_modes(model: torch.nn.Module) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        model.train(modes[0][1])
        for module, mode in modes[1:]:
            if module.training != mode:
                module.train(mode)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class TextValidationRunner:
    """Own Text SFT validation data and forward semantics.

    The callback only schedules this runner. This owner is installed by
    ``TextTrainer`` alone so composed trainers cannot silently inherit causal-LM
    evaluation semantics that do not match their objectives.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer
        self.parallel_state = get_parallel_state()
        self._validate_support()
        self.dataset, self.dataloader = self._build_data()

    @staticmethod
    def is_requested(trainer: "BaseTrainer") -> bool:
        args = trainer.args
        return args.data.eval_path is not None and bool(args.train.eval_steps or args.train.eval_epochs)

    def _validate_support(self) -> None:
        args = self.trainer.args
        parallel_state = self.parallel_state
        unsupported = []
        if args.data.data_type not in {"conversation", "classification"}:
            unsupported.append(f"data_type={args.data.data_type!r}")
        if args.data.eval_path.endswith(".yaml"):
            unsupported.append("multisource evaluation YAML")
        if args.data.dataloader.type != "native":
            unsupported.append(f"data.dataloader.type={args.data.dataloader.type!r}")
        if parallel_state.sp_enabled:
            unsupported.append("sequence/context parallelism")
        if parallel_state.tp_enabled:
            unsupported.append("tensor parallelism")
        if parallel_state.pp_enabled:
            unsupported.append("pipeline parallelism")
        if parallel_state.any_extra_parallel_enabled:
            unsupported.append("ExtraParallel/EP")
        if parallel_state.async_enabled:
            unsupported.append("async sequence parallelism")
        if args.train.chunk_mbs_config.enable:
            unsupported.append("ChunkMBS")
        if args.train.torch_compile.enable:
            unsupported.append("torch.compile")
        if args.train.moe_load_balance_monitor_interval > 0 and hasattr(self.trainer.model_config, "num_experts"):
            unsupported.append("MoE router monitoring")
        if args.train.profile.enable:
            unsupported.append("torch profiler")
        if unsupported:
            raise ValueError(
                "Training-time validation currently supports TextTrainer with map-style data and pure DP/FSDP2 "
                f"only; unsupported configuration: {', '.join(unsupported)}."
            )

    def _build_data(self):
        args = self.trainer.args
        eval_path = args.data.eval_path
        data_kwargs = asdict(args.data)
        data_kwargs.update(train_path=eval_path, shuffle=False, split_by_node=False)
        dataset = build_dataset(
            dataset_name=args.data.datasets_type,
            transform=self.trainer.data_transform,
            seed=args.train.seed,
            **data_kwargs,
        )
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "Training-time validation requires a map-style dataset so every sample can be evaluated exactly "
                "once with collective-safe rank cardinality."
            )

        batch_sampler = ExactDistributedBatchSampler(
            dataset_size=len(dataset),
            batch_size=args.train.micro_batch_size,
            num_replicas=self.parallel_state.dp_size,
            rank=self.parallel_state.dp_rank,
        )
        self.dataloader_generator = torch.Generator().manual_seed(args.train.seed)
        dataloader_kwargs = asdict(args.data.dataloader)
        dataloader_type = dataloader_kwargs.pop("type")
        dataloader_kwargs.pop("use_background_prefetcher", None)
        dataloader_kwargs["in_order"] = True
        dataloader_kwargs["persistent_workers"] = False
        dataloader = build_dataloader(
            dataloader_type=dataloader_type,
            dataset=dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.micro_batch_size * self.parallel_state.dp_size,
            dataloader_batch_size=args.train.micro_batch_size,
            max_seq_len=args.data.max_seq_len,
            train_steps=len(batch_sampler),
            dyn_bsz=False,
            seed=args.train.seed,
            collate_fn=self.trainer.collate_fn,
            batch_sampler=batch_sampler,
            generator=self.dataloader_generator,
            **dataloader_kwargs,
        )
        return dataset, dataloader

    def _count_loss_units(self, labels: torch.Tensor) -> torch.Tensor:
        if self.trainer.args.data.data_type == "classification":
            return torch.count_nonzero(labels != IGNORE_INDEX)
        if labels.ndim == 0 or labels.shape[-1] < 2:
            return labels.new_zeros((), dtype=torch.long)
        return torch.count_nonzero(labels[..., 1:] != IGNORE_INDEX)

    @torch.no_grad()
    def run(self) -> dict[str, float]:
        trainer = self.trainer
        self.dataloader_generator.manual_seed(trainer.args.train.seed)
        self.dataloader.set_epoch(0)
        loss_sum = torch.zeros((), dtype=torch.float32, device=trainer.device)
        loss_units = torch.zeros((), dtype=torch.int64, device=trainer.device)

        with _preserve_module_training_modes(trainer.model):
            for micro_batches in self.dataloader:
                if len(micro_batches) != 1:
                    raise RuntimeError(
                        "Validation requires exactly one micro-batch per forward, "
                        f"but the dataloader produced {len(micro_batches)}."
                    )
                micro_batch = _move_to_device(micro_batches[0], trainer.device)
                labels = micro_batch.get("labels")
                if not isinstance(labels, torch.Tensor):
                    raise TypeError("Validation batches must contain a tensor 'labels' field.")
                with (
                    use_parallel_state("base"),
                    trainer.model_fwd_context,
                    set_batch_invariant_mode(trainer.args.train.enable_batch_invariant_mode),
                ):
                    outputs = trainer.model(**micro_batch, use_cache=False)
                loss = outputs.loss
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Text validation expects model outputs.loss to be a scalar tensor.")
                unit_count = self._count_loss_units(labels)
                weighted_loss = loss.detach().to(torch.float32) * unit_count
                loss_sum += torch.where(unit_count > 0, weighted_loss, loss_sum.new_zeros(()))
                loss_units += unit_count

        if self.parallel_state.dp_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=self.parallel_state.dp_group)
            dist.all_reduce(loss_units, op=dist.ReduceOp.SUM, group=self.parallel_state.dp_group)
        if loss_units.item() == 0:
            raise ValueError("Validation produced no non-ignored loss targets.")

        validation_loss = (loss_sum / loss_units).item()
        logger.info_rank0(f"Validation completed: loss={validation_loss:.6f}, loss_units={loss_units.item()}.")
        return {"loss": validation_loss}
