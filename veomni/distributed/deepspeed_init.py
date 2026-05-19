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

"""DeepSpeed engine initialization and config builder."""

import json
from typing import TYPE_CHECKING, Tuple

import torch

from veomni.utils import logging


if TYPE_CHECKING:
    from ..arguments import TrainingArguments

logger = logging.get_logger(__name__)


def build_ds_config(train_args: "TrainingArguments") -> dict:
    """Translate TrainingArguments into a DeepSpeed config dict.

    If ``train_args.deepspeed.config_path`` is set, load that JSON verbatim
    and only patch ``train_batch_size`` / ``gradient_accumulation_steps``.
    Otherwise, build from the individual ``deepspeed.*`` fields.
    """
    ds = train_args.deepspeed

    if ds.config_path:
        with open(ds.config_path) as f:
            config = json.load(f)
        config.setdefault(
            "train_batch_size",
            train_args.world_size * train_args.micro_batch_size * train_args.gradient_accumulation_steps,
        )
        config.setdefault("gradient_accumulation_steps", train_args.gradient_accumulation_steps)
        return config

    config = {
        "train_batch_size": train_args.world_size
        * train_args.micro_batch_size
        * train_args.gradient_accumulation_steps,
        "micro_batch_size_per_gpu": train_args.micro_batch_size,
        "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
        "gradient_clipping": train_args.optimizer.max_grad_norm,
        "zero_optimization": {
            "stage": ds.zero_stage,
            "overlap_comm": ds.overlap_comm,
            "contiguous_gradients": ds.contiguous_gradients,
        },
        "bf16": {"enabled": train_args.accelerator.fsdp_config.mixed_precision.enable},
        "steps_per_print": 1,
    }

    zero = config["zero_optimization"]
    if ds.offload_optimizer:
        offload_opt = {"device": ds.offload_optimizer}
        if ds.offload_optimizer == "nvme":
            offload_opt["nvme_path"] = ds.nvme_path
        zero["offload_optimizer"] = offload_opt

    if ds.offload_param:
        offload_param = {"device": ds.offload_param}
        if ds.offload_param == "nvme":
            offload_param["nvme_path"] = ds.nvme_path
        zero["offload_param"] = offload_param

    return config


def init_deepspeed_engine(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_args: "TrainingArguments",
    ds_config: dict,
) -> Tuple:
    """Initialize DeepSpeed engine.

    Returns ``(engine, ds_optimizer, ds_lr_scheduler)``.
    The engine wraps the model; access original via ``engine.module``.
    """
    import deepspeed

    engine, ds_optimizer, _, ds_lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=ds_config,
    )

    logger.info_rank0(
        f"DeepSpeed engine initialized. ZeRO stage={train_args.deepspeed.zero_stage}, "
        f"offload_optimizer={train_args.deepspeed.offload_optimizer}, "
        f"offload_param={train_args.deepspeed.offload_param}"
    )

    return engine, ds_optimizer, ds_lr_scheduler
