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


import math
from typing import TYPE_CHECKING, Dict, Literal

from torch.optim.lr_scheduler import LambdaLR

from ..utils import logging


if TYPE_CHECKING:
    from torch.optim import Optimizer


logger = logging.get_logger(__name__)


class MultiLRScheduler(dict):
    """
    Mapping of name -> LR scheduler with convenience methods to mirror a single scheduler.
    """

    _is_multi_lr_scheduler: bool = True

    def step(self) -> None:
        for sched in self.values():
            sched.step()

    def state_dict(self) -> Dict[str, any]:
        return {name: sched.state_dict() for name, sched in self.items()}

    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        for name, sched in self.items():
            if name in state_dict:
                sched.load_state_dict(state_dict[name])

    def get_last_lr(self):
        # Return the first scheduler's LR for logging consistency
        if not self:
            return [0.0]
        first = next(iter(self.values()))
        return first.get_last_lr()


def build_lr_scheduler(
    optimizer: "Optimizer",
    train_steps: int,
    lr: float = 1e-3,
    lr_decay_style: Literal["constant", "linear", "cosine"] = "constant",
    lr_decay_ratio: float = 1.0,
    lr_warmup_ratio: float = 0.0,
    lr_min: float = 1e-7,
    lr_start: float = 0.0,
):
    # Handle MultiOptimizer by creating one scheduler per underlying optimizer
    if hasattr(optimizer, "_is_multi_optimizer") or isinstance(optimizer, dict):
        schedulers = {}
        for key_name in optimizer.key_names:
            schedulers[key_name] = build_lr_scheduler(
                optimizer=optimizer.optimizers_dict[key_name],
                train_steps=train_steps,
                lr=lr,
                lr_decay_style=lr_decay_style,
                lr_decay_ratio=lr_decay_ratio,
                lr_warmup_ratio=lr_warmup_ratio,
                lr_min=lr_min,
                lr_start=lr_start,
            )
        return MultiLRScheduler(schedulers)

    lr_warmup_steps = int(train_steps * lr_warmup_ratio)
    if lr_decay_style == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            lr_start=lr_start,
            init_lr=lr,
        )

    if lr_decay_style == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            lr_start=lr_start,
        )

    if lr_decay_style == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=train_steps,
            init_lr=lr,
            lr_decay_ratio=lr_decay_ratio,
            min_lr=lr_min,
            lr_start=lr_start,
        )

    raise ValueError(f"Unknown learning rate decay style: {lr_decay_style}.")


def get_constant_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    last_epoch: int = -1,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """

    def _lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: "Optimizer",
    num_warmup_steps: int,
    num_training_steps: int,
    init_lr: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_decay_ratio: float = 1.0,
    min_lr: float = 1e-7,
    lr_start: float = 0.0,
):
    """
    Creates a schedule with a learning rate that decreases following the values of the cosine function between
    the initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0
    and the initial lr set in the optimizer.
    """

    def lr_lambda(current_step: int):
        lr_decay_steps = int(num_training_steps * lr_decay_ratio)
        if current_step < num_warmup_steps:
            return (lr_start + (init_lr - lr_start) * current_step / max(1, num_warmup_steps)) / init_lr

        min_lr_ratio = min_lr / init_lr
        if current_step > lr_decay_steps:
            return min_lr_ratio

        progress = float(current_step - num_warmup_steps) / float(max(1, lr_decay_steps - num_warmup_steps))
        assert 0 <= progress <= 1
        factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1 - min_lr_ratio) + min_lr_ratio
        return max(0, factor)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
