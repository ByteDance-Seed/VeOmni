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

import time
from typing import TYPE_CHECKING, Any, Dict, List

from ....utils.dist_utils import all_reduce
from ....utils.omni_helper import OmniEnvironMeter
from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ....arguments import VeOmniArguments
    from ...omni.omni_trainer import OmniTrainer


class OmniEnvironMeterCallback(Callback):
    """Per-module metric metering for OmniModel V2.

    The single-model :class:`EnvironMeterCallback` can't meter ``OmniModel``: it
    is a composition of sub-modules with no single ``model_type`` to estimate
    FLOPs on, and the entry batch carries only ``conversation_list`` (no
    ``input_ids`` / ``attention_mask`` to count tokens from).  So **FLOPs / MFU**
    are computed per-module by each module's
    :class:`~veomni.models.seed_omni.mixins.metric_meter_mixin.MetricMeterMixin`.  This callback drives the
    timing, collects every module's ``(theoretical_flops, seqlens)`` via
    :meth:`~veomni.trainer.omni.omni_trainer.OmniTrainer.collect_step_metrics`,
    and hands them to :class:`~veomni.utils.omni_helper.OmniEnvironMeter`, which owns the **global**
    roll-up — merged batch-token statistics, multi-source accounting, and
    device/host memory.

    ``self.trainer`` here is the :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`
    orchestrator; it writes :attr:`~OmniTrainer.environ_meter`,
    :attr:`~OmniTrainer.step_env_metrics`, and :attr:`~OmniTrainer.step_train_metrics`
    each step for :class:`~veomni.trainer.callbacks.WandbTraceCallback` /
    :class:`~veomni.trainer.callbacks.TqdmCallback` / :class:`ChannelLossCallback`.
    """

    trainer: "OmniTrainer"

    def __init__(self, trainer: "OmniTrainer") -> None:
        super().__init__(trainer)
        args: "VeOmniArguments" = trainer.args
        trainer.environ_meter = OmniEnvironMeter(
            global_batch_size=args.train.global_batch_size,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            empty_cache_steps=args.train.empty_cache_steps,
            gc_steps=args.train.gc_steps,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches or []:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time

        module_metrics = self.trainer.collect_step_metrics()
        step_env_metrics = self.trainer.environ_meter.step(delta_time, state.global_step, module_metrics)

        step_train_metrics = {
            "total_loss": loss,
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=self.parallel_state.fsdp_group) for k, v in step_train_metrics.items()
        }

        if self.trainer.lr_scheduler is not None:
            lr = max(self.trainer.lr_scheduler.get_last_lr())
            step_train_metrics["training/lr"] = lr

        step_env_metrics.update(step_train_metrics)

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


__all__ = ["OmniEnvironMeterCallback"]
