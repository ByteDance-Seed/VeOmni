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

import torch.distributed as dist

from ....utils.dist_utils import all_reduce
from ....utils.omni_helper import OmniEnvironMeter
from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


class OmniStepMetricsCallback(Callback):
    """Per-step training + efficiency metrics for OmniModel.

    The single-model :class:`EnvironMeterCallback` cannot run here: an
    ``OmniModel`` has no single ``model_type`` to estimate FLOPs on, and its
    batch carries only ``conversation_list`` (no ``input_ids`` to count tokens
    from). Instead each metered module reports its own tokens and FLOPs
    (:meth:`~veomni.models.seed_omni.accelerated.OmniModelRuntime.metric_meter_collect`)
    and :class:`~veomni.utils.omni_helper.OmniEnvironMeter` rolls them up over
    the whole-step wall-clock. On top of that it publishes total loss and grad
    norm averaged over the FSDP group, and each node's loss averaged over the
    ranks whose batch produced it.

    It writes :attr:`~OmniTrainer.step_train_metrics` (read by
    :class:`~veomni.trainer.callbacks.TqdmCallback`) and
    :attr:`~OmniTrainer.step_env_metrics` (logged by
    :class:`~veomni.trainer.callbacks.WandbTraceCallback`), so it must run before
    both in the handler list.
    """

    trainer: "OmniTrainer"

    def __init__(self, trainer: "OmniTrainer") -> None:
        super().__init__(trainer)
        args = trainer.args
        trainer.environ_meter = OmniEnvironMeter(
            global_batch_size=args.train.global_batch_size,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            empty_cache_steps=args.train.empty_cache_steps,
            gc_steps=args.train.gc_steps,
            parallel_state=self.parallel_state,
        )

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        step_env_metrics = self.trainer.environ_meter.step(
            delta_time,
            global_step=state.global_step,
            module_metrics=self.trainer.model.metric_meter_collect(),
        )

        group = self.parallel_state.fsdp_group
        # A node records a loss only when its batch produced one, and ranks see
        # different modality mixes: reduce the union of keys in one order, or
        # ranks issue different all_reduce sequences and hang or mis-pair.
        rank_keys: List[List[str]] = [[] for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(rank_keys, sorted(loss_dict), group=group)
        node_keys = sorted(set().union(*rank_keys))

        step_train_metrics = {
            "training/total_loss": all_reduce(loss, group=group),
            "training/grad_norm": all_reduce(grad_norm, group=group),
        }
        if node_keys:
            # Average each node over the ranks that produced it: a rank without
            # the node must not pull its logged loss toward zero.
            values = [loss_dict.get(key, 0.0) for key in node_keys]
            counts = [float(key in loss_dict) for key in node_keys]
            sums = all_reduce(values + counts, op="sum", group=group)
            for i, key in enumerate(node_keys):
                step_train_metrics[f"training/{key}"] = sums[i] / sums[len(node_keys) + i]
        # None only for offline_cache, which freezes every module.
        lr_scheduler = self.trainer.model.lr_scheduler
        if lr_scheduler is not None:
            step_train_metrics["training/lr"] = max(lr_scheduler.get_last_lr())

        step_env_metrics.update(step_train_metrics)

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics


__all__ = ["OmniStepMetricsCallback"]
