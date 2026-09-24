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

from typing import TYPE_CHECKING, Dict, List

import torch.distributed as dist

from ....utils.dist_utils import all_reduce
from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


class OmniStepMetricsCallback(Callback):
    """Per-step training metrics (loss / per-node losses / grad norm / lr) for OmniModel.

    The single-model :class:`EnvironMeterCallback` cannot run here: an
    ``OmniModel`` has no single ``model_type`` to estimate FLOPs on, and its
    batch carries only ``conversation_list`` (no ``input_ids`` to count tokens
    from). This callback therefore publishes only what the train step itself
    knows: total loss and grad norm averaged over the FSDP group, and each
    node's loss averaged over the ranks whose batch produced it.

    It writes :attr:`~OmniTrainer.step_train_metrics` (read by
    :class:`~veomni.trainer.callbacks.TqdmCallback`) and
    :attr:`~OmniTrainer.step_env_metrics` (logged by
    :class:`~veomni.trainer.callbacks.WandbTraceCallback`), so it must run before
    both in the handler list.
    """

    trainer: "OmniTrainer"

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
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
        step_train_metrics["training/lr"] = max(self.trainer.lr_scheduler.get_last_lr())

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = dict(step_train_metrics)


__all__ = ["OmniStepMetricsCallback"]
