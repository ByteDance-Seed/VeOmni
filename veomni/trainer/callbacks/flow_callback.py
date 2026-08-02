# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Per-micro-step RNG identity for flow-matching (diffusion) training."""

from typing import TYPE_CHECKING, Any, Dict, List

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = helper.create_logger(__name__)


class FlowStepContextCallback(Callback):
    """Stamp each micro-batch with the logical identity of its flow RNG draw.

    Flow-matching samples a posterior, a timestep, and diffusion noise per
    micro-batch. Every SP/EP rank of one logical micro-batch must draw the same
    values, and a DCP resume must replay the same sequence, so the identity is
    derived rather than drawn from ambient RNG: ``veomni.schedulers.flow_matching``
    hashes ``(train_seed, data_replica_rank, optimizer_step, micro_step)`` into a
    per-stream seed. The rank is the pure data-parallel replica rank, so ranks that
    differ only by SP/EP share one identity and therefore one draw.

    Inert unless the model config carries a ``flow`` recipe, so text/VLM runs pay
    nothing.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        model_config = getattr(trainer, "model_config", None)
        self.enabled = getattr(model_config, "flow", None) is not None
        if self.enabled:
            logger.info_rank0(f"Flow-matching training enabled with config: {model_config.flow}")

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        if not self.enabled or not micro_batches:
            return
        train_seed = int(self.trainer.args.train.seed)
        data_replica_rank = int(self.parallel_state.dp_rank)
        for micro_step, micro_batch in enumerate(micro_batches):
            micro_batch["flow_step_context"] = {
                "train_seed": train_seed,
                "data_replica_rank": data_replica_rank,
                "optimizer_step": int(state.global_step),
                "micro_step": micro_step,
            }
