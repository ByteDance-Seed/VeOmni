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

import os
from typing import TYPE_CHECKING, Any, Dict, List

from ...models.seed_omni.graphs import GraphProfiler
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ...arguments import OmniArguments
    from ..omni.omni_trainer import OmniTrainer


class GraphProfileCallback(Callback):
    """Own the SeedOmni graph-profiler lifecycle at **step** granularity.

    One graph profile is emitted per training step: :meth:`on_step_begin` builds a
    fresh profiler (gated to rank-0 inside ``[train_start_step, train_end_step]``)
    and stashes it on the orchestrator as ``trainer.graph_profiler`` for the
    forward pass to consume; :meth:`on_step_end` writes the records out and clears
    the slot. With gradient accumulation the same profiler spans every micro-batch
    of the step, so its records cover the whole step (node lines repeat per
    micro-batch and memory-peak is the whole-step peak).

    :class:`~veomni.models.seed_omni.graphs.profiling.GraphProfiler` itself stays
    in ``graphs/profiling.py`` — the inferencer builds it directly (ungated, per
    request), so only the *train-side lifecycle* is centralized here.

    ``self.trainer`` is the :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`
    orchestrator; args / output_dir live under ``self.trainer.base``.
    """

    def __init__(self, trainer: "OmniTrainer") -> None:
        super().__init__(trainer)
        # The slot the forward pass reads (``base.model(profiler=trainer.graph_profiler)``).
        trainer.graph_profiler = None

    def _should_save(self, state: TrainerState) -> bool:
        # Gated: rank-0 only, within the [train_start_step, train_end_step] window,
        # and only when a detail switch is on — so most steps carry no profiler.
        args: "OmniArguments" = self.trainer.base.args
        profile = args.graph_profile
        if args.train.global_rank != 0:
            return False
        if not profile.enable_graph_profiling():
            return False
        return profile.train_start_step <= state.global_step <= profile.train_end_step

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        if not self._should_save(state):
            self.trainer.graph_profiler = None
            return
        self.trainer.graph_profiler = GraphProfiler.from_config(self.trainer.base.args.graph_profile)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        profiler = self.trainer.graph_profiler
        if profiler is None:
            return

        args: "OmniArguments" = self.trainer.base.args
        trace_dir = os.path.join(args.train.checkpoint.output_dir, "graph_trace")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(
            trace_dir,
            f"step_{state.global_step:06d}_rank_{args.train.global_rank}.txt",
        )
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(profiler.save_records()) + "\n")
        logger.info_rank0(f"OmniTrainer: graph profile trace → {trace_path}")
        self.trainer.graph_profiler = None
