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

"""Scheduler for SeedOmni graph-profiler lifecycle at **step** granularity.

Owns the config gate (``enable_*`` detail switches, rank 0, ``train_start_step`` /
``train_end_step``).  Init / flush of per-model profilers lives on
:meth:`OmniTrainer.init_graph_profile` / :meth:`OmniTrainer.flush_graph_profile`.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


class GraphProfileCallback(Callback):
    """Schedule graph-profiler init / flush from ``train.graph_profile``."""

    trainer: "OmniTrainer"

    def __init__(self, trainer: "OmniTrainer") -> None:
        super().__init__(trainer)
        profile = trainer.args.train.graph_profile
        self._enabled = profile.enable_graph_profiling()
        self._start_step = profile.train_start_step
        self._end_step = profile.train_end_step

    def _should_profile(self, state: TrainerState) -> bool:
        if not self._enabled or self.trainer.args.train.global_rank != 0:
            return False
        return self._start_step <= state.global_step <= self._end_step

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        if self._should_profile(state):
            self.trainer.init_graph_profile()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        # Always flush: no-op when this step was outside the window / not rank 0.
        self.trainer.flush_graph_profile(state)


__all__ = ["GraphProfileCallback"]
