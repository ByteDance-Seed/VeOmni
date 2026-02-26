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

from typing import TYPE_CHECKING

from veomni.trainer.callbacks.base import TrainerState

from .base import Callback


if TYPE_CHECKING:
    from ..base import Arguments


class EvaluateCallback(Callback):
    def on_epoch_end(self, state: TrainerState, **kwargs):
        args: "Arguments" = self.trainer.args
        if args.train.eval_epochs and (state.epoch + 1) % args.train.eval_epochs == 0:
            self._evaluate(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "Arguments" = self.trainer.args
        if args.train.eval_steps and state.global_step % args.train.eval_steps == 0:
            self._evaluate(state)

    def _evaluate(self, state: TrainerState):
        # TODO: implement evaluate
        pass
