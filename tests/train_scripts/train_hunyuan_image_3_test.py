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

"""Test entry for the Hunyuan Image 3 T2I end-to-end run.

Uses the REAL VLMTrainer generation path (transform / collator hooks / flow
injection) and only adds a per-step loss logger so the e2e test can assert finite
losses. Unlike train_vlm_test.py it does not stub the data transform.

Also captures per-step timing and bucket-sampler state when available (P4
verification) — the extra fields are inert for non-HI3 trainers.
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict

from veomni.arguments import parse_args
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer


class HunyuanImage3TestTrainer(VLMTrainer):
    def __init__(self, args: VeOmniVLMArguments):
        super().__init__(args)
        self.base.logdictsave_callback = _LogDictSaveCallback(self.base)

    def on_train_begin(self):
        super().on_train_begin()
        # AFTER the CheckpointerCallback restored the sampler (if resuming), so
        # our snapshot reflects the resumed cursors.
        self.base.logdictsave_callback.on_train_begin(self.base.state)

    def on_train_end(self):
        super().on_train_end()
        self.base.logdictsave_callback.on_train_end(self.base.state)

    def on_step_begin(self, **kwargs):
        super().on_step_begin(**kwargs)
        self.base.logdictsave_callback.on_step_begin(self.base.state, **kwargs)

    def on_step_end(self, **kwargs):
        super().on_step_end(**kwargs)
        self.base.logdictsave_callback.on_step_end(self.base.state, **kwargs)


class _LogDictSaveCallback(Callback):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.log_dict = defaultdict(list)
        self._step_start_ns: int | None = None

    def _append_bucket_state(self, tag: str) -> None:
        """P4 telemetry: snapshot the ``HunyuanImage3BucketBatchSampler`` state so
        the smoke driver can verify the sampler advanced deterministically and
        (post-resume) picked up from the saved cursors.
        """
        sampler = getattr(self.trainer, "bucket_batch_sampler", None)
        if sampler is None:
            return
        try:
            state = sampler.state_dict()
        except Exception:  # noqa: BLE001 — telemetry must never fail training
            return
        # Serialize dicts with int keys → JSON needs str keys.
        state["epochs"] = {str(k): v for k, v in state.get("epochs", {}).items()}
        state["cursors"] = {str(k): v for k, v in state.get("cursors", {}).items()}
        self.log_dict[tag].append(state)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        # Snapshot AFTER the checkpoint-load callback has restored the sampler
        # cursors (P4). This is what a resumed run reports as its "initial" state.
        self._append_bucket_state("bucket_sampler_state_pre_train")

    def on_step_begin(self, state: TrainerState, **kwargs) -> None:
        self._step_start_ns = time.monotonic_ns()

    def on_step_end(self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs):
        self.log_dict["loss"].append(loss)
        for key, value in loss_dict.items():
            self.log_dict[key].append(value)
        self.log_dict["grad_norm"].append(grad_norm)
        self.log_dict["global_step"].append(int(state.global_step))
        if self._step_start_ns is not None:
            self.log_dict["step_wall_ms"].append((time.monotonic_ns() - self._step_start_ns) / 1e6)
            self._step_start_ns = None
        # step_env_metrics is populated by EnvironMeterCallback.on_step_end BEFORE
        # our on_step_end fires (base callbacks run first via super().on_step_end).
        # Grab whichever keys the meter chose to publish — mfu, tokens_per_second,
        # tflops, memory — so the throughput smoke can compute steady-state stats.
        env_metrics = getattr(self.trainer, "step_env_metrics", None)
        if env_metrics is not None:
            # Filter to picklable primitives; skip tensors/objects.
            picklable = {k: v for k, v in env_metrics.items() if isinstance(v, (int, float, str, bool))}
            self.log_dict["step_env_metrics"].append(picklable)
        self._append_bucket_state("bucket_sampler_state")

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.trainer.args.train.global_rank == 0:
            output_dir = self.trainer.args.train.checkpoint.output_dir
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "log_dict.json"), "w") as f:
                json.dump(self.log_dict, f, indent=4, default=str)


if __name__ == "__main__":
    args = parse_args(VeOmniVLMArguments)
    trainer = HunyuanImage3TestTrainer(args)
    trainer.train()
