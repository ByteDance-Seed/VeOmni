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
"""`OmniTrainer` must hand every module where the step sits, before it runs it.

A module whose forward draws randomness per micro-batch keys the draw on
``(seed, global_step, micro_step)`` so a resume reproduces it. The step half comes from
this hook, and the micro-batch counter resets here — so deleting the call would make the
draw depend on whatever the previous step left behind, with nothing in the logs to show
it. That is what a module-level test cannot see, which is why the wiring is pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace

from veomni.trainer.omni.omni_trainer import OmniTrainer


class _RecordingModule:
    """A module that records what the trainer tells it, in the order it arrives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def on_micro_batches_ready(self, micro_batches):
        self.calls.append(("micro_batches_ready", {"n": len(micro_batches)}))

    def on_step_context(self, *, seed, global_step, batch_invariant):
        self.calls.append(
            ("step_context", {"seed": seed, "global_step": global_step, "batch_invariant": batch_invariant})
        )


def _trainer_with(module, *, global_step=7, seed=42, batch_invariant=False):
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(seed=seed, enable_batch_invariant_mode=batch_invariant),
    )
    trainer.state = SimpleNamespace(global_step=global_step)
    # `_notify_modules_of_step` unwraps each runtime's model; a bare module is its own
    # unwrapped form, so a one-entry map is the whole contract it needs.
    trainer.model = SimpleNamespace(module_runtimes={"dg": SimpleNamespace(model=module)})
    return trainer


def test_the_step_context_follows_the_sample_count():
    """Order matters: the counter resets before the first `pre_forward` reads it."""
    module = _RecordingModule()

    _trainer_with(module)._notify_modules_of_step([{}, {}, {}])

    assert [name for name, _ in module.calls] == ["micro_batches_ready", "step_context"]
    assert module.calls[1][1] == {"seed": 42, "global_step": 7, "batch_invariant": False}


def test_the_global_step_is_the_one_already_advanced():
    """A step's draw is keyed on the step it belongs to, and `train_step` increments
    `state.global_step` before notifying. Passing the pre-increment value would key step
    N's masks on N-1 and break the resume replay."""
    module = _RecordingModule()

    _trainer_with(module, global_step=9)._notify_modules_of_step([{}])

    assert module.calls[-1][1]["global_step"] == 9


def test_a_module_without_the_hook_is_left_alone():
    """The hook is new, so every module but DiffusionGemma's has no such method."""
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(seed=1, enable_batch_invariant_mode=False),
    )
    trainer.state = SimpleNamespace(global_step=1)
    trainer.model = SimpleNamespace(module_runtimes={"plain": SimpleNamespace(model=object())})

    trainer._notify_modules_of_step([{}])
