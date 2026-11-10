"""
Qwen3 + Qwen3-VL visual-instruction-tuning recipe, driven through ``OmniTrainer``.

The composition acceptance test (``tests/seed_omni/modules/qwen3/
test_qwen3_visual_instruction_tuning.py``) builds the graph in-process, which
pins the wiring but not the launcher: config resolution, checkpoint conversion,
the data pipeline and the per-module optimizers are all bypassed there. This
runner takes the user-facing path instead — ``tasks/omni/train_omni.py``'
trainer, the recipe's own YAML, split checkpoints written by
``scripts/seed_omni/convert_model.py`` — and records the loss per step so
``tests/models/test_omni_visual_instruction_tuning.py`` can compare SP=1 against
SP=4 and check that the toy run descends.

Launched via ``torchrun`` with ``--ulysses-size``; paths come from the
environment (``OMNI_VIT_RUNNER_DIR``) so the pytest side owns the fixtures.
"""

from __future__ import annotations

import json
import os

from veomni.arguments.omni_arguments_types import OmniArguments
from veomni.arguments.omni_parser import parse_omni_args
from veomni.trainer.callbacks import Callback, TrainerState
from veomni.trainer.omni import OmniTrainer


os.environ.setdefault("NCCL_DEBUG", "OFF")

RUNNER_DIR = os.environ["OMNI_VIT_RUNNER_DIR"]


class _LossRecorder(Callback):
    """Append every step's loss (already reduced) to a JSON file on rank 0."""

    def __init__(self, trainer: OmniTrainer) -> None:
        """Start with empty curves; ``on_step_end`` fills them."""
        super().__init__(trainer)
        self.losses: list[float] = []
        self.grad_norms: list[float] = []

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        """Record the step metrics ``OmniStepMetricsCallback`` just reduced.

        Read rather than re-reduced: that callback already all-reduces the total
        loss and grad norm over the FSDP group and stores the result on the
        trainer, so a second collective here would be both redundant and a way
        to desynchronise the ranks.
        """
        del state, kwargs
        metrics = self.trainer.step_train_metrics or {}
        self.losses.append(float(metrics["training/total_loss"]))
        self.grad_norms.append(float(metrics["training/grad_norm"]))

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        """Write the recorded curve to ``losses.json`` in the runner directory."""
        del state, kwargs
        if self.trainer.args.train.global_rank != 0:
            return
        with open(os.path.join(RUNNER_DIR, "losses.json"), "w", encoding="utf-8") as f:
            json.dump({"losses": self.losses, "grad_norms": self.grad_norms}, f, indent=2)


class _RecordingTrainer(OmniTrainer):
    """``OmniTrainer`` with the loss recorder appended after the standard callbacks."""

    def _init_callbacks(self) -> None:
        """Install the standard callbacks, then the loss recorder."""
        super()._init_callbacks()
        self.loss_recorder = _LossRecorder(self)
        self._callbacks.append(self.loss_recorder)


def main() -> None:
    """Run the recipe once at the ``--ulysses-size`` the launcher asked for."""
    args = parse_omni_args(OmniArguments, preload_path_fields=("model.model_config.modules",))
    trainer = _RecordingTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.destroy_distributed()


if __name__ == "__main__":
    main()
