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

from typing import Dict, List

import torch

from veomni.data import build_evaluator, build_validation_dataloader
from veomni.trainer.callbacks.base import TrainerState
from veomni.utils import logging

from .base import Callback


logger = logging.get_logger(__name__)


class EvaluateCallback(Callback):
    """Runs validation evaluation at configured intervals.

    Triggered by ``train.eval_steps`` (per-step) and ``train.eval_epochs``
    (per-epoch) from :class:`VeOmniArguments`.  When triggered, builds a
    validation dataloader from ``data.eval_path`` (cached on first use),
    runs the model in ``eval`` mode under ``torch.no_grad``, computes the
    metrics listed in ``train.validation_metrics``, and logs results to
    stdout and wandb (if enabled).
    """

    _val_dataloader = None
    _evaluators: List = None

    def on_epoch_end(self, state: TrainerState, **kwargs):
        args = self.trainer.args
        if args.train.eval_epochs and (state.epoch + 1) % args.train.eval_epochs == 0:
            self._evaluate(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args = self.trainer.args
        if args.train.eval_steps and state.global_step % args.train.eval_steps == 0:
            self._evaluate(state)

    def _ensure_built(self):
        """Lazily build the validation dataloader and evaluators."""
        if self._val_dataloader is not None:
            return

        args = self.trainer.args

        # Build validation dataloader
        self._val_dataloader = build_validation_dataloader(
            args=args,
            tokenizer=self.trainer.tokenizer,
            chat_template=getattr(self.trainer, "chat_template", None),
        )

        # Build evaluators from config
        metric_names = getattr(args.train, "validation_metrics", ["loss"]) or ["loss"]
        self._evaluators = [build_evaluator(name) for name in metric_names]

        if self._val_dataloader is None:
            logger.warning_rank0(
                "Validation is enabled (eval_steps/eval_epochs > 0) but data.eval_path is not set. "
                "Skipping evaluation."
            )

    def _evaluate(self, state: TrainerState):
        """Run a full validation pass and log metrics."""
        self._ensure_built()
        if self._val_dataloader is None:
            return

        args = self.trainer.args
        model = self.trainer.model

        # Guard: skip evaluation for models that don't produce logits
        # (e.g. DiT diffusion models use a different forward signature)
        if not hasattr(model, "use_cache"):
            logger.info_rank0(
                "Skipping validation: model does not support use_cache "
                "(likely a non-causal-LM model). Evaluation is only supported "
                "for text generation models."
            )
            return

        # Switch to eval mode
        was_training = model.training
        model.eval()

        # Set epoch on the sampler for deterministic sharding
        if hasattr(self._val_dataloader, "set_epoch"):
            self._val_dataloader.set_epoch(state.epoch)

        logger.info_rank0(
            f"Starting validation at step {state.global_step}, epoch {state.epoch}."
        )

        # Accumulate partial sums per evaluator (each gets its own dict)
        accumulated: List[Dict[str, torch.Tensor]] = [{} for _ in self._evaluators]

        with torch.no_grad():
            for batch_idx, micro_batches in enumerate(self._val_dataloader):
                # The dataloader yields a list of micro-batches (same format as training).
                # For validation we process each micro-batch and accumulate.
                if not isinstance(micro_batches, list):
                    micro_batches = [micro_batches]

                for micro_batch in micro_batches:
                    # Move tensors to device
                    moved = {}
                    for k, v in micro_batch.items():
                        if isinstance(v, torch.Tensor):
                            moved[k] = v.to(self.trainer.device, non_blocking=True)
                        else:
                            moved[k] = v

                    # Forward pass
                    outputs = model(**moved, use_cache=False)

                    logits = getattr(outputs, "logits", None)
                    labels = moved.get("labels")
                    if logits is None or labels is None:
                        continue

                    # Compute partial metrics for this batch, per evaluator
                    for i, evaluator in enumerate(self._evaluators):
                        partial = evaluator.compute(
                            logits, labels, loss=outputs.loss,
                        )
                        for k, v in partial.items():
                            if k in accumulated[i]:
                                accumulated[i][k] += v.detach()
                            else:
                                accumulated[i][k] = v.detach()

        # Aggregate using each evaluator's own aggregate() method
        results: Dict[str, float] = {}
        for i, evaluator in enumerate(self._evaluators):
            results.update(evaluator.aggregate(accumulated[i]))

        # Log results
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in results.items())
        logger.info_rank0(
            f"Validation results at step {state.global_step}: {metrics_str}"
        )

        # Log to wandb if enabled
        if args.train.wandb.enable:
            try:
                import wandb

                wandb.log(
                    {f"val/{k}": v for k, v in results.items()},
                    step=state.global_step,
                )
            except Exception:
                logger.warning_rank0("Failed to log validation metrics to wandb.")

        # Restore training mode
        if was_training:
            model.train()
