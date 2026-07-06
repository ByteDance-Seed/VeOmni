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

"""``DraftModelTrainer`` — train DeepSpec draft models on VeOmni's engine.

This trainer subclasses VeOmni's ``BaseTrainer`` and reuses almost all of it
(distributed init, FSDP2 parallelization, meta-init weight loading, DCP
checkpointing, callbacks, optimizer / LR scheduler). It overrides exactly the
five seams where the DeepSpec draft-model recipe differs from a vanilla causal
LM:

1. ``_build_model``     — build the draft model with ``flex_attention`` (DeepSpec
                          requires it) and, under meta-init, load frozen target
                          embeddings + lm_head from the prepared init checkpoint.
2. ``_freeze_model_module`` — mark ``embed_tokens`` / ``lm_head`` frozen so
                          VeOmni's optimizer skips them (it only takes
                          ``requires_grad`` params).
3. ``_build_dataset`` / ``_build_collate_fn`` — feed DeepSpec's target-cache
                          dataset + collator instead of the text pipeline.
4. ``forward_backward_step`` — call the model with DeepSpec's batch keys and run
                          DeepSpec's own loss function; DeepSpec scales the loss
                          by the DP world size to cancel FSDP gradient averaging.
5. ``train_step``       — divide the per-micro-batch loss by the number of micro
                          batches (DeepSpec-style grad accumulation) and flush
                          DeepSpec's metric buffer into VeOmni's logging.

Why the DeepSpec loss is *correct* under FSDP2 data parallel: DeepSpec's
``compute_dspark_loss`` / ``compute_eagle3_loss`` compute a globally-correct mean
by all-reducing loss denominators over the whole process group and then
multiplying the backward loss by ``world_size``. FSDP2 later divides gradients
by the shard-group size. When the *only* parallelism is FSDP data parallelism
(``dp_shard`` spans the whole world, everything else size 1), the shard group ==
the world, so ``×world_size`` exactly cancels the ``÷world_size`` FSDP applies —
identical semantics to DeepSpec's original ``FSDP.NO_SHARD`` setup, but now with
real parameter/optimizer/gradient sharding for the memory + throughput win.
That equivalence breaks under SP / TP / EP / PP / HSDP-replicate, so the trainer
asserts those are all disabled (see ``_validate_parallelism``).
"""

from collections import defaultdict
from typing import Any, Dict, List

import torch

from ...distributed.clip_grad_norm import veomni_clip_grad_norm
from ...distributed.parallel_state import get_parallel_state
from ...integrations.deepspec import ensure_deepspec_importable
from ...models import build_foundation_model, build_tokenizer
from ...ops.config.singleton import get_ops_config
from ...utils import helper
from ...utils.device import synchronize
from ..base import BaseTrainer


logger = helper.create_logger(__name__)


class DraftModelTrainer(BaseTrainer):
    """Trainer for DeepSpec draft models (DSpark / DFlash / Eagle3)."""

    # DeepSpec's dataloader yields a flat batch that we split into micro batches
    # through VeOmni's fixed-sample dataloader; no chat template / tokenized text.
    LOG_SAMPLE = True

    def __init__(self, args):
        ensure_deepspec_importable()
        # Which DeepSpec algorithm family: inferred from the draft architecture
        # recorded in the model config (``*DSparkModel`` vs ``*Eagle3Model``).
        self._algo = None
        super().__init__(args)

    # ------------------------------------------------------------------ #
    # Setup guards
    # ------------------------------------------------------------------ #
    def _setup(self):
        super()._setup()
        self._validate_parallelism()

    def _validate_parallelism(self):
        """DeepSpec's loss math is only correct under pure FSDP data parallel.

        DeepSpec computes a globally-correct mean by all-reducing loss
        denominators over the whole world and multiplying the backward loss by
        ``world_size`` to undo FSDP's gradient averaging. That identity holds
        only when the FSDP shard group spans the entire world — i.e. every other
        parallel dim is size 1. Fail loudly otherwise instead of silently
        mis-scaling gradients.
        """
        acc = self.args.train.accelerator
        offending = {
            "ulysses_size (sequence parallel)": acc.ulysses_size,
            "tp_size (tensor parallel)": acc.tp_size,
            "pp_size (pipeline parallel)": acc.pp_size,
            "cp_size (context parallel)": acc.cp_size,
            "ep_size (expert parallel)": acc.ep_size,
            "dp_replicate_size (HSDP replicate)": acc.dp_replicate_size,
        }
        bad = {name: size for name, size in offending.items() if size and int(size) > 1}
        if bad:
            raise ValueError(
                "DeepSpec draft-model training currently supports pure FSDP data "
                "parallelism only (so its world-size loss scaling cancels FSDP "
                "gradient averaging exactly). Disable these parallel dims (set to "
                f"1): {bad}. Sequence/expert/tensor/pipeline parallel support is a "
                "follow-up: the loss reduction in deepspec would need to reduce "
                "over the fsdp/dp group and account for the sequence split."
            )
        # flex_attention (DeepSpec's training attention) is incompatible with
        # VeOmni's SP-aware flash-attn kernel; make the requirement explicit.
        ops_config = get_ops_config()
        if ops_config is not None and ops_config.attn_implementation not in ("flex_attention", "eager", "sdpa"):
            logger.warning_rank0(
                f"DeepSpec draft models train with flex_attention; the configured "
                f"attn_implementation={ops_config.attn_implementation!r} will be "
                "overridden to 'flex_attention' when building the model."
            )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    def _build_model(self):
        args = self.args
        logger.info_rank0("Build DeepSpec draft model")
        mixed_precision = args.train.accelerator.fsdp_config.mixed_precision.enable

        # DeepSpec models require flex_attention. VeOmni's OpsImplementationConfig
        # cannot express that literal (and rewrites flash_attention_2 -> SP kernel).
        # build_foundation_model overwrites its attn_implementation arg with
        # ops_implementation.attn_implementation whenever ops_implementation is
        # passed, so pre-install the ops singleton ourselves and pass
        # ops_implementation=None + an explicit attn_implementation. The
        # "singleton already installed" branch then honours our explicit value.
        from ...ops import apply_ops_config

        apply_ops_config(args.model.ops_implementation)

        self.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if mixed_precision else "bfloat16",
            attn_implementation="flex_attention",
            init_device=args.train.init_device,
            ops_implementation=None,
            config_kwargs=args.model.model_config,
        )
        self.model_config = self.model.config
        self._algo = _infer_algorithm(self.model_config)
        logger.info_rank0(f"DeepSpec draft algorithm: {self._algo}")

    def _freeze_model_module(self):
        """Freeze the target-copied embeddings and lm_head.

        DeepSpec never trains ``embed_tokens`` / ``lm_head`` (they are copied
        from the frozen target). VeOmni's ``build_optimizer`` only optimizes
        ``requires_grad`` params, so flipping ``requires_grad`` here is enough to
        keep them out of the optimizer and out of grad-norm clipping.
        """
        model = self.model
        if hasattr(model, "set_embedding_head_trainable"):
            model.set_embedding_head_trainable(False)
        else:  # defensive: freeze by name if the helper is absent
            for name, param in model.named_parameters():
                if name.startswith("embed_tokens") or name.startswith("lm_head"):
                    param.requires_grad_(False)

        from ...utils.model_utils import pretty_print_trainable_parameters

        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building draft model")

    def _build_model_assets(self):
        # Draft models are evaluated against the target tokenizer; save it with
        # the checkpoint so downstream eval can load it. tokenizer_path defaults
        # to the (target) config_path when unset.
        self.tokenizer = build_tokenizer(self.args.model.tokenizer_path)
        self.model_assets = [self.model_config, self.tokenizer]

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _build_data_transform(self):
        # No text transform: the target cache already stores tensors.
        self.data_transform = None

    def _build_dataset(self):
        from ...data.deepspec import build_target_cache_dataset

        args = self.args
        cache_dir = args.data.train_path
        self.train_dataset = build_target_cache_dataset(cache_dir)
        self._validate_cache_against_model(self.train_dataset)

        dataset_length = len(self.train_dataset) / get_parallel_state().dp_size
        args.compute_train_steps(dataset_length)
        self.train_steps = args.train_steps
        logger.info_rank0(
            f"Target cache: {len(self.train_dataset)} samples, "
            f"dp_size={get_parallel_state().dp_size}, train_steps={self.train_steps}."
        )

    def _validate_cache_against_model(self, dataset):
        """Mirror DeepSpec's ``validate_train_cache`` on the built model."""
        model = self.model
        expected_layer_ids = [int(x) for x in getattr(model, "target_layer_ids", [])]
        cache_layer_ids = [int(x) for x in dataset.target_layer_ids]
        if expected_layer_ids and cache_layer_ids != expected_layer_ids:
            raise ValueError(
                "Target cache target_layer_ids do not match the draft model: "
                f"{cache_layer_ids} != {expected_layer_ids}."
            )
        if int(dataset.hidden_size) != int(self.model_config.hidden_size):
            raise ValueError(
                "Target cache hidden_size does not match the draft model hidden "
                f"size: {dataset.hidden_size} != {self.model_config.hidden_size}."
            )

    def _build_collate_fn(self):
        from ...data.deepspec import build_cache_collator

        self.collate_fn = build_cache_collator()

    # ------------------------------------------------------------------ #
    # Forward / backward
    # ------------------------------------------------------------------ #
    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        micro_batch = {
            key: (value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in micro_batch.items()
        }
        # input_ids come from the cache as int32; the modeling code and embedding
        # lookup expect long. Do the tiny cast on-device.
        if "input_ids" in micro_batch:
            micro_batch["input_ids"] = micro_batch["input_ids"].long()
        if getattr(self, "LOG_SAMPLE", False):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    def forward_backward_step(self, micro_batch: Dict[str, torch.Tensor]):
        micro_batch = self.preforward(micro_batch)

        with self.model_fwd_context:
            loss = self._compute_deepspec_loss(micro_batch)

        # DeepSpec already scales the loss by world_size to cancel FSDP grad
        # averaging; divide only by the number of micro batches for grad accum.
        loss = loss / self._num_micro_steps

        with self.model_bwd_context:
            loss.backward()

        del micro_batch
        return loss, {}

    def _compute_deepspec_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._algo == "dspark":
            from deepspec.modeling.dspark.loss import compute_dspark_loss

            outputs = self.model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch["target_last_hidden_states"],
            )
            weights = self._dspark_loss_weights()
            return compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=weights["loss_decay_gamma"],
                ce_loss_alpha=float(weights["ce_loss_alpha"]),
                l1_loss_alpha=float(weights["l1_loss_alpha"]),
                confidence_head_alpha=float(weights["confidence_head_alpha"]),
            )
        elif self._algo == "eagle3":
            from deepspec.modeling.eagle3.loss import compute_eagle3_loss

            return compute_eagle3_loss(
                model=self.model,
                batch=batch,
                ttt_length=int(getattr(self.model, "ttt_length", self.model_config.ttt_length)),
                step_loss_decay=float(getattr(self.model, "step_loss_decay", self.model_config.step_loss_decay)),
            )
        raise ValueError(f"Unknown DeepSpec algorithm {self._algo!r}.")

    def _dspark_loss_weights(self) -> Dict[str, Any]:
        """Resolve DSpark loss weights: model config first, YAML override last.

        The weights are baked into the draft config.json by
        ``prepare_draft_init.py`` (single source of truth) and may be overridden
        per run via ``model.model_config`` in the YAML.
        """
        overrides = self.args.model.model_config or {}
        defaults = {
            "loss_decay_gamma": 4.0,
            "ce_loss_alpha": 1.0,
            "l1_loss_alpha": 0.0,
            "confidence_head_alpha": 0.0,
        }
        resolved = {}
        for key, default in defaults.items():
            if key in overrides:
                resolved[key] = overrides[key]
            else:
                resolved[key] = getattr(self.model_config, key, default)
        return resolved

    # ------------------------------------------------------------------ #
    # Train step (DeepSpec-style grad accumulation + metric bridging)
    # ------------------------------------------------------------------ #
    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        args = self.args
        self.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)
        self._num_micro_steps = max(1, len(micro_batches))

        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, self._num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch)
            total_loss += loss.item()
            for key, value in loss_dict.items():
                total_loss_dict[key] += value.item() if torch.is_tensor(value) else float(value)

        grad_norm = veomni_clip_grad_norm(self.model, args.train.optimizer.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        # Bridge DeepSpec's own metric buffer (ce_loss, l1_loss, accept_rate@k,
        # confidence_*, tau_probabilistic, ...) into VeOmni's step metrics.
        total_loss_dict.update(self._flush_deepspec_metrics())

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def _flush_deepspec_metrics(self) -> Dict[str, float]:
        """Flush DeepSpec's distributed metric accumulator (all ranks call it).

        ``deepspec.utils.metrics.flush`` performs collective all-reduces and
        asserts a consistent metric schema across ranks, so every rank must call
        it exactly once per step.
        """
        try:
            from deepspec.utils import metrics as deepspec_metrics
        except Exception:
            return {}
        try:
            summary = deepspec_metrics.flush()
        except Exception as exc:  # never let metric logging break training
            logger.warning_rank0(f"Failed to flush DeepSpec metrics: {exc}")
            deepspec_metrics.reset()
            return {}
        # Strip the ``train/`` prefix DeepSpec adds; VeOmni re-namespaces.
        return {key.split("/", 1)[-1]: value for key, value in summary.items()}


def _infer_algorithm(config) -> str:
    """Return ``"dspark"`` or ``"eagle3"`` from the draft config architecture."""
    architectures = getattr(config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    if "Eagle3" in arch:
        return "eagle3"
    if "DSpark" in arch:
        return "dspark"
    # Fall back to config field presence (Eagle3 configs carry ttt_length).
    if hasattr(config, "ttt_length"):
        return "eagle3"
    return "dspark"


__all__ = ["DraftModelTrainer"]
