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

"""
OmniTrainer — trainer for OmniModel V2.

Design
------
Each OmniModule in OmniModel owns its full build lifecycle: it constructs its
foundation model, calls build_parallelize_model, and exposes its model assets
(tokenizer, processor) — exactly like a BaseTrainer does for a monolithic model.

OmniTrainer orchestrates this via OmniModel.build_from_args(), which calls
OmniModule.build() per sub-module. There is no separate _build_parallelized_model
step; parallelisation happens inside each module's build().

Comparison with VLMTrainer
--------------------------
  BaseTrainer._build_model             → OmniTrainer._build_model
    calls build_foundation_model         calls OmniModel.build_from_args()
    and build_parallelize_model            which calls OmniModule.build() per module

  BaseTrainer._build_model_assets      → OmniTrainer._build_model_assets
    builds tokenizer / processor         collects assets from all modules via
                                          OmniModel.collect_assets()

Usage
-----
    from veomni.arguments import parse_args
    from veomni.trainer.omni_trainer import OmniTrainer, VeOmniOmniArguments

    if __name__ == "__main__":
        args = parse_args(VeOmniOmniArguments)
        trainer = OmniTrainer(args)
        trainer.train()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_data_transform
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..models.seed_omni import OmniBuildArgs, OmniConfig, OmniModel
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper
from ..utils.device import synchronize
from ..utils.model_utils import pretty_print_trainable_parameters
from .base import BaseTrainer


logger = helper.create_logger(__name__)


# ── Custom argument dataclasses ────────────────────────────────────────────────


@dataclass
class OmniModelArguments(ModelArguments):
    omni_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the OmniModel YAML config (e.g. configs/seed_omni/janus_1.3b.yaml)."},
    )


@dataclass
class VeOmniOmniArguments(VeOmniArguments):
    model: "OmniModelArguments" = field(default_factory=OmniModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


# ── OmniTrainer ────────────────────────────────────────────────────────────────


class OmniTrainer:
    """Trainer for OmniModel V2 — composable multi-modal training.

    Follows the same BaseTrainer delegation pattern as VLMTrainer and
    TextTrainer: a BaseTrainer instance is created via __new__ and its
    private _build_* helpers are called one-by-one, with OmniModel-specific
    overrides where needed.

    Build flow
    ----------
    ::

        OmniTrainer.__init__
          └── _build_model
                ├── load OmniConfig from omni_config_path
                ├── create OmniBuildArgs from VeOmniArguments
                └── OmniModel.build_from_args(config, build_args)
                      └── for each module:
                            OmniModule.build(cfg, build_args)
                              ├── OmniModule._build_nn_module(cfg, init_device)
                              └── build_parallelize_model(module, ...)

        OmniTrainer.__init__
          └── _build_model_assets
                └── OmniModel.collect_assets()   ← tokenizer / processor from modules
    """

    base: BaseTrainer

    def __init__(self, args: VeOmniOmniArguments):
        # BaseTrainer.__init__ is NOT called; individual helpers are invoked
        # explicitly so the sequence is transparent and overridable.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()

        # _build_model handles both model construction AND parallelisation.
        # There is no separate _build_parallelized_model step.
        self._build_model()
        self._freeze_model_module()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()

    # ── Build helpers ──────────────────────────────────────────────────────────

    def _build_model(self):
        """Load OmniConfig, create OmniBuildArgs, and build OmniModel.

        Parallelisation happens inside OmniModel.build_from_args() → each
        OmniModule.build() calls build_parallelize_model independently, so
        there is no separate _build_parallelized_model step.
        """
        args: VeOmniOmniArguments = self.base.args
        logger.info_rank0("Building OmniModel via OmniModel.build_from_args()")

        omni_config = _load_omni_config(args.model.omni_config_path)
        build_args = _make_build_args(args)

        self.base.model = OmniModel.build_from_args(omni_config, build_args)
        self.base.model_config = omni_config

    def _freeze_model_module(self):
        pretty_print_trainable_parameters(self.base.model)
        helper.print_device_mem_info("VRAM usage after building OmniModel")

    def _build_model_assets(self):
        """Collect model assets (tokenizer, processor) from all sub-modules.

        Each OmniModule.get_assets() is called via OmniModel.collect_assets().
        The first tokenizer found is stored as self.base.tokenizer; the full
        list is stored as self.base.model_assets.
        """
        assets = self.base.model.collect_assets()
        # Expose the first tokenizer found for the data pipeline
        from transformers import PreTrainedTokenizerBase

        tokenizer = next((a for a in assets if isinstance(a, PreTrainedTokenizerBase)), None)
        self.base.tokenizer = tokenizer
        self.base.model_assets = [self.base.model_config] + assets

    def _build_data_transform(self):
        """Build data transform (standard text by default)."""
        args: VeOmniOmniArguments = self.base.args
        self.base.data_transform = build_data_transform(
            args.data.data_type,
            tokenizer=self.base.tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )

    def _build_collate_fn(self):
        """Pick the collator that matches the active ``data_type``.

        For ``data_type == "seedomni"`` the per-sample shape is
        ``{"conversation_list": [...]}`` (list of dicts with possibly
        heterogeneous image tensor shapes), and the V2 design contract
        defers all sequence packing / SP slicing to module
        ``pre_forward`` hooks — so we use the list-only
        ``SeedOmniCollator``.

        For all other ``data_type``s (legacy ``conversation`` / ``dpo`` /
        ``classification`` / ``qwen*_vl`` / ...) we fall back to
        ``BaseTrainer._build_collate_fn`` which builds a ``MainCollator``
        with packing + SP — the V1 contract is preserved verbatim.
        """
        args: VeOmniOmniArguments = self.base.args
        if args.data.data_type == "seedomni":
            from ..data import SeedOmniCollator

            self.base.collate_fn = SeedOmniCollator()
            logger.info_rank0("OmniTrainer: using SeedOmniCollator (list-only) for data_type='seedomni'")
        else:
            self.base._build_collate_fn()

    # ── Forward / backward ─────────────────────────────────────────────────────

    def forward_backward_step(
        self,
        micro_batch: Dict[str, Any],
        num_micro_steps: int,
    ) -> tuple:
        """Forward + backward for one gradient-accumulation micro-batch.

        OmniModel.forward() returns ``{"loss": tensor, "loss_dict": {...}, ...}``.
        The loss is divided by *num_micro_steps* for gradient accumulation.
        """
        micro_batch = self.base.preforward(micro_batch)

        with self.base.model_fwd_context, set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode):
            result: Dict[str, Any] = self.base.model(**micro_batch)

        total_loss: torch.Tensor = result["loss"]
        loss_dict: Dict[str, torch.Tensor] = result.get("loss_dict", {})

        loss = total_loss / num_micro_steps

        with self.base.model_bwd_context, set_batch_invariant_mode(self.base.args.train.enable_batch_invariant_mode):
            loss.backward()

        del micro_batch
        return loss, loss_dict

    def _model_reshard(self, micro_step: int, num_micro_steps: int):
        """Apply set_reshard_after_backward to each FSDP2 sub-module.

        OmniModel itself is not wrapped as a monolith, so we iterate over its
        modules_dict to apply the FSDP2 reshard-after-backward optimisation.
        """
        fsdp_cfg = self.base.args.train.accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward or num_micro_steps <= 1:
            return

        try:
            from torch.distributed._composable.fsdp import FSDPModule
        except ImportError:
            return

        for mod in self.base.model.modules_dict.values():
            if isinstance(mod, FSDPModule):
                if micro_step == 0:
                    mod.set_reshard_after_backward(False)
                elif micro_step == num_micro_steps - 1:
                    mod.set_reshard_after_backward(True)

    # ── Callbacks (delegate to base) ───────────────────────────────────────────

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ── Train loop ─────────────────────────────────────────────────────────────

    def train_step(self, data_iterator: Any) -> None:
        args: VeOmniOmniArguments = self.base.args
        self.base.state.global_step += 1

        micro_batches: List[Dict[str, Any]] = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)

        synchronize()

        total_loss = 0.0
        total_loss_dict: Dict[str, float] = defaultdict(float)
        num_micro_steps = len(micro_batches)

        for micro_step, micro_batch in enumerate(micro_batches):
            self._model_reshard(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch, num_micro_steps)
            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() / num_micro_steps

        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)

        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=dict(total_loss_dict), grad_norm=grad_norm)

    def train(self):
        args: VeOmniOmniArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch

            self.on_epoch_begin()
            data_iterator = iter(self.base.train_dataloader)

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()
            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()
        synchronize()
        self.base.destroy_distributed()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_omni_config(path: Optional[str]) -> OmniConfig:
    """Load OmniConfig from a YAML file."""
    import yaml

    if path is None:
        raise ValueError(
            "model.omni_config_path must be set. "
            "Point it to the OmniModel YAML (e.g. configs/seed_omni/janus_1.3b.yaml)."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return OmniConfig.from_dict(data)


def _make_build_args(args: VeOmniOmniArguments) -> OmniBuildArgs:
    """Translate VeOmniArguments into OmniBuildArgs for OmniModule.build()."""
    fsdp_cfg = args.train.accelerator.fsdp_config
    torch_dtype = "float32" if fsdp_cfg.mixed_precision.enable else "bfloat16"
    return OmniBuildArgs(
        init_device=args.train.init_device,
        torch_dtype=torch_dtype,
        enable_full_shard=fsdp_cfg.full_shard,
        enable_reshard_after_forward=fsdp_cfg.reshard_after_forward,
        mixed_precision=fsdp_cfg.mixed_precision,
        enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
        enable_fsdp_offload=fsdp_cfg.offload,
        enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
        enable_forward_prefetch=fsdp_cfg.forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
        max_load_broadcast_size=fsdp_cfg.max_load_broadcast_size,
        extra_basic_modules=list(args.model.basic_modules or []),
    )
