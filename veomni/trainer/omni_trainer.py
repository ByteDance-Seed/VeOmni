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
OmniTrainer — trainer for OmniModel V2 (interim shell).

Status (D2 stage 1 / stale-cleanup PR)
--------------------------------------
This module previously imported a pair of names that no longer exist
after the OmniModel V2 refactor (``OmniBuildArgs`` and
``OmniModel.build_from_args``).  Those references made the file
unimportable in any environment that pulled it transitively.  This PR
removes the stale references so:

* the file is importable,
* :meth:`OmniTrainer._build_collate_fn` (the D1 list-only collator
  dispatch) can be unit-tested without going through the broken
  ``__init__`` path,

but the actual build flow (``_build_model`` / ``_build_model_assets`` /
``_build_parallelized_model``) is **not yet wired** — those three
helpers raise :class:`NotImplementedError` with a pointer to the
follow-up PRs.

The full V2 build flow lands in two follow-up PRs (the design is
already in :doc:`design.md` § "Build & 权重加载"):

#. **D2 stage 2 — extend `build_parallelize_model`**.  Add an opt-in
   ``weights_path: Mapping[str, str]`` branch in
   :func:`veomni.distributed.torch_parallelize.parallelize_model_fsdp2`
   so the parallelize step can load each sub-module from its own
   directory after meta-init.  Keeps the existing ``str`` / ``None``
   behaviour byte-for-byte; covered by its own unit tests so other
   trainers (BaseTrainer / VLMTrainer / TextTrainer / DiTTrainer) are
   provably unaffected.
#. **D2 stage 3 — re-implement OmniTrainer build flow**.  Register
   each V2 sub-module (``janus_siglip`` / ``janus_vqvae`` /
   ``janus_llama`` / ``janus_text_encoder`` / ``text_encoder``) into
   :data:`veomni.models.loader.MODELING_REGISTRY` so
   :func:`veomni.models.build_foundation_model` can dispatch to the
   right mixin class (today they live only in HF
   ``AutoConfig`` / ``AutoModel`` registries, which
   ``build_foundation_model`` does not consult when
   ``MODELING_BACKEND='veomni'``).  Then:

   * ``_build_model`` calls ``build_foundation_model(init_device='meta',
     weights_path=None)`` per declared module — pure empty meta init,
     no per-rank divergence — and composes them into ``OmniModel``.
   * ``_build_parallelized_model`` calls
     ``build_parallelize_model(self.base.model, weights_path={
     name: cfg["weights_path"] for ...})`` exactly once on the whole
     ``OmniModel``; the new dict branch loads each sub-tree from disk
     after FSDP wrap.
   * end-to-end smoke tests exercise the real ``build_foundation_model``
     path (no internal mocking of the loader / parallelize) so that
     the registry / loader assumptions are validated.

D1's data-layer features are stable and shipped — only the trainer
build flow is staged.

Usage (today, intentionally a soft fail)
----------------------------------------
::

    from veomni.arguments import parse_args
    from veomni.trainer.omni_trainer import OmniTrainer, VeOmniOmniArguments

    if __name__ == "__main__":
        args = parse_args(VeOmniOmniArguments)
        trainer = OmniTrainer(args)        # raises NotImplementedError
        trainer.train()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_data_transform
from ..distributed.clip_grad_norm import veomni_clip_grad_norm
from ..ops.batch_invariant_ops import set_batch_invariant_mode
from ..utils import helper
from ..utils.device import synchronize
from .base import BaseTrainer


logger = helper.create_logger(__name__)


# ── Custom argument dataclasses ────────────────────────────────────────────────


@dataclass
class OmniModelArguments(ModelArguments):
    omni_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the OmniModel YAML config (e.g. "
                "configs/seed_omni/janus_1.3b/train_joint.yaml).  Sub-module "
                "weights are read from each ``modules.<name>.weights_path`` "
                "entry inside that YAML."
            )
        },
    )


@dataclass
class VeOmniOmniArguments(VeOmniArguments):
    model: "OmniModelArguments" = field(default_factory=OmniModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


# ── OmniTrainer ────────────────────────────────────────────────────────────────


# Common message for the three build helpers that haven't been re-implemented
# yet — kept in one place so the wording is identical across helpers.
_BUILD_NOT_WIRED_MSG = (
    "OmniTrainer build flow is staged across two follow-up PRs (extend "
    "`build_parallelize_model` to accept a per-sub-module ``weights_path`` "
    "mapping, then rewrite `_build_model` to use meta-init + the new "
    "multi-path parallelize). See ``omni_trainer.py`` module docstring "
    "for the full plan. The class itself is importable so the D1 collator "
    "wiring (`_build_collate_fn`) can be unit-tested in the meantime."
)


class OmniTrainer:
    """Trainer for OmniModel V2 — composable multi-modal training.

    Mirrors the ``BaseTrainer.__new__`` delegation pattern used by
    :class:`VLMTrainer` / :class:`TextTrainer`.  At this stage only
    :meth:`_build_collate_fn` (the D1 list-only collator dispatch) is
    wired; :meth:`_build_model` and :meth:`_build_model_assets`
    intentionally raise ``NotImplementedError``.  The follow-up
    ``_build_parallelized_model`` step is **not yet defined on this
    class** — it will be added in D2.3 once
    :func:`veomni.distributed.torch_parallelize.build_parallelize_model`
    learns the multi-path ``weights_path`` form (D2.2).  See the module
    docstring for the staged plan.
    """

    base: BaseTrainer

    def __init__(self, args: VeOmniOmniArguments):
        # Even constructing the trainer is currently a soft fail: the
        # build helpers below raise as soon as they're invoked.  We
        # still walk through ``_setup`` (parallel state init, args
        # logging) so callers get a deterministic failure point right
        # at ``_build_model`` rather than later in the optimizer
        # construction or the train loop.
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()

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
        """Stub — to be implemented in the D2 stage-3 follow-up PR.

        See module docstring for the planned implementation
        (meta-init per module + ``OmniModel`` compose + dict-path
        ``build_parallelize_model``).
        """
        raise NotImplementedError("OmniTrainer._build_model is not implemented yet. " + _BUILD_NOT_WIRED_MSG)

    def _freeze_model_module(self):  # pragma: no cover — unreachable until _build_model is wired
        """No-op placeholder — same shape as ``BaseTrainer._freeze_model_module``.

        In the follow-up PR this will print parameter counts and
        delegate to ``_setup_lora`` so ``args.model.lora_config`` is
        honoured (it is silently ignored today).
        """
        return None

    def _build_model_assets(self):
        """Stub — to be implemented in the D2 stage-3 follow-up PR.

        Will collect ``tokenizer`` from ``OmniConfig.tokenizer_path``
        (kept global until D4 moves it onto ``text_encoder``) plus
        per-module assets via ``OmniModel.collect_assets``.
        """
        raise NotImplementedError("OmniTrainer._build_model_assets is not implemented yet. " + _BUILD_NOT_WIRED_MSG)

    def _build_data_transform(self):  # pragma: no cover — unreachable until _build_model_assets is wired
        """Build the data transform.

        Same body as :meth:`BaseTrainer._build_data_transform` for
        signature parity; will be exercised end-to-end once the build
        flow is wired.  ``data_type='seedomni'`` ignores ``tokenizer``
        / ``max_seq_len`` / ``text_keys`` (they're forwarded for
        legacy contract compatibility).
        """
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

        This helper is intentionally wired in this stage-1 PR so D1
        data-layer testing can validate the dispatch even though the
        rest of the build flow is still stubbed.
        """
        args: VeOmniOmniArguments = self.base.args
        if args.data.data_type == "seedomni":
            from ..data import SeedOmniCollator

            self.base.collate_fn = SeedOmniCollator()
            logger.info_rank0("OmniTrainer: using SeedOmniCollator (list-only) for data_type='seedomni'")
        else:
            self.base._build_collate_fn()

    # ── Forward / backward ─────────────────────────────────────────────────────

    def forward_backward_step(  # pragma: no cover — unreachable until build flow is wired
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

    def _model_reshard(  # pragma: no cover — unreachable until build flow is wired
        self, micro_step: int, num_micro_steps: int
    ):
        """Apply set_reshard_after_backward to FSDP2 sub-modules.

        Walks every nested FSDP2 unit so the optimisation also covers
        ``basic_modules``-carved blocks, not just the top-level wrap.
        """
        fsdp_cfg = self.base.args.train.accelerator.fsdp_config
        if fsdp_cfg.fsdp_mode != "fsdp2" or fsdp_cfg.reshard_after_backward or num_micro_steps <= 1:
            return

        try:
            from torch.distributed._composable.fsdp import FSDPModule
        except ImportError:
            return

        for mod in self.base.model.modules():
            if isinstance(mod, FSDPModule):
                if micro_step == 0:
                    mod.set_reshard_after_backward(False)
                elif micro_step == num_micro_steps - 1:
                    mod.set_reshard_after_backward(True)

    # ── Callbacks (delegate to base) ───────────────────────────────────────────
    # The pragma annotations below mirror the build helpers' rationale: the
    # whole train loop is unreachable until D2.3 wires `_build_model`.  Drop
    # these pragmas as part of the D2.3 cleanup so coverage tracks the
    # callback delegation once it actually runs.

    def on_train_begin(self):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_train_begin()

    def on_train_end(self):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_train_end()

    def on_epoch_begin(self):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_epoch_begin()

    def on_epoch_end(self):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(
        self, loss=None, loss_dict=None, grad_norm=None
    ):  # pragma: no cover — unreachable until build flow is wired
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    # ── Train loop ─────────────────────────────────────────────────────────────

    def train_step(self, data_iterator: Any) -> None:  # pragma: no cover — unreachable until build flow is wired
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

    def train(self):  # pragma: no cover — unreachable until build flow is wired
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
