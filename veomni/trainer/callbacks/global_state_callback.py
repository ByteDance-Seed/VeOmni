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

"""Job-level checkpoint callback, as distinct from per-model checkpoint I/O.

Nothing here belongs to a model: where the dataloader is, the rng, the metric
meters. Written per rank into two directories — ``loader/`` for the dataloader
cursor, ``extra_state/`` for the rest. They are separate because the cursor is
the part a job may want to replace or drop on its own: an Energon or
multisource-sampler state is large, and resuming weights onto a different
dataset means keeping ``extra_state/`` while discarding ``loader/``.

This callback also publishes ``checkpoint_manifest.json``. It runs after
:mod:`~veomni.trainer.callbacks.checkpoint_callback` in the dispatch list, so by
the time it finishes it is the last writer of the step's resume tree — which is
exactly what the marker has to attest to. Model weights, optimizer, HF/LoRA
export and the tokenizer/config sidecars are scheduled by that other callback.

On-disk contract: ``docs/usage/checkpoint.md``.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist

from ...checkpoint import layout
from ...utils import helper
from ...utils.device import get_device_type
from ...utils.dist_utils import raise_if_any_rank_failed
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)

# Keys of ``state_dict`` that belong to ``loader/`` rather than ``extra_state/``.
_LOADER_KEYS = ("train_dataloader",)


class GlobalStateCallback(Callback):
    """Save and resume the state that belongs to the job rather than to a model.

    Written per rank, not once on rank 0. The cursor in here is rank-local by
    construction: iterable datasets are ``split_dataset_by_node``-sharded on
    ``dp_rank``, the multisource sampler filters on ``_global_sample_idx %
    dp_size == dp_rank``, and Energon takes ``dp_rank`` in its ``WorkerConfig``.
    Restoring one rank's cursor everywhere would make every rank resume on rank
    0's shard — replaying that slice and skipping the rest.
    """

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1

    @property
    def rank(self) -> int:
        return self.trainer.args.train.global_rank

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.load_global_state()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        due = bool(self.every_n_steps) and state.global_step % self.every_n_steps == 0
        if due or self._model_saved_but_unpublished(state):
            self.save_global_state(state)

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        due = bool(self.every_n_epochs) and (state.epoch + 1) % self.every_n_epochs == 0
        if (due or self._model_saved_but_unpublished(state)) and state.global_step != self._last_saved_step:
            self.save_global_state(state)

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self._model_saved_but_unpublished(state):
            self.save_global_state(state)

    def _model_saved_but_unpublished(self, state: TrainerState) -> bool:
        """Whether this step has model state on disk that nothing has published.

        An HF or LoRA export writes the step's DCP when the step does not
        already have one (``ModelCheckpointManager._prepare_export``), so an
        export cadence of its own — or an export at train end — puts a complete
        ``model/`` at a step this callback never ran for. Without the cursor
        files and the manifest that step is invisible to ``load_path: auto``,
        and a resume goes back to the last cadence step to retrain weights that
        are already on disk. Publishing it costs two small per-rank files beside
        an export that already holds the whole model.

        What decides is the manager's record of its last save rather than a
        probe of the filesystem: every rank runs the same save calls and so
        agrees on it, whereas a directory one rank wrote need not be visible to
        the others — and ranks disagreeing here would split on a collective.

        Read defensively, like ``module_names`` above: a trainer whose manager
        does not report a last save keeps the cadence-only behaviour rather than
        failing.
        """
        checkpoint = getattr(self.trainer, "checkpoint", None)
        saved = getattr(checkpoint, "last_saved_step", None)
        return saved == state.global_step and saved != self._last_saved_step

    def state_dict(self, state: TrainerState) -> Dict[str, Any]:
        if hasattr(self.trainer, "data_iterator") and hasattr(self.trainer.data_iterator, "state_dict"):
            train_dataloader_state = self.trainer.data_iterator.state_dict()
        elif self.trainer.train_dataloader is not None:
            train_dataloader_state = self.trainer.train_dataloader.state_dict()
        else:
            train_dataloader_state = {}

        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        channel_loss_state = channel_loss_callback.state_dict() if channel_loss_callback is not None else {}

        return {
            "global_step": state.global_step,
            "train_dataloader": train_dataloader_state,
            "environ_meter": self.trainer.environ_meter.state_dict(),
            "channel_loss_callback": channel_loss_state,
            "torch_rng_state": torch.get_rng_state(),
        }

    def module_names(self) -> List[str]:
        """Names of the models this job checkpoints, for the manifest.

        Empty for a single-model job. A multi-module trainer overrides this to
        list every module it saved.
        """
        checkpoint = getattr(self.trainer, "checkpoint", None)
        name = getattr(checkpoint, "module_name", "")
        return [name] if name else []

    def save_global_state(self, state: TrainerState) -> None:
        # Drain a pending async DCP save first. CheckpointCallback returns while
        # that write is still in flight; a cursor file that lands before the
        # shards would resume a step whose weights never made it to disk. The
        # manifest published below depends on this too: it claims the whole step
        # is on disk, which is false while a shard write is still running.
        checkpoint = getattr(self.trainer, "checkpoint", None)
        if checkpoint is not None:
            checkpoint.wait_for_pending_save()

        args: "VeOmniArguments" = self.trainer.args
        step_root = layout.step_dir(args.train.checkpoint.save_path, state.global_step)
        payload = self.state_dict(state)
        loader_payload = {key: payload[key] for key in _LOADER_KEYS if key in payload}
        extra_payload = {key: value for key, value in payload.items() if key not in _LOADER_KEYS}

        # Each rank writes its own files, so a full disk or a bad pickle starts out
        # visible to that rank alone. Reduce before the manifest: a rank that
        # raised here would otherwise leave its peers publishing a step whose
        # state is incomplete, or waiting in a collective it never reaches.
        write_error: Optional[BaseException] = None
        try:
            for path, blob in (
                (layout.loader_path(step_root, self.rank), loader_payload),
                (layout.extra_state_path(step_root, self.rank), extra_payload),
            ):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(blob, path)
        except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {self.rank}] failed to write trainer state under {step_root}", exc_info=True)
            write_error = e
        raise_if_any_rank_failed(write_error, "writing the trainer state")

        # Every rank's files are down by now, so the step can be advertised.
        manifest_error: Optional[BaseException] = None
        if self.rank == 0:
            try:
                layout.write_manifest(
                    step_root,
                    global_step=state.global_step,
                    world_size=args.train.world_size,
                    modules=self.module_names(),
                )
            except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
                logger.error(f"[RANK {self.rank}] failed to publish the manifest under {step_root}", exc_info=True)
                manifest_error = e
        raise_if_any_rank_failed(manifest_error, "publishing the checkpoint manifest")

        self._last_saved_step = state.global_step

    def _read_current(self, load_path: str) -> Optional[Dict[str, Any]]:
        """Merge this rank's ``extra_state/`` and ``loader/`` back into one dict.

        ``extra_state/`` is the one that decides whether a current-layout state
        exists: it holds ``global_step``, without which there is nothing to
        resume. A missing ``loader/`` file is not an error — dropping it is how a
        run resumes onto different data — so the cursor is simply absent and the
        dataloader starts from the beginning.
        """
        extra_path = layout.extra_state_path(load_path, self.rank)
        if not os.path.exists(extra_path):
            return None
        merged = torch.load(extra_path, map_location="cpu", weights_only=False)

        loader_file = layout.loader_path(load_path, self.rank)
        if os.path.exists(loader_file):
            merged.update(torch.load(loader_file, map_location="cpu", weights_only=False))
        else:
            logger.warning_rank0(f"No dataloader cursor at {loader_file}; the dataloader restarts from its beginning.")
        return merged

    def load_global_state(self) -> Optional[Dict[str, Any]]:
        args: "VeOmniArguments" = self.trainer.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return None

        state_path = layout.extra_state_path(load_path, self.rank)
        # A file that is present but unreadable is not the same as an absent one:
        # the reduction below treats absence as "resume weights only", which would
        # silently drop a corrupt cursor. Reduce the read failure separately, and
        # before that reduction, so a rank that raised cannot strand its peers.
        read_error: Optional[BaseException] = None
        current_state = legacy_state = None
        try:
            current_state = self._read_current(load_path)
            if current_state is None:
                # Delete this import (and veomni/checkpoint/legacy_v0_1_12.py) to drop
                # resume from the pre-split layouts.
                from ...checkpoint.legacy_v0_1_12 import apply_legacy_global_state

                legacy_state = apply_legacy_global_state(load_path, self.rank)
        except BaseException as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {self.rank}] failed to read trainer state under {load_path}", exc_info=True)
            read_error = e
        raise_if_any_rank_failed(read_error, "reading the trainer state")

        found = current_state is not None or legacy_state is not None
        if dist.is_initialized():
            flag = torch.tensor([int(found)], dtype=torch.int32, device=get_device_type())
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            found = bool(flag.item())
            if not found:
                logger.warning_rank0("Trainer state missing on at least one rank; resuming weights only.")
                return None
        elif not found:
            logger.warning(f"No trainer state at {state_path}; resuming weights only.")
            return None

        global_state = current_state if current_state is not None else legacy_state
        self.trainer.state.global_step = global_state["global_step"]
        self._restore_position(global_state)

        channel_loss_state = global_state.get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        if self.trainer.train_dataloader is not None and global_state.get("train_dataloader") is not None:
            self.trainer.train_dataloader.load_state_dict(global_state["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(global_state["environ_meter"])
        rng_state = global_state.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
            iter(self.trainer.train_dataloader)

        logger.info_rank0(
            f"Restored trainer state from {state_path} (global_step={self.trainer.state.global_step}, "
            f"start_epoch={self.trainer.start_epoch}, start_step={self.trainer.start_step})."
        )
        return global_state

    def _restore_position(self, global_state: Dict[str, Any]) -> None:
        args: "VeOmniArguments" = self.trainer.args
        global_step = global_state["global_step"]
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps


__all__ = ["GlobalStateCallback"]
