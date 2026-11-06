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

"the rng" spans three streams: the CPU default generator, the accelerator
default generator (diffusion condition models sample noise and timesteps from
it), and any per-run ``torch.Generator`` a condition model owns. Restoring only
the first resumes a different noise stream than an uninterrupted run.

This callback also writes ``checkpoint_manifest.json``, which records that the
files above are down. It says nothing about the model state: that is covered by
DCP's own ``.metadata``, one per directory under ``model/``, and a step counts
as resumable only when both halves are there. Model weights, optimizer, HF/LoRA
export and the tokenizer/config sidecars are scheduled by
:mod:`~veomni.trainer.callbacks.checkpoint_callback`, on the same cadences.

On-disk contract: ``docs/usage/checkpoint.md``.
"""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

from ...checkpoint import layout
from ...utils import helper
from ...utils.device import get_device_rng_state, get_device_type, get_torch_device, set_device_rng_state
from ...utils.dist_utils import raise_if_any_rank_failed
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)

# Keys of ``state_dict`` that belong to ``loader/`` rather than ``extra_state/``.
_LOADER_KEYS = ("train_dataloader",)


def _multi_optimizer_checkpoints(trainer):
    """Find actual model checkpoint owners, including Omni module runtimes."""
    root = getattr(trainer, "model", None)
    runtimes = [("model", root)]
    modules = getattr(root, "module_runtimes", None)
    if isinstance(modules, Mapping):
        runtimes.extend((f"module:{name}", runtime) for name, runtime in sorted(modules.items()))
    result = []
    seen = set()
    for name, runtime in runtimes:
        if runtime is None:
            continue
        optimizer = getattr(runtime, "optimizer", None)
        checkpoint = getattr(runtime, "checkpoint", None)
        loaded_step = getattr(checkpoint, "_loaded_optimizer_global_step", None)
        saved_generation = getattr(checkpoint, "_last_saved_optimizer_generation", None)
        if (
            getattr(optimizer, "_is_multi_optimizer", False) is not True
            and type(loaded_step) is not int
            and not isinstance(saved_generation, str)
        ):
            continue
        if checkpoint is None or id(checkpoint) in seen:
            raise RuntimeError("Restored MultiOptimizer has no unique checkpoint owner")
        owner = getattr(checkpoint, "runtime", None)
        if owner is not None and owner is not runtime:
            raise RuntimeError("Restored MultiOptimizer checkpoint belongs to another runtime")
        seen.add(id(checkpoint))
        result.append((name, checkpoint))
    return result


def _restored_multi_optimizer_state(trainer):
    """Read the proven optimizer step and generations loaded by the runtimes."""
    steps, generations = [], {}
    for name, checkpoint in _multi_optimizer_checkpoints(trainer):
        step = getattr(checkpoint, "_loaded_optimizer_global_step", None)
        generation = getattr(checkpoint, "_loaded_optimizer_generation", None)
        if type(step) is not int or step < 0 or not isinstance(generation, str) or not generation:
            raise RuntimeError("Restored MultiOptimizer step or generation is missing")
        steps.append(step)
        generations[name] = generation
    if len(set(steps)) > 1:
        raise RuntimeError("Model runtimes restored different MultiOptimizer steps")
    return (steps[0] if steps else None), generations


def _saved_multi_optimizer_generations(trainer, step):
    generations = {}
    for name, checkpoint in _multi_optimizer_checkpoints(trainer):
        generation = getattr(checkpoint, "_last_saved_optimizer_generation", None)
        if getattr(checkpoint, "_last_saved_step", None) != step or not isinstance(generation, str) or not generation:
            raise RuntimeError("Trainer state has no matching MultiOptimizer checkpoint generation")
        generations[name] = generation
    return generations


def _same_output_resume_owners(trainer, load_path, step):
    """Return verified owners only when this completed step is the output step."""
    config = trainer.args.train.checkpoint
    same_output, error = False, None
    try:
        same_output = layout.same_checkpoint_path(load_path, layout.step_dir(config.save_path, step))
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking restored checkpoint output path")
    if dist.is_initialized():
        decisions = [None] * dist.get_world_size()
        dist.all_gather_object(decisions, same_output)
        if any(decision != same_output for decision in decisions):
            raise RuntimeError("Restored checkpoint output path differs across ranks")
    if not same_output:
        return []

    owners, error = [], None
    try:
        from ...models.checkpoint.manager import ModelCheckpointManager

        manifest = layout.read_manifest(load_path)
        if (
            not isinstance(manifest, dict)
            or not layout.checkpoint_is_complete(load_path)
            or manifest.get("model_state") is not True
            or manifest.get("global_step") != step
            or manifest.get("world_size") != trainer.args.train.world_size
        ):
            raise RuntimeError("Cannot reuse an incomplete restored optimizer checkpoint")
        root = getattr(trainer, "model", None)
        modules = getattr(root, "module_runtimes", None)
        runtimes = list(modules.values()) if isinstance(modules, Mapping) and modules else [root]
        for runtime in runtimes:
            checkpoint = getattr(runtime, "checkpoint", None)
            if checkpoint is None and getattr(runtime, "has_trainable_parameters", None) is False:
                continue
            loaded_path = getattr(checkpoint, "_loaded_checkpoint_path", None)
            if (
                not isinstance(checkpoint, ModelCheckpointManager)
                or loaded_path is None
                or not layout.same_checkpoint_path(loaded_path, load_path)
            ):
                raise RuntimeError("Cannot reuse an unverified model checkpoint owner")
            owners.append(checkpoint)
        if not owners:
            raise RuntimeError("Cannot reuse a checkpoint without a model owner")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating the restored checkpoint for reuse")
    return owners


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
        # The same cadences ``CheckpointCallback`` runs on, read from the same
        # config. An HF export writes the step's DCP whether or not the DCP
        # cadence reaches that step (``ModelCheckpointManager.prepare_export``),
        # so the cursor has to follow both: model state at a step with no cursor
        # beside it is a step nothing can resume from.
        ckpt = args.train.checkpoint
        self.dcp_every_n_steps = ckpt.save_steps
        self.dcp_every_n_epochs = ckpt.save_epochs
        self.save_hf_weights = ckpt.save_hf_weights
        self.hf_every_n_steps = ckpt.hf_save_steps
        self.hf_every_n_epochs = ckpt.hf_save_epochs
        self._last_saved_step: int = -1

    @property
    def rank(self) -> int:
        return self.trainer.args.train.global_rank

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.load_global_state()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        dcp_due = self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0
        hf_due = self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0
        if dcp_due or hf_due:
            self.save_global_state(state)

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        dcp_due = self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0
        hf_due = self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0
        if dcp_due or hf_due:
            if state.global_step != self._last_saved_step:
                self.save_global_state(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate trainer state save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.save_hf_weights:
            if state.global_step != self._last_saved_step:
                self.save_global_state(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate trainer state save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )

    def state_dict(self, state: TrainerState) -> Dict[str, Any]:
        if hasattr(self.trainer, "data_iterator") and hasattr(self.trainer.data_iterator, "state_dict"):
            train_dataloader_state = self.trainer.data_iterator.state_dict()
        elif self.trainer.train_dataloader is not None:
            train_dataloader_state = self.trainer.train_dataloader.state_dict()
        else:
            train_dataloader_state = {}

        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        channel_loss_state = channel_loss_callback.state_dict() if channel_loss_callback is not None else {}

        # Flushes before it reports, so the rows this cursor covers are on disk.
        offline_cache_writer = getattr(self.trainer, "offline_cache_writer", None)
        offline_cache_writer_state = offline_cache_writer.state_dict() if offline_cache_writer is not None else None

        return {
            "global_step": state.global_step,
            "train_dataloader": train_dataloader_state,
            "environ_meter": self.trainer.environ_meter.state_dict(),
            "channel_loss_callback": channel_loss_state,
            "offline_cache_writer": offline_cache_writer_state,
            "torch_rng_state": torch.get_rng_state(),
            "device_rng_state": get_device_rng_state(),
        }

    def save_global_state(self, state: TrainerState) -> None:
        """Clear the step's manifest, write this rank's cursor files, write it back.

        Nothing here waits on the DCP. The cursor does not depend on those
        shards, and the manifest claims only what this method wrote -- the model
        state answers for itself, through the ``.metadata`` DCP puts in each
        directory it owns. Blocking for an async save here is what used to leave
        ``save_async`` overlapping nothing.

        Both ends of the step's trainer half are this method's, which is the
        point: the manifest is the only marker it writes, so it is the only one
        it clears.
        """
        args: "VeOmniArguments" = self.trainer.args
        step_root, path_error = None, None
        try:
            step_root = layout.step_dir(args.train.checkpoint.save_path, state.global_step)
        except Exception as exc:  # noqa: BLE001 - all ranks must agree before invalidating a manifest
            path_error = exc
        raise_if_any_rank_failed(path_error, "preparing the trainer checkpoint path")
        assert step_root is not None

        # A restarted run reaching this step again overwrites the cursor files, so
        # the manifest an earlier attempt left has to go before the first of them
        # lands -- otherwise a rewrite that dies half-way leaves a step that still
        # reads as complete. Only this file: DCP's markers belong to
        # ``DistributedCheckpointer``, which drops them when it rewrites a module.
        # Rank 0 owns it, so rank 0 clears it, and every rank waits for that
        # before writing anything of its own.
        remove_error: Optional[Exception] = None
        if self.rank == 0:
            try:
                layout.remove_manifest(step_root)
            except Exception as e:  # noqa: BLE001 - re-raised once every rank has agreed
                logger.error(f"[RANK {self.rank}] failed to remove the manifest under {step_root}", exc_info=True)
                remove_error = e
        raise_if_any_rank_failed(remove_error, "removing the stale checkpoint manifest")

        payload, state_error = None, None
        try:
            payload = self.state_dict(state)
            if not isinstance(payload, dict):
                raise TypeError("Trainer checkpoint state must be a dictionary")
        except BaseException as exc:  # noqa: BLE001 - synchronize a rank-local cursor failure
            state_error = exc
        raise_if_any_rank_failed(state_error, "collecting trainer checkpoint state")
        assert payload is not None
        loader_payload = {key: payload[key] for key in _LOADER_KEYS if key in payload}
        extra_payload = {key: value for key, value in payload.items() if key not in _LOADER_KEYS}

        # Each rank writes its own files, so a full disk or a bad pickle starts out
        # visible to that rank alone. Reduce before the manifest: the manifest
        # claims every rank's cursor is down, and rank 0 only knows about its own.
        # A rank that raised here would otherwise leave its peers recording a step
        # whose state is incomplete, or waiting in a collective it never reaches.
        generations, generation_error = {}, None
        try:
            generations = _saved_multi_optimizer_generations(self.trainer, state.global_step)
        except BaseException as exc:  # noqa: BLE001 - all ranks must fail before writing
            generation_error = exc
        raise_if_any_rank_failed(generation_error, "preparing trainer checkpoint state")
        if generations:
            extra_payload["_veomni_optimizer_generations"] = generations
        write_error: Optional[Exception] = None
        try:
            for path, blob in (
                (layout.loader_path(step_root, self.rank), loader_payload),
                (layout.extra_state_path(step_root, self.rank), extra_payload),
            ):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(blob, path)
        except Exception as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {self.rank}] failed to write trainer state under {step_root}", exc_info=True)
            write_error = e
        raise_if_any_rank_failed(write_error, "writing the trainer state")

        # Every rank's cursor is down by now, so the trainer-level half of the
        # step can be recorded. Model completeness is DCP's ``.metadata`` under
        # ``model/``, discovered from the filesystem — this file does not list
        # modules. A trainer whose ``saves_model_state`` is False writes no
        # ``model/``, so the manifest says the cursor is the whole step.
        manifest_error: Optional[Exception] = None
        if self.rank == 0:
            try:
                layout.write_manifest(
                    step_root,
                    global_step=state.global_step,
                    world_size=args.train.world_size,
                    model_state=bool(getattr(self.trainer, "saves_model_state", True)),
                )
            except Exception as e:  # noqa: BLE001 - re-raised once every rank has agreed
                logger.error(f"[RANK {self.rank}] failed to write the manifest under {step_root}", exc_info=True)
                manifest_error = e
        raise_if_any_rank_failed(manifest_error, "writing the checkpoint manifest")

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
            logger.warning(f"No dataloader cursor at {loader_file}; the dataloader restarts from its beginning.")
        return merged

    def load_global_state(self) -> Optional[Dict[str, Any]]:
        args: "VeOmniArguments" = self.trainer.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return None

        restored_step, generations, optimizer_error = None, {}, None
        try:
            restored_step, generations = _restored_multi_optimizer_state(self.trainer)
        except Exception as exc:  # noqa: BLE001 - coordinate before any rank reads state
            optimizer_error = exc
        raise_if_any_rank_failed(optimizer_error, "checking the restored MultiOptimizer step")

        state_path = layout.extra_state_path(load_path, self.rank)
        # A file that is present but unreadable is not the same as an absent one:
        # the reduction below treats absence as "resume weights only", which would
        # silently drop a corrupt cursor. Reduce the read failure separately, and
        # before that reduction, so a rank that raised cannot strand its peers.
        read_error: Optional[Exception] = None
        current_state = legacy_state = None
        try:
            current_state = self._read_current(load_path)
            if current_state is None:
                # Delete this import (and veomni/checkpoint/legacy_v0_1_12.py) to drop
                # resume from the pre-split layouts.
                from ...checkpoint.legacy_v0_1_12 import apply_legacy_global_state

                legacy_state = apply_legacy_global_state(load_path, self.rank)
        except Exception as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {self.rank}] failed to read trainer state under {load_path}", exc_info=True)
            read_error = e
        raise_if_any_rank_failed(read_error, "reading the trainer state")

        found = current_state is not None or legacy_state is not None
        if dist.is_initialized():
            device = "cpu" if dist.get_backend() == "gloo" else get_device_type()
            flag = torch.tensor([int(found)], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            found = bool(flag.item())
            strict = torch.tensor([int(restored_step is not None)], dtype=torch.int32, device=device)
            dist.all_reduce(strict, op=torch.distributed.ReduceOp.MAX)
            if strict.item():
                status = torch.tensor(
                    [restored_step if restored_step is not None else -1], dtype=torch.int64, device=device
                )
                statuses = [torch.empty_like(status) for _ in range(dist.get_world_size())]
                dist.all_gather(statuses, status)
                if len({int(item.item()) for item in statuses}) != 1:
                    raise RuntimeError("Restored MultiOptimizer steps differ across ranks")
        if not found:
            if restored_step is not None:
                raise RuntimeError("Trainer state is missing; exact MultiOptimizer resume is unsafe")
            if dist.is_initialized():
                logger.warning_rank0("Trainer state missing on at least one rank; resuming weights only.")
            else:
                logger.warning(f"No trainer state at {state_path}; resuming weights only.")
            return None

        global_state = current_state if current_state is not None else legacy_state
        error = None
        if restored_step is not None:
            if not isinstance(global_state, dict) or type(global_state.get("global_step")) is not int:
                error = RuntimeError("Trainer checkpoint step is missing or invalid")
            elif global_state["global_step"] != restored_step:
                error = RuntimeError("Trainer and MultiOptimizer checkpoint steps differ")
            elif global_state.get("_veomni_optimizer_generations") != generations:
                error = RuntimeError("Trainer and MultiOptimizer checkpoint generations differ or are missing")
        raise_if_any_rank_failed(error, "checking the trainer checkpoint step")
        reuse_owners = (
            _same_output_resume_owners(self.trainer, load_path, restored_step) if restored_step is not None else []
        )
        error = None
        if reuse_owners:
            cpu_rng = global_state.get("torch_rng_state")
            device_rng = global_state.get("device_rng_state")
            cursor = global_state.get("train_dataloader")
            if not torch.is_tensor(cpu_rng):
                error = RuntimeError("Cannot reuse a checkpoint without the trainer RNG state")
            elif (
                get_device_type() == "cpu"
                and device_rng is not None
                and (not torch.is_tensor(device_rng) or not torch.equal(cpu_rng, device_rng))
            ):
                error = RuntimeError("Cannot reuse a checkpoint with conflicting CPU RNG states")
            elif get_device_type() != "cpu" and (
                device_rng is None or not callable(getattr(get_torch_device(), "set_rng_state", None))
            ):
                error = RuntimeError("Cannot reuse a checkpoint without the device RNG state")
            elif self.trainer.train_dataloader is not None and (
                not isinstance(cursor, dict) or (restored_step > 0 and not cursor)
            ):
                error = RuntimeError("Cannot reuse a checkpoint without a proven data cursor")
            raise_if_any_rank_failed(error, "checking restored checkpoint RNG and cursor")

        def restore_local(action, description):
            if restored_step is None:
                action()
                return
            error = None
            try:
                action()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, description)

        def restore_position():
            self.trainer.state.global_step = global_state["global_step"]
            self._restore_position(global_state)

        restore_local(restore_position, "restoring trainer position")
        channel_loss_state = global_state.get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        restore_local(
            lambda: channel_loss_callback.load_state_dict(channel_loss_state)
            if channel_loss_state is not None and channel_loss_callback is not None
            else None,
            "restoring channel-loss state",
        )
        offline_cache_writer_state = global_state.get("offline_cache_writer")
        offline_cache_writer = getattr(self.trainer, "offline_cache_writer", None)
        restore_local(
            lambda: offline_cache_writer.load_state_dict(offline_cache_writer_state)
            if offline_cache_writer_state is not None and offline_cache_writer is not None
            else None,
            "restoring offline-cache state",
        )
        restore_local(
            lambda: self.trainer.train_dataloader.load_state_dict(global_state["train_dataloader"])
            if self.trainer.train_dataloader is not None and global_state.get("train_dataloader") is not None
            else None,
            "restoring the training data cursor",
        )
        restore_local(
            lambda: self.trainer.environ_meter.load_state_dict(global_state["environ_meter"])
            if global_state.get("environ_meter") is not None
            else None,
            "restoring training meter",
        )

        def restore_rng():
            rng_state = global_state.get("torch_rng_state")
            if rng_state is not None:
                torch.set_rng_state(rng_state)
            set_device_rng_state(global_state.get("device_rng_state"))

        restore_local(restore_rng, "restoring training RNG")
        restore_local(
            lambda: iter(self.trainer.train_dataloader)
            if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None
            else None,
            "initializing the restored data iterator",
        )

        if reuse_owners:
            for checkpoint in reuse_owners:
                checkpoint._last_saved_step = restored_step
                checkpoint._last_saved_optimizer_generation = getattr(checkpoint, "_loaded_optimizer_generation", None)
            self._last_saved_step = restored_step
            from .checkpoint_callback import CheckpointCallback

            checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
            if isinstance(checkpoint_callback, CheckpointCallback):
                checkpoint_callback._last_dcp_step = restored_step

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
