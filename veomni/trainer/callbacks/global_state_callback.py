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

import hashlib
import json
import os
import re
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


def _has_exact_sft_dynamic_resume(trainer):
    """Keep iterable sources on their legacy cursor path, without relaxing mapping validation."""
    from ...data.dynamic_batching import DynamicBatchSizeDataLoader

    loader = getattr(trainer, "train_dataloader", None)
    if (
        getattr(trainer, "_sft_dynamic_exact_resume", False) is not True
        or type(loader) is not DynamicBatchSizeDataLoader
    ):
        return False
    dataset = getattr(loader._dataloader, "dataset", None)
    return not isinstance(dataset, torch.utils.data.IterableDataset)


def _multi_optimizer_checkpoints(trainer):
    """Find every MultiOptimizer owner without silently dropping one."""
    root = getattr(trainer, "model", None)
    runtimes = [("model", root)]
    modules = getattr(root, "module_runtimes", None)
    if isinstance(modules, Mapping) and modules:
        if getattr(root, "checkpoint", None) is None:
            runtimes = []
        runtimes.extend((f"module:{name}", module) for name, module in sorted(modules.items()))
    result, seen = [], set()
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
        result.append((name, checkpoint))
        seen.add(id(checkpoint))
    return result


def _legacy_optimizer_checkpoints(trainer):
    """Find the model runtimes responsible for legacy checkpoint recovery."""
    root = getattr(trainer, "model", None)
    runtimes = [("model", root)]
    modules = getattr(root, "module_runtimes", None)
    if isinstance(modules, Mapping) and modules:
        if getattr(root, "checkpoint", None) is None:
            # A composed runtime delegates checkpoint ownership to its modules.
            runtimes = []
        runtimes.extend((f"module:{name}", module) for name, module in sorted(modules.items()))
    result, unknown, seen = [], [], set()
    for name, runtime in runtimes:
        if runtime is None:
            continue
        optimizer = getattr(runtime, "optimizer", None)
        checkpoint = getattr(runtime, "checkpoint", None)
        confirm = getattr(checkpoint, "confirm_legacy_loaded_checkpoint_for_reuse", None)
        if checkpoint is not None and id(checkpoint) in seen:
            continue
        if optimizer is None:
            if getattr(runtime, "has_trainable_parameters", None) is not False:
                unknown.append(name)
            continue
        multi = getattr(optimizer, "_is_multi_optimizer", False)
        if multi is False and callable(confirm):
            result.append((name, checkpoint))
            seen.add(id(checkpoint))
        elif multi is True and checkpoint is not None and hasattr(checkpoint, "_loaded_optimizer_generation"):
            seen.add(id(checkpoint))
        else:
            unknown.append(name)
    return result, unknown


def _dynamic_resume_context(args, epoch):
    """Capture the source and epoch settings needed for a dynamic cursor."""
    data = getattr(args, "data", None)
    dataloader = getattr(data, "dataloader", None)
    train = getattr(args, "train", None)
    model = getattr(args, "model", None)
    accelerator = getattr(model, "accelerator", None)
    fields = (
        "train_path",
        "data_type",
        "datasets_type",
        "dataset_repeat",
        "text_keys",
        "max_seq_len",
        "dyn_bsz_buffer_size",
        "resume_input_contract",
        "resume_input_deterministic",
    )
    return {
        "version": 1,
        "epoch_steps": getattr(args, "train_steps", None),
        "saved_epoch": epoch,
        "data": {key: getattr(data, key, None) for key in fields},
        "train": {
            key: getattr(train, key, None)
            for key in ("pad_to_length", "micro_batch_size", "global_batch_size", "seed")
        },
        "model": {key: getattr(model, key, None) for key in ("model_path", "config_path", "tokenizer_path")},
        "parallel": {
            key: getattr(accelerator, key, None)
            for key in ("ulysses_size", "cp_size", "ep_size", "dp_shard_size", "dp_replicate_size")
        },
        "background_prefetch": getattr(dataloader, "use_background_prefetcher", None),
    }


def _json_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()


def _agree_restored_position(global_step, saved_epoch, group):
    """Check loaded state, not the job configuration replicated by the launcher."""
    if not dist.is_initialized():
        return
    signature = _json_signature({"global_step": global_step, "saved_epoch": saved_epoch})
    device = "cpu" if dist.get_backend(group) == "gloo" else get_device_type()
    low = torch.tensor(list(signature), dtype=torch.int32, device=device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        raise RuntimeError("Restored training position differs across ranks")


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
        self._unadvanced_boundary_state = None
        data = getattr(args, "data", None)
        if _has_exact_sft_dynamic_resume(trainer):
            digest = getattr(data, "resume_input_contract", None)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                or getattr(data, "resume_input_deterministic", False) is not True
            ):
                logger.warning(
                    "Dynamic SFT trainer-state resume will be rejected unless an input contract digest and "
                    "deterministic-input attestation are configured before training. To start a new data stream, "
                    "load model weights as a new run instead of resuming the trainer state."
                )

    @property
    def rank(self) -> int:
        return self.trainer.args.train.global_rank

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.load_global_state()

    def on_step_begin(self, state: TrainerState, **kwargs) -> None:
        self._unadvanced_boundary_state = None

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        dcp_due = self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0
        hf_due = self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0
        if dcp_due or hf_due:
            self.save_global_state(state, require_optimizer_checkpoint=bool(dcp_due))

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        dcp_due = self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0
        hf_due = self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0
        if dcp_due or hf_due:
            if state.global_step != self._last_saved_step:
                self.save_global_state(state, require_optimizer_checkpoint=bool(dcp_due))
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
        pending = getattr(self, "_unadvanced_boundary_state", None)
        if pending is not None and state.global_step == pending["global_step"]:
            return pending.copy()
        self._unadvanced_boundary_state = None
        loader = self.trainer.train_dataloader
        if _has_exact_sft_dynamic_resume(self.trainer):
            loader._strict_resume_contract = True
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

        result = {
            "global_step": state.global_step,
            "train_dataloader": train_dataloader_state,
            "environ_meter": self.trainer.environ_meter.state_dict(),
            "channel_loss_callback": channel_loss_state,
            "offline_cache_writer": offline_cache_writer_state,
            "torch_rng_state": torch.get_rng_state(),
            "device_rng_state": get_device_rng_state(),
        }
        if _has_exact_sft_dynamic_resume(self.trainer):
            result["_veomni_dynamic_context"] = _dynamic_resume_context(
                self.trainer.args, getattr(state, "epoch", None)
            )
        return result

    def save_global_state(self, state: TrainerState, *, require_optimizer_checkpoint: bool = False) -> None:
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
            receipts = {}
            for name, checkpoint in _multi_optimizer_checkpoints(self.trainer):
                generation = getattr(checkpoint, "_last_saved_optimizer_generation", None)
                valid = not (
                    getattr(checkpoint, "_last_saved_step", None) != state.global_step
                    or not isinstance(generation, str)
                    or not generation
                )
                if not valid:
                    raise RuntimeError("Trainer cursor requires the same-step MultiOptimizer checkpoint generation")
                if valid:
                    receipts[name] = generation
            if receipts:
                payload["_veomni_optimizer_generations"] = receipts
            else:
                payload.pop("_veomni_optimizer_generations", None)
        except BaseException as exc:
            state_error = exc
        group = dist.group.WORLD if dist.is_initialized() else None
        raise_if_any_rank_failed(state_error, "collecting the trainer state", group=group)
        loader_payload = {key: payload[key] for key in _LOADER_KEYS if key in payload}
        extra_payload = {key: value for key, value in payload.items() if key not in _LOADER_KEYS}

        # Each rank writes its own files, so a full disk or a bad pickle starts out
        # visible to that rank alone. Reduce before the manifest: the manifest
        # claims every rank's cursor is down, and rank 0 only knows about its own.
        # A rank that raised here would otherwise leave its peers recording a step
        # whose state is incomplete, or waiting in a collective it never reaches.
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
        resume. A missing ``loader/`` leaves the cursor absent; exact SFT resume
        rejects it.
        """
        extra_path = layout.extra_state_path(load_path, self.rank)
        if not os.path.exists(extra_path):
            return None
        merged = torch.load(extra_path, map_location="cpu", weights_only=False)

        loader_file = layout.loader_path(load_path, self.rank)
        if os.path.exists(loader_file):
            merged.update(torch.load(loader_file, map_location="cpu", weights_only=False))
        else:
            logger.warning(f"No dataloader cursor at {loader_file}; exact data continuation is unavailable.")
        return merged

    def load_global_state(self) -> Optional[Dict[str, Any]]:
        from .checkpoint_callback import CheckpointCallback

        self._last_saved_step = -1
        checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
        if isinstance(checkpoint_callback, CheckpointCallback):
            checkpoint_callback._last_dcp_step = -1
            checkpoint_callback._last_hf_step = -1
        args: "VeOmniArguments" = self.trainer.args
        from ...data.dynamic_batching import DynamicBatchSizeDataLoader

        strict_sft = _has_exact_sft_dynamic_resume(self.trainer)
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return None
        checkpoints, error = [], None
        try:
            checkpoints = _multi_optimizer_checkpoints(self.trainer)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking restored optimizer providers")
        steps = [getattr(checkpoint, "_loaded_optimizer_global_step", None) for _, checkpoint in checkpoints]
        generations = {
            name: getattr(checkpoint, "_loaded_optimizer_generation", None) for name, checkpoint in checkpoints
        }
        error = None
        if any(type(step) is not int for step in steps) or len(set(steps)) > 1:
            error = RuntimeError("Model modules restored different optimizer checkpoint steps")
        elif any(not isinstance(generation, str) or not generation for generation in generations.values()):
            error = RuntimeError("Model modules did not restore a proven optimizer checkpoint generation")
        raise_if_any_rank_failed(error, "checking restored optimizer steps")
        optimizer_step = steps[0] if steps else None
        _agree_restored_position(optimizer_step, None, dist.group.WORLD if dist.is_initialized() else None)
        strict_optimizer = type(optimizer_step) is int
        strict_resume = strict_sft or strict_optimizer
        legacy_checkpoints, unknown_checkpoints, error = [], [], None
        try:
            legacy_checkpoints, unknown_checkpoints = _legacy_optimizer_checkpoints(self.trainer)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking restored optimizer providers")
        if dist.is_initialized():
            local_policy = (
                strict_sft,
                tuple(name for name, _ in legacy_checkpoints),
                tuple(unknown_checkpoints),
                tuple(sorted(generations.items())),
            )
            policies = [None] * dist.get_world_size()
            dist.all_gather_object(policies, local_policy)
            if any(policy != local_policy for policy in policies):
                raise RuntimeError("Checkpoint data and optimizer resume policy differs across ranks")
        coordinated_resume = strict_resume or bool(legacy_checkpoints or unknown_checkpoints)

        if strict_optimizer:
            completeness_error = None
            try:
                manifest = layout.read_manifest(load_path)
                if not layout.checkpoint_is_complete(load_path) or manifest is None:
                    raise RuntimeError("Trainer checkpoint is incomplete; optimizer continuation is unsafe")
                if manifest.get("model_state") is not True:
                    raise RuntimeError("Trainer checkpoint does not include model and optimizer state")
                if manifest.get("global_step") != optimizer_step:
                    raise RuntimeError("Trainer and optimizer checkpoint steps differ")
                if manifest.get("world_size") != args.train.world_size:
                    raise RuntimeError("Trainer checkpoint world size differs")
            except BaseException as exc:
                completeness_error = exc
            raise_if_any_rank_failed(completeness_error, "checking trainer checkpoint completeness")

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
        read_group = dist.group.WORLD if dist.is_initialized() else None
        raise_if_any_rank_failed(read_error, "reading the trainer state", group=read_group)

        found = current_state is not None or legacy_state is not None
        if dist.is_initialized():
            device = "cpu" if dist.get_backend(dist.group.WORLD) == "gloo" else get_device_type()
            flag = torch.tensor([int(found)], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            found = bool(flag.item())
        if not found:
            if strict_resume:
                raise RuntimeError("Trainer state is missing on at least one rank; exact cursor resume is unsafe")
            same_output, error = False, None
            try:
                save_root = getattr(self.trainer.args.train.checkpoint, "save_path", None)
                source_path = os.fspath(load_path)
                if "://" not in source_path:
                    source_path = os.path.realpath(source_path)
                name = os.path.basename(os.path.normpath(source_path))
                prefix = "global_step_"
                if save_root and name.startswith(prefix) and name[len(prefix) :].isdecimal():
                    source_step = int(name[len(prefix) :])
                    same_output = layout.same_checkpoint_path(layout.step_dir(save_root, source_step), load_path)
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "checking weights-only checkpoint output path")
            # Resolved paths depend on each node's actual filesystem, not just
            # the common launcher configuration.
            if dist.is_initialized():
                decisions = [None] * dist.get_world_size()
                dist.all_gather_object(decisions, same_output)
                if any(decision != same_output for decision in decisions):
                    raise RuntimeError("Weights-only checkpoint output path differs across ranks")
            if same_output and (legacy_checkpoints or unknown_checkpoints):
                raise RuntimeError(
                    "Cannot safely reuse a checkpoint without trainer state in its output directory; "
                    "use a different output directory."
                )
            logger.warning_rank0("Trainer state is missing; resuming weights only.")
            return None

        global_state = current_state if current_state is not None else legacy_state
        # This cursor format cannot be resumed after the one-shot drain path
        # was removed. Check it before *any* rank enters loader restoration;
        # custom loaders may perform their own collectives during restore.
        cursor = None
        global_step = None
        cursor_error = None
        sft_next_epoch = False
        prepared_sft_buffer = None
        saved_epoch = None
        try:
            cursor = global_state.get("train_dataloader")
            global_step = global_state["global_step"]
            if type(global_step) is not int or global_step < 0:
                raise ValueError("Trainer checkpoint global_step must be a non-negative integer")
            loader = self.trainer.train_dataloader
            saved_sft_context = "_veomni_dynamic_context" in global_state
            saved_geometry = isinstance(cursor, dict) and "_veomni_resume_geometry" in cursor
            is_dpo = getattr(self.trainer, "_dpo_resume_proven", False) is True
            if not strict_sft and (saved_sft_context or (saved_geometry and not is_dpo)):
                raise ValueError(
                    "Cannot restore a strict dynamic SFT cursor into a different data path; "
                    "use a verified migration or start a new weights-only run"
                )
            if strict_optimizer:
                if global_step != optimizer_step:
                    raise ValueError("Trainer and optimizer checkpoint steps differ")
                if global_state.get("_veomni_optimizer_generations") != generations:
                    raise ValueError("Trainer cursor and optimizer checkpoint generations differ")
                dpo_proven = getattr(self.trainer, "_dpo_resume_proven", False) is True
                if loader is not None and not strict_sft and not dpo_proven:
                    logger.warning_rank0(
                        "Resuming the existing loader cursor without a verified exact data-resume contract; "
                        "the optimizer state is validated, but the subsequent data sequence may differ."
                    )
                if loader is not None and (cursor is None or (global_step > 0 and not cursor)):
                    raise ValueError("Trainer dataloader cursor is missing; optimizer continuation is unsafe")
                rng = global_state.get("torch_rng_state")
                if not isinstance(rng, torch.Tensor):
                    raise ValueError("Trainer CPU RNG state is missing; optimizer continuation is unsafe")
                try:
                    torch.Generator(device="cpu").set_state(rng)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError("Trainer CPU RNG state is invalid") from exc
                device_rng = global_state.get("device_rng_state")
                if (
                    get_device_type() == "cpu"
                    and device_rng is not None
                    and (not isinstance(device_rng, torch.Tensor) or not torch.equal(device_rng, rng))
                ):
                    raise ValueError("Trainer CPU and device RNG states conflict")
                if get_device_type() != "cpu" and device_rng is None:
                    raise ValueError("Trainer accelerator RNG state is missing; optimizer continuation is unsafe")
                if get_device_type() != "cpu" and not callable(getattr(get_torch_device(), "set_rng_state", None)):
                    raise ValueError("Trainer accelerator RNG cannot be restored by this device")
            if type(loader) is DynamicBatchSizeDataLoader and strict_sft:
                if cursor is None:
                    raise ValueError("dynamic loader checkpoint cursor is missing; exact resume is unsafe")
            if type(loader) is DynamicBatchSizeDataLoader and cursor is not None and strict_sft:
                context = global_state.get("_veomni_dynamic_context")
                if not isinstance(context, dict):
                    raise RuntimeError(
                        "Cannot safely restore dynamic batching at an epoch boundary or from a legacy cursor "
                        "without saved source and epoch geometry; use a verified migration."
                    )
                epoch = context.get("saved_epoch")
                if (
                    type(global_step) is not int
                    or global_step < 0
                    or type(epoch) is not int
                    or epoch < 0
                    or type(args.train_steps) is not int
                    or args.train_steps <= 0
                    or _json_signature(context) != _json_signature(_dynamic_resume_context(args, epoch))
                ):
                    raise ValueError("dynamic loader checkpoint source or epoch geometry differs")
                if global_step == 0:
                    raise ValueError("exact dynamic loader resume requires a completed training step")
                loader.validate_resume_state(cursor)
                contract = context["data"]["resume_input_contract"]
                if not isinstance(contract, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", contract) is None:
                    raise ValueError(
                        "exact dynamic loader resume requires an attested input contract digest covering the immutable "
                        "data snapshot, tokenizer, collator and deterministic transforms"
                    )
                if context["data"]["resume_input_deterministic"] is not True:
                    raise ValueError(
                        "exact dynamic loader resume requires an explicit deterministic-input attestation"
                    )
                fingerprint = cursor["_veomni_resume_geometry"]["dataset_fingerprint"]
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError("exact dynamic loader resume requires a source dataset fingerprint")
                source = loader._dataloader
                if getattr(source, "num_workers", 0) and getattr(source, "in_order", True) is not True:
                    raise ValueError("exact dynamic loader resume requires in-order data workers")
                from torchdata.stateful_dataloader.sampler import BatchSampler, StatefulDistributedSampler

                from ...data.data_loader import DistributedDataloader
                from ...data.dataset import InterleavedMappingDataset, MappingDataset

                if (
                    type(source) is not DistributedDataloader
                    or type(getattr(source, "sampler", None)) is not StatefulDistributedSampler
                    or type(getattr(source, "dataset", None)) not in (MappingDataset, InterleavedMappingDataset)
                    or type(getattr(source, "batch_sampler", None)) is not BatchSampler
                    or source.batch_sampler.sampler is not source.sampler
                    or source.batch_sampler.batch_size != source.batch_size
                    or source.batch_sampler.drop_last != source.drop_last
                    or source.snapshot_every_n_steps != 1
                    or source.generator is not None
                ):
                    raise ValueError(
                        "exact dynamic loader resume requires the standard native mapping source and sampler"
                    )
                rng = global_state.get("torch_rng_state")
                if not isinstance(rng, torch.Tensor):
                    raise ValueError("exact dynamic loader resume requires the saved CPU RNG state")
                try:
                    torch.Generator(device="cpu").set_state(rng)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError("invalid CPU RNG state in dynamic loader checkpoint") from exc
                device_rng = global_state.get("device_rng_state")
                if get_device_type() == "cpu" and device_rng is not None:
                    if not isinstance(device_rng, torch.Tensor) or not torch.equal(device_rng, rng):
                        raise ValueError("conflicting CPU and device RNG states in dynamic loader checkpoint")
                if get_device_type() != "cpu" and global_state.get("device_rng_state") is None:
                    raise ValueError("exact dynamic loader resume requires the saved accelerator RNG state")
                if get_device_type() != "cpu" and not callable(getattr(get_torch_device(), "set_rng_state", None)):
                    raise ValueError("the accelerator runtime cannot restore its saved RNG state")
                expected_step = global_step - epoch * args.train_steps
                if expected_step < 0 or expected_step > args.train_steps or cursor.get("step") != expected_step:
                    raise ValueError("dynamic loader cursor and saved epoch position differ")
                if expected_step == 0 or cursor.get("_veomni_resume_phase") != "after_yield":
                    raise ValueError("dynamic loader cursor was not captured at a completed training step")
                if context["background_prefetch"] is not False:
                    raise ValueError("exact dynamic loader resume requires background prefetch to be disabled")
                if global_step > 0 and expected_step == args.train_steps:
                    from datasets import Dataset as HFDataset

                    source = loader._dataloader
                    dataset = getattr(source, "dataset", None)
                    sft_next_epoch = (
                        strict_sft
                        and type(source) is DistributedDataloader
                        and type(getattr(source, "sampler", None)) is StatefulDistributedSampler
                        and type(dataset) in (MappingDataset, InterleavedMappingDataset)
                        and type(dataset._data) in (list, tuple, HFDataset)
                        and not source.persistent_workers
                        and (source.num_workers == 0 or source.in_order is True)
                        and isinstance(cursor.get("dataloader_state"), dict)
                        and bool(cursor["dataloader_state"])
                    )
                    if not sft_next_epoch:
                        raise RuntimeError(
                            "Cannot safely restore dynamic batching at this epoch boundary with the current source "
                            "or worker configuration."
                        )
                    prepared_sft_buffer = loader.prepare_sft_next_epoch_buffer(cursor)
            elif (
                type(loader) is DynamicBatchSizeDataLoader
                and cursor is not None
                and global_step > 0
                and args.train_steps > 0
                and global_step % args.train_steps == 0
            ):
                raise RuntimeError(
                    "Cannot safely restore main-process dynamic batching at an epoch boundary: "
                    "the saved cursor cannot prove the next epoch's buffered samples and source position."
                )
            if isinstance(cursor, dict) and "_veomni_source_exhausted" in cursor:
                from ...data.dynamic_batching import _check_legacy_exhausted_cursor

                if isinstance(self.trainer.train_dataloader, DynamicBatchSizeDataLoader):
                    _check_legacy_exhausted_cursor(cursor)
            if strict_sft:
                context = global_state.get("_veomni_dynamic_context")
                saved_epoch = context.get("saved_epoch") if isinstance(context, dict) else None
        except BaseException as exc:
            cursor_error = exc
        default_group = dist.group.WORLD if dist.is_initialized() else None
        raise_if_any_rank_failed(cursor_error, "validating the training data cursor", group=default_group)
        _agree_restored_position(global_step, saved_epoch, default_group)

        def restore_local(action, description):
            if not coordinated_resume:
                action()
                return
            error = None
            try:
                action()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, description, group=default_group)

        def restore_position():
            self.trainer.state.global_step = global_step
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

        restore_error = None
        try:
            if self.trainer.train_dataloader is not None and cursor is not None:
                if strict_sft and type(self.trainer.train_dataloader) is DynamicBatchSizeDataLoader:
                    self.trainer.train_dataloader._strict_resume_contract = True
                # Set the sampler epoch before restoring worker snapshots.
                if self.trainer.start_step:
                    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

                    from ...data.data_loader import DistributedDataloader
                    from ...data.dynamic_batching import DynamicBatchSizeDataLoader

                    loader = self.trainer.train_dataloader
                    if (
                        type(loader) is DynamicBatchSizeDataLoader
                        and type(loader._dataloader) is DistributedDataloader
                        and type(loader._dataloader.sampler) is StatefulDistributedSampler
                    ):
                        loader.set_epoch(self.trainer.start_epoch)
                if sft_next_epoch:
                    self.trainer.train_dataloader.load_sft_next_epoch_buffer(prepared_sft_buffer)
                else:
                    self.trainer.train_dataloader.load_state_dict(cursor)
        except BaseException as exc:
            if not coordinated_resume:
                raise
            restore_error = exc
        if coordinated_resume:
            raise_if_any_rank_failed(restore_error, "restoring the training data cursor", group=default_group)

        # OmniTrainer checkpoints written before it had a meter store None here.
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
        if sft_next_epoch:
            # The live loader now points at the next epoch. If train_end exports
            # immediately without a step, serialize the original boundary
            # cursor rather than overwriting it with an unconsumed new epoch.
            self._unadvanced_boundary_state = global_state
        restore_local(
            lambda: iter(self.trainer.train_dataloader)
            if (
                not strict_sft
                and self.trainer.start_step == 0
                and not sft_next_epoch
                and self.trainer.train_dataloader is not None
            )
            else None,
            "initializing the restored data iterator",
        )

        legacy_cursor_proven = True
        if legacy_checkpoints:
            error = None
            try:
                cpu_rng = global_state.get("torch_rng_state")
                device_rng = global_state.get("device_rng_state")
                if get_device_type() == "cpu":
                    device_proven = device_rng is None or (
                        torch.is_tensor(cpu_rng) and torch.is_tensor(device_rng) and torch.equal(cpu_rng, device_rng)
                    )
                else:
                    device_proven = device_rng is not None and callable(
                        getattr(get_torch_device(), "set_rng_state", None)
                    )
                loader = self.trainer.train_dataloader
                cursor_proven = loader is None or (isinstance(cursor, dict) and (global_step == 0 or bool(cursor)))
                legacy_cursor_proven = torch.is_tensor(cpu_rng) and device_proven and cursor_proven
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "checking legacy trainer state completeness")
            if dist.is_initialized():
                proofs = [None] * dist.get_world_size()
                dist.all_gather_object(proofs, legacy_cursor_proven)
                legacy_cursor_proven = all(proofs)
            if not legacy_cursor_proven:
                self._last_saved_step = -1
                if strict_optimizer:
                    raise RuntimeError("Mixed optimizer resume lacks proven legacy trainer RNG or cursor state")
        if strict_optimizer or legacy_checkpoints or unknown_checkpoints:
            self._confirm_same_output_checkpoint(
                global_step,
                load_path,
                checkpoints,
                generations,
                legacy_checkpoints,
                unknown_checkpoints,
                legacy_cursor_proven,
            )

        logger.info_rank0(
            f"Restored trainer state from {state_path} (global_step={self.trainer.state.global_step}, "
            f"start_epoch={self.trainer.start_epoch}, start_step={self.trainer.start_step})."
        )
        return global_state

    def _confirm_same_output_checkpoint(
        self,
        step,
        load_path,
        checkpoints,
        generations,
        legacy_checkpoints=(),
        unknown_checkpoints=(),
        legacy_cursor_proven=True,
    ):
        """Reuse a proven checkpoint only at its original output location."""
        same_output, error = False, None
        try:
            save_root = getattr(self.trainer.args.train.checkpoint, "save_path", None)
            same_output = bool(save_root and layout.same_checkpoint_path(layout.step_dir(save_root, step), load_path))
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking restored checkpoint output path")
        # Resolving symlinks and mount visibility is an actual filesystem
        # result and must agree before any rank enters the receipt collectives.
        if dist.is_initialized():
            decisions = [None] * dist.get_world_size()
            dist.all_gather_object(decisions, same_output)
            if any(decision != same_output for decision in decisions):
                raise RuntimeError("Restored checkpoint output path differs across ranks")
        if not same_output:
            return
        if legacy_checkpoints:
            if not legacy_cursor_proven:
                raise RuntimeError(
                    "Cannot safely reuse a legacy checkpoint without proven trainer RNG and cursor state; "
                    "use a different output directory."
                )
            complete, error = False, None
            try:
                manifest = layout.read_manifest(load_path)
                complete = bool(
                    isinstance(manifest, dict)
                    and layout.checkpoint_is_complete(load_path)
                    and manifest.get("model_state") is True
                    and manifest.get("global_step") == step
                    and manifest.get("world_size") == self.trainer.args.train.world_size
                )
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, "checking legacy checkpoint completeness")
            if not complete:
                self._last_saved_step = -1
                error = RuntimeError(
                    "Cannot safely reuse an incomplete legacy checkpoint in its output directory; "
                    "use a different output directory."
                )
            raise_if_any_rank_failed(error, "checking legacy checkpoint completeness")
        error = None
        try:
            if unknown_checkpoints:
                raise RuntimeError(f"Checkpoint providers cannot confirm optimizer identity: {unknown_checkpoints}")
            for name, checkpoint in checkpoints:
                loaded_path = getattr(checkpoint, "_loaded_checkpoint_path", None)
                if loaded_path is None or not layout.same_checkpoint_path(loaded_path, load_path):
                    raise RuntimeError("Optimizer owner restored from a different checkpoint path")
                confirm = getattr(type(checkpoint), "confirm_loaded_checkpoint_for_reuse", None)
                if not callable(confirm):
                    raise RuntimeError("Strict checkpoint provider cannot confirm restored optimizer identity")
                confirm(checkpoint, step, generations[name])
            for _, checkpoint in legacy_checkpoints:
                checkpoint.confirm_legacy_loaded_checkpoint_for_reuse(step, load_path)
        except BaseException as exc:
            error = exc
        try:
            raise_if_any_rank_failed(error, "confirming restored optimizer checkpoint")
        except BaseException:
            for _, checkpoint in (*checkpoints, *legacy_checkpoints):
                invalidate = getattr(checkpoint, "_invalidate_saved_receipt", None)
                if callable(invalidate):
                    invalidate()
                else:
                    checkpoint._last_saved_step = -1
                    checkpoint._last_saved_optimizer_generation = None
            raise
        self._last_saved_step = step
        from .checkpoint_callback import CheckpointCallback

        callback = getattr(self.trainer, "checkpoint_callback", None)
        if isinstance(callback, CheckpointCallback):
            callback._last_dcp_step = step

    def _restore_position(self, global_state: Dict[str, Any]) -> None:
        args: "VeOmniArguments" = self.trainer.args
        global_step = global_state["global_step"]
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps


__all__ = ["GlobalStateCallback"]
