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

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import timedelta
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


def _agree_dpo_output_checkpoint(value, group=None):
    if not dist.is_initialized():
        return value
    device = "cpu" if dist.get_backend(group) == "gloo" else get_device_type()
    low = torch.tensor([int(value)], dtype=torch.int32, device=device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        raise RuntimeError("DPO ranks disagree on whether the output checkpoint is the loaded checkpoint")
    return value


def _has_exact_sft_dynamic_resume(trainer):
    """Enable the strict SFT cursor contract for mapping and supported multi-source loaders."""
    from ...data.dataset import WeightedMultiSourceDataset
    from ...data.dynamic_batching import DynamicBatchSizeDataLoader

    loader = getattr(trainer, "train_dataloader", None)
    if (
        getattr(trainer, "_sft_dynamic_exact_resume", False) is not True
        or type(loader) is not DynamicBatchSizeDataLoader
    ):
        return False
    dataset = getattr(loader._dataloader, "dataset", None)
    return not isinstance(dataset, torch.utils.data.IterableDataset) or type(dataset) is WeightedMultiSourceDataset


def _external_data_contract(trainer):
    """Return an explicitly requested external exact-data provider."""
    if getattr(trainer, "_veomni_require_exact_data_resume", False) is not True:
        return None
    loader = getattr(trainer, "train_dataloader", None)
    contract = getattr(loader, "_veomni_exact_checkpoint_contract", None)
    required = (
        "install",
        "checkpoint_identity",
        "checkpoint_global_identity",
        "validate_before_checkpoint",
        "validate_cursor",
    )
    if not all(callable(getattr(contract, name, None)) for name in required):
        raise RuntimeError("Exact data continuation requires an installed checkpoint provider")
    return contract


def _loader_requests_exact_resume(loader):
    from ...data.dynamic_batching import DynamicBatchSizeDataLoader

    seen = set()
    while loader is not None:
        if id(loader) in seen:
            raise RuntimeError("Cycle in dynamic data-loader wrappers")
        seen.add(id(loader))
        if getattr(loader, "_veomni_exact_resume_requested", False) is True:
            return True
        if type(loader) is not DynamicBatchSizeDataLoader:
            return False
        loader = loader._dataloader
    return False


def prepare_external_data_contract(trainer, state=None, *, install=False):
    """Coordinate rank-local provider errors before model checkpoint I/O.

    Provider methods must not call distributed collectives: every invocation
    is followed by its own error agreement before the next stage.
    """
    enabled = getattr(trainer, "_veomni_require_exact_data_resume", False) is True
    if dist.is_initialized():
        device = "cpu" if dist.get_backend() == "gloo" else get_device_type()
        flags = torch.tensor([int(enabled), int(not enabled)], dtype=torch.int32, device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if flags.tolist() == [1, 1]:
            raise RuntimeError("Ranks disagree on the exact external-data checkpoint mode")
    contract, error = None, None
    try:
        contract = _external_data_contract(trainer)
        if contract is None and _loader_requests_exact_resume(getattr(trainer, "train_dataloader", None)):
            raise RuntimeError("The requested exact-data loader requires an exact-resume trainer integration")
        if contract is not None:
            if callable(getattr(type(trainer), "validate_dpo_provenance_checkpoint", None)):
                raise RuntimeError("Exact external-data providers do not yet support the DPO recovery protocol")
            checkpoints = _multi_optimizer_checkpoints(trainer)
            legacy, unknown = _legacy_optimizer_checkpoints(trainer)
            if not checkpoints or legacy or unknown:
                raise RuntimeError("Exact external-data resume requires proven MultiOptimizer owners")
            if getattr(trainer.args.train.checkpoint, "save_async", False) is not False:
                raise RuntimeError("Exact external-data resume requires synchronous checkpoints")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking exact external-data providers")
    if contract is None:
        return
    if install:
        error = None
        try:
            contract.install()
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "installing exact external-data provider")
    error, replay_signature = None, None
    try:
        global_callback = getattr(trainer, "global_state_callback", None)
        pending = getattr(global_callback, "_unadvanced_boundary_state", None)
        reuse_boundary = (
            state is not None
            and isinstance(pending, dict)
            and pending.get("global_step") == state.global_step
            and pending.get("_veomni_external_data") is not None
        )
        contract.validate_before_checkpoint(trainer, None if reuse_boundary else state)
        common_identity = contract.checkpoint_global_identity()
        if not isinstance(common_identity, str) or not common_identity or len(common_identity) > 256:
            raise RuntimeError("Exact data checkpoint provider has an invalid global identity")
        if contract is not None:
            identity = contract.checkpoint_identity()
            if not isinstance(identity, str) or not identity or len(identity) > 256:
                raise RuntimeError("Exact data checkpoint provider has an invalid identity")
            if state is not None:
                position = getattr(trainer, "_veomni_exact_epoch_position", None)
                _validate_external_position(trainer, position, state.global_step)
                if state.epoch != position["epoch"]:
                    raise ValueError("Exact external-data loader and trainer positions differ")
                if reuse_boundary:
                    _validate_external_data_state(
                        trainer,
                        contract,
                        pending["_veomni_external_data"],
                        pending.get("train_dataloader"),
                        state.global_step,
                    )
                elif getattr(trainer.train_dataloader, "_veomni_consumed_in_epoch", None) != position["step_in_epoch"]:
                    raise ValueError("Exact external-data loader and trainer positions differ")
        # Construct the digest before the coordinated error reduction. A
        # rank-local non-serializable setting must not strand peers in the
        # following collective.
        replay_signature = _json_signature(
            {
                "source": common_identity,
                "context": _dynamic_resume_context(trainer.args, None),
                "optimizer_owners": sorted(name for name, _ in checkpoints),
                # Composite runtimes save module owners in insertion order.
                # Equal sets with different order can enter different DCP
                # collectives on different ranks.
                "module_order": list(getattr(trainer.model, "module_runtimes", {}) or {}),
                # Readiness involves a collective only when a save is due.
                # Agree the schedule before training so every rank enters it.
                "checkpoint_schedule": {
                    key: getattr(trainer.args.train.checkpoint, key, None)
                    for key in ("save_steps", "save_epochs", "save_hf_weights", "hf_save_steps", "hf_save_epochs")
                },
            }
        ).hex()
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating exact external data before checkpoint")
    _agree_restored_position(
        getattr(state, "global_step", None),
        getattr(state, "epoch", None),
        dist.group.WORLD if dist.is_initialized() else None,
        replay_contract=replay_signature,
    )


def external_data_checkpoint_ready(trainer, state):
    """Agree a non-mutating provider readiness check before any checkpoint I/O.

    A provider without this optional method retains its existing fail-closed
    validation behavior. A false result only defers a scheduled checkpoint;
    it never certifies the cursor or suppresses the regular save preflight.
    """
    error, ready = None, True
    try:
        contract = _external_data_contract(trainer)
        check = getattr(contract, "checkpoint_ready", None)
        if check is not None:
            if not callable(check):
                raise TypeError("External checkpoint readiness must be callable")
            ready = check(trainer, state)
            if type(ready) is not bool:
                raise TypeError("External checkpoint readiness must return bool")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking external checkpoint readiness")
    if dist.is_initialized():
        device = "cpu" if dist.get_backend() == "gloo" else get_device_type()
        ready_tensor = torch.tensor([int(ready)], dtype=torch.int32, device=device)
        dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
        ready = bool(ready_tensor.item())
    return ready


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


def _saved_multi_optimizer_generations(trainer, step):
    generations = {}
    for name, checkpoint in _multi_optimizer_checkpoints(trainer):
        generation = getattr(checkpoint, "_last_saved_optimizer_generation", None)
        if getattr(checkpoint, "_last_saved_step", None) != step or not isinstance(generation, str) or not generation:
            raise RuntimeError("Trainer state has no matching same-step MultiOptimizer checkpoint generation")
        generations[name] = generation
    return generations


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


def _validate_external_position(trainer, position, global_step):
    steps = trainer.args.train_steps
    epochs = getattr(trainer.args.train, "num_train_epochs", None)
    if (
        not isinstance(position, dict)
        or set(position) != {"epoch", "step_in_epoch"}
        or type(position["epoch"]) is not int
        or type(position["step_in_epoch"]) is not int
        or type(steps) is not int
        or steps <= 0
        or type(epochs) is not int
        or type(global_step) is not int
        or not 0 <= position["epoch"] < epochs
        or not 0 < position["step_in_epoch"] <= steps
        or global_step != position["epoch"] * steps + position["step_in_epoch"]
    ):
        raise ValueError("Exact external-data checkpoint epoch position is invalid")


def _validate_external_data_state(trainer, contract, external, cursor, global_step):
    if not isinstance(external, dict) or external.get("version") != 1:
        raise ValueError("Exact external-data checkpoint metadata is missing")
    position = external.get("position")
    _validate_external_position(trainer, position, global_step)
    if (
        external.get("identity") != contract.checkpoint_identity()
        or external.get("global_identity") != contract.checkpoint_global_identity()
        or _json_signature(external.get("context"))
        != _json_signature(_dynamic_resume_context(trainer.args, position["epoch"]))
    ):
        raise ValueError("Exact external-data checkpoint source or configuration differs")
    contract.validate_cursor(trainer, cursor, require_cursor=True)
    if cursor.get("_veomni_consumed_in_epoch") != position["step_in_epoch"]:
        raise ValueError("Exact external-data cursor and trainer position differ")
    return position["epoch"]


def _agree_restored_position(global_step, saved_epoch, group, replay_contract=None):
    """Check loaded state, not the job configuration replicated by the launcher."""
    if not dist.is_initialized():
        return
    signature = _json_signature(
        {"global_step": global_step, "saved_epoch": saved_epoch, "replay_contract": replay_contract}
    )
    device = "cpu" if dist.get_backend(group) == "gloo" else get_device_type()
    low = torch.tensor(list(signature), dtype=torch.int32, device=device)
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        raise RuntimeError("Restored training position differs across ranks, including its data-replay contract")


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

    def _handle_exact_external_event(self, state, event):
        if getattr(self.trainer, "_veomni_require_exact_data_resume", False) is not True:
            return False
        from .checkpoint_callback import CheckpointCallback

        checkpoint = getattr(self.trainer, "checkpoint_callback", None)
        if not isinstance(checkpoint, CheckpointCallback):
            raise RuntimeError("Exact external-data checkpoint scheduling requires its model callback")
        decision = checkpoint._external_checkpoint_event
        if decision is None:
            raise RuntimeError("Exact external-data model callback did not run before its data callback")
        event_name, step, status, dcp = decision
        if event_name != event or step != state.global_step or status not in ("none", "deferred", "saved"):
            raise RuntimeError("Exact external-data checkpoint callbacks disagree on their event")
        if status == "saved" and state.global_step != self._last_saved_step:
            self.save_global_state(state, require_optimizer_checkpoint=dcp)
        return True

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if self._handle_exact_external_event(state, "step_end"):
            return
        dcp_due = self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0
        hf_due = self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0
        if dcp_due or hf_due:
            self.save_global_state(state, require_optimizer_checkpoint=bool(dcp_due))

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        if self._handle_exact_external_event(state, "epoch_end"):
            return
        dcp_due = self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0
        hf_due = self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0
        dpo_saved_this_step = (
            callable(getattr(type(self.trainer), "validate_dpo_provenance_checkpoint", None))
            and state.global_step == self._last_saved_step
        )
        if dcp_due or hf_due or dpo_saved_this_step:
            if state.global_step != self._last_saved_step or dpo_saved_this_step:
                self.save_global_state(
                    state, require_optimizer_checkpoint=bool(dcp_due), refresh_epoch_state=dpo_saved_this_step
                )
            else:
                logger.info_rank0(
                    f"Skipping duplicate trainer state save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self._handle_exact_external_event(state, "train_end"):
            return
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
        contract = _external_data_contract(self.trainer)
        if contract is not None:
            contract.validate_before_checkpoint(self.trainer, state)
        if _has_exact_sft_dynamic_resume(self.trainer):
            loader._strict_resume_contract = True
        is_dpo = callable(getattr(type(self.trainer), "validate_dpo_provenance_checkpoint", None))
        pending_cursor = getattr(self.trainer, "_dpo_pending_loader_state", None) if is_dpo else None
        pending_epoch = getattr(self.trainer, "_dpo_pending_epoch_end", None) if is_dpo else None
        if pending_epoch is not None and pending_cursor is not None:
            # The old cursor still describes the buffered tail of this epoch.
            # Reading the fresh next-epoch loader here may start workers and
            # prefetch using the old sampler epoch before set_epoch runs.
            train_dataloader_state = pending_cursor
        elif is_dpo and getattr(self, "_dpo_finished_cursor", None) is not None:
            # A completed run exported into a different output directory has
            # no new iterator. Keep the saved cursor rather than initializing
            # a fresh loader and overwriting the final recovery position.
            train_dataloader_state = self._dpo_finished_cursor
        elif hasattr(self.trainer, "data_iterator") and hasattr(self.trainer.data_iterator, "state_dict"):
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
        dpo_protocol = getattr(type(self.trainer), "dpo_provenance_checkpoint_state", None)
        if callable(dpo_protocol):
            dpo_protocol_state = dpo_protocol(self.trainer)
            if dpo_protocol_state is not None:
                result["dpo_provenance_protocol"] = dpo_protocol_state
                result["dpo_position"] = {
                    "epoch": state.epoch,
                    "step_in_epoch": self.trainer._dpo_epoch_step,
                    "epoch_complete": self.trainer._dpo_epoch_complete,
                }
        if contract is not None:
            position = getattr(self.trainer, "_veomni_exact_epoch_position", None)
            result["_veomni_external_data"] = {
                "version": 1,
                "identity": contract.checkpoint_identity(),
                "global_identity": contract.checkpoint_global_identity(),
                "position": dict(position),
                "context": _dynamic_resume_context(self.trainer.args, state.epoch),
            }
            _validate_external_data_state(
                self.trainer, contract, result["_veomni_external_data"], train_dataloader_state, state.global_step
            )
        if _has_exact_sft_dynamic_resume(self.trainer):
            result["_veomni_dynamic_context"] = _dynamic_resume_context(
                self.trainer.args, getattr(state, "epoch", None)
            )
        return result

    def save_global_state(
        self, state: TrainerState, *, require_optimizer_checkpoint: bool = False, refresh_epoch_state: bool = False
    ) -> None:
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
        else:
            extra_payload.pop("_veomni_optimizer_generations", None)
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
        external_enabled = getattr(self.trainer, "_veomni_require_exact_data_resume", False) is True
        strict_resume = strict_sft or strict_optimizer or external_enabled
        legacy_checkpoints, unknown_checkpoints, error = [], [], None
        try:
            legacy_checkpoints, unknown_checkpoints = _legacy_optimizer_checkpoints(self.trainer)
        except BaseException as exc:
            error = exc
        raise_if_any_rank_failed(error, "checking restored optimizer providers")
        if dist.is_initialized():
            local_generations = tuple(sorted(generations.items()))
            restored_generations = [None] * dist.get_world_size()
            dist.all_gather_object(restored_generations, local_generations)
            if any(value != local_generations for value in restored_generations):
                raise RuntimeError("Restored optimizer checkpoint generations differ across ranks")
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
        dpo_sampler = None
        try:
            current_state = self._read_current(load_path)
            if current_state is None:
                # Delete this import (and veomni/checkpoint/legacy_v0_1_12.py) to drop
                # resume from the pre-split layouts.
                from ...checkpoint.legacy_v0_1_12 import apply_legacy_global_state

                legacy_state = apply_legacy_global_state(load_path, self.rank)
            loaded_state = current_state if current_state is not None else legacy_state
            validator = getattr(type(self.trainer), "validate_dpo_provenance_checkpoint", None)
            if callable(validator):
                validator(
                    self.trainer, loaded_state.get("dpo_provenance_protocol") if loaded_state is not None else None
                )
                if (
                    loaded_state is None
                    or not isinstance(loaded_state.get("train_dataloader"), dict)
                    or not loaded_state["train_dataloader"]
                ):
                    raise ValueError(
                        "DPO checkpoint recovery requires the rank-local dataloader cursor; "
                        "start from weights only or restore the complete trainer state"
                    )
                if type(loaded_state.get("global_step")) is not int or loaded_state["global_step"] < 0:
                    raise ValueError("DPO checkpoint recovery requires a valid nonnegative global_step")
                rng = loaded_state.get("torch_rng_state")
                if not isinstance(rng, torch.Tensor):
                    raise ValueError("DPO exact recovery requires the saved CPU RNG state")
                try:
                    torch.Generator(device="cpu").set_state(rng)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError("DPO checkpoint has invalid CPU RNG state") from exc
                device_rng = loaded_state.get("device_rng_state")
                if get_device_type() == "cpu" and device_rng is not None:
                    if not isinstance(device_rng, torch.Tensor) or not torch.equal(device_rng, rng):
                        raise ValueError("DPO checkpoint has conflicting CPU and device RNG states")
                if get_device_type() != "cpu" and (
                    device_rng is None or not callable(getattr(get_torch_device(), "set_rng_state", None))
                ):
                    raise ValueError("DPO exact recovery requires a restorable accelerator RNG state")
                position = loaded_state.get("dpo_position")
                if (
                    type(position) is not dict
                    or set(position) != {"epoch", "step_in_epoch", "epoch_complete"}
                    or type(position["epoch"]) is not int
                    or not 0 <= position["epoch"] < args.train.num_train_epochs
                    or type(position["step_in_epoch"]) is not int
                    or not 0 <= position["step_in_epoch"] <= args.train_steps
                    or type(position["epoch_complete"]) is not bool
                    or position["step_in_epoch"] > loaded_state["global_step"]
                ):
                    raise ValueError("DPO checkpoint recovery requires a valid epoch position")
                from ...data.dynamic_batching import (
                    DynamicBatchSizeDataLoader,
                    _check_dpo_cursor_fields,
                    _check_dpo_cursor_strategy,
                )

                if self.trainer.train_dataloader is None:
                    raise ValueError("DPO checkpoint recovery requires an initialized dataloader")
                if isinstance(self.trainer.train_dataloader, DynamicBatchSizeDataLoader):
                    if type(self.trainer.train_dataloader) is not DynamicBatchSizeDataLoader:
                        raise ValueError("DPO exact recovery does not support a custom dynamic loader")
                    cursor = loaded_state["train_dataloader"]
                    if "_veomni_resume_phase" not in cursor:
                        raise ValueError("DPO dynamic loader cursor predates the exact-resume phase protocol")
                    if (
                        type(cursor.get("num_micro_batch")) is not int
                        or cursor["num_micro_batch"] != self.trainer.train_dataloader.num_micro_batch
                    ):
                        raise ValueError("DPO dynamic loader microbatch geometry differs from the checkpoint")
                    if type(cursor.get("step")) is not int or cursor["step"] != position["step_in_epoch"]:
                        raise ValueError("DPO dynamic loader cursor step differs from the checkpoint position")
                    _check_dpo_cursor_fields(cursor)
                    _check_dpo_cursor_strategy(cursor, self.trainer.train_dataloader.batching_strategy)
                    if self.trainer.train_dataloader._strict_resume_contract:
                        self.trainer.train_dataloader.validate_resume_state(cursor)
                    if position["step_in_epoch"] > 0 and cursor.get("_veomni_resume_phase") == "running":
                        raise ValueError("DPO cursor was not captured at a completed training step")
                    if not position["epoch_complete"] and cursor.get("_veomni_resume_phase") == "done":
                        raise ValueError("DPO cursor cannot be exhausted before its epoch is complete")
                    if (
                        loaded_state["global_step"] > 0
                        and position["step_in_epoch"] == 0
                        and not position["epoch_complete"]
                    ):
                        raise ValueError("DPO cursor at an unconsumed epoch start is not an exact resume point")

                # A restored worker iterator must be created with the saved
                # epoch, not the sampler's freshly constructed epoch zero.
                # Only the native, standard sampler has an idempotent epoch
                # setter that we can safely invoke before loading its cursor.
                loader = self.trainer.train_dataloader
                from torchdata.stateful_dataloader.sampler import BatchSampler, StatefulDistributedSampler

                from ...data.data_loader import DistributedDataloader
                from ...data.dataset import MappingDataset

                if isinstance(loader, DynamicBatchSizeDataLoader) != args.train.dyn_bsz:
                    raise ValueError("DPO dataloader batching mode differs from its saved recovery configuration")
                inner = loader._dataloader if isinstance(loader, DynamicBatchSizeDataLoader) else loader
                sampler = getattr(inner, "sampler", None)
                if (
                    type(inner) is DistributedDataloader
                    and type(inner.dataset) is MappingDataset
                    and type(sampler) is StatefulDistributedSampler
                ):
                    loader_config = args.data.dataloader
                    parallel = self.trainer.policy_model.parallel_state
                    if (
                        inner.num_workers != loader_config.num_workers
                        or inner.persistent_workers != (loader_config.persistent_workers and inner.num_workers > 0)
                        or inner.in_order != loader_config.in_order
                        or (inner.num_workers > 0 and inner.in_order is not True)
                        or inner.batch_size != args.train.dataloader_batch_size
                        or inner.drop_last != loader_config.drop_last
                        or inner.generator is not None
                        or inner.snapshot_every_n_steps != 1
                        or type(inner.batch_sampler) is not BatchSampler
                        or inner.batch_sampler.sampler is not inner.sampler
                        or inner.batch_sampler.batch_size != inner.batch_size
                        or inner.batch_sampler.drop_last != inner.drop_last
                        or (
                            isinstance(loader, DynamicBatchSizeDataLoader)
                            and loader._drop_last != loader_config.drop_last
                        )
                        or sampler.shuffle != loader_config.shuffle
                        or sampler.drop_last is not False
                        or sampler.seed != args.train.seed
                        or inner.dataset.seed != args.train.seed
                        or sampler.dataset is not inner.dataset
                        or sampler.num_replicas != parallel.dp_size
                        or sampler.rank != parallel.dp_rank
                    ):
                        raise ValueError("DPO native dataloader runtime differs from its saved recovery configuration")
                    if isinstance(loader, DynamicBatchSizeDataLoader):
                        dpo_sampler = sampler
                else:
                    raise ValueError("DPO exact recovery requires the native dataloader and stateful sampler")
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
        dpo_recovery = callable(getattr(type(self.trainer), "validate_dpo_provenance_checkpoint", None))
        # Reject an incompatible legacy cursor on every rank before any
        # loader or DPO position restoration can perform its own collectives.
        cursor = None
        global_step = None
        cursor_error = None
        sft_next_epoch = False
        prepared_sft_buffer = None
        saved_epoch = None
        weighted_replay = False
        replay_timeout = None
        replay_workers = None
        try:
            cursor = global_state.get("train_dataloader")
            global_step = global_state["global_step"]
            if type(global_step) is not int or global_step < 0:
                raise ValueError("Trainer checkpoint global_step must be a non-negative integer")
            loader = self.trainer.train_dataloader
            contract = _external_data_contract(self.trainer)
            external = global_state.get("_veomni_external_data")
            if contract is not None:
                if not strict_optimizer:
                    raise ValueError("Exact external-data resume requires a verified optimizer generation")
                saved_epoch = _validate_external_data_state(self.trainer, contract, external, cursor, global_step)
            elif external is not None:
                raise ValueError("Exact external-data checkpoint requires its original provider")
            saved_sft_context = "_veomni_dynamic_context" in global_state
            saved_geometry = isinstance(cursor, dict) and "_veomni_resume_geometry" in cursor
            if not strict_sft and (saved_sft_context or (saved_geometry and not dpo_recovery)):
                raise ValueError(
                    "Cannot restore a strict dynamic SFT cursor into a different data path; "
                    "use a verified migration or start a new weights-only run"
                )
            if strict_optimizer:
                if global_step != optimizer_step:
                    raise ValueError("Trainer and optimizer checkpoint steps differ")
                if global_state.get("_veomni_optimizer_generations") != generations:
                    raise ValueError("Trainer cursor and optimizer checkpoint generations differ")
                if loader is not None and not strict_sft and not dpo_recovery and contract is None:
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
                from torch.utils.data.dataloader import _InfiniteConstantSampler
                from torchdata.stateful_dataloader.sampler import BatchSampler, StatefulDistributedSampler

                from ...data.data_loader import DistributedDataloader
                from ...data.dataset import InterleavedMappingDataset, MappingDataset, WeightedMultiSourceDataset

                dataset = getattr(source, "dataset", None)
                sampler = getattr(source, "sampler", None)
                batch_sampler = getattr(source, "batch_sampler", None)
                batch_size = getattr(source, "batch_size", None)
                batch_layout = (batch_size is None and batch_sampler is None) or (
                    type(batch_sampler) is BatchSampler
                    and batch_sampler.sampler is sampler
                    and batch_sampler.batch_size == batch_size
                    and batch_sampler.drop_last == getattr(source, "drop_last", None)
                )
                mapping_source = (
                    type(dataset) in (MappingDataset, InterleavedMappingDataset)
                    and type(sampler) is StatefulDistributedSampler
                    and type(batch_sampler) is BatchSampler
                )
                weighted_source = (
                    type(dataset) is WeightedMultiSourceDataset
                    and type(sampler) is _InfiniteConstantSampler
                    and bool(dataset._fingerprint)
                )
                if (
                    type(source) is not DistributedDataloader
                    or not batch_layout
                    or not (mapping_source or weighted_source)
                    or source.snapshot_every_n_steps != 1
                    or source.generator is not None
                ):
                    raise ValueError(
                        "exact dynamic loader resume requires a standard native mapping or replayable multi-source loader"
                    )
                if weighted_source:
                    source.validate_weighted_resume_state(cursor.get("dataloader_state"), expected_epoch=epoch)
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
                    mapping_epoch_source = (
                        type(getattr(source, "sampler", None)) is StatefulDistributedSampler
                        and type(dataset) in (MappingDataset, InterleavedMappingDataset)
                        and type(dataset._data) in (list, tuple, HFDataset)
                    )
                    weighted_epoch_source = type(dataset) is WeightedMultiSourceDataset and bool(dataset._fingerprint)
                    sft_next_epoch = (
                        strict_sft
                        and type(source) is DistributedDataloader
                        and (mapping_epoch_source or weighted_epoch_source)
                        and (
                            not source.persistent_workers
                            or (weighted_epoch_source and dataset.supports_persistent_epoch_cold_resume)
                        )
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
                and not dpo_recovery
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
            from ...data.data_loader import DistributedDataloader

            raw_loader = loader._dataloader if type(loader) is DynamicBatchSizeDataLoader else loader
            if (
                isinstance(raw_loader, DistributedDataloader)
                and type(raw_loader) is not DistributedDataloader
                and raw_loader._weighted_resume_dataset is not None
            ):
                raise ValueError("exact multi-source recovery requires the native DistributedDataloader")
            weighted_loader = (
                type(raw_loader) is DistributedDataloader and raw_loader._weighted_resume_dataset is not None
            )
            if weighted_loader and global_step > 0 and not cursor:
                raise ValueError("multi-source trainer checkpoint is missing its data cursor")
            weighted_replay = weighted_loader and cursor is not None
            if weighted_replay:
                if dpo_recovery:
                    raise ValueError("DPO recovery does not support replayable multi-source streams")
                replay_timeout = raw_loader.resume_replay_timeout_seconds
                replay_workers = raw_loader.num_workers
                if not strict_sft:
                    raw_cursor = (
                        cursor.get("dataloader_state") if type(loader) is DynamicBatchSizeDataLoader else cursor
                    )
                    raw_loader.validate_weighted_resume_state(raw_cursor)
                if dist.is_initialized() and raw_loader.num_workers == 0:
                    raise ValueError("distributed exact streaming replay requires at least one data worker")
                if dist.is_initialized() and not dist.is_gloo_available():
                    raise RuntimeError("distributed exact streaming replay requires a Gloo control group")
            if strict_sft:
                context = global_state.get("_veomni_dynamic_context")
                saved_epoch = context.get("saved_epoch") if isinstance(context, dict) else None
        except BaseException as exc:
            cursor_error = exc
        default_group = dist.group.WORLD if dist.is_initialized() else None
        raise_if_any_rank_failed(cursor_error, "validating the training data cursor", group=default_group)
        _agree_restored_position(
            global_step,
            saved_epoch,
            default_group,
            replay_contract={"weighted": weighted_replay, "timeout": replay_timeout, "workers": replay_workers},
        )

        if dpo_recovery and dist.is_initialized():
            steps = [None] * dist.get_world_size()
            position = global_state["dpo_position"]
            local_position = (
                global_state["global_step"],
                position["epoch"],
                position["step_in_epoch"],
                position["epoch_complete"],
            )
            dist.all_gather_object(steps, local_position)
            if len(set(steps)) != 1:
                raise ValueError(f"DPO checkpoint position differs across ranks: {steps}")
        if dpo_recovery:
            epoch_error = None
            try:
                position = global_state["dpo_position"]
                same_epoch = not position["epoch_complete"] and position["step_in_epoch"] != args.train_steps
                if same_epoch and dpo_sampler is not None:
                    dpo_sampler.set_epoch(position["epoch"])
            except BaseException as exc:
                epoch_error = exc
            raise_if_any_rank_failed(epoch_error, "preparing the DPO sampler epoch", group=default_group)

        same_output_checkpoint = False
        if dpo_recovery:
            path_error = None
            try:
                save_path = getattr(args.train.checkpoint, "save_path", None)
                save_step_path = layout.step_dir(save_path, global_step) if save_path else None
                same_output_checkpoint = bool(
                    save_step_path and layout.same_checkpoint_path(save_step_path, load_path)
                )
            except BaseException as exc:
                path_error = exc
            raise_if_any_rank_failed(path_error, "checking DPO output checkpoint path", group=default_group)
            same_output_checkpoint = _agree_dpo_output_checkpoint(same_output_checkpoint, group=default_group)

        def restore_local(action, description, group=default_group):
            if not (coordinated_resume or dpo_recovery or weighted_replay):
                action()
                return
            error = None
            try:
                action()
            except BaseException as exc:
                error = exc
            raise_if_any_rank_failed(error, description, group=group)

        def restore_position():
            self.trainer.state.global_step = global_step
            self._restore_position(global_state)
            if dpo_recovery:
                position = global_state["dpo_position"]
                self.trainer.state.epoch = position["epoch"]
                self.trainer._dpo_epoch_step = position["step_in_epoch"]
                self.trainer._dpo_epoch_complete = position["epoch_complete"]
                self.trainer._dpo_pending_loader_state = (
                    cursor if getattr(self.trainer, "_dpo_pending_epoch_end", None) is not None else None
                )
                self._dpo_finished_cursor = cursor if self.trainer.start_epoch >= args.train.num_train_epochs else None
                self._last_saved_step = global_step if same_output_checkpoint else -1

        restore_local(restore_position, "restoring trainer position")
        dpo_new_epoch = dpo_recovery and self.trainer.start_epoch != global_state["dpo_position"]["epoch"]
        if dpo_recovery:
            self.trainer._dpo_restore_fixed_iterator_rng = bool(
                not dpo_new_epoch and not isinstance(self.trainer.train_dataloader, DynamicBatchSizeDataLoader)
            )
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

        def restore_loader():
            loader = self.trainer.train_dataloader
            if dpo_new_epoch and isinstance(loader, DynamicBatchSizeDataLoader):
                loader.load_next_epoch_buffer(cursor)
                return
            if dpo_new_epoch or loader is None or cursor is None:
                return
            if strict_sft and type(loader) is DynamicBatchSizeDataLoader:
                loader._strict_resume_contract = True
            if not dpo_recovery and self.trainer.start_step:
                from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

                from ...data.data_loader import DistributedDataloader

                if (
                    type(loader) is DynamicBatchSizeDataLoader
                    and type(loader._dataloader) is DistributedDataloader
                    and type(loader._dataloader.sampler) is StatefulDistributedSampler
                ):
                    loader.set_epoch(self.trainer.start_epoch)
            if sft_next_epoch:
                loader.load_sft_next_epoch_buffer(prepared_sft_buffer)
                if weighted_replay and raw_loader.persistent_workers:
                    # An uninterrupted persistent loader reuses its workers at
                    # the next epoch and does not draw a new base seed. Prime
                    # replacement workers before restoring the trainer RNG.
                    loader.set_epoch(self.trainer.start_epoch)
                    raw_loader.prime_persistent_epoch_for_resume()
            else:

                def prepare_resume_epoch():
                    prepare = getattr(loader, "prepare_resume_epoch", None)
                    if self.trainer.start_step != 0 and callable(prepare):
                        prepare(self.trainer.start_epoch)

                prepare_resume_epoch()
                needs_copy = dpo_recovery and not isinstance(loader, DynamicBatchSizeDataLoader)
                loader.load_state_dict(copy.deepcopy(cursor) if needs_copy else cursor)
                if dpo_recovery:
                    # Validate deferred source errors without consuming a training sample.
                    iter(loader)
                    prepare_resume_epoch()
                    loader.load_state_dict(cursor)

        replay_group = None
        try:
            if weighted_replay and dist.is_initialized():
                # Each worker startup can wait up to the loader timeout. Leave
                # time for every worker timeout plus rank error coordination.
                group_timeout = replay_timeout * (replay_workers + 1) + 30
                replay_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=group_timeout))
            restore_local(
                restore_loader,
                "restoring the training data cursor",
                group=replay_group if replay_group is not None else default_group,
            )
        finally:
            if replay_group is not None:
                dist.destroy_process_group(replay_group)
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
        if external_enabled:
            position = global_state["_veomni_external_data"]["position"]
            self.trainer._veomni_exact_epoch_position = dict(position)
            self.trainer.state.epoch = position["epoch"]
            if position["step_in_epoch"] == args.train_steps:
                # The loader has entered the next epoch already. An immediate
                # export must reuse the validated, unadvanced boundary cursor.
                self._unadvanced_boundary_state = global_state
        if sft_next_epoch:
            self._unadvanced_boundary_state = global_state
        restore_local(
            lambda: iter(self.trainer.train_dataloader)
            if (
                not strict_sft
                and not dpo_recovery
                and self.trainer.start_step == 0
                and not sft_next_epoch
                and self.trainer.train_dataloader is not None
                # A stateful loader may already have started its restored
                # iterator in load_state_dict. The training loop will call
                # iter(loader); priming it here as well can discard its queue.
                and (
                    cursor is None
                    or getattr(self.trainer.train_dataloader, "_veomni_resume_iterator_prepared", False) is not True
                )
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
        if callable(getattr(type(self.trainer), "validate_dpo_provenance_checkpoint", None)):
            position = global_state["dpo_position"]
            epoch, step = position["epoch"], position["step_in_epoch"]
            self.trainer._dpo_pending_epoch_end = None
            if step == args.train_steps and not position["epoch_complete"]:
                self.trainer._dpo_pending_epoch_end = (epoch, step)
            if position["epoch_complete"] or step == args.train_steps:
                epoch, step = epoch + 1, 0
            self.trainer.start_epoch = epoch
            self.trainer.start_step = step
            return
        global_step = global_state["global_step"]
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps


__all__ = ["GlobalStateCallback"]
