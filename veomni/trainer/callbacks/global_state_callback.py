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
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.api import CheckpointException

from ...checkpoint import layout
from ...utils import helper
from ...utils.device import get_device_rng_state, get_device_type, set_device_rng_state
from ...utils.dist_utils import raise_if_any_rank_failed, require_same_rank_value
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)

# Keys of ``state_dict`` that belong to ``loader/`` rather than ``extra_state/``.
_LOADER_KEYS = ("train_dataloader",)


def _validate_exact_environ_meter(trainer, saved, step):
    """Reject absent or malformed meter state before applying any cursor."""
    meter = getattr(trainer, "environ_meter", None)
    if meter is None or not isinstance(saved, Mapping):
        raise RuntimeError("File-backed full resume requires saved environment meter state")
    if not getattr(meter, "enable_multisource", False):
        raise RuntimeError("File-backed exact resume currently requires the verified multisource meter")

    def valid_count(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    if not all(valid_count(saved.get(name)) for name in ("consume_tokens", "consume_chunks")):
        raise RuntimeError("File-backed resume requires complete environment meter counters")
    if getattr(meter, "enable_multisource", False):
        tracker = saved.get("multisource_tracker")
        if not isinstance(tracker, Mapping):
            raise RuntimeError("File-backed multisource resume requires the sampler tracker")
        batch_idx, counters = tracker.get("batch_idx"), tracker.get("accumulate_counter")
        if type(batch_idx) is not int or batch_idx != step:
            raise RuntimeError("File-backed multisource resume requires a valid sampler batch index")
        if not isinstance(counters, Mapping):
            raise RuntimeError("File-backed multisource resume requires accumulated sampler counters")
        live_tracker = getattr(meter, "multisource_tracker", None)
        names = getattr(live_tracker, "names", None)
        if not isinstance(names, (list, tuple)):
            raise RuntimeError("File-backed multisource resume requires the active source inventory")
        for key, value in counters.items():
            if type(key) is not int or key < 0 or key >= len(names):
                raise RuntimeError("File-backed multisource resume has an invalid source index")
            if not isinstance(value, helper.MultiSourceCounterItem) or not all(
                type(getattr(value, name, None)) is int and getattr(value, name) >= 0
                for name in ("num_tokens", "num_samples", "num_steps")
            ):
                raise RuntimeError("File-backed multisource resume has invalid sampler counters")
        # The meter and tracker consume the same sequences, but token totals
        # use FP32 all-reduce in the meter. Permit its small rounding error.
        tokens = sum(item.num_tokens for item in counters.values())
        chunks = sum(item.num_samples for item in counters.values())
        if (
            not math.isclose(tokens, saved["consume_tokens"], rel_tol=1e-6, abs_tol=1.0)
            or chunks != saved["consume_chunks"]
        ):
            raise RuntimeError("File-backed multisource resume has inconsistent accumulated counters")


def _validate_exact_epoch(trainer, step, epoch):
    """Fail closed when an early data exhaustion makes step-derived epoch ambiguous."""
    steps = trainer.args.train_steps
    if (
        type(step) is not int
        or type(epoch) is not int
        or type(steps) is not int
        or steps <= 0
        or step <= 0
        or step % steps == 0
        or epoch != step // steps
    ):
        raise RuntimeError("File-backed exact resume has an unvalidated epoch or data exhaustion boundary")


def _has_file_profile(trainer) -> bool:
    from ...checkpoint.dcp_checkpointer import _file_backed_mode
    from ...optim.ple_world_profile import _Receipt

    runtime = getattr(trainer, "model", None)
    return isinstance(getattr(runtime, "_ple_profile_receipt", None), _Receipt) or _file_backed_mode(
        getattr(runtime, "optimizer", None)
    )


def _resume_identity(trainer) -> str:
    """Bind an exact-resume cursor to data, geometry and update semantics."""

    runtime_identity = os.environ.get("VEOMNI_EXACT_RESUME_RUNTIME_ID", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_identity):
        raise RuntimeError("File-backed exact resume requires an attested runtime/source identity")

    def plain(value):
        if is_dataclass(value):
            value = asdict(value)
        elif hasattr(value, "__dict__"):
            value = vars(value)
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    args = trainer.args
    if getattr(args, "dpo_config", None) is not None or getattr(args.data, "data_type", None) == "dpo":
        raise RuntimeError("File-backed exact resume has not validated multisource preference/DPO data")
    accelerator = args.model.accelerator
    optimizer = getattr(args.model, "optimizer", None)
    fsdp_config = getattr(accelerator, "fsdp_config", None)
    geometry = (
        "dp_replicate_size",
        "dp_shard_size",
        "tp_size",
        "ep_size",
        "ep_outside",
        "extra_parallel_sizes",
        "extra_parallel_placement_innermost",
        "extra_parallel_names",
        "pp_size",
        "ulysses_size",
        "cp_size",
        "allow_hybrid_cp",
    )
    batching = (
        "global_batch_size",
        "micro_batch_size",
        "dataloader_batch_size",
        "dyn_bsz",
        "dyn_bsz_runtime",
        "dyn_bsz_count_mode",
        "dyn_bsz_physical_overflow_ratio",
        "pad_to_length",
        "bsz_warmup_ratio",
        "bsz_warmup_init_mbtoken",
        "seed",
        "text_sft",
        "text_sft_renderer",
        "qwen38_text_sft",
        "loss_chunk_size",
        "enable_moe_lb_loss",
        "freeze_vit",
        "freeze_qsa_indexer",
        "freeze_audio_tower",
        "qwen38_freeze_prefix_layers",
        "qwen38_cpu_ngram",
        "enable_batch_invariant_mode",
        "enable_full_determinism",
    )
    # LambdaLR does not serialize the closure used to construct the schedule.
    # Binding only its step counter would silently change the next learning
    # rate if the resumed job changed the total duration or these arguments.
    # Exclude scratch paths and other storage/performance settings: they do not
    # affect the mathematical update and may legitimately differ on restart.
    update = (
        "type",
        "lr",
        "lr_min",
        "lr_start",
        "lr_warmup_ratio",
        "lr_decay_style",
        "lr_decay_ratio",
        "weight_decay",
        "no_decay_modules",
        "no_decay_params",
        "betas",
        "muon_momentum",
        "max_grad_norm",
        "grad_clip_scope",
    )
    resolved_config = getattr(getattr(trainer, "model", None), "model_config", None)
    if resolved_config is not None and callable(getattr(resolved_config, "to_dict", None)):
        resolved_config = resolved_config.to_dict()
    optimizer_config = plain(optimizer)
    if isinstance(optimizer_config, dict):
        # Local scratch paths are deliberately not part of the algorithm and
        # can change when a checkpoint is restored on a different machine.
        optimizer_config = {k: v for k, v in optimizer_config.items() if k != "qwen38_ple_moments_scratch_dir"}
    identity = {
        "version": 2,
        "world_size": args.train.world_size,
        "geometry": {key: plain(getattr(accelerator, key, None)) for key in geometry},
        "batching": {key: plain(getattr(args.train, key, None)) for key in batching},
        "train_steps": args.train_steps,
        "num_train_epochs": getattr(args.train, "num_train_epochs", None),
        "optimizer_update": {key: plain(getattr(optimizer, key, None)) for key in update},
        "optimizer_config": optimizer_config,
        "fsdp_config": plain(fsdp_config),
        "fsdp_mixed_precision": plain(getattr(fsdp_config, "mixed_precision", None)),
        "fsdp_execution": {key: plain(getattr(fsdp_config, key, None)) for key in ("fsdp_mode", "fsdp_scope")},
        "gradient_checkpointing": plain(getattr(accelerator, "gradient_checkpointing", None)),
        "torch_compile": plain(getattr(accelerator, "torch_compile", None)),
        "ulysses_async": plain(getattr(accelerator, "enable_async", None)),
        "low_precision_reduce_scatter": getattr(fsdp_config, "low_precision_reduce_scatter_comm", None),
        "model_config_override": plain(getattr(args.model, "model_config", None)),
        "resolved_model_config": plain(resolved_config),
        "model_ops": plain(getattr(args.model, "ops_implementation", None)),
        "data_contract": _exact_checkpoint_contract(trainer, require_identity=True).checkpoint_identity(),
        "attested_runtime": runtime_identity,
        "tokenizer_path": getattr(args.model, "tokenizer_path", None),
        "chat_template": plain(getattr(args.model, "chat_template", None)),
        "processor_config": plain(getattr(args.model, "processor_config", None)),
        "data": plain(args.data),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_checkpoint_contract(trainer, *, require_identity=False):
    """Return the data provider's explicit exact-resume contract, or fail closed."""
    loader = getattr(trainer, "train_dataloader", None)
    contract = getattr(loader, "_veomni_exact_checkpoint_contract", None)
    if not all(
        callable(getattr(contract, name, None))
        for name in ("install", "validate_cursor", "validate_before_checkpoint", "checkpoint_identity")
    ):
        raise RuntimeError("File-backed exact resume requires a supported dataloader checkpoint contract")
    if require_identity:
        identity = contract.checkpoint_identity()
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            raise RuntimeError("Dataloader exact-checkpoint contract has an invalid identity")
    return contract


def _validate_exact_loader(trainer, cursor=None, *, require_cursor=False):
    loader = getattr(trainer, "train_dataloader", None)
    if not callable(getattr(loader, "state_dict", None)) or not callable(getattr(loader, "load_state_dict", None)):
        raise RuntimeError("File-backed exact resume requires a stateful, restorable dataloader")
    if require_cursor and (not isinstance(cursor, Mapping) or not cursor):
        raise RuntimeError("File-backed exact resume requires a nonempty dataloader cursor")
    _exact_checkpoint_contract(trainer, require_identity=True).validate_cursor(
        trainer, cursor, require_cursor=require_cursor
    )


def validate_resumed_prefetch_is_drained(trainer, state=None):
    """Agree on the provider's exact-resume checkpoint preconditions."""
    if not _has_file_profile(trainer):
        return
    error = None
    try:
        _exact_checkpoint_contract(trainer, require_identity=True).validate_before_checkpoint(trainer, state)
    except (Exception, CheckpointException) as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating exact data before checkpoint")
    if state is None:
        checkpoint = trainer.args.train.checkpoint
        require_same_rank_value(
            (
                checkpoint.save_steps,
                checkpoint.save_epochs,
                checkpoint.save_hf_weights,
                checkpoint.hf_save_steps,
                checkpoint.hf_save_epochs,
                getattr(trainer, "_veomni_exact_verified_stop_after_steps", None),
                getattr(trainer.args.train, "stop_after_steps", None),
            ),
            "exact checkpoint cadence",
        )


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
        step_root = layout.step_dir(args.train.checkpoint.save_path, state.global_step)
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

        validate_resumed_prefetch_is_drained(self.trainer, state)
        # Even payload construction can fail. The old manifest must already be
        # invalidated so it cannot certify the new DCP together with old cursors.
        loader_payload, extra_payload, payload_error = None, None, None
        try:
            payload = self.state_dict(state)
            if not isinstance(payload, dict):
                raise RuntimeError("Checkpoint cursor payload must be a dictionary")
            if _has_file_profile(self.trainer):
                _validate_exact_loader(self.trainer, payload.get("train_dataloader"), require_cursor=True)
                _validate_exact_epoch(self.trainer, state.global_step, state.epoch)
                _validate_exact_environ_meter(self.trainer, payload.get("environ_meter"), state.global_step)
                payload["file_profile_epoch"] = state.epoch
                if payload.get("torch_rng_state") is None or payload.get("device_rng_state") is None:
                    raise RuntimeError("File-backed checkpoint requires complete RNG state")
                payload["file_profile_resume_identity"] = _resume_identity(self.trainer)
            loader_payload = {key: payload[key] for key in _LOADER_KEYS if key in payload}
            extra_payload = {key: value for key, value in payload.items() if key not in _LOADER_KEYS}
        except (Exception, CheckpointException) as exc:  # noqa: BLE001 - all participants leave together
            payload_error = exc
        raise_if_any_rank_failed(payload_error, "preparing checkpoint cursor payload")

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
        except (Exception, CheckpointException) as e:  # noqa: BLE001 - re-raised once every rank has agreed
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
        exact_file_resume = _has_file_profile(self.trainer)

        state_path = layout.extra_state_path(load_path, self.rank)
        # A file that is present but unreadable is not the same as an absent one:
        # the reduction below treats absence as "resume weights only", which would
        # silently drop a corrupt cursor. Reduce the read failure separately, and
        # before that reduction, so a rank that raised cannot strand its peers.
        read_error: Optional[Exception] = None
        current_state = legacy_state = None
        try:
            if exact_file_resume:
                manifest = layout.read_manifest(load_path)
                if not isinstance(manifest, dict) or manifest.get("world_size") != args.train.world_size:
                    raise RuntimeError("File-backed resume requires the original checkpoint world size")
            current_state = self._read_current(load_path)
            if current_state is None and not exact_file_resume:
                # Delete this import (and veomni/checkpoint/legacy_v0_1_12.py) to drop
                # resume from the pre-split layouts.
                from ...checkpoint.legacy_v0_1_12 import apply_legacy_global_state

                legacy_state = apply_legacy_global_state(load_path, self.rank)
        except (Exception, CheckpointException) as e:  # noqa: BLE001 - re-raised once every rank has agreed
            logger.error(f"[RANK {self.rank}] failed to read trainer state under {load_path}", exc_info=True)
            read_error = e
        raise_if_any_rank_failed(read_error, "reading the trainer state")

        found = current_state is not None or legacy_state is not None
        if dist.is_initialized():
            flag = torch.tensor([int(found)], dtype=torch.int32, device=get_device_type())
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            found = bool(flag.item())
            if not found:
                if exact_file_resume:
                    raise RuntimeError("File-backed full resume requires every rank's trainer state")
                logger.warning_rank0("Trainer state missing on at least one rank; resuming weights only.")
                return None
        elif not found:
            if exact_file_resume:
                raise RuntimeError("File-backed full resume requires trainer state")
            logger.warning(f"No trainer state at {state_path}; resuming weights only.")
            return None

        global_state = current_state if current_state is not None else legacy_state
        step, apply_error = None, None
        try:
            if exact_file_resume:
                required = {"global_step", "train_dataloader", "torch_rng_state", "device_rng_state"}
                if (
                    not required.issubset(global_state)
                    or global_state["torch_rng_state"] is None
                    or global_state["device_rng_state"] is None
                ):
                    raise RuntimeError("File-backed full resume requires complete cursor and RNG state")
                _validate_exact_loader(self.trainer, global_state["train_dataloader"], require_cursor=True)
                _validate_exact_epoch(
                    self.trainer, global_state["global_step"], global_state.get("file_profile_epoch")
                )
                _validate_exact_environ_meter(
                    self.trainer, global_state.get("environ_meter"), global_state["global_step"]
                )
                if global_state.get("file_profile_resume_identity") != _resume_identity(self.trainer):
                    raise RuntimeError("File-backed resume data or parallel geometry differs from the checkpoint")
            step = int(global_state["global_step"])
            if exact_file_resume and (
                not isinstance(manifest.get("global_step"), int) or manifest["global_step"] != step
            ):
                raise RuntimeError("File-backed checkpoint cursor and manifest step differ")
            if exact_file_resume and step > 0 and (args.train_steps <= 0 or step % args.train_steps == 0):
                raise RuntimeError("File-backed exact resume at an epoch boundary is not yet validated")
        except (Exception, CheckpointException) as exc:  # noqa: BLE001 - every participant must leave the load together
            apply_error = exc
        raise_if_any_rank_failed(apply_error, "reading checkpoint global step")
        require_same_rank_value(step, "checkpoint global step")

        apply_error = None
        try:
            self.trainer.state.global_step = step
            self._restore_position(global_state)

            channel_loss_state = global_state.get("channel_loss_callback")
            channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
            if channel_loss_state is not None and channel_loss_callback is not None:
                channel_loss_callback.load_state_dict(channel_loss_state)

            offline_cache_writer_state = global_state.get("offline_cache_writer")
            offline_cache_writer = getattr(self.trainer, "offline_cache_writer", None)
            if offline_cache_writer_state is not None and offline_cache_writer is not None:
                offline_cache_writer.load_state_dict(offline_cache_writer_state)

            if self.trainer.train_dataloader is not None and global_state.get("train_dataloader") is not None:
                self.trainer.train_dataloader.load_state_dict(global_state["train_dataloader"])

            # OmniTrainer checkpoints written before it had a meter store None here.
            if global_state.get("environ_meter") is not None:
                self.trainer.environ_meter.load_state_dict(global_state["environ_meter"])
            rng_state = global_state.get("torch_rng_state")
            if rng_state is not None:
                torch.set_rng_state(rng_state)
            # Older checkpoints and some device namespaces omit RNG state.
            set_device_rng_state(global_state.get("device_rng_state"))
            if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
                iter(self.trainer.train_dataloader)
        except (Exception, CheckpointException) as exc:  # noqa: BLE001 - reject partial restore on every rank
            apply_error = exc
        raise_if_any_rank_failed(apply_error, "applying checkpoint cursor and RNG state")

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
