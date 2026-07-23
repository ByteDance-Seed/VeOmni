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

"""Detached per-channel loss logging for causal language-model training.

The callback computes per-token cross entropy as an observability side channel.
It never contributes to the returned model loss and all tensors are detached
under ``torch.no_grad()`` before the extra CE work runs.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections import defaultdict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakSet

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...distributed.parallel_state import get_parallel_state
from ...utils.constants import IGNORE_INDEX
from ...utils.device import empty_cache, synchronize
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ...distributed.parallel_state import ParallelState
    from ..base import BaseTrainer


ChannelKey = Hashable
_LOSS_CALL_POSITIONAL_KEYS = ("logits", "labels", "vocab_size", "num_items_in_batch", "ignore_index")
_ACTIVE_CHANNEL_LOSS_COMPUTER: ContextVar[ChannelLossComputer | None] = ContextVar(
    "veomni_active_channel_loss_computer", default=None
)


class _OpSlotDispatcher:
    """Shared dispatcher for a module-global causal-loss OpSlot."""

    def __init__(self, slot: Any, original_kernel: Callable[..., Any]) -> None:
        self.slot = slot
        self.original_kernel = original_kernel
        self.owners: WeakSet[Any] = WeakSet()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original_kernel(*args, **kwargs)
        owner = _ACTIVE_CHANNEL_LOSS_COMPUTER.get()
        if owner is not None and owner in self.owners:
            owner._observe_opslot_call(self.original_kernel, args, kwargs)
        return result


_OP_SLOT_DISPATCHERS: dict[Any, _OpSlotDispatcher] = {}


class ChannelLossMetadataError(ValueError):
    """Raised when channel metadata cannot be aligned with packed segments."""


class ChannelLossCollectiveError(RuntimeError):
    """Raised when an observer collective fails and its process group is unsafe."""


@dataclass(frozen=True, eq=False)
class _SPCaptureDescriptor:
    """Authoritative cached SP topology for one outer capture."""

    enabled: bool
    parallel_state: ParallelState | None
    group: Any | None
    world: int
    rank: int
    resolution_error: str | None = None


@dataclass
class _PendingSPObservation:
    """Fully materialized SP payload prepared before forward-finally preflight."""

    source_ids: list[ChannelKey]
    device: torch.device
    parallel_state: ParallelState
    include_data_stats: bool
    group: Any | None
    world: int
    capture_descriptor: _SPCaptureDescriptor
    segment_count: int
    loss_payload: torch.Tensor
    integer_payload: torch.Tensor
    gathered_segment_counts: list[int]
    gathered_loss_payloads: list[torch.Tensor]
    gathered_integer_payloads: list[torch.Tensor]


def _sp_descriptor_from_parallel_state(parallel_state: ParallelState | None) -> _SPCaptureDescriptor:
    if parallel_state is None or not bool(parallel_state.sp_enabled):
        return _SPCaptureDescriptor(enabled=False, parallel_state=parallel_state, group=None, world=1, rank=0)
    group = parallel_state.sp_group
    rank = int(parallel_state.sp_rank)
    if rank < 0:
        rank = dist.get_rank(group) if group is not None else 0
    return _SPCaptureDescriptor(
        enabled=True,
        parallel_state=parallel_state,
        group=group,
        world=int(parallel_state.sp_size),
        rank=rank,
    )


def _sp_descriptors_match(left: _SPCaptureDescriptor, right: _SPCaptureDescriptor) -> bool:
    return (
        left.enabled == right.enabled
        and left.parallel_state is right.parallel_state
        and left.group is right.group
        and left.world == right.world
        and left.rank == right.rank
        and left.resolution_error == right.resolution_error
    )


def _unresolved_sp_descriptor(
    parallel_state: ParallelState | None,
    message: str,
) -> _SPCaptureDescriptor:
    return _SPCaptureDescriptor(
        enabled=False,
        parallel_state=parallel_state,
        group=None,
        world=1,
        rank=0,
        resolution_error=message,
    )


def _resolve_cached_sp_capture_descriptor(parallel_state: ParallelState) -> _SPCaptureDescriptor:
    try:
        return _sp_descriptor_from_parallel_state(parallel_state)
    except Exception as error:
        return _unresolved_sp_descriptor(
            parallel_state,
            f"cached SP descriptor resolution failed: {type(error).__name__}: {error}",
        )


def _release_observer_cache(device: torch.device) -> None:
    """Release detached CE workspaces before the training backward pass."""
    if device.type == "cpu":
        return
    synchronize()
    empty_cache()


class ChannelLossComputer:
    """Computes detached per-source CE from model loss-function inputs."""

    EAGER_CHUNK_SIZE = 512

    def __init__(self, parallel_state: ParallelState | None = None) -> None:
        # Cached at construction (same pattern as ``Callback.parallel_state``) so
        # SP/DP collectives never depend on ambient ``get_parallel_state()``.
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self._original_loss_fn: Callable[..., Any] | None = None
        self._model_ref: torch.nn.Module | None = None
        self._installed = False
        self._wrapped_opslots: list[Any] = []

        self._source_ids: list[ChannelKey] = []
        self._position_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._result: list[dict[str, Any]] | None = None
        self._pending_observations: list[_PendingSPObservation | list[dict[str, Any]]] = []
        self._observation_attempt_count = 0
        self._capture_errors: list[tuple[str, str]] = []
        self._observer_devices: list[torch.device] = []
        self._capture_sp_descriptor: _SPCaptureDescriptor | None = None
        self._per_mb_source_ids: list[list[ChannelKey]] = []
        self._per_mb_position_ids: list[torch.Tensor | None] = []
        self._per_mb_attention_masks: list[torch.Tensor | None] = []
        self._micro_step = 0
        self._micro_step_observation_count = 0
        self._micro_step_failed = False
        self.missing_observation_micro_steps: list[int] = []
        self.strict_observation_errors: list[tuple[int, str]] = []
        self.step_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor]] = {}
        self.step_data_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.source_names: dict[ChannelKey, str] = {}
        self.strict = False
        self.release_cache = False

    @staticmethod
    def _resolve_loss_fn_host(model: torch.nn.Module) -> torch.nn.Module | None:
        """Walk common wrappers to find the module whose forward calls ``loss_function``."""

        visited = set()
        cur: torch.nn.Module | None = model
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))
            cls_name = type(cur).__name__
            is_native_lora = any(cls.__name__ == "VeOmniLoraModel" for cls in type(cur).__mro__)
            if is_native_lora:
                get_base_model = getattr(cur, "get_base_model", None)
                nxt = get_base_model() if callable(get_base_model) else None
                if isinstance(nxt, torch.nn.Module) and nxt is not cur:
                    cur = nxt
                    continue
            is_omni_thinker_wrapper = any(
                cls.__name__
                in {
                    "Qwen2_5OmniForConditionalGeneration",
                    "Qwen3OmniMoeForConditionalGeneration",
                }
                for cls in type(cur).__mro__
            )
            if is_omni_thinker_wrapper:
                nxt = getattr(cur, "thinker", None)
                if isinstance(nxt, torch.nn.Module) and nxt is not cur:
                    cur = nxt
                    continue
            is_seed_omni = any(cls.__name__ == "SeedOmniModel" for cls in type(cur).__mro__)
            if is_seed_omni:
                nxt = getattr(cur, "foundation", None)
                if isinstance(nxt, torch.nn.Module) and nxt is not cur:
                    cur = nxt
                    continue
            if cls_name.startswith("Peft"):
                nxt = getattr(cur, "base_model", None)
                if isinstance(nxt, torch.nn.Module):
                    cur = nxt
                    continue
            if cls_name in ("LoraModel", "AdaLoraModel"):
                nxt = getattr(cur, "model", None)
                if isinstance(nxt, torch.nn.Module):
                    cur = nxt
                    continue
            nxt = getattr(cur, "module", None)
            if isinstance(nxt, torch.nn.Module):
                cur = nxt
                continue
            break
        return cur

    def install(self, model: torch.nn.Module) -> None:
        if self._installed:
            return

        host = self._resolve_loss_fn_host(model)
        if host is not None and any(
            cls.__name__ == "Qwen3MoeFoundationModel"
            and ".seed_omni.foundation.qwen3_moe_foundation." in cls.__module__
            for cls in type(host).__mro__
        ):
            raise ValueError(
                "train.channel_loss does not support SeedOmni Qwen3MoeFoundationModel because it "
                "computes causal-LM loss directly instead of using loss_function or a loss OpSlot."
            )
        if host is None or not hasattr(host, "loss_function"):
            logger.warning_rank0(
                "Channel loss: could not locate model.loss_function "
                f"(outer_type={type(model).__name__}, inner_type={type(host).__name__ if host else None}). "
                "Only OpSlot-based fused CE paths, if any, can be observed."
            )
        else:
            self._original_loss_fn = host.loss_function
            host.loss_function = self._wrapped_loss_fn
            self._model_ref = host
            logger.info_rank0(
                f"Channel loss: wrapped {type(host).__name__}.loss_function (outer={type(model).__name__})."
            )

        self._wrap_causal_loss_opslots(model, host)
        self._installed = self._original_loss_fn is not None or bool(self._wrapped_opslots)
        if not self._installed:
            logger.warning_rank0("Channel loss: no causal loss hook was installed.")

    def uninstall(self) -> None:
        if self._model_ref is not None and self._original_loss_fn is not None:
            self._model_ref.loss_function = self._original_loss_fn

        for slot in reversed(self._wrapped_opslots):
            dispatcher = _OP_SLOT_DISPATCHERS.get(slot)
            if dispatcher is None:
                continue
            dispatcher.owners.discard(self)
            if dispatcher.owners:
                continue
            if getattr(slot, "_kernel", None) is dispatcher:
                slot._kernel = dispatcher.original_kernel
            _OP_SLOT_DISPATCHERS.pop(slot, None)

        self._original_loss_fn = None
        self._model_ref = None
        self._wrapped_opslots = []
        self._installed = False

    def _wrap_causal_loss_opslots(
        self,
        model: torch.nn.Module,
        host: torch.nn.Module | None,
    ) -> None:
        modules = []
        seen = set()
        for root in (model, host):
            if root is None:
                continue
            for cls in type(root).__mro__:
                module_name = getattr(cls, "__module__", None)
                if not module_name or module_name in seen:
                    continue
                seen.add(module_name)
                try:
                    modules.append(importlib.import_module(module_name))
                except Exception:
                    continue

        for module in modules:
            slot = getattr(module, "veomni_causal_lm_loss", None)
            if slot is None or not getattr(slot, "use_non_eager_impl", False):
                continue
            original_kernel = getattr(slot, "_kernel", None)
            if original_kernel is None:
                continue
            dispatcher = _OP_SLOT_DISPATCHERS.get(slot)
            if dispatcher is None:
                dispatcher = _OpSlotDispatcher(slot, original_kernel)
                _OP_SLOT_DISPATCHERS[slot] = dispatcher
                slot._kernel = dispatcher
                logger.info_rank0(f"Channel loss: installed dispatcher for {module.__name__}.veomni_causal_lm_loss.")
            elif getattr(slot, "_kernel", None) is not dispatcher:
                logger.warning_rank0(
                    f"Channel loss: {module.__name__}.veomni_causal_lm_loss was rebound after interception; "
                    "skipping this slot."
                )
                continue

            dispatcher.owners.add(self)
            self._wrapped_opslots.append(slot)

    @property
    def capture_active(self) -> bool:
        return _ACTIVE_CHANNEL_LOSS_COMPUTER.get() is self

    @contextmanager
    def capture(self) -> Iterator[None]:
        nested = self.capture_active
        if not nested:
            self._reset_capture_state()
            self._capture_sp_descriptor = _resolve_cached_sp_capture_descriptor(self.parallel_state)
        token = _ACTIVE_CHANNEL_LOSS_COMPUTER.set(self)
        try:
            yield
        finally:
            _ACTIVE_CHANNEL_LOSS_COMPUTER.reset(token)
            if not nested:
                self._finalize_capture()

    def _reset_capture_state(self) -> None:
        self._pending_observations = []
        self._observation_attempt_count = 0
        self._capture_errors = []
        self._observer_devices = []

    def _finalize_capture(self) -> None:
        """Finalize detached observations before model backward can begin."""

        observer_devices = list(self._observer_devices)
        collective_failed = False
        try:
            descriptor = self._capture_sp_descriptor
            descriptor_error = self._preflight_capture_descriptor(descriptor)
            if descriptor_error is not None:
                self._record_capture_failure(descriptor_error)
            elif descriptor is not None and descriptor.enabled:
                self._finalize_sp_capture(descriptor)
            else:
                self._finalize_local_capture()
        except ChannelLossCollectiveError:
            # A failed process-group operation cannot safely be converted into
            # rank-local observer state or followed by another collective.
            collective_failed = True
            raise
        except Exception as error:
            # Local preparation/reconstruction failures must never escape a
            # rank before strict world coordination at step end.
            self._record_capture_failure(
                f"channel-loss forward finalization failed with {type(error).__name__}: {error}"
            )
        finally:
            if self.release_cache and not collective_failed:
                released_devices: list[torch.device] = []
                for device in observer_devices:
                    if device in released_devices:
                        continue
                    released_devices.append(device)
                    try:
                        _release_observer_cache(device)
                    except Exception:
                        logger.warning_rank0(
                            "Channel loss: failed to release detached CE allocator cache; continuing training.",
                            exc_info=True,
                        )
            self._pending_observations = []
            self._capture_errors = []
            self._observer_devices = []
            self._capture_sp_descriptor = None

    @staticmethod
    def _preflight_capture_descriptor(descriptor: _SPCaptureDescriptor | None) -> str | None:
        """Reject rank-local SP topology divergence before any SP collective."""

        if not dist.is_available() or not dist.is_initialized():
            if descriptor is None:
                return "Channel loss: capture has no authoritative SP descriptor."
            if descriptor.resolution_error is not None:
                return f"Channel loss: capture SP descriptor resolution failed: {descriptor.resolution_error}."
            if descriptor.enabled and (
                descriptor.world < 1
                or descriptor.rank < 0
                or descriptor.rank >= descriptor.world
                or (descriptor.world > 1 and descriptor.group is None)
            ):
                return f"Channel loss: capture has an invalid SP descriptor: {descriptor}."
            return None

        try:
            global_rank = dist.get_rank()
            global_world = dist.get_world_size()
        except Exception as error:
            raise ChannelLossCollectiveError(
                "Channel loss could not inspect the initialized default process group for descriptor preflight."
            ) from error

        errors: list[str] = []
        group_ranks: tuple[int, ...] = ()
        if descriptor is None:
            errors.append("missing capture descriptor")
            enabled = False
            sp_world = -1
            sp_rank = -1
        else:
            enabled = descriptor.enabled
            sp_world = descriptor.world
            sp_rank = descriptor.rank
            if descriptor.resolution_error is not None:
                errors.append(descriptor.resolution_error)
            if enabled:
                if descriptor.group is None:
                    if sp_world == 1:
                        group_ranks = (global_rank,)
                    else:
                        errors.append("enabled multi-rank SP descriptor has no process group")
                else:
                    try:
                        get_process_group_ranks = getattr(dist, "get_process_group_ranks", None)
                        if get_process_group_ranks is not None:
                            group_ranks = tuple(int(rank) for rank in get_process_group_ranks(descriptor.group))
                        else:
                            group_ranks = tuple(
                                int(dist.get_global_rank(descriptor.group, local_rank))
                                for local_rank in range(sp_world)
                            )
                    except Exception as error:
                        errors.append(f"failed to resolve SP group membership: {type(error).__name__}: {error}")

        status = {
            "global_rank": global_rank,
            "global_world": global_world,
            "enabled": enabled,
            "sp_world": sp_world,
            "sp_rank": sp_rank,
            "group_ranks": group_ranks,
            "errors": errors,
        }
        gathered_statuses: list[dict[str, Any] | None] = [None] * global_world
        try:
            dist.all_gather_object(gathered_statuses, status)
        except Exception as error:
            raise ChannelLossCollectiveError(
                "Channel loss world descriptor preflight collective failed; the process group is unsafe."
            ) from error
        statuses = [item or {} for item in gathered_statuses]
        return ChannelLossComputer._validate_capture_descriptor_preflight(statuses)

    @staticmethod
    def _validate_capture_descriptor_preflight(statuses: list[dict[str, Any]]) -> str | None:
        if not statuses:
            return "Channel loss: world SP descriptor preflight returned no statuses."

        expected_ranks = list(range(len(statuses)))
        global_ranks = [status.get("global_rank") for status in statuses]
        global_worlds = [status.get("global_world") for status in statuses]
        enabled_values = [status.get("enabled") for status in statuses]
        errors = [status.get("errors") or [] for status in statuses]
        valid = (
            global_ranks == expected_ranks
            and all(world == len(statuses) for world in global_worlds)
            and all(isinstance(enabled, bool) for enabled in enabled_values)
            and len(set(enabled_values)) == 1
            and all(not rank_errors for rank_errors in errors)
        )

        if valid and bool(enabled_values[0]):
            for status_idx, status in enumerate(statuses):
                sp_world = status.get("sp_world")
                sp_rank = status.get("sp_rank")
                group_ranks = status.get("group_ranks")
                if (
                    not isinstance(sp_world, int)
                    or not isinstance(sp_rank, int)
                    or not isinstance(group_ranks, (list, tuple))
                    or sp_world < 1
                    or len(group_ranks) != sp_world
                    or sp_rank < 0
                    or sp_rank >= sp_world
                    or any(
                        not isinstance(member, int) or member < 0 or member >= len(statuses) for member in group_ranks
                    )
                    or len(set(group_ranks)) != sp_world
                    or group_ranks[sp_rank] != status_idx
                ):
                    valid = False
                    break
                member_tuple = tuple(group_ranks)
                if any(tuple(statuses[member].get("group_ranks") or ()) != member_tuple for member in member_tuple):
                    valid = False
                    break

        if valid:
            return None
        return (
            "Channel loss: world preflight rejected rank-local SP capture descriptor divergence before "
            f"group-specific collectives (statuses={statuses})."
        )

    def _finalize_local_capture(self) -> None:
        errors = list(self._capture_errors)
        if self._source_ids and self._observation_attempt_count == 0:
            errors.append(("metadata", "source metadata was present but no causal-LM loss hook was observed"))
        if len(self._pending_observations) != self._observation_attempt_count:
            errors.append(
                (
                    "metadata",
                    "successful observation count does not match the number of observed causal-LM loss calls",
                )
            )
        if any(isinstance(observation, _PendingSPObservation) for observation in self._pending_observations):
            errors.append(("generic", "an SP payload was prepared while sequence parallelism was disabled"))
        if errors:
            self._record_capture_failure(_format_capture_errors(errors))
            return

        observations = [observation for observation in self._pending_observations if isinstance(observation, list)]
        self._append_observations(observations)

    def _finalize_sp_capture(self, descriptor: _SPCaptureDescriptor) -> None:
        group = descriptor.group
        world = descriptor.world
        pending_sp = [
            observation for observation in self._pending_observations if isinstance(observation, _PendingSPObservation)
        ]
        descriptor_errors: list[tuple[str, str]] = []
        if not descriptor.enabled or descriptor.world < 1:
            descriptor_errors.append(("generic", "capture has an invalid enabled SP descriptor"))
        for observation_idx, observation in enumerate(pending_sp):
            if not _sp_descriptors_match(observation.capture_descriptor, descriptor):
                descriptor_errors.append(
                    (
                        "generic",
                        f"observation {observation_idx} was prepared with a different SP capture descriptor",
                    )
                )
            if (
                observation.parallel_state is not descriptor.parallel_state
                or observation.group is not descriptor.group
                or observation.world != descriptor.world
            ):
                descriptor_errors.append(
                    (
                        "generic",
                        f"observation {observation_idx} SP group/world does not match its capture descriptor",
                    )
                )
        status = {
            "has_source": bool(self._source_ids),
            "observation_count": self._observation_attempt_count,
            "successful_count": len(self._pending_observations),
            "source_ids": _source_id_signature(self._source_ids),
            "observation_source_ids": [_source_id_signature(observation.source_ids) for observation in pending_sp],
            "sp_payload_count": len(pending_sp),
            "payload_specs": [
                (
                    observation.loss_payload.numel(),
                    observation.integer_payload.size(1),
                    observation.include_data_stats,
                )
                for observation in pending_sp
            ],
            "segment_counts": [observation.segment_count for observation in pending_sp],
            "descriptor_spec": (descriptor.enabled, descriptor.world, descriptor.group is not None),
            "errors": [*self._capture_errors, *descriptor_errors],
        }

        if world > 1 and group is None:
            status["errors"].append(("generic", "SP process group is unavailable"))

        if world > 1 and group is not None and dist.is_available() and dist.is_initialized():
            gathered_statuses: list[dict[str, Any] | None] = [None] * world
            # This is the sole SP collective that decides whether any
            # conditional observation payload collective may follow.
            try:
                dist.all_gather_object(gathered_statuses, status, group=group)
            except Exception as error:
                raise ChannelLossCollectiveError(
                    "Channel loss SP payload preflight collective failed; the process group is unsafe."
                ) from error
            statuses = [item or {} for item in gathered_statuses]
        else:
            statuses = [status]

        preflight_error = self._validate_sp_preflight(statuses)
        if preflight_error is not None:
            self._record_capture_failure(preflight_error)
            return
        for observation_idx, observation in enumerate(pending_sp):
            observation.gathered_segment_counts = [
                int(status["segment_counts"][observation_idx]) for status in statuses
            ]

        # No observer-owned tensor construction, conversion, or validation
        # is permitted in this loop. All buffers were materialized before the
        # preflight, so approved ranks execute an unconditional fixed sequence.
        for observation in pending_sp:
            self._gather_prepared_sp_payload(observation, descriptor)

        reduced_observations: list[list[dict[str, Any]]] = []
        reconstruction_errors: list[tuple[str, str]] = []
        for observation_idx, observation in enumerate(pending_sp):
            try:
                reduced = self._reconstruct_prepared_sp_payload(observation, strict=True)
                if not reduced:
                    raise ChannelLossMetadataError("SP reduction produced no source-aligned CE payload")
                reduced_observations.append(reduced)
            except Exception as error:
                reconstruction_errors.append(
                    (
                        "metadata" if isinstance(error, ChannelLossMetadataError) else "generic",
                        f"observation {observation_idx}: {type(error).__name__}: {error}",
                    )
                )

        if reconstruction_errors:
            self._record_capture_failure(_format_capture_errors(reconstruction_errors))
            return
        self._append_observations(reduced_observations)

    @staticmethod
    def _validate_sp_preflight(statuses: list[dict[str, Any]]) -> str | None:
        has_source = [bool(status.get("has_source")) for status in statuses]
        observation_counts = [int(status.get("observation_count", -1)) for status in statuses]
        successful_counts = [int(status.get("successful_count", -1)) for status in statuses]
        payload_counts = [int(status.get("sp_payload_count", -1)) for status in statuses]
        source_ids = [status.get("source_ids") for status in statuses]
        errors = [status.get("errors") or [] for status in statuses]
        observation_sources = [status.get("observation_source_ids") or [] for status in statuses]
        payload_specs = [status.get("payload_specs") or [] for status in statuses]
        segment_counts = [status.get("segment_counts") or [] for status in statuses]
        descriptor_specs = [status.get("descriptor_spec") for status in statuses]
        expected_observation_count = observation_counts[0] if observation_counts else -1
        segment_count_contract_valid = True
        for rank_specs, rank_counts in zip(payload_specs, segment_counts):
            if len(rank_specs) != expected_observation_count or len(rank_counts) != expected_observation_count:
                segment_count_contract_valid = False
                break
            for observation_idx, segment_count in enumerate(rank_counts):
                if (
                    not isinstance(segment_count, int)
                    or segment_count < 0
                    or segment_count > int(rank_specs[observation_idx][0])
                ):
                    segment_count_contract_valid = False
                    break

        valid = (
            bool(statuses)
            and all(has_source)
            and len(set(observation_counts)) == 1
            and expected_observation_count > 0
            and successful_counts == observation_counts
            and payload_counts == observation_counts
            and all(source_id == source_ids[0] for source_id in source_ids)
            and all(payload_spec == payload_specs[0] for payload_spec in payload_specs)
            and all(descriptor_spec == descriptor_specs[0] for descriptor_spec in descriptor_specs)
            and all(not rank_errors for rank_errors in errors)
            and segment_count_contract_valid
            and all(
                len(rank_sources) == expected_observation_count
                and all(observation_source == source_ids[rank_idx] for observation_source in rank_sources)
                for rank_idx, rank_sources in enumerate(observation_sources)
            )
        )
        if valid:
            return None

        return (
            "Channel loss: SP preflight rejected observation payload before conditional collectives "
            f"(has_source={has_source}, observation_counts={observation_counts}, "
            f"successful_counts={successful_counts}, payload_counts={payload_counts}, "
            f"source_ids={source_ids}, payload_specs={payload_specs}, "
            f"segment_counts={segment_counts}, descriptor_specs={descriptor_specs}, errors={errors})."
        )

    def _append_observations(self, observations: list[list[dict[str, Any]]]) -> None:
        if not observations or (self.strict and self._micro_step_failed):
            return
        if self._result is None:
            self._result = []
        for observation in observations:
            self._result.extend(observation)
        self._micro_step_observation_count += len(observations)

    def _record_capture_failure(self, message: str) -> None:
        if self.strict:
            # A strict micro-step is atomic evidence: one failed observation
            # invalidates successful observations from the same micro-step.
            self._micro_step_failed = True
            self._result = None
            self._micro_step_observation_count = 0
            self.strict_observation_errors.append((self._micro_step, message))
            logger.warning_rank0(
                f"Channel loss: deferring strict observation failure until globally synchronized step end: {message}"
            )
            return
        logger.warning_rank0(f"Channel loss: {message} Skipping this model forward.")

    def _observe_opslot_call(
        self,
        original_kernel: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        call_args = _bind_loss_call_args(original_kernel, args, kwargs)
        labels = call_args.get("labels")
        ignore_index = call_args.get("ignore_index", IGNORE_INDEX)
        if ignore_index is None:
            ignore_index = IGNORE_INDEX
        if self.capture_active and self._source_ids and labels is not None:
            self.compute_side_channel(
                logits=call_args.get("logits"),
                labels=labels,
                vocab_size=call_args.get("vocab_size"),
                hidden_states=call_args.get("hidden_states"),
                weights=call_args.get("weights"),
                shift_labels=call_args.get("shift_labels"),
                ignore_index=ignore_index,
            )

    def begin_step(
        self,
        per_mb_source_ids: list[list[ChannelKey]],
        per_mb_position_ids: list[torch.Tensor | None],
        per_mb_attention_masks: list[torch.Tensor | None],
    ) -> None:
        self._per_mb_source_ids = per_mb_source_ids
        self._per_mb_position_ids = per_mb_position_ids
        self._per_mb_attention_masks = per_mb_attention_masks
        self._micro_step = 0
        self._micro_step_observation_count = 0
        self._micro_step_failed = False
        self.missing_observation_micro_steps = []
        self.strict_observation_errors = []
        self.step_totals = {}
        self.step_data_totals = {}

    def before_micro_step(self) -> None:
        step = self._micro_step
        if step < len(self._per_mb_source_ids):
            self._source_ids = self._per_mb_source_ids[step]
            self._position_ids = self._per_mb_position_ids[step]
            self._attention_mask = self._per_mb_attention_masks[step]
        else:
            self._source_ids = []
            self._position_ids = None
            self._attention_mask = None
        self._result = None
        self._micro_step_observation_count = 0
        self._micro_step_failed = False
        self._reset_capture_state()

    def after_micro_step(self) -> None:
        if self.strict and self._source_ids and self._micro_step_observation_count == 0:
            self.missing_observation_micro_steps.append(self._micro_step)
        if self._result:
            for item in self._result:
                source_id = item["source_id"]
                loss_sum = item["loss_sum"].detach()
                token_count = item["token_count"].detach()
                sample_count = item["sample_count"].detach()
                input_token_count = item["input_token_count"].detach()
                previous = self.step_totals.get(source_id)
                if previous is None:
                    self.step_totals[source_id] = (loss_sum, token_count)
                else:
                    self.step_totals[source_id] = (previous[0] + loss_sum, previous[1] + token_count)

                data_previous = self.step_data_totals.get(source_id)
                if data_previous is None:
                    self.step_data_totals[source_id] = (sample_count, input_token_count, token_count)
                else:
                    self.step_data_totals[source_id] = (
                        data_previous[0] + sample_count,
                        data_previous[1] + input_token_count,
                        data_previous[2] + token_count,
                    )
        self._source_ids = []
        self._position_ids = None
        self._attention_mask = None
        self._result = None
        self._reset_capture_state()
        self._micro_step += 1

    def record_source_names(self, source_ids: list[ChannelKey], source_names: list[str]) -> None:
        if not source_ids or not source_names:
            return
        if len(source_names) == 1:
            for source_id in source_ids:
                self.source_names[source_id] = source_names[0]
            return
        for source_id, name in zip(source_ids, source_names):
            self.source_names[source_id] = name

    def _wrapped_loss_fn(self, *args: Any, **kwargs: Any) -> Any:
        assert self._original_loss_fn is not None
        main_loss = self._original_loss_fn(*args, **kwargs)

        call_args = _bind_loss_call_args(self._original_loss_fn, args, kwargs)
        logits = call_args.get("logits")
        labels = call_args.get("labels")
        vocab_size = call_args.get("vocab_size")
        ignore_index = call_args.get("ignore_index", IGNORE_INDEX)
        if ignore_index is None:
            ignore_index = IGNORE_INDEX
        if self.capture_active and self._source_ids and labels is not None:
            self.compute_side_channel(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=call_args.get("hidden_states"),
                weights=call_args.get("weights"),
                shift_labels=call_args.get("shift_labels"),
                ignore_index=ignore_index,
            )

        return main_loss

    def compute_side_channel(
        self,
        logits: torch.Tensor | None,
        labels: torch.Tensor,
        vocab_size: int | None,
        hidden_states: torch.Tensor | None,
        weights: torch.Tensor | None,
        shift_labels: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        observer_device = next(
            (tensor.device for tensor in (logits, hidden_states, weights, labels) if isinstance(tensor, torch.Tensor)),
            None,
        )
        standalone = not self.capture_active
        if standalone:
            self._reset_capture_state()
            self._capture_sp_descriptor = _resolve_cached_sp_capture_descriptor(self.parallel_state)
        self._observation_attempt_count += 1
        if observer_device is not None:
            self._observer_devices.append(observer_device)
        try:
            result = self._compute_channel_loss(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=hidden_states,
                weights=weights,
                attention_mask=self._attention_mask,
                shift_labels=shift_labels,
                ignore_index=ignore_index,
                defer_sp=True,
            )
            if result:
                self._pending_observations.append(result)
            else:
                self._capture_errors.append(
                    ("metadata", "channel-loss observation produced no source-aligned CE payload")
                )
        except ChannelLossMetadataError as error:
            self._capture_errors.append(("metadata", str(error)))
        except Exception as error:
            self._capture_errors.append(("generic", f"{type(error).__name__}: {error}"))
        finally:
            if standalone:
                self._finalize_capture()

    @torch.no_grad()
    def _compute_channel_loss(
        self,
        logits: torch.Tensor | None,
        labels: torch.Tensor,
        vocab_size: int | None,
        hidden_states: torch.Tensor | None,
        weights: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        shift_labels: torch.Tensor | None,
        ignore_index: int,
        defer_sp: bool = False,
    ) -> list[dict[str, Any]] | _PendingSPObservation:
        descriptor = self._capture_sp_descriptor
        if descriptor is None:
            descriptor = _resolve_cached_sp_capture_descriptor(self.parallel_state)
        sp_enabled = descriptor.enabled
        if sp_enabled:
            effective_labels = labels
        elif shift_labels is not None:
            effective_labels = shift_labels
        else:
            effective_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()

        labels_flat = effective_labels.reshape(-1)
        attention_mask_flat = _position_aligned_attention_mask_flat(
            attention_mask=attention_mask,
            labels=labels,
            effective_labels=effective_labels,
        )
        per_token_loss = self._per_token_ce(logits, labels_flat, vocab_size, hidden_states, weights, ignore_index)
        if per_token_loss is None:
            return []
        labels_flat = labels_flat.to(per_token_loss.device)
        if attention_mask_flat is not None:
            attention_mask_flat = attention_mask_flat.to(per_token_loss.device)

        return self._aggregate_by_source(
            per_token_loss=per_token_loss,
            labels_flat=labels_flat,
            attention_mask_flat=attention_mask_flat,
            source_ids=self._source_ids,
            position_ids=self._position_ids,
            ignore_index=ignore_index,
            sp_enabled=sp_enabled,
            parallel_state=descriptor.parallel_state if sp_enabled else None,
            capture_descriptor=descriptor,
            strict=self.strict,
            include_data_stats=True,
            defer_sp=defer_sp,
        )

    def _per_token_ce(
        self,
        logits: torch.Tensor | None,
        labels_flat: torch.Tensor,
        vocab_size: int | None,
        hidden_states: torch.Tensor | None,
        weights: torch.Tensor | None,
        ignore_index: int,
    ) -> torch.Tensor | None:
        if hidden_states is not None and weights is not None:
            hs = hidden_states.detach().reshape(-1, hidden_states.size(-1))
            w = weights.detach()
            return self._eager_chunked(hs, w, labels_flat.to(hs.device), ignore_index)

        if logits is not None:
            vocab = vocab_size if vocab_size is not None else logits.size(-1)
            flat_logits = logits.detach().reshape(-1, vocab)
            return self._logits_chunked(flat_logits, labels_flat.to(flat_logits.device), ignore_index)

        return None

    @classmethod
    def _logits_chunked(
        cls,
        logits: torch.Tensor,
        labels_flat: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        seq_len = logits.size(0)
        result = torch.empty(seq_len, device=logits.device, dtype=torch.float32)
        for start in range(0, seq_len, cls.EAGER_CHUNK_SIZE):
            end = min(start + cls.EAGER_CHUNK_SIZE, seq_len)
            result[start:end] = F.cross_entropy(
                logits[start:end].float(),
                labels_flat[start:end],
                ignore_index=ignore_index,
                reduction="none",
            )
        return result

    @classmethod
    def _eager_chunked(
        cls,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        labels_flat: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        seq_len = hidden_states.size(0)
        result = torch.empty(seq_len, device=hidden_states.device, dtype=torch.float32)
        for start in range(0, seq_len, cls.EAGER_CHUNK_SIZE):
            end = min(start + cls.EAGER_CHUNK_SIZE, seq_len)
            chunk_logits = F.linear(hidden_states[start:end], weights).float()
            result[start:end] = F.cross_entropy(
                chunk_logits,
                labels_flat[start:end],
                ignore_index=ignore_index,
                reduction="none",
            )
            del chunk_logits
        return result

    @staticmethod
    def _aggregate_by_source(
        per_token_loss: torch.Tensor,
        labels_flat: torch.Tensor,
        attention_mask_flat: torch.Tensor | None,
        source_ids: list[ChannelKey],
        position_ids: torch.Tensor | None,
        ignore_index: int,
        sp_enabled: bool = False,
        parallel_state: ParallelState | None = None,
        capture_descriptor: _SPCaptureDescriptor | None = None,
        strict: bool = False,
        include_data_stats: bool = False,
        defer_sp: bool = False,
    ) -> list[dict[str, Any]] | _PendingSPObservation:
        if not source_ids:
            return []

        seq_len = min(per_token_loss.numel(), labels_flat.numel())
        per_token_loss = per_token_loss[:seq_len]
        labels_flat = labels_flat[:seq_len]

        segments: list[tuple[int, int, int]] = []
        pos_flat: torch.Tensor | None = None
        if position_ids is not None:
            pos = position_ids
            if pos.dim() == 3:
                pos = pos[:, 0, :]
            if pos.dim() == 1:
                pos_2d = pos.reshape(1, -1)
            else:
                pos_2d = pos.reshape(pos.size(0), -1)
            pos_flat = pos_2d.reshape(-1)[:seq_len]

            if sp_enabled:
                return ChannelLossComputer._aggregate_sp(
                    per_token_loss=per_token_loss,
                    labels_flat=labels_flat,
                    attention_mask_flat=attention_mask_flat,
                    source_ids=source_ids,
                    positions=pos_2d,
                    seq_len=seq_len,
                    ignore_index=ignore_index,
                    parallel_state=parallel_state,
                    capture_descriptor=capture_descriptor,
                    strict=strict,
                    include_data_stats=include_data_stats,
                    defer_reduce=defer_sp,
                )

            row_width = pos_2d.size(1)
            for batch_idx, row_pos in enumerate(pos_2d):
                row_len = min(row_width, max(seq_len - batch_idx * row_width, 0))
                if row_len <= 0:
                    continue
                row_starts = row_pos[:row_len].eq(0).nonzero(as_tuple=True)[0].cpu()
                if row_starts.numel() == 0:
                    continue
                row_ends = torch.cat([row_starts[1:], torch.tensor([row_len])])
                row_offset = batch_idx * row_width
                segments.extend(
                    (batch_idx, row_offset + int(start), row_offset + int(end) - 1)
                    for start, end in zip(row_starts.tolist(), row_ends.tolist())
                )
        elif sp_enabled:
            msg = "Channel loss: sequence-parallel source alignment requires position_ids; skipping this micro-batch."
            if strict:
                raise ChannelLossMetadataError(msg)
            logger.warning_rank0(msg)
            return []
        else:
            segments = [(0, 0, seq_len - 1)]

        if len(segments) > len(source_ids):
            segments = _filter_surplus_tail_padding_segments(
                segments=segments,
                source_count=len(source_ids),
                labels_flat=labels_flat,
                positions_flat=pos_flat,
                attention_mask_flat=attention_mask_flat,
                ignore_index=ignore_index,
            )

        n_segments = len(segments)
        if not _validate_segment_count(n_segments, len(source_ids), strict, "packed segment count"):
            return []

        channel_loss: list[dict[str, Any]] = []
        for i, (_, start, end) in enumerate(segments):
            labels_slice = labels_flat[start : end + 1]
            loss_slice = per_token_loss[start : end + 1]
            mask_slice = attention_mask_flat[start : end + 1] if attention_mask_flat is not None else None
            valid = labels_slice != ignore_index
            token_count = valid.sum()
            loss_sum = loss_slice[valid].sum()
            input_token_count = (
                mask_slice.to(torch.bool).sum(dtype=torch.int64)
                if mask_slice is not None
                else torch.as_tensor(labels_slice.numel(), device=labels_slice.device, dtype=torch.int64)
            )
            item = {
                "source_id": source_ids[i],
                "loss_sum": loss_sum,
                "token_count": token_count,
            }
            if include_data_stats:
                item["sample_count"] = torch.ones((), device=labels_slice.device, dtype=torch.int64)
                item["input_token_count"] = input_token_count
            channel_loss.append(item)
        return channel_loss

    @staticmethod
    def _aggregate_sp(
        per_token_loss: torch.Tensor,
        labels_flat: torch.Tensor,
        attention_mask_flat: torch.Tensor | None,
        source_ids: list[ChannelKey],
        positions: torch.Tensor,
        seq_len: int,
        ignore_index: int,
        parallel_state: ParallelState | None = None,
        capture_descriptor: _SPCaptureDescriptor | None = None,
        strict: bool = False,
        include_data_stats: bool = False,
        defer_reduce: bool = False,
    ) -> list[dict[str, Any]] | _PendingSPObservation:
        if capture_descriptor is not None:
            sp_rank = capture_descriptor.rank
        elif parallel_state is None:
            raise ValueError("parallel_state is required for SP channel-loss aggregation.")
        else:
            sp_group = parallel_state.sp_group
            sp_rank = int(parallel_state.sp_rank)
            if sp_rank < 0:
                sp_rank = dist.get_rank(sp_group) if sp_group is not None else 0
        segments: list[dict[str, Any]] = []

        if positions.dim() == 1:
            positions_2d = positions.reshape(1, -1)
        else:
            positions_2d = positions.reshape(positions.size(0), -1)

        local_width = positions_2d.size(1)
        # Segment ordering is host-side; copy positions once instead of synchronizing per segment.
        positions_cpu = positions_2d.detach().cpu()

        def _partial(
            batch_idx: int,
            start: int,
            end: int,
            is_start: bool,
            local_order: int,
        ) -> None:
            if end <= start:
                return
            flat_start = batch_idx * local_width + start
            flat_end = min(batch_idx * local_width + end, seq_len)
            labels_slice = labels_flat[flat_start:flat_end]
            loss_slice = per_token_loss[flat_start:flat_end]
            pos_slice = positions_cpu[batch_idx, start:end].reshape(-1)[: labels_slice.numel()]
            mask_slice = None
            if attention_mask_flat is not None:
                mask_slice = attention_mask_flat[flat_start:flat_end]
            valid = labels_slice != ignore_index
            token_count = valid.sum(dtype=torch.int64)
            loss_sum = loss_slice[valid].sum()
            input_token_count = (
                mask_slice.to(torch.bool).sum(dtype=torch.int64)
                if mask_slice is not None
                else torch.as_tensor(labels_slice.numel(), device=labels_slice.device, dtype=torch.int64)
            )
            false_flag = torch.zeros((), dtype=torch.bool, device=labels_slice.device)
            has_explicit_padding_mask = (
                ~mask_slice.to(torch.bool).any()
                if labels_slice.numel() > 0 and mask_slice is not None and mask_slice.numel() > 0
                else false_flag
            )
            has_only_zero_positions = bool(pos_slice.eq(0).all().tolist()) if pos_slice.numel() > 0 else False
            segment = {
                "batch_idx": batch_idx,
                "sp_rank": sp_rank,
                "local_order": local_order,
                "is_start": is_start,
                "has_explicit_padding_mask": has_explicit_padding_mask,
                "has_only_zero_positions": has_only_zero_positions,
                "loss_sum": loss_sum,
                "token_count": token_count,
            }
            if include_data_stats:
                segment["sample_count"] = torch.as_tensor(int(is_start), device=labels_slice.device, dtype=torch.int64)
                segment["input_token_count"] = input_token_count
            segments.append(segment)

        for batch_idx, row_pos in enumerate(positions_cpu):
            row_len = min(local_width, max(seq_len - batch_idx * local_width, 0))
            if row_len <= 0:
                continue
            local_starts = row_pos[:row_len].eq(0).nonzero(as_tuple=True)[0].tolist()
            local_order = 0
            if not local_starts:
                _partial(batch_idx, 0, row_len, is_start=False, local_order=local_order)
                continue

            first = local_starts[0]
            if first > 0:
                _partial(batch_idx, 0, first, is_start=False, local_order=local_order)
                local_order += 1
            ends = local_starts[1:] + [row_len]
            for start, end in zip(local_starts, ends):
                _partial(batch_idx, start, end, is_start=True, local_order=local_order)
                local_order += 1

        if defer_reduce:
            return ChannelLossComputer._prepare_sp_observation_payload(
                local_segments=segments,
                source_ids=source_ids,
                device=per_token_loss.device,
                payload_capacity=max(seq_len, 1),
                parallel_state=parallel_state,
                capture_descriptor=capture_descriptor,
                include_data_stats=include_data_stats,
            )

        reduced = ChannelLossComputer._reduce_sp(
            segments,
            source_ids,
            device=per_token_loss.device,
            parallel_state=parallel_state,
            strict=strict,
            include_data_stats=include_data_stats,
        )
        result = []
        for item in reduced:
            converted = {
                "source_id": item["source_id"],
                "loss_sum": torch.as_tensor(item["loss_sum"], device=per_token_loss.device, dtype=torch.float32),
                "token_count": torch.as_tensor(item["token_count"], device=per_token_loss.device, dtype=torch.int64),
            }
            if include_data_stats:
                converted["sample_count"] = torch.as_tensor(
                    item["sample_count"], device=per_token_loss.device, dtype=torch.int64
                )
                converted["input_token_count"] = torch.as_tensor(
                    item["input_token_count"], device=per_token_loss.device, dtype=torch.int64
                )
            result.append(converted)
        return result

    @staticmethod
    def _prepare_sp_observation_payload(
        local_segments: list[dict[str, Any]],
        source_ids: list[ChannelKey],
        device: torch.device,
        payload_capacity: int,
        parallel_state: ParallelState | None = None,
        capture_descriptor: _SPCaptureDescriptor | None = None,
        include_data_stats: bool = False,
    ) -> _PendingSPObservation:
        """Materialize every rank-local buffer before the SP preflight."""

        if payload_capacity < len(local_segments):
            raise ChannelLossMetadataError(
                "Channel loss: local SP segment count exceeds its deterministic payload capacity "
                f"({len(local_segments)} > {payload_capacity})."
            )
        if capture_descriptor is None:
            if parallel_state is None:
                raise ValueError("parallel_state is required to prepare an SP channel-loss observation.")
            capture_descriptor = _sp_descriptor_from_parallel_state(parallel_state)
        group = capture_descriptor.group
        world = capture_descriptor.world

        loss_values: list[torch.Tensor] = []
        integer_rows: list[torch.Tensor] = []
        integer_column_count = 9 if include_data_stats else 7
        for segment in local_segments:
            loss_values.append(
                _as_scalar_tensor(segment["loss_sum"], device=device, dtype=torch.float32, name="loss_sum")
            )
            integer_values = [
                _as_scalar_tensor(segment["batch_idx"], device=device, dtype=torch.int64, name="batch_idx"),
                _as_scalar_tensor(segment["sp_rank"], device=device, dtype=torch.int64, name="sp_rank"),
                _as_scalar_tensor(segment["local_order"], device=device, dtype=torch.int64, name="local_order"),
                _as_scalar_tensor(segment["is_start"], device=device, dtype=torch.int64, name="is_start"),
                _as_scalar_tensor(segment["token_count"], device=device, dtype=torch.int64, name="token_count"),
            ]
            if include_data_stats:
                integer_values.extend(
                    [
                        _as_scalar_tensor(
                            segment["sample_count"], device=device, dtype=torch.int64, name="sample_count"
                        ),
                        _as_scalar_tensor(
                            segment["input_token_count"],
                            device=device,
                            dtype=torch.int64,
                            name="input_token_count",
                        ),
                    ]
                )
            integer_values.extend(
                [
                    _as_scalar_tensor(
                        segment["has_explicit_padding_mask"],
                        device=device,
                        dtype=torch.int64,
                        name="has_explicit_padding_mask",
                    ),
                    _as_scalar_tensor(
                        segment["has_only_zero_positions"],
                        device=device,
                        dtype=torch.int64,
                        name="has_only_zero_positions",
                    ),
                ]
            )
            integer_rows.append(torch.stack(integer_values))

        loss_payload = torch.zeros(payload_capacity, dtype=torch.float32, device=device)
        integer_payload = torch.zeros((payload_capacity, integer_column_count), dtype=torch.int64, device=device)
        if loss_values:
            loss_payload[: len(loss_values)].copy_(torch.stack(loss_values))
            integer_payload[: len(integer_rows)].copy_(torch.stack(integer_rows))

        if world > 1:
            gathered_loss_payloads = [torch.empty_like(loss_payload) for _ in range(world)]
            gathered_integer_payloads = [torch.empty_like(integer_payload) for _ in range(world)]
        else:
            gathered_loss_payloads = [loss_payload]
            gathered_integer_payloads = [integer_payload]

        return _PendingSPObservation(
            source_ids=list(source_ids),
            device=device,
            parallel_state=parallel_state,
            include_data_stats=include_data_stats,
            group=group,
            world=world,
            capture_descriptor=capture_descriptor,
            segment_count=len(local_segments),
            loss_payload=loss_payload,
            integer_payload=integer_payload,
            gathered_segment_counts=[len(local_segments)] if world <= 1 else [0] * world,
            gathered_loss_payloads=gathered_loss_payloads,
            gathered_integer_payloads=gathered_integer_payloads,
        )

    @staticmethod
    def _gather_prepared_sp_payload(
        observation: _PendingSPObservation,
        descriptor: _SPCaptureDescriptor,
    ) -> None:
        if descriptor.group is None or descriptor.world <= 1:
            return
        try:
            dist.all_gather(
                observation.gathered_loss_payloads,
                observation.loss_payload,
                group=descriptor.group,
            )
            dist.all_gather(
                observation.gathered_integer_payloads,
                observation.integer_payload,
                group=descriptor.group,
            )
        except Exception as error:
            raise ChannelLossCollectiveError(
                "Channel loss payload collective failed; the SP process group must not be reused by the observer."
            ) from error

    @staticmethod
    def _reconstruct_prepared_sp_payload(
        observation: _PendingSPObservation,
        strict: bool,
    ) -> list[dict[str, Any]]:
        all_segments: list[dict[str, Any]] = []
        for rank_idx, segment_count in enumerate(observation.gathered_segment_counts):
            metadata_values = (
                observation.gathered_integer_payloads[rank_idx][:segment_count, :4].to(device="cpu").tolist()
            )
            for segment_idx, (batch_idx, sp_rank, local_order, is_start) in enumerate(metadata_values):
                if (
                    int(batch_idx) < 0
                    or int(sp_rank) != rank_idx
                    or int(local_order) < 0
                    or int(is_start) not in (0, 1)
                ):
                    raise ChannelLossMetadataError(
                        "Channel loss: gathered SP segment metadata does not match its fixed payload rank/order."
                    )
                integer_stats = observation.gathered_integer_payloads[rank_idx][segment_idx]
                stats_offset = 2 if observation.include_data_stats else 0
                segment = {
                    "batch_idx": int(batch_idx),
                    "sp_rank": int(sp_rank),
                    "local_order": int(local_order),
                    "is_start": bool(is_start),
                    "loss_sum": observation.gathered_loss_payloads[rank_idx][segment_idx],
                    "token_count": integer_stats[4],
                    "has_explicit_padding_mask": integer_stats[5 + stats_offset].to(torch.bool),
                    "has_only_zero_positions": integer_stats[6 + stats_offset].to(torch.bool),
                }
                if observation.include_data_stats:
                    segment["sample_count"] = integer_stats[5]
                    segment["input_token_count"] = integer_stats[6]
                all_segments.append(segment)

        return ChannelLossComputer._merge_sp_segments(
            all_segments,
            observation.source_ids,
            strict=strict,
            include_data_stats=observation.include_data_stats,
        )

    @staticmethod
    def _reduce_sp(
        local_segments: list[dict[str, Any]],
        source_ids: list[ChannelKey],
        device: torch.device,
        parallel_state: ParallelState | None = None,
        strict: bool = False,
        include_data_stats: bool = False,
    ) -> list[dict[str, Any]]:
        if local_segments:
            local_zero_position_flags = torch.tensor(
                [segment["has_only_zero_positions"] for segment in local_segments],
                dtype=torch.bool,
                device=device,
            )
            for segment, zero_position_flag in zip(local_segments, local_zero_position_flags.unbind()):
                segment["has_only_zero_positions"] = zero_position_flag

        if parallel_state is None:
            raise ValueError("parallel_state is required for SP channel-loss reduction.")
        group = parallel_state.sp_group
        world = parallel_state.sp_size if group is not None else 1
        if group is None or world <= 1:
            all_segments = local_segments
        else:
            local_metadata = [
                {
                    "batch_idx": segment["batch_idx"],
                    "sp_rank": segment["sp_rank"],
                    "local_order": segment["local_order"],
                    "is_start": segment["is_start"],
                }
                for segment in local_segments
            ]
            gathered_metadata: list[list[dict[str, Any]] | None] = [None] * world
            dist.all_gather_object(gathered_metadata, local_metadata, group=group)
            rank_segment_counts = [len(rank_metadata or []) for rank_metadata in gathered_metadata]
            max_segment_count = max(rank_segment_counts, default=0)

            all_segments = []
            if max_segment_count > 0:
                # Keep losses and exact integer counts in dtype-specific compact collectives.
                if local_segments:
                    local_loss_sums = torch.stack(
                        [segment["loss_sum"].to(device=device, dtype=torch.float32) for segment in local_segments]
                    )
                    local_token_counts = torch.stack([segment["token_count"] for segment in local_segments]).to(
                        device=device, dtype=torch.int64
                    )
                    local_explicit_padding_flags = torch.stack(
                        [segment["has_explicit_padding_mask"] for segment in local_segments]
                    ).to(device=device, dtype=torch.int64)
                    local_zero_position_flags = torch.stack(
                        [segment["has_only_zero_positions"] for segment in local_segments]
                    ).to(device=device, dtype=torch.int64)
                    integer_columns = [local_token_counts]
                    if include_data_stats:
                        integer_columns.extend(
                            [
                                torch.stack([segment["sample_count"] for segment in local_segments]).to(
                                    device=device, dtype=torch.int64
                                ),
                                torch.stack([segment["input_token_count"] for segment in local_segments]).to(
                                    device=device, dtype=torch.int64
                                ),
                            ]
                        )
                    integer_columns.extend([local_explicit_padding_flags, local_zero_position_flags])
                    local_integer_stats = torch.stack(integer_columns, dim=1)
                else:
                    local_loss_sums = torch.empty(0, dtype=torch.float32, device=device)
                    integer_column_count = 5 if include_data_stats else 3
                    local_integer_stats = torch.empty((0, integer_column_count), dtype=torch.int64, device=device)

                loss_padding = local_loss_sums.new_zeros(max_segment_count - local_loss_sums.size(0))
                integer_padding = local_integer_stats.new_zeros(
                    (max_segment_count - local_integer_stats.size(0), local_integer_stats.size(1))
                )
                padded_loss_sums = torch.cat([local_loss_sums, loss_padding])
                padded_integer_stats = torch.cat([local_integer_stats, integer_padding])
                gathered_loss_sums = [torch.empty_like(padded_loss_sums) for _ in range(world)]
                gathered_integer_stats = [torch.empty_like(padded_integer_stats) for _ in range(world)]
                dist.all_gather(gathered_loss_sums, padded_loss_sums, group=group)
                dist.all_gather(gathered_integer_stats, padded_integer_stats, group=group)

                for rank_idx, rank_metadata in enumerate(gathered_metadata):
                    for segment_idx, metadata in enumerate(rank_metadata or []):
                        integer_stats = gathered_integer_stats[rank_idx][segment_idx]
                        stats_offset = 2 if include_data_stats else 0
                        segment = {
                            **metadata,
                            "loss_sum": gathered_loss_sums[rank_idx][segment_idx],
                            "token_count": integer_stats[0],
                            "has_explicit_padding_mask": integer_stats[1 + stats_offset].to(torch.bool),
                            "has_only_zero_positions": integer_stats[2 + stats_offset].to(torch.bool),
                        }
                        if include_data_stats:
                            segment["sample_count"] = integer_stats[1]
                            segment["input_token_count"] = integer_stats[2]
                        all_segments.append(segment)

        return ChannelLossComputer._merge_sp_segments(
            all_segments,
            source_ids,
            strict=strict,
            include_data_stats=include_data_stats,
        )

    @staticmethod
    def _merge_sp_segments(
        all_segments: list[dict[str, Any]],
        source_ids: list[ChannelKey],
        strict: bool,
        include_data_stats: bool,
    ) -> list[dict[str, Any]]:
        all_segments.sort(
            key=lambda segment: (
                int(segment.get("batch_idx", 0)),
                int(segment.get("sp_rank", 0)),
                int(segment.get("local_order", 0)),
            )
        )
        start_count = sum(1 for segment in all_segments if segment["is_start"])
        if start_count > len(source_ids):
            token_counts = torch.stack([segment["token_count"] for segment in all_segments])
            explicit_padding_flags = torch.stack([segment["has_explicit_padding_mask"] for segment in all_segments])
            zero_position_flags = torch.stack([segment["has_only_zero_positions"] for segment in all_segments])
            is_padding_only = token_counts.eq(0) & (explicit_padding_flags | zero_position_flags)
            padding_flags = torch.stack([is_padding_only, explicit_padding_flags], dim=1).to(
                device="cpu", dtype=torch.bool
            )
            for segment, (is_padding_only, has_explicit_padding_mask) in zip(all_segments, padding_flags.tolist()):
                segment["is_padding_only"] = is_padding_only
                segment["has_explicit_padding_mask"] = has_explicit_padding_mask
            all_segments = _filter_surplus_tail_padding_sp_segments(all_segments, len(source_ids))
            start_count = sum(1 for segment in all_segments if segment["is_start"])
        if not _validate_segment_count(start_count, len(source_ids), strict, "SP packed segment count"):
            return []

        doc_idx = 0
        result: list[dict[str, Any]] = []
        last_result_idx_by_batch: dict[int, int] = {}
        for segment in all_segments:
            batch_idx = int(segment.get("batch_idx", 0))
            if segment["is_start"]:
                item = {
                    "source_id": source_ids[doc_idx],
                    "loss_sum": segment["loss_sum"],
                    "token_count": segment["token_count"],
                }
                if include_data_stats:
                    item["sample_count"] = segment["sample_count"]
                    item["input_token_count"] = segment["input_token_count"]
                result.append(item)
                last_result_idx_by_batch[batch_idx] = len(result) - 1
                doc_idx += 1
                continue

            result_idx = last_result_idx_by_batch.get(batch_idx)
            if result_idx is None:
                msg = (
                    "Channel loss: SP continuation segment appeared before any source segment "
                    f"for batch_idx={batch_idx}; skipping this partial segment."
                )
                if strict:
                    raise ChannelLossMetadataError(msg)
                logger.warning_rank0(msg)
                continue
            result[result_idx]["loss_sum"] += segment["loss_sum"]
            result[result_idx]["token_count"] += segment["token_count"]
            if include_data_stats:
                result[result_idx]["sample_count"] += segment["sample_count"]
                result[result_idx]["input_token_count"] += segment["input_token_count"]
        return result


class ChannelLossCallback(Callback):
    """Trainer callback for detached per-source causal-LM loss metrics."""

    def __init__(self, trainer: BaseTrainer) -> None:
        super().__init__(trainer)
        self.config = trainer.args.train.channel_loss
        data_type = getattr(getattr(trainer.args, "data", None), "data_type", None)
        if self.config.enable and data_type == "classification":
            raise ValueError(
                "train.channel_loss is only supported by causal-LM objectives; "
                "data.data_type='classification' uses sequence-classification loss."
            )
        is_base_rl_trainer = any(
            cls.__name__ == "BaseRLTrainer" and cls.__module__ == "veomni.trainer.base_rl_trainer"
            for cls in type(trainer).__mro__
        )
        if self.config.enable and is_base_rl_trainer:
            raise ValueError(
                "train.channel_loss does not support BaseRLTrainer because it packs metadata during preforward, "
                "after the common channel-loss step lifecycle captures alignment metadata."
            )
        self.computer = ChannelLossComputer(parallel_state=self.parallel_state)
        self._strip_keys = set(self.config.source_id_keys)
        self._strip_keys.update(self.config.source_name_keys)
        self._strip_keys.update(self.config.extra_strip_keys)
        self.computer.strict = bool(self.config.strict)
        self.computer.release_cache = bool(self.config.release_cache)
        self._collect_step = False
        self._source_registry: dict[ChannelKey, str | None] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enable)

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_registry": sorted(
                self._source_registry.items(),
                key=lambda item: _channel_sort_key(item[0]),
            )
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        entries = state_dict.get("source_registry", [])
        if not isinstance(entries, (list, tuple)):
            raise ValueError("Channel loss checkpoint source_registry must be a list of pairs.")

        restored: dict[ChannelKey, str | None] = {}
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("Channel loss checkpoint source_registry entries must be (source_id, source_name).")
            raw_source_id, raw_source_name = entry
            source_ids = _as_channel_key_list(raw_source_id)
            if len(source_ids) != 1:
                raise ValueError(f"Invalid channel loss checkpoint source ID: {raw_source_id!r}.")
            source_id = source_ids[0]
            source_name = None if raw_source_name is None else str(raw_source_name)
            if source_id in restored and restored[source_id] != source_name:
                raise ValueError(f"Duplicate channel loss checkpoint source ID with conflicting names: {source_id!r}.")
            restored[source_id] = source_name

        self._source_registry = restored
        self.computer.source_names.update(
            {source_id: source_name for source_id, source_name in restored.items() if source_name is not None}
        )

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self.computer.install(self.trainer.model)
        logger.info_rank0(
            "Channel loss enabled "
            f"(interval={self.config.interval}, source_id_keys={self.config.source_id_keys}, "
            f"source_name_keys={self.config.source_name_keys})."
        )

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        if self.enabled:
            self.computer.uninstall()

    def on_step_begin(
        self,
        state: TrainerState,
        micro_batches: list[dict[str, Any]] = None,
        source_repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            self._collect_step = False
            return
        if source_repeat < 1:
            raise ValueError(f"Channel loss source_repeat must be >= 1, got {source_repeat}.")

        self._collect_step = state.global_step % self.config.interval == 0
        if not self._collect_step:
            self.computer.begin_step([], [], [])
            if self.config.strict:
                missing_source_micro_steps = []
                for micro_step, micro_batch in enumerate(micro_batches or []):
                    source_ids, _, _, _ = self._extract_metadata(micro_batch)
                    if not source_ids:
                        missing_source_micro_steps.append(micro_step)
                self._raise_if_missing_source_ids(missing_source_micro_steps)
            return

        per_mb_source_ids: list[list[ChannelKey]] = []
        per_mb_position_ids: list[torch.Tensor | None] = []
        per_mb_attention_masks: list[torch.Tensor | None] = []
        missing_source_micro_steps = []
        for micro_step, micro_batch in enumerate(micro_batches or []):
            source_ids, source_names, position_ids, attention_mask = self._extract_metadata(micro_batch)
            if self.config.strict and not source_ids:
                missing_source_micro_steps.append(micro_step)
            source_ids, source_names = self._repeat_source_metadata(source_ids, source_names, source_repeat)
            self.computer.record_source_names(source_ids, source_names)
            per_mb_source_ids.append(source_ids)
            per_mb_position_ids.append(position_ids)
            per_mb_attention_masks.append(attention_mask)

        self.computer.begin_step(per_mb_source_ids, per_mb_position_ids, per_mb_attention_masks)
        if self.config.strict:
            self._raise_if_missing_source_ids(missing_source_micro_steps)

    @staticmethod
    def _repeat_source_metadata(
        source_ids: list[ChannelKey],
        source_names: list[str],
        repeat: int,
    ) -> tuple[list[ChannelKey], list[str]]:
        if repeat == 1:
            return source_ids, source_names

        repeated_ids = [source_id for source_id in source_ids for _ in range(repeat)]
        if len(source_names) == len(source_ids):
            source_names = [source_name for source_name in source_names for _ in range(repeat)]
        return repeated_ids, source_names

    def _raise_if_missing_source_ids(self, local_missing_micro_steps: list[int]) -> None:
        if not self._synchronize_strict_failure(bool(local_missing_micro_steps)):
            return
        raise ValueError(
            "train.channel_loss.enable=True but no configured source ID was found in a micro-batch on at least "
            f"one rank. Checked keys: {self.config.source_id_keys}; "
            f"local missing micro-batch indices: {local_missing_micro_steps}."
        )

    def on_micro_step_begin(
        self,
        state: TrainerState,
        micro_batch: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if not self._collect_step:
            return
        self.computer.before_micro_step()

    def strip_model_inputs(self, micro_batch: dict[str, Any]) -> None:
        if not self.enabled or not isinstance(micro_batch, dict):
            return
        for key in self._strip_keys:
            micro_batch.pop(key, None)

    def on_micro_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        if self._collect_step:
            self.computer.after_micro_step()

    @contextmanager
    def micro_step_context(self, state: TrainerState, micro_batch: dict[str, Any]) -> Iterator[None]:
        if not self._collect_step:
            yield
            return
        self.on_micro_step_begin(state, micro_batch=micro_batch)
        try:
            yield
        finally:
            self.on_micro_step_end(state)

    @contextmanager
    def model_forward_context(self) -> Iterator[None]:
        if not self._collect_step:
            yield
            return
        with self.computer.capture():
            yield

    def on_step_end(
        self,
        state: TrainerState,
        loss: float,
        loss_dict: dict[str, float],
        grad_norm: float,
        **kwargs: Any,
    ) -> None:
        if not self._collect_step:
            return

        if self.config.strict and self._synchronize_missing_observation():
            raise ChannelLossMetadataError(
                "Channel loss: sampled micro-step had source metadata but produced no causal-LM CE observation "
                "or raised a channel-loss observation error "
                f"(local missing micro-step indices: {self.computer.missing_observation_micro_steps}; "
                f"local observation errors: {self.computer.strict_observation_errors})."
            )

        strict_reduction_errors: list[str] = []
        totals, source_names = self._reduce_step_totals(
            self.computer.step_totals,
            strict_errors=strict_reduction_errors if self.config.strict else None,
        )
        if not totals:
            if self.config.strict and any(self.computer._per_mb_source_ids):
                strict_reduction_errors.append(
                    "Channel loss: sampled optimizer step had source metadata but produced no causal-LM CE totals. "
                    "The model forward did not invoke an observed loss path."
                )
            if self.config.strict:
                self._raise_if_strict_post_collective_failure(strict_reduction_errors)
            return

        data_totals = self._reduce_step_data_totals(self.computer.step_data_totals, list(totals))
        if self.config.strict:
            self._raise_if_strict_post_collective_failure(strict_reduction_errors)

        metrics = self._build_metrics(totals, source_names)
        metrics.update(self._build_data_metrics(data_totals, source_names))
        if not metrics:
            return

        if getattr(self.trainer, "step_train_metrics", None) is None:
            self.trainer.step_train_metrics = {}
        if getattr(self.trainer, "step_env_metrics", None) is None:
            self.trainer.step_env_metrics = {}

        self.trainer.step_train_metrics.update(metrics)
        self.trainer.step_env_metrics.update(metrics)

    def _synchronize_missing_observation(self) -> bool:
        missing = bool(self.computer.missing_observation_micro_steps or self.computer.strict_observation_errors)
        return self._synchronize_strict_failure(missing)

    def _synchronize_strict_failure(self, failed: bool) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return failed

        device = self._totals_device(self.computer.step_totals)
        failure_count = torch.tensor(int(failed), dtype=torch.int64, device=device)
        # This helper is called only at a common world boundary, before any
        # group-specific collective or after every such collective completes.
        dist.all_reduce(failure_count)
        return bool(failure_count.cpu().item())

    def _raise_if_strict_post_collective_failure(self, local_errors: list[str]) -> None:
        if not self._synchronize_strict_failure(bool(local_errors)):
            return
        local_detail = " | ".join(local_errors) if local_errors else "none on this rank"
        raise ChannelLossMetadataError(
            "Channel loss: strict reduction/registry validation failed on at least one rank after all "
            f"group-specific collectives completed (local errors: {local_detail})."
        )

    def _reduce_step_totals(
        self,
        local_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor]],
        *,
        strict_errors: list[str] | None = None,
    ) -> tuple[dict[ChannelKey, tuple[float, int]], dict[ChannelKey, str]]:
        ps = self.parallel_state
        dp_size = ps.dp_size
        dp_group = ps.dp_group

        distributed = dp_size > 1 and dist.is_available() and dist.is_initialized()
        local_metadata = [
            (source_id, self._lookup_source_name(source_id))
            for source_id in sorted(local_totals, key=_channel_sort_key)
        ]
        if distributed:
            gathered_metadata: list[list[tuple[ChannelKey, str | None]] | None] = [None] * dp_size
            dist.all_gather_object(gathered_metadata, local_metadata, group=dp_group)
        else:
            gathered_metadata = [local_metadata]

        source_ids = sorted(
            {source_id for rank_metadata in gathered_metadata if rank_metadata for source_id, _ in rank_metadata},
            key=_channel_sort_key,
        )
        if not source_ids:
            return {}, {}

        name_candidates: dict[ChannelKey, set[str]] = defaultdict(set)
        for rank_metadata in gathered_metadata:
            for source_id, source_name in rank_metadata or []:
                if source_name:
                    name_candidates[source_id].add(source_name)

        synchronized_names: dict[ChannelKey, str] = {}
        for source_id in source_ids:
            candidates = sorted(name_candidates[source_id])
            if len(candidates) > 1:
                msg = f"Channel loss: source_id={source_id!r} has inconsistent names across DP ranks: {candidates}."
                if self.config.strict:
                    if strict_errors is None:
                        raise ChannelLossMetadataError(msg)
                    strict_errors.append(msg)
                else:
                    logger.warning_rank0(msg + " Using the lexicographically first name.")
            if candidates:
                synchronized_names[source_id] = candidates[0]

        device = self._totals_device(local_totals)
        loss_sums = torch.zeros(len(source_ids), dtype=torch.float32, device=device)
        token_counts = torch.zeros(len(source_ids), dtype=torch.int64, device=device)
        source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
        for source_id, (loss_sum, token_count) in local_totals.items():
            index = source_indices[source_id]
            loss_sums[index] = torch.as_tensor(loss_sum, dtype=torch.float32, device=device)
            token_counts[index] = torch.as_tensor(token_count, dtype=torch.int64, device=device)

        if distributed:
            dist.all_reduce(loss_sums, group=dp_group)
            dist.all_reduce(token_counts, group=dp_group)

        loss_values = loss_sums.cpu().tolist()
        token_values = token_counts.cpu().tolist()
        totals = {
            source_id: (float(loss_values[index]), int(token_values[index]))
            for index, source_id in enumerate(source_ids)
        }
        registered_names = self._register_sources(
            source_ids,
            synchronized_names,
            strict_errors=strict_errors,
        )
        return totals, registered_names

    def _reduce_step_data_totals(
        self,
        local_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        source_ids: list[ChannelKey],
    ) -> dict[ChannelKey, tuple[int, int, int]]:
        if not source_ids:
            return {}

        device = self._data_totals_device(local_totals)
        counts = torch.zeros((len(source_ids), 3), dtype=torch.int64, device=device)
        source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
        for source_id, (sample_count, input_token_count, label_token_count) in local_totals.items():
            index = source_indices.get(source_id)
            if index is None:
                continue
            counts[index] = torch.stack(
                [
                    torch.as_tensor(sample_count, dtype=torch.int64, device=device),
                    torch.as_tensor(input_token_count, dtype=torch.int64, device=device),
                    torch.as_tensor(label_token_count, dtype=torch.int64, device=device),
                ]
            )

        ps = self.parallel_state
        if ps.dp_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, group=ps.dp_group)

        values = counts.cpu().tolist()
        return {
            source_id: (int(values[index][0]), int(values[index][1]), int(values[index][2]))
            for index, source_id in enumerate(source_ids)
        }

    def _register_sources(
        self,
        source_ids: list[ChannelKey],
        source_names: dict[ChannelKey, str] | None = None,
        *,
        strict_errors: list[str] | None = None,
    ) -> dict[ChannelKey, str]:
        source_names = source_names or {}
        for source_id in source_ids:
            incoming_name = source_names.get(source_id)
            if source_id not in self._source_registry:
                self._source_registry[source_id] = incoming_name
                continue

            registered_name = self._source_registry[source_id]
            if incoming_name is None or incoming_name == registered_name:
                continue
            if registered_name is None:
                self._source_registry[source_id] = incoming_name
                continue

            candidates = sorted({registered_name, incoming_name})
            msg = f"Channel loss: source_id={source_id!r} has inconsistent names across sampled steps: {candidates}."
            if self.config.strict:
                if strict_errors is None:
                    raise ChannelLossMetadataError(msg)
                strict_errors.append(msg)
            else:
                logger.warning_once(msg + " Using the lexicographically first name.")
            self._source_registry[source_id] = candidates[0]

        registered_names = {
            source_id: source_name
            for source_id, source_name in self._source_registry.items()
            if source_name is not None
        }
        self.computer.source_names.update(registered_names)
        return {source_id: registered_names[source_id] for source_id in source_ids if source_id in registered_names}

    def _totals_device(
        self,
        local_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.device:
        for loss_sum, _ in local_totals.values():
            if isinstance(loss_sum, torch.Tensor):
                return loss_sum.device

        trainer_device = getattr(self.trainer, "device", None)
        if trainer_device is not None:
            return torch.device(trainer_device)

        model = getattr(self.trainer, "model", None)
        if model is not None:
            try:
                return next(model.parameters()).device
            except (AttributeError, StopIteration):
                pass
        return torch.device("cpu")

    def _data_totals_device(
        self,
        local_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.device:
        for counts in local_totals.values():
            for count in counts:
                if isinstance(count, torch.Tensor):
                    return count.device
        return self._totals_device({})

    def _build_metrics(
        self,
        totals: dict[ChannelKey, tuple[float, int]],
        source_names: dict[ChannelKey, str] | None = None,
    ) -> dict[str, float]:
        total_tokens = sum(token_count for _, token_count in totals.values())
        metric_names = self._render_metric_names(totals, source_names)
        metrics: dict[str, float] = {}
        for source_id, (loss_sum, token_count) in sorted(totals.items(), key=lambda item: _channel_sort_key(item[0])):
            if token_count <= 0:
                continue
            name = metric_names[source_id]
            metrics[f"{self.config.loss_metric_prefix}/{name}"] = loss_sum / token_count
            if self.config.log_weighted_loss and total_tokens > 0:
                metrics[f"{self.config.weighted_loss_metric_prefix}/{name}"] = loss_sum / total_tokens
            if self.config.log_token_count:
                metrics[f"{self.config.token_count_metric_prefix}/{name}"] = float(token_count)
        return metrics

    def _build_data_metrics(
        self,
        totals: dict[ChannelKey, tuple[int, int, int]],
        source_names: dict[ChannelKey, str] | None = None,
    ) -> dict[str, float]:
        metric_names = self._render_data_metric_names(totals, source_names)
        metrics: dict[str, float] = {}
        for source_id, (sample_count, input_token_count, label_token_count) in sorted(
            totals.items(), key=lambda item: _channel_sort_key(item[0])
        ):
            if sample_count <= 0:
                continue
            name = metric_names[source_id]
            metrics[f"samples/{name}"] = float(sample_count)
            metrics[f"input_tokens/{name}"] = float(input_token_count)
            metrics[f"label_tokens/{name}"] = float(label_token_count)
            metrics[f"label_tokens_per_sample/{name}"] = label_token_count / sample_count
        return metrics

    def _render_data_metric_names(
        self,
        totals: dict[ChannelKey, tuple[int, int, int]],
        source_names: dict[ChannelKey, str] | None,
    ) -> dict[ChannelKey, str]:
        base_names = self._render_base_metric_names(totals, source_names)
        name_counts: dict[str, int] = defaultdict(int)
        for name in base_names.values():
            name_counts[name] += 1
        qualified_names = {
            source_id: f"{_stable_source_metric_fragment(source_id)}__{base_names[source_id]}"
            for source_id in base_names
        }
        qualified_values = set(qualified_names.values())
        return {
            source_id: (
                base_names[source_id]
                if name_counts[base_names[source_id]] == 1 and base_names[source_id] not in qualified_values
                else qualified_names[source_id]
            )
            for source_id in totals
        }

    def _render_metric_names(
        self,
        totals: dict[ChannelKey, tuple[float, int]],
        source_names: dict[ChannelKey, str] | None,
    ) -> dict[ChannelKey, str]:
        base_names = self._render_base_metric_names(totals, source_names)
        return {
            source_id: f"{_stable_source_metric_fragment(source_id)}__{base_names[source_id]}" for source_id in totals
        }

    def _render_base_metric_names(
        self,
        totals: dict[ChannelKey, tuple[Any, ...]],
        source_names: dict[ChannelKey, str] | None,
    ) -> dict[ChannelKey, str]:
        self._register_sources(list(totals), source_names)
        rendered: dict[ChannelKey, str] = {}
        for source_id in self._source_registry:
            source_name = self._source_registry[source_id]
            rendered[source_id] = (
                _sanitize_metric_fragment(source_name)
                if source_name is not None
                else self._resolve_source_name(source_id)
            )
        return rendered

    def _resolve_source_name(self, source_id: ChannelKey) -> str:
        source_name = self._lookup_source_name(source_id)
        if source_name is not None:
            return _sanitize_metric_fragment(source_name)

        return _sanitize_metric_fragment(f"source_{source_id}")

    def _lookup_source_name(self, source_id: ChannelKey) -> str | None:
        if source_id in self.computer.source_names:
            return str(self.computer.source_names[source_id])

        meter = getattr(self.trainer, "environ_meter", None)
        tracker = getattr(meter, "multisource_tracker", None) if meter is not None else None
        names = getattr(tracker, "names", None) if tracker is not None else None
        if isinstance(names, dict) and source_id in names:
            return str(names[source_id])
        if isinstance(source_id, int) and isinstance(names, (list, tuple)) and 0 <= source_id < len(names):
            return str(names[source_id])

        return None

    def _extract_metadata(
        self,
        micro_batch: Any,
    ) -> tuple[list[ChannelKey], list[str], torch.Tensor | None, torch.Tensor | None]:
        if isinstance(micro_batch, dict):
            source_ids = self._first_present_list(micro_batch, self.config.source_id_keys, _as_channel_key_list)
            source_names = self._first_present_list(micro_batch, self.config.source_name_keys, _as_str_list)
            position_ids = micro_batch.get("position_ids")
            attention_mask = micro_batch.get("attention_mask")
            return (
                source_ids,
                source_names,
                position_ids if isinstance(position_ids, torch.Tensor) else None,
                attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            )

        if isinstance(micro_batch, (list, tuple)):
            source_ids: list[ChannelKey] = []
            source_names: list[str] = []
            for sample in micro_batch:
                if not isinstance(sample, dict):
                    continue
                source_ids.extend(self._first_present_list(sample, self.config.source_id_keys, _as_channel_key_list))
                source_names.extend(self._first_present_list(sample, self.config.source_name_keys, _as_str_list))
            return source_ids, source_names, None, None

        return [], [], None, None

    @staticmethod
    def _first_present_list(
        batch: dict[str, Any],
        keys: list[str],
        converter: Callable[[Any], list[Any]],
    ) -> list[Any]:
        for key in keys:
            if key in batch:
                return converter(batch[key])
        return []


def _as_flat_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_as_flat_list(item))
        return result
    return [value]


def _bind_loss_call_args(
    loss_fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(loss_fn)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        bound_args = dict(bound.arguments)
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                extra_kwargs = bound_args.pop(name, {})
                if isinstance(extra_kwargs, dict):
                    bound_args.update(extra_kwargs)
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                bound_args.pop(name, None)

        for key, value in zip(_LOSS_CALL_POSITIONAL_KEYS, args):
            bound_args.setdefault(key, value)
        return bound_args
    except (TypeError, ValueError):
        bound_args = dict(zip(_LOSS_CALL_POSITIONAL_KEYS, args))
        bound_args.update(kwargs)
        return bound_args


def _position_aligned_attention_mask_flat(
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor,
    effective_labels: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    if attention_mask.numel() == labels.numel() or attention_mask.numel() == effective_labels.numel():
        return attention_mask.reshape(-1)
    return None


def _filter_surplus_tail_padding_segments(
    segments: list[tuple[int, int, int]],
    source_count: int,
    labels_flat: torch.Tensor,
    positions_flat: torch.Tensor | None,
    attention_mask_flat: torch.Tensor | None,
    ignore_index: int,
) -> list[tuple[int, int, int]]:
    surplus = len(segments) - source_count
    if surplus <= 0:
        return segments

    indexed_by_batch: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for segment_idx, (batch_idx, start, end) in enumerate(segments):
        indexed_by_batch[batch_idx].append((segment_idx, start, end))

    candidates_by_batch: dict[int, list[tuple[bool, int]]] = defaultdict(list)
    for batch_idx, row_segments in indexed_by_batch.items():
        for segment_idx, start, end in reversed(row_segments):
            mask_slice = attention_mask_flat[start : end + 1] if attention_mask_flat is not None else None
            if not _is_padding_only_segment(
                labels_slice=labels_flat[start : end + 1],
                positions_slice=positions_flat[start : end + 1] if positions_flat is not None else None,
                attention_mask_slice=mask_slice,
                ignore_index=ignore_index,
            ):
                break
            candidates_by_batch[batch_idx].append((_has_explicit_padding_mask(mask_slice), segment_idx))

    selected = _select_unambiguous_tail_candidates(candidates_by_batch, surplus)
    if selected is None:
        return segments
    removed = set(selected)
    return [segment for segment_idx, segment in enumerate(segments) if segment_idx not in removed]


def _filter_surplus_tail_padding_sp_segments(
    segments: list[dict[str, Any]],
    source_count: int,
) -> list[dict[str, Any]]:
    surplus = sum(1 for segment in segments if segment["is_start"]) - source_count
    if surplus <= 0:
        return segments

    indexed_by_batch: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for segment_idx, segment in enumerate(segments):
        indexed_by_batch[int(segment.get("batch_idx", 0))].append((segment_idx, segment))

    candidates_by_batch: dict[int, list[tuple[bool, list[int]]]] = defaultdict(list)
    for batch_idx, row_partials in indexed_by_batch.items():
        logical_segments: list[list[tuple[int, dict[str, Any]]]] = []
        for segment_idx, segment in row_partials:
            if segment["is_start"]:
                logical_segments.append([])
            if logical_segments:
                logical_segments[-1].append((segment_idx, segment))

        for logical_segment in reversed(logical_segments):
            if not logical_segment or not all(partial.get("is_padding_only", False) for _, partial in logical_segment):
                break
            candidates_by_batch[batch_idx].append(
                (
                    all(partial.get("has_explicit_padding_mask", False) for _, partial in logical_segment),
                    [segment_idx for segment_idx, _ in logical_segment],
                )
            )

    selected = _select_unambiguous_tail_candidates(candidates_by_batch, surplus)
    if selected is None:
        return segments
    removed: set[int] = set()
    for segment_indices in selected:
        removed.update(segment_indices)
    return [segment for segment_idx, segment in enumerate(segments) if segment_idx not in removed]


def _is_padding_only_segment(
    labels_slice: torch.Tensor,
    positions_slice: torch.Tensor | None,
    attention_mask_slice: torch.Tensor | None,
    ignore_index: int,
) -> bool:
    if labels_slice.numel() == 0 or labels_slice.ne(ignore_index).any().item():
        return False

    if attention_mask_slice is not None and attention_mask_slice.numel() > 0:
        if not attention_mask_slice.to(torch.bool).any().item():
            return True

    if positions_slice is not None and positions_slice.numel() > 0:
        return bool(positions_slice.eq(0).all().item())

    return False


def _has_explicit_padding_mask(attention_mask_slice: torch.Tensor | None) -> bool:
    return bool(
        attention_mask_slice is not None
        and attention_mask_slice.numel() > 0
        and not attention_mask_slice.to(torch.bool).any().item()
    )


def _select_unambiguous_tail_candidates(
    candidates_by_batch: dict[int, list[tuple[bool, Any]]],
    surplus: int,
) -> list[Any] | None:
    # Each layer stores compact backpointers instead of copying payload prefixes.
    batch_ids = sorted(candidates_by_batch)
    states: dict[int, tuple[int, bool]] = {0: (0, True)}
    layers: list[dict[int, tuple[int, bool, int, int]]] = []
    for batch_idx in batch_ids:
        row_candidates = candidates_by_batch[batch_idx]
        explicit_prefix = [0]
        for has_explicit_mask, _ in row_candidates:
            explicit_prefix.append(explicit_prefix[-1] + int(has_explicit_mask))

        next_layer: dict[int, tuple[int, bool, int, int]] = {}
        for removed_count, (score, is_unique) in states.items():
            for row_removed in range(len(row_candidates) + 1):
                new_count = removed_count + row_removed
                if new_count > surplus:
                    break
                candidate_score = score + explicit_prefix[row_removed]
                existing = next_layer.get(new_count)
                if existing is None or candidate_score > existing[0]:
                    next_layer[new_count] = (candidate_score, is_unique, removed_count, row_removed)
                elif candidate_score == existing[0]:
                    next_layer[new_count] = (existing[0], False, existing[2], existing[3])
        layers.append(next_layer)
        states = {removed_count: (entry[0], entry[1]) for removed_count, entry in next_layer.items()}

    best = states.get(surplus)
    if best is None or not best[1]:
        return None

    selected: list[Any] = []
    removed_count = surplus
    for batch_idx, layer in zip(reversed(batch_ids), reversed(layers)):
        _, _, previous_count, row_removed = layer[removed_count]
        selected.extend(payload for _, payload in candidates_by_batch[batch_idx][:row_removed])
        removed_count = previous_count
    return selected


def _validate_segment_count(segment_count: int, source_count: int, strict: bool, context: str) -> bool:
    if segment_count == source_count:
        return True

    msg = (
        f"Channel loss: source metadata count ({source_count}) does not match {context} "
        f"({segment_count}); skipping this micro-batch."
    )
    if strict:
        raise ChannelLossMetadataError(msg)
    logger.warning_rank0(msg)
    return False


def _as_channel_key_list(value: Any) -> list[ChannelKey]:
    result: list[ChannelKey] = []
    for item in _as_flat_list(value):
        if item is None:
            continue
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        if isinstance(item, Integral):
            result.append(int(item))
            continue
        text = str(item)
        if text:
            result.append(text)
    return result


def _as_str_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _as_flat_list(value):
        if item is None:
            continue
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        text = str(item)
        if text:
            result.append(text)
    return result


def _as_scalar_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() != 1:
        raise ChannelLossMetadataError(
            f"Channel loss: local SP {name} must be scalar, got shape={tuple(tensor.shape)}."
        )
    return tensor.reshape(())


def _source_id_signature(source_ids: list[ChannelKey]) -> tuple[tuple[str, str], ...]:
    return tuple((type(source_id).__name__, repr(source_id)) for source_id in source_ids)


def _format_capture_errors(errors: list[tuple[str, str]]) -> str:
    return "Channel loss observation failed: " + " | ".join(
        f"{error_type}: {message}" for error_type, message in errors
    )


def _sanitize_metric_fragment(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def _stable_source_metric_fragment(value: ChannelKey) -> str:
    if isinstance(value, int):
        return f"source-i-{value}"
    encoded = str(value).encode("utf-8").hex()
    return f"source-s-{encoded or 'empty'}"


def _channel_sort_key(value: ChannelKey) -> tuple[int, str]:
    return (0 if isinstance(value, int) else 1, str(value))
