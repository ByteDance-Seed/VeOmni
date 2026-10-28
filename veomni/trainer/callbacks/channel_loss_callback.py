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

import hashlib
import inspect
import pickle
import re
from collections import defaultdict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from numbers import Integral
from threading import local
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...data.data_collator import CHANNEL_LOSS_PROVENANCE, _ChannelLossProvenance
from ...distributed.parallel_state import get_parallel_state
from ...utils.constants import IGNORE_INDEX
from ...utils.device import empty_cache, get_torch_device, stream_synchronize
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


def _pop_channel_loss_provenance(batch: Any) -> _ChannelLossProvenance | None:
    if isinstance(batch, dict) and isinstance(batch.get(CHANNEL_LOSS_PROVENANCE), _ChannelLossProvenance):
        return batch.pop(CHANNEL_LOSS_PROVENANCE)
    return None


if TYPE_CHECKING:
    from ...distributed.parallel_state import ParallelState
    from ..base import BaseTrainer


ChannelKey = Hashable
_LOSS_CALL_POSITIONAL_KEYS = ("logits", "labels", "vocab_size", "num_items_in_batch", "ignore_index")
_ACTIVE_CHANNEL_LOSS_COMPUTER: ContextVar[ChannelLossComputer | None] = ContextVar(
    "veomni_active_channel_loss_computer", default=None
)
_CHUNK_LOSS_CAPTURE = local()


@contextmanager
def capture_chunk_loss_per_token(
    consumer: Callable[[torch.Tensor], None], on_error: Callable[[str], None] | None = None
) -> Iterator[None]:
    """Enable detached token-loss capture for one sampled model-loss call."""
    from ...ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

    previous = getattr(_CHUNK_LOSS_CAPTURE, "value", None)
    capture = {}
    token = chunk_loss._CAPTURE.set(capture)
    _CHUNK_LOSS_CAPTURE.value = (consumer, on_error)
    completed = False
    try:
        yield
        completed = True
    finally:
        chunk_loss._CAPTURE.reset(token)
        _CHUNK_LOSS_CAPTURE.value = previous
        if completed:
            if capture.get("op_calls", 0) > 1:
                capture["error"] = "multiple chunk-loss observations in one causal-loss call"
            elif capture.get("error") and capture.get("op_calls", 0) != 1:
                capture["error"] = "chunk-loss observation failed without a verifiable single op"
            if capture.get("error"):
                if on_error is None:
                    raise RuntimeError(capture["error"])
                on_error(capture["error"])
            elif capture.get("per_token") is not None:
                try:
                    consumer(capture["per_token"])
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(f"{type(error).__name__}: {error}")


def get_chunk_loss_consumer() -> Callable[[torch.Tensor], None] | None:
    capture = getattr(_CHUNK_LOSS_CAPTURE, "value", None)
    return capture[0] if capture is not None else None


def report_chunk_loss_observer_error(error: Exception) -> bool:
    capture = getattr(_CHUNK_LOSS_CAPTURE, "value", None)
    handler = capture[1] if capture is not None else None
    if handler is None:
        return False
    handler(f"{type(error).__name__}: {error}")
    return True


class ChannelLossMetadataError(ValueError):
    """Raised when channel metadata cannot be aligned with packed segments."""


class ChannelLossCollectiveError(RuntimeError):
    """Raised when an observer collective fails and its process group is unsafe."""


def _release_observer_cache(device: torch.device) -> None:
    """Release detached CE allocator cache before backward, when explicitly requested."""
    if device.type == "cpu":
        return
    accelerator = get_torch_device()
    with accelerator.device(device):
        stream_synchronize()
        empty_cache()


def _source_signature_digest(source_ids: list[ChannelKey]) -> str:
    return hashlib.sha256(repr(_source_id_signature(source_ids)).encode("utf-8", errors="surrogatepass")).hexdigest()


def _serialize_channel_sp_observation(observation):
    segments = observation["segments"]
    device = observation["device"]
    losses = (
        torch.stack([s["loss_sum"].detach().to(device=device, dtype=torch.float32) for s in segments]).cpu().tolist()
        if segments
        else []
    )
    columns = ("token_count", "has_explicit_padding_mask")
    if observation["data"]:
        columns += ("sample_count", "input_token_count")
    integer = (
        torch.stack([torch.stack([s[k].to(device=device, dtype=torch.int64) for k in columns]) for s in segments])
        .cpu()
        .tolist()
        if segments
        else []
    )
    integer = [
        row[:2] + [int(segment["has_only_zero_positions"])] + row[2:]
        for segment, row in zip(segments, integer, strict=True)
    ]
    return {
        "mode": observation["mode"],
        "data": observation["data"],
        "source_digest": _source_signature_digest(observation["source_ids"]),
        "layout_digest": observation.get("layout_digest"),
        "geometry_candidates": observation.get("geometry_candidates", 3),
        "segments": [
            (int(s["batch_idx"]), int(s["sp_rank"]), int(s["local_order"]), bool(s["is_start"]), loss, values)
            for s, loss, values in zip(segments, losses, integer, strict=True)
        ],
    }


def _channel_sp_topology(parallel_state, *, initialized, global_rank):
    status = {"enabled": False, "sp_rank": 0, "sp_size": 1, "members": (), "topology_error": ""}
    try:
        status["enabled"] = bool(parallel_state.sp_enabled)
        if status["enabled"]:
            size = int(parallel_state.sp_size)
            group = parallel_state.sp_group
            rank = int(parallel_state.sp_rank)
            if rank < 0:
                rank = int(dist.get_rank(group)) if initialized and group is not None else 0
            if size < 1 or rank < 0 or rank >= size or (size > 1 and (group is None or not initialized)):
                raise ValueError("invalid or unavailable SP group")
            if group is None:
                members = (global_rank,)
            elif hasattr(dist, "get_process_group_ranks"):
                members = tuple(int(r) for r in dist.get_process_group_ranks(group))
            else:
                members = tuple(int(dist.get_global_rank(group, r)) for r in range(size))
            status.update(sp_rank=rank, sp_size=size, members=members)
    except Exception as error:
        status["topology_error"] = f"{type(error).__name__}: {error}"
    return status


def _validate_channel_sp_topologies(statuses):
    if not statuses or any(s["topology_error"] for s in statuses):
        return f"SP topology could not be resolved on all ranks: {[s['topology_error'] for s in statuses]}"
    if len({s["enabled"] for s in statuses}) != 1:
        return "SP enablement differs across ranks"
    if not statuses[0]["enabled"]:
        return None
    count = len(statuses)
    for rank, status in enumerate(statuses):
        members, local_rank, size = status["members"], status["sp_rank"], status["sp_size"]
        if (
            len(members) != size
            or not 0 <= local_rank < size
            or len(set(members)) != size
            or any(not 0 <= member < count for member in members)
            or members[local_rank] != rank
            or any(statuses[member]["members"] != members for member in members)
        ):
            return "SP group membership differs across ranks"
    return None


def _validate_channel_sp_group(statuses):
    if not statuses:
        return "empty SP group"
    signatures = [s["source_digest"] for s in statuses]
    source_counts = [s["source_count"] for s in statuses]
    attempts = [s["attempts"] for s in statuses]
    data_indices = [s["data_observation_index"] for s in statuses]
    if not any(source_counts) and all(n == 0 for n in attempts) and not any(s["errors"] for s in statuses):
        return None
    if (
        any(s["errors"] for s in statuses)
        or any(not count for count in source_counts)
        or any(count != source_counts[0] for count in source_counts)
        or any(sig != signatures[0] for sig in signatures)
        or len(set(attempts)) != 1
        or attempts[0] < 1
        or any(index != data_indices[0] for index in data_indices)
        or any(index is not None and not 0 <= index < attempts[0] for index in data_indices)
        or any(len(s["observations"]) != attempts[0] for s in statuses)
    ):
        return "SP source metadata or observation differs across ranks"
    for index in range(attempts[0]):
        observations = [s["observations"][index] for s in statuses]
        if (
            len({o["mode"] for o in observations}) != 1
            or len({o["data"] for o in observations}) != 1
            or observations[0]["mode"] not in ("packed_cu", "positions", "source")
            or len({o.get("layout_digest") for o in observations}) != 1
        ):
            return "SP observation layouts differ across ranks"
        for rank, observation in enumerate(observations):
            if observation["source_digest"] != signatures[rank]:
                return "SP observation source metadata changed during capture"
            if observation["mode"] in ("packed_cu", "source") and observation["segment_count"] != source_counts[rank]:
                return "rank-local packed spans do not match source count"
            if not observation["valid_rows"]:
                return "SP observation segment metadata is invalid"
        geometry = 3
        for observation in observations:
            geometry &= observation.get("geometry_candidates", 3)
        if not geometry:
            return "SP ranks do not share a valid packed shard geometry"
    return None


def _materialize_channel_sp_segments(payloads, observation_index, *, device):
    observations = [payload[observation_index] for payload in payloads]
    records = [segment for observation in observations for segment in observation["segments"]]
    if not records:
        return []
    losses = torch.tensor([row[4] for row in records], dtype=torch.float32, device=device)
    integers = torch.tensor([row[5] for row in records], dtype=torch.int64, device=device)
    include_data = observations[0]["data"]
    result = []
    for index, (batch_idx, sp_rank, local_order, is_start, _, _) in enumerate(records):
        segment = {
            "batch_idx": batch_idx,
            "sp_rank": sp_rank,
            "local_order": local_order,
            "is_start": is_start,
            "loss_sum": losses[index],
            "token_count": integers[index, 0],
            "has_explicit_padding_mask": integers[index, 1].bool(),
            "has_only_zero_positions": integers[index, 2].bool(),
        }
        if include_data:
            segment["sample_count"] = integers[index, 3]
            segment["input_token_count"] = integers[index, 4]
        result.append(segment)
    return result


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

        self._source_ids: list[ChannelKey] = []
        self._position_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._packed_cu_seqlens: Any | None = None
        self._packed_global_cu_seqlens: Any | None = None
        self._packed_global_physical_seq_len: int | None = None
        self._tail_padding_length: Any | None = None
        self._source_provenance: dict[str, Any] | None = None
        self._require_source_provenance = True
        self._dpo_attention_cu: Any | None = None
        self._result: list[dict[str, Any]] | None = None
        self._primary_labels: torch.Tensor | None = None
        self._require_primary_labels = False
        self._mtp_labels: torch.Tensor | None = None
        self._dpo_segments = False
        self._source_repeat = 1
        self._data_result: list[dict[str, Any]] | None = None
        self._pending_data_observation_index: int | None = None
        self._pending_observations: list[dict[str, Any] | list[dict[str, Any]]] = []
        self._observation_attempt_count = 0
        self._capture_errors: list[tuple[str, str]] = []
        self._capture_topology: dict[str, Any] | None = None
        self._capture_group: Any | None = None
        self._observer_devices: list[torch.device] = []
        self._per_mb_source_ids: list[list[ChannelKey]] = []
        self._per_mb_position_ids: list[torch.Tensor | None] = []
        self._per_mb_attention_masks: list[torch.Tensor | None] = []
        self._per_mb_packed_cu_seqlens: list[Any | None] = []
        self._per_mb_packed_global_cu_seqlens: list[Any | None] = []
        self._per_mb_packed_global_physical_seq_lens: list[int | None] = []
        self._per_mb_tail_padding_lengths: list[Any | None] = []
        self._per_mb_source_provenances: list[dict[str, Any] | None] = []
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
        cur: Any = model
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))
            if not isinstance(cur, torch.nn.Module):
                nxt = getattr(cur, "unwrapped_module", None)
                if nxt is None:
                    nxt = getattr(cur, "model", None)
                if isinstance(nxt, torch.nn.Module) and nxt is not cur:
                    cur = nxt
                    continue
                return None
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

        # ``trainer.model`` is a VeOmniModelRuntime handle: reads forward via
        # ``__getattr__``, writes do not. Peel it (and DDP) before patching.
        if not isinstance(model, torch.nn.Module):
            unwrapped = getattr(model, "unwrapped_module", None)
            if not isinstance(unwrapped, torch.nn.Module):
                unwrapped = getattr(model, "model", None)
            if isinstance(unwrapped, torch.nn.Module):
                model = unwrapped

        host = self._resolve_loss_fn_host(model)
        if host is None or not hasattr(host, "loss_function"):
            logger.warning_rank0(
                "Channel loss: could not locate model.loss_function "
                f"(outer_type={type(model).__name__}, inner_type={type(host).__name__ if host else None})."
            )
        else:
            self._original_loss_fn = host.loss_function
            wrapper = self._wrapped_loss_fn
            host.loss_function = wrapper
            if host.loss_function is not wrapper:
                raise RuntimeError(
                    "Channel loss wrap did not stick on "
                    f"{type(host).__name__} (outer={type(model).__name__}); "
                    "the runtime handle forwards reads but not writes — pass "
                    "the nn.Module."
                )
            self._model_ref = host
            logger.info_rank0(
                f"Channel loss: wrapped {type(host).__name__}.loss_function (outer={type(model).__name__})."
            )

        self._installed = self._original_loss_fn is not None
        if not self._installed:
            logger.warning_rank0("Channel loss: no causal loss hook was installed.")

    def uninstall(self) -> None:
        if self._model_ref is not None and self._original_loss_fn is not None:
            self._model_ref.loss_function = self._original_loss_fn

        self._original_loss_fn = None
        self._model_ref = None
        self._installed = False

    @property
    def capture_active(self) -> bool:
        return _ACTIVE_CHANNEL_LOSS_COMPUTER.get() is self

    @contextmanager
    def capture(self) -> Iterator[None]:
        nested = self.capture_active
        if not nested:
            self._reset_capture_state()
            self._capture_topology = self._snapshot_topology()
        pending_count = len(self._pending_observations)
        error_count = len(self._capture_errors)
        attempts = self._observation_attempt_count
        data_index = self._pending_data_observation_index
        devices_count = len(self._observer_devices)
        previous_result, previous_data = self._result, self._data_result
        token = _ACTIVE_CHANNEL_LOSS_COMPUTER.set(self)
        completed = False
        try:
            yield
            completed = True
        finally:
            _ACTIVE_CHANNEL_LOSS_COMPUTER.reset(token)
            if not completed:
                del self._pending_observations[pending_count:]
                del self._capture_errors[error_count:]
                del self._observer_devices[devices_count:]
                self._observation_attempt_count = attempts
                self._pending_data_observation_index = data_index
                self._result, self._data_result = previous_result, previous_data
            if not nested:
                if completed:
                    self._finalize_capture()
                else:
                    self._reset_capture_state()

    def _reset_capture_state(self) -> None:
        self._pending_observations = []
        self._pending_data_observation_index = None
        self._observation_attempt_count = 0
        self._capture_errors = []
        self._capture_topology = None
        self._capture_group = None
        self._observer_devices = []

    def _snapshot_topology(self) -> dict[str, Any]:
        initialized = dist.is_available() and dist.is_initialized()
        try:
            rank = dist.get_rank() if initialized else 0
            topology = _channel_sp_topology(self.parallel_state, initialized=initialized, global_rank=rank)
            if topology["enabled"] and not topology["topology_error"]:
                self._capture_group = self.parallel_state.sp_group
            return topology
        except Exception as error:
            return {
                "enabled": False,
                "sp_rank": 0,
                "sp_size": 1,
                "members": (),
                "topology_error": f"{type(error).__name__}: {error}",
            }

    def _finalize_capture(self) -> None:
        """Coordinate detached observations and surface strict failures before backward."""
        initialized = dist.is_available() and dist.is_initialized()
        collective_failed = False
        try:
            try:
                world = dist.get_world_size() if initialized else 1
            except Exception as error:
                raise ChannelLossCollectiveError(
                    "Channel loss could not inspect the default process group."
                ) from error
            status = dict(self._capture_topology or self._snapshot_topology())
            status.update(
                source_count=0,
                source_digest="",
                attempts=self._observation_attempt_count,
                data_observation_index=self._pending_data_observation_index,
                strict=self.strict,
                errors=[(kind, message[:512]) for kind, message in self._capture_errors],
                observations=[],
            )
            payload = b""
            try:
                status["source_count"] = len(self._source_ids)
                status["source_digest"] = _source_signature_digest(self._source_ids)
                if self._source_ids and self._observation_attempt_count == 0:
                    status["errors"].append(("metadata", "source metadata was present but no loss hook was observed"))
                if len(self._pending_observations) != self._observation_attempt_count:
                    status["errors"].append(("metadata", "successful observations do not match loss-hook calls"))
                if status["enabled"]:
                    local_observations = [
                        _serialize_channel_sp_observation(item) for item in self._pending_observations
                    ]
                    for observation in local_observations:
                        rows = observation["segments"]
                        status["observations"].append(
                            {
                                "mode": observation["mode"],
                                "data": observation["data"],
                                "source_digest": observation["source_digest"],
                                "layout_digest": observation.get("layout_digest"),
                                "geometry_candidates": observation.get("geometry_candidates", 3),
                                "segment_count": len(rows),
                                "valid_rows": all(
                                    row[0] >= 0 and row[1] == status["sp_rank"] and row[2] >= 0 for row in rows
                                ),
                            }
                        )
                    # Prepare the exact CPU bytes before the status exchange.
                    payload = pickle.dumps(local_observations, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as error:
                failure = ("generic", f"observation preparation failed: {type(error).__name__}: {str(error)[:512]}")
                status["errors"].append(failure)
                self._capture_errors.append(failure)
            if initialized:
                statuses = [None] * world
                try:
                    dist.all_gather_object(statuses, status)
                except Exception as error:
                    raise ChannelLossCollectiveError(
                        "Channel loss world status exchange failed; the process group is unsafe."
                    ) from error
            else:
                statuses = [status]
            topology_error = _validate_channel_sp_topologies(statuses)
            strict_values = [entry["strict"] for entry in statuses]
            strict_enabled = any(strict_values)
            world_error = topology_error
            if len(set(strict_values)) != 1:
                world_error = "channel-loss strict mode differs across ranks"
            if strict_enabled and world_error is None:
                for rank, entry in enumerate(statuses):
                    if entry["errors"] or not entry["source_count"]:
                        world_error = f"rank {rank}: {entry['errors'] or 'source metadata is missing'}"
                        break
                if world_error is None and status["enabled"]:
                    for members in {entry["members"] for entry in statuses}:
                        group_error = _validate_channel_sp_group([statuses[member] for member in members])
                        if group_error:
                            world_error = f"SP group {members}: {group_error}"
                            break
            if strict_enabled and world_error:
                self._record_capture_failure(world_error)
                raise ChannelLossMetadataError(world_error)

            local_error = None
            try:
                if topology_error:
                    local_error = topology_error
                elif not status["enabled"]:
                    prior_errors = len(self.strict_observation_errors)
                    self._finalize_local_capture()
                    if self.strict and len(self.strict_observation_errors) > prior_errors:
                        local_error = self.strict_observation_errors[-1][1]
                else:
                    group_statuses = [statuses[member] for member in status["members"]]
                    local_error = _validate_channel_sp_group(group_statuses)
                    if not local_error and self._pending_observations:
                        if status["sp_size"] > 1:
                            group_payloads = [None] * status["sp_size"]
                            try:
                                dist.all_gather_object(group_payloads, payload, group=self._capture_group)
                            except Exception as error:
                                raise ChannelLossCollectiveError(
                                    "Channel loss SP observation exchange failed; the process group is unsafe."
                                ) from error
                        else:
                            group_payloads = [payload]
                        decoded = [pickle.loads(item) for item in group_payloads]
                        reduced = []
                        for index, observation in enumerate(self._pending_observations):
                            segments = _materialize_channel_sp_segments(decoded, index, device=observation["device"])
                            aligned = self._merge_sp_segments(
                                segments, self._source_ids, strict=True, include_data_stats=observation["data"]
                            )
                            if not aligned:
                                raise ChannelLossMetadataError("SP reduction produced no source-aligned CE payload")
                            reduced.append(aligned)
                        self._append_observations(reduced)
            except ChannelLossCollectiveError:
                raise
            except Exception as error:
                local_error = (
                    f"channel-loss forward finalization failed with {type(error).__name__}: {str(error)[:512]}"
                )

            if strict_enabled:
                if initialized:
                    failures = [None] * world
                    try:
                        dist.all_gather_object(failures, local_error)
                    except Exception as error:
                        raise ChannelLossCollectiveError(
                            "Channel loss strict error exchange failed; the process group is unsafe."
                        ) from error
                else:
                    failures = [local_error]
                if any(failures):
                    message = f"channel-loss observation failed before backward: {failures}"
                    self._record_capture_failure(message)
                    raise ChannelLossMetadataError(message)
            elif local_error:
                self._record_capture_failure(local_error)
        except ChannelLossCollectiveError:
            collective_failed = True
            raise
        finally:
            if self.release_cache and not collective_failed:
                for device in dict.fromkeys(self._observer_devices):
                    try:
                        _release_observer_cache(device)
                    except Exception:
                        logger.warning_rank0(
                            "Channel loss: failed to release detached CE allocator cache; continuing training.",
                            exc_info=True,
                        )
            self._reset_capture_state()

    def _finalize_local_capture(self) -> None:
        errors = list(self._capture_errors)
        if self._source_ids and self._observation_attempt_count == 0:
            errors.append(("metadata", "source metadata was present but no causal-LM loss hook was observed"))
        if len(self._pending_observations) != self._observation_attempt_count:
            errors.append(
                ("metadata", "successful observation count does not match the number of observed causal-LM loss calls")
            )
        if any(not isinstance(observation, list) for observation in self._pending_observations):
            errors.append(("generic", "an SP payload was prepared while sequence parallelism was disabled"))
        if errors:
            self._record_capture_failure(_format_capture_errors(errors))
            return
        self._append_observations(self._pending_observations)

    def _append_observations(self, observations: list[list[dict[str, Any]]]) -> None:
        if not observations or (self.strict and self._micro_step_failed):
            return
        if self._result is None:
            self._result = []
        for observation in observations:
            if self._dpo_segments and self._source_provenance is None:
                for index, item in enumerate(observation):
                    if "sample_count" in item:
                        item["sample_count"] = item["sample_count"] * int(index % self._source_repeat == 0)
            self._result.extend(observation)
        # Losses may include auxiliary objectives. Physical samples/tokens and
        # supervised-token counts describe the primary batch exactly once.
        if self._data_result is None and self._pending_data_observation_index is not None:
            self._data_result = observations[self._pending_data_observation_index]
        self._micro_step_observation_count += len(observations)

    def _record_capture_failure(self, message: str) -> None:
        if self.strict:
            # A strict micro-step is atomic evidence: one failed observation
            # invalidates successful observations from the same micro-step.
            self._micro_step_failed = True
            self._result = None
            self._data_result = None
            self._micro_step_observation_count = 0
            self.strict_observation_errors.append((self._micro_step, message))
            logger.warning_rank0(f"Channel loss: strict observation failure: {message}")
            return
        logger.warning_rank0(f"Channel loss: {message} Skipping this model forward.")

    def begin_step(
        self,
        per_mb_source_ids: list[list[ChannelKey]],
        per_mb_position_ids: list[torch.Tensor | None],
        per_mb_attention_masks: list[torch.Tensor | None],
        per_mb_packed_cu_seqlens: list[Any | None] | None = None,
        per_mb_packed_global_cu_seqlens: list[Any | None] | None = None,
        per_mb_packed_global_physical_seq_lens: list[int | None] | None = None,
        per_mb_tail_padding_lengths: list[Any | None] | None = None,
        per_mb_source_provenances: list[dict[str, Any] | None] | None = None,
    ) -> None:
        self._per_mb_source_ids = per_mb_source_ids
        self._per_mb_position_ids = per_mb_position_ids
        self._per_mb_attention_masks = per_mb_attention_masks
        self._per_mb_packed_cu_seqlens = (
            per_mb_packed_cu_seqlens if per_mb_packed_cu_seqlens is not None else [None] * len(per_mb_source_ids)
        )
        self._per_mb_packed_global_cu_seqlens = (
            per_mb_packed_global_cu_seqlens
            if per_mb_packed_global_cu_seqlens is not None
            else [None] * len(per_mb_source_ids)
        )
        self._per_mb_packed_global_physical_seq_lens = (
            per_mb_packed_global_physical_seq_lens
            if per_mb_packed_global_physical_seq_lens is not None
            else [None] * len(per_mb_source_ids)
        )
        self._per_mb_tail_padding_lengths = (
            per_mb_tail_padding_lengths if per_mb_tail_padding_lengths is not None else [None] * len(per_mb_source_ids)
        )
        self._per_mb_source_provenances = (
            per_mb_source_provenances if per_mb_source_provenances is not None else [None] * len(per_mb_source_ids)
        )
        metadata_lengths = {
            "position_ids": len(self._per_mb_position_ids),
            "attention_masks": len(self._per_mb_attention_masks),
            "packed_cu_seqlens": len(self._per_mb_packed_cu_seqlens),
            "packed_global_cu_seqlens": len(self._per_mb_packed_global_cu_seqlens),
            "packed_global_physical_seq_lens": len(self._per_mb_packed_global_physical_seq_lens),
            "tail_padding_lengths": len(self._per_mb_tail_padding_lengths),
            "source_provenances": len(self._per_mb_source_provenances),
        }
        mismatched = {name: length for name, length in metadata_lengths.items() if length != len(per_mb_source_ids)}
        if mismatched:
            raise ChannelLossMetadataError(
                "Channel loss: per-microbatch metadata list lengths do not match source metadata "
                f"(source_microbatches={len(per_mb_source_ids)}, mismatched={mismatched})."
            )
        self._micro_step = 0
        self._micro_step_observation_count = 0
        self._micro_step_failed = False
        self.missing_observation_micro_steps = []
        self.strict_observation_errors = []
        self.step_totals = {}
        self.step_data_totals = {}

    def before_micro_step(
        self, primary_labels: torch.Tensor | None = None, mtp_labels: torch.Tensor | None = None
    ) -> None:
        step = self._micro_step
        if step < len(self._per_mb_source_ids):
            self._source_ids = self._per_mb_source_ids[step]
            self._position_ids = self._per_mb_position_ids[step]
            self._attention_mask = self._per_mb_attention_masks[step]
            self._packed_cu_seqlens = self._per_mb_packed_cu_seqlens[step]
            self._packed_global_cu_seqlens = self._per_mb_packed_global_cu_seqlens[step]
            self._packed_global_physical_seq_len = self._per_mb_packed_global_physical_seq_lens[step]
            self._tail_padding_length = self._per_mb_tail_padding_lengths[step]
            self._source_provenance = self._per_mb_source_provenances[step]
        else:
            self._source_ids = []
            self._position_ids = None
            self._attention_mask = None
            self._packed_cu_seqlens = None
            self._packed_global_cu_seqlens = None
            self._packed_global_physical_seq_len = None
            self._tail_padding_length = None
            self._source_provenance = None
        self._result = None
        self._primary_labels = primary_labels
        self._require_primary_labels = True
        self._mtp_labels = mtp_labels
        self._data_result = None
        self._micro_step_observation_count = 0
        self._micro_step_failed = False
        self._reset_capture_state()

    def after_micro_step(self) -> None:
        if self._result and self._data_result is None:
            self._record_capture_failure("no observed causal-LM loss matched the primary micro-batch labels")
        if self.strict and self._source_ids and self._micro_step_observation_count == 0:
            self.missing_observation_micro_steps.append(self._micro_step)
        if self._result:
            for item in self._result:
                source_id = item["source_id"]
                loss_sum = item["loss_sum"].detach()
                token_count = item["token_count"].detach()
                previous = self.step_totals.get(source_id)
                if previous is None:
                    self.step_totals[source_id] = (loss_sum, token_count)
                else:
                    self.step_totals[source_id] = (previous[0] + loss_sum, previous[1] + token_count)

        if self._data_result:
            for item in self._data_result:
                source_id = item["source_id"]
                token_count = item["token_count"].detach()
                sample_count = item.get("sample_count", token_count.new_ones(()).to(dtype=torch.int64)).detach()
                input_token_count = item.get("input_token_count", token_count).detach()
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
        self._packed_cu_seqlens = None
        self._packed_global_cu_seqlens = None
        self._packed_global_physical_seq_len = None
        self._tail_padding_length = None
        self._result = None
        self._primary_labels = None
        self._require_primary_labels = False
        self._mtp_labels = None
        self._source_provenance = None
        self._data_result = None
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
        if not self._source_ids or self._dpo_segments or not self.capture_active:
            return self._original_loss_fn(*args, **kwargs)
        from ...models.loss_utils.cross_entropy_loss import ForCausalLMLoss
        from ...ops import VeomniOp
        from ...ops.kernels.loss.cross_entropy_loss.standard import chunk_loss, eager, liger_kernel

        original_loss, effective_args, effective_kwargs = _unpack_partial_loss(self._original_loss_fn, args, kwargs)
        # Custom loss functions may opt in only when their token CE and token
        # order are exactly the causal CE defined by the outer loss arguments.
        try:
            custom_contract = any(
                getattr(fn, "__veomni_channel_loss_contract__", None) == "outer_causal_ce_v1"
                for fn in (self._original_loss_fn, original_loss)
            )
            if custom_contract:
                custom_signature = inspect.signature(original_loss)
                custom_bound = custom_signature.bind_partial(*effective_args, **effective_kwargs)
                canonical = {
                    "logits",
                    "labels",
                    "vocab_size",
                    "num_items_in_batch",
                    "ignore_index",
                    "shift_labels",
                    "hidden_states",
                    "weights",
                    "logits_bias",
                    "logits_bias_weight",
                    "loss_reduction_group",
                    "op",
                }
                for name, parameter in custom_signature.parameters.items():
                    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                        custom_contract &= not bool(custom_bound.arguments.get(name))
                    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                        custom_contract &= not bool(set(custom_bound.arguments.get(name, {})) - canonical)
                    elif name not in canonical:
                        custom_contract = False
        except Exception:  # Observer inspection must not affect the training loss.
            custom_contract = False
        proven_loss = custom_contract
        if original_loss is ForCausalLMLoss:
            try:
                bound = inspect.signature(original_loss).bind_partial(*effective_args, **effective_kwargs)
                bound.apply_defaults()
                op = bound.arguments.get("op")
                if op is None and not bound.arguments.get("kwargs", {}).get("return_log_probs", False):
                    impl = (
                        "chunk_loss" if bound.arguments.get("kwargs", {}).get("logits_bias") is not None else "eager"
                    )
                    op = VeomniOp("cross_entropy_loss", "standard", impl)
                builtin = {"eager": eager, "chunk_loss": chunk_loss, "liger_kernel": liger_kernel}.get(
                    getattr(op, "impl", None)
                )
                proven_loss = (
                    type(op) is VeomniOp
                    and op.op == "cross_entropy_loss"
                    and op.variant == "standard"
                    and builtin is not None
                    and op.entry.forward is builtin.forward
                    and op.entry.backward is builtin.backward
                    and op.entry._generated_forward is builtin.forward
                    and op.entry._generated_backward is builtin.backward
                    and op.entry.wrapper is op.entry._generated_wrapper
                    and not bound.arguments.get("kwargs", {}).get("return_log_probs", False)
                )
            except Exception:  # Observer lookup must never replace the training loss failure.
                proven_loss = False
        reused = []
        errors = []
        with capture_chunk_loss_per_token(reused.append, on_error=errors.append):
            main_loss = self._original_loss_fn(*args, **kwargs)
        if custom_contract and original_loss is not ForCausalLMLoss:
            # A custom loss may invoke unrelated chunk ops, including compiled
            # ones that are invisible to capture. Use its declared outer-CE
            # contract instead of assuming an observed op is the objective.
            reused.clear()
            errors.clear()
        if self._require_source_provenance and not proven_loss:
            self._observation_attempt_count += 1
            self._capture_errors.append(("metadata", "custom causal loss cannot prove source-aligned token losses"))
            return main_loss
        if errors:
            ambiguous = any(
                error
                in (
                    "multiple chunk-loss observations in one causal-loss call",
                    "chunk-loss observation failed without a verifiable single op",
                )
                for error in errors
            )
            if ambiguous or not proven_loss or not self._require_source_provenance:
                self._observation_attempt_count += 1
                self._capture_errors.extend(("generic", error) for error in errors)
                return main_loss
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            logger.warning(
                f"Channel loss: rank {rank} chunk observation failed; recomputing detached metrics: {errors[0][:256]}"
            )
            reused.clear()
        if len(reused) > 1:
            self._observation_attempt_count += 1
            self._capture_errors.append(("generic", "multiple chunk-loss observations in one causal loss call"))
            return main_loss

        try:
            call_args = _bind_loss_call_args(self._original_loss_fn, args, kwargs)
        except Exception as exc:  # Coordinate observer errors, preserving the already-computed training loss.
            self._observation_attempt_count += 1
            self._capture_errors.append(("generic", f"cannot inspect loss arguments: {exc}"))
            return main_loss
        logits = call_args.get("logits")
        labels = call_args.get("labels")
        if labels is None:
            labels = call_args.get("shift_labels")
        vocab_size = call_args.get("vocab_size")
        ignore_index = call_args.get("ignore_index", IGNORE_INDEX)
        if ignore_index is None:
            ignore_index = IGNORE_INDEX
        if self.capture_active and self._source_ids and labels is not None:
            observer_kwargs = dict(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=call_args.get("hidden_states"),
                weights=call_args.get("weights"),
                shift_labels=call_args.get("shift_labels"),
                ignore_index=ignore_index,
            )
            extras = {
                "reused_per_token_loss": reused[0] if reused else None,
                "logits_bias": call_args.get("logits_bias"),
                "logits_bias_weight": call_args.get("logits_bias_weight"),
            }
            accepted = _supported_observer_kwargs(self.compute_side_channel, extras)
            if any(extras[key] is not None and key not in accepted for key in ("logits_bias", "logits_bias_weight")):
                self._observation_attempt_count += 1
                self._capture_errors.append(("metadata", "custom channel observer cannot receive logits bias"))
                return main_loss
            observer_kwargs.update(accepted)
            try:
                self.compute_side_channel(**observer_kwargs)
            except Exception as error:
                self._observation_attempt_count += 1
                self._capture_errors.append(
                    ("generic", f"custom channel observer failed: {type(error).__name__}: {error}")
                )

        return main_loss

    def observe_dpo_log_probs(self, log_probs: torch.Tensor, labels: torch.Tensor, attention_cu: Any) -> None:
        """Observe expanded policy log-probs, after any model-side row compaction."""
        if not self._dpo_segments or not self._source_ids or not self.capture_active:
            return
        try:
            extras = {"reused_per_token_loss": -log_probs.detach()}
            observer_extras = _supported_observer_kwargs(self.compute_side_channel, extras)
            compute_extras = _supported_observer_kwargs(self._compute_channel_loss, extras)
            if "reused_per_token_loss" not in observer_extras or "reused_per_token_loss" not in compute_extras:
                self._observation_attempt_count += 1
                self._capture_errors.append(
                    ("metadata", "custom DPO observers must accept expanded per-token log-probs")
                )
                return
            self._dpo_attention_cu = attention_cu
            self.compute_side_channel(
                logits=None,
                labels=labels,
                vocab_size=None,
                hidden_states=None,
                weights=None,
                **observer_extras,
            )
        except Exception as error:
            self._observation_attempt_count += 1
            self._capture_errors.append(
                ("generic", f"DPO log-prob observation failed: {type(error).__name__}: {error}")
            )

    def compute_side_channel(
        self,
        logits: torch.Tensor | None,
        labels: torch.Tensor,
        vocab_size: int | None,
        hidden_states: torch.Tensor | None,
        weights: torch.Tensor | None,
        shift_labels: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
        reused_per_token_loss: torch.Tensor | None = None,
        logits_bias: torch.Tensor | None = None,
        logits_bias_weight: torch.Tensor | None = None,
    ) -> None:
        standalone = not self.capture_active
        if standalone:
            self._reset_capture_state()
            self._capture_topology = self._snapshot_topology()
        self._observation_attempt_count += 1
        if self.release_cache:
            device = next(
                (
                    tensor.device
                    for tensor in (reused_per_token_loss, logits, hidden_states, weights, labels)
                    if isinstance(tensor, torch.Tensor)
                ),
                None,
            )
            if device is not None:
                self._observer_devices.append(device)
        try:
            compute_kwargs = dict(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                hidden_states=hidden_states,
                weights=weights,
                attention_mask=self._attention_mask,
                shift_labels=shift_labels,
                ignore_index=ignore_index,
            )
            extras = {
                "defer_sp": True,
                "reused_per_token_loss": reused_per_token_loss,
                "logits_bias": logits_bias,
                "logits_bias_weight": logits_bias_weight,
            }
            accepted = _supported_observer_kwargs(self._compute_channel_loss, extras)
            if self.parallel_state.sp_enabled and "defer_sp" not in accepted:
                raise ChannelLossMetadataError("Custom channel loss computation cannot defer SP reduction.")
            if any(extras[key] is not None and key not in accepted for key in ("logits_bias", "logits_bias_weight")):
                raise ChannelLossMetadataError("Custom channel loss computation cannot receive logits bias.")
            compute_kwargs.update(accepted)
            result = self._compute_channel_loss(**compute_kwargs)
            if result:
                is_mtp = isinstance(result, dict) and "depth_observations" in result
                observations = result["depth_observations"] if is_mtp else [result]
                if not observations or any(not item for item in observations):
                    raise ChannelLossMetadataError("an MTP depth produced no source-aligned CE payload")
                if not is_mtp and self._data_result is None and self._pending_data_observation_index is None:
                    primary = self._primary_labels
                    # Identity is the usual trainer path. Equal copied labels
                    # also support models that make a contiguous/view copy.
                    if (
                        primary is None
                        or labels is primary
                        or (
                            labels.shape == primary.shape
                            and labels.device == primary.device
                            and torch.equal(labels, primary)
                        )
                    ):
                        self._pending_data_observation_index = len(self._pending_observations)
                self._pending_observations.extend(observations)
                self._observation_attempt_count += len(observations) - 1
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

    def _matched_mtp_targets(
        self, labels: torch.Tensor, shift_labels: torch.Tensor | None, ignore_index: int
    ) -> torch.Tensor | None:
        """Match the real MTP targets, including the all-ignored safe-label case."""
        primary = self._primary_labels
        if self._require_primary_labels and not isinstance(primary, torch.Tensor):
            raise ChannelLossMetadataError("Channel loss: primary micro-batch labels are missing")
        same_primary = (
            isinstance(primary, torch.Tensor)
            and labels.shape == primary.shape
            and labels.device == primary.device
            and (labels is primary or torch.equal(labels, primary))
        )
        if labels is primary or (same_primary and shift_labels is None):
            return None

        targets = self._mtp_labels
        candidate = shift_labels if shift_labels is not None else labels
        if isinstance(targets, torch.Tensor) and targets.ndim == 3 and candidate.ndim == 2:
            batch, depth, width = targets.shape
            if candidate.shape == (batch * depth, width) and candidate.device == targets.device:
                flat = targets.reshape_as(candidate)
                matched = torch.equal(candidate, flat)
                if flat.numel():
                    safe = flat.clone()
                    has_valid = flat.ne(ignore_index).any()
                    safe.reshape(-1)[0] = torch.where(has_valid, safe.reshape(-1)[0], safe.new_zeros(()))
                    matched = matched or torch.equal(candidate, safe)
                if matched:
                    if same_primary:
                        raise ChannelLossMetadataError("Channel loss: primary and MTP targets are ambiguous")
                    return targets
        if same_primary:
            return None
        if primary is not None:
            raise ChannelLossMetadataError(
                "Channel loss: auxiliary labels cannot be aligned with primary or recorded MTP targets"
            )
        return None

    def _dpo_tail_mask(self, target: torch.Tensor, topology: dict[str, Any]) -> torch.Tensor:
        """Match DPO's per-segment causal mask without changing training labels."""
        positions = self._position_ids
        if not isinstance(positions, torch.Tensor) or target.ndim != 2:
            raise ChannelLossMetadataError("DPO channel loss requires two-dimensional labels and position IDs")
        if positions.dim() == 3:
            positions = positions[:, 0, :]
        if positions.shape != target.shape:
            raise ChannelLossMetadataError("DPO channel loss position IDs do not match the local labels")
        batch, width = target.shape
        if not topology["enabled"] or topology["sp_size"] == 1:
            return F.pad(positions[:, 1:].eq(0), (0, 1), value=True).to(target.device)

        if int(getattr(self.parallel_state, "cp_size", 1)) > 1:
            raise ChannelLossMetadataError("DPO channel loss cannot prove segment tails for context parallelism")

        local_points, local_error = _packed_cu_points(self._packed_cu_seqlens)
        global_points, global_error = _packed_cu_points(self._packed_global_cu_seqlens)
        expected_width = batch * width * topology["sp_size"]
        if (
            local_error
            or global_error
            or local_points is None
            or global_points is None
            or local_points != global_points
            or global_points[-1] != expected_width
        ):
            raise ChannelLossMetadataError("DPO channel loss cannot prove segment tails for the current SP/CP layout")
        mask = torch.zeros((batch, width), dtype=torch.bool)
        global_row_width = width * topology["sp_size"]
        for endpoint in global_points[1:]:
            global_tail = endpoint - 1
            row, offset = divmod(global_tail, global_row_width)
            offset -= topology["sp_rank"] * width
            if 0 <= row < batch and 0 <= offset < width:
                mask[row, offset] = True
        return mask.to(target.device)

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
        reused_per_token_loss: torch.Tensor | None = None,
        logits_bias: torch.Tensor | None = None,
        logits_bias_weight: torch.Tensor | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        topology = self._capture_topology or self._snapshot_topology()
        sp_enabled = topology["enabled"]
        mtp_targets = self._matched_mtp_targets(labels, shift_labels, ignore_index)
        if mtp_targets is not None:
            effective_labels = mtp_targets.reshape(-1, mtp_targets.size(-1))
        elif shift_labels is not None:
            effective_labels = shift_labels
        elif not sp_enabled:
            effective_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        else:
            effective_labels = labels

        provenance = self._source_provenance
        if provenance is not None or self._require_source_provenance:
            shard_labels = mtp_targets.unbind(dim=1) if mtp_targets is not None else (effective_labels,)
            for target in shard_labels:
                provenance = _validate_source_provenance(
                    provenance, self._source_ids, self.parallel_state, target, ignore_index
                )
        if provenance is not None and sp_enabled and self._packed_global_cu_seqlens is not None:
            global_cu, error = _packed_cu_points(self._packed_global_cu_seqlens)
            if error or global_cu is None:
                raise ChannelLossMetadataError("Channel loss global attention boundaries are invalid.")
            provenance = {**provenance, "global_cu": tuple(global_cu)}
        if self._dpo_segments:
            if mtp_targets is not None:
                raise ChannelLossMetadataError("DPO channel loss cannot combine MTP with segment-tail masking")
            if provenance is not None:
                tail_mask, dpo_cu = _dpo_source_tail_mask(provenance, self._dpo_attention_cu, effective_labels)
                provenance = {**provenance, "dpo_cu": dpo_cu}
            else:
                tail_mask = self._dpo_tail_mask(effective_labels, topology)
            effective_labels = effective_labels.masked_fill(tail_mask, ignore_index)

        if reused_per_token_loss is not None:
            reused_per_token_loss = self._align_reused_per_token_loss(
                reused_per_token_loss,
                effective_labels,
                sp_enabled=sp_enabled,
            )

        labels_flat = effective_labels.reshape(-1)
        attention_mask_flat = _position_aligned_attention_mask_flat(
            attention_mask=attention_mask,
            labels=labels,
            effective_labels=effective_labels,
        )
        per_token_loss = (
            reused_per_token_loss.reshape(-1)
            if reused_per_token_loss is not None
            else self._per_token_ce(
                logits, labels_flat, vocab_size, hidden_states, weights, ignore_index, logits_bias, logits_bias_weight
            )
        )
        if per_token_loss is None:
            return []
        per_token_loss = per_token_loss.float()
        labels_flat = labels_flat.to(per_token_loss.device)
        if attention_mask_flat is not None:
            attention_mask_flat = attention_mask_flat.to(per_token_loss.device)

        def aggregate(loss, target, mask):
            if provenance is not None:
                return _aggregate_provenance(
                    loss.reshape(-1),
                    target.reshape(-1),
                    self._source_ids,
                    provenance,
                    ignore_index,
                    strict=self.strict,
                    defer_sp=defer_sp,
                )
            return self._aggregate_by_source(
                per_token_loss=loss.reshape(-1),
                labels_flat=target.reshape(-1),
                attention_mask_flat=mask,
                source_ids=self._source_ids,
                position_ids=self._position_ids,
                packed_cu_seqlens=self._packed_cu_seqlens,
                packed_global_cu_seqlens=self._packed_global_cu_seqlens,
                packed_global_physical_seq_len=self._packed_global_physical_seq_len,
                tail_padding_length=self._tail_padding_length,
                ignore_index=ignore_index,
                sp_enabled=sp_enabled,
                parallel_state=self.parallel_state if sp_enabled else None,
                sp_rank_override=topology["sp_rank"] if sp_enabled else None,
                sp_size_override=topology["sp_size"] if sp_enabled else None,
                strict=self.strict,
                include_data_stats=True,
                defer_sp=defer_sp,
            )

        if mtp_targets is not None:
            batch, depth, width = mtp_targets.shape
            losses = per_token_loss.reshape(batch, depth, width)
            targets = mtp_targets.to(per_token_loss.device)
            observations = []
            for index in range(depth):
                target = targets[:, index, :]
                mask = _position_aligned_attention_mask_flat(
                    attention_mask=attention_mask, labels=target, effective_labels=target
                )
                if mask is not None:
                    mask = mask.to(per_token_loss.device)
                observations.append(aggregate(losses[:, index, :], target, mask))
            return {"depth_observations": observations}
        return aggregate(per_token_loss, labels_flat, attention_mask_flat)

    @staticmethod
    def _align_reused_per_token_loss(
        per_token_loss: torch.Tensor,
        effective_labels: torch.Tensor,
        *,
        sp_enabled: bool,
    ) -> torch.Tensor:
        """Align causal ``S-1`` main-loss observations with callback ``S`` labels."""

        # The registered CE consumes flattened targets selected by ForCausalLMLoss.
        if per_token_loss.numel() == effective_labels.numel():
            return per_token_loss.reshape_as(effective_labels)
        if (
            not sp_enabled
            and per_token_loss.dim() == effective_labels.dim()
            and per_token_loss.shape[:-1] == effective_labels.shape[:-1]
            and per_token_loss.size(-1) + 1 == effective_labels.size(-1)
        ):
            return F.pad(per_token_loss, (0, 1), value=0.0)
        raise ChannelLossMetadataError(
            "Channel loss: reused main-loss CE shape does not align with effective labels "
            f"({tuple(per_token_loss.shape)} vs {tuple(effective_labels.shape)})."
        )

    def _per_token_ce(
        self,
        logits: torch.Tensor | None,
        labels_flat: torch.Tensor,
        vocab_size: int | None,
        hidden_states: torch.Tensor | None,
        weights: torch.Tensor | None,
        ignore_index: int,
        logits_bias: torch.Tensor | None = None,
        logits_bias_weight: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if hidden_states is not None and weights is not None:
            hs = hidden_states.detach().reshape(-1, hidden_states.size(-1))
            w = weights.detach()
            bias = logits_bias.detach().reshape(-1, logits_bias.size(-1)) if logits_bias is not None else None
            bias_weight = logits_bias_weight.detach() if logits_bias_weight is not None else None
            return self._eager_chunked(hs, w, labels_flat.to(hs.device), ignore_index, bias, bias_weight)

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
        logits_bias: torch.Tensor | None = None,
        logits_bias_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.size(0)
        result = torch.empty(seq_len, device=hidden_states.device, dtype=torch.float32)
        for start in range(0, seq_len, cls.EAGER_CHUNK_SIZE):
            end = min(start + cls.EAGER_CHUNK_SIZE, seq_len)
            chunk_logits = F.linear(hidden_states[start:end], weights).float()
            if logits_bias is not None:
                bias = logits_bias[start:end]
                if logits_bias_weight is not None:
                    bias = F.linear(bias, logits_bias_weight)
                chunk_logits = chunk_logits + bias.float()
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
        packed_cu_seqlens: Any | None = None,
        packed_global_cu_seqlens: Any | None = None,
        packed_global_physical_seq_len: int | None = None,
        tail_padding_length: Any | None = None,
        sp_enabled: bool = False,
        parallel_state: ParallelState | None = None,
        sp_rank_override: int | None = None,
        sp_size_override: int | None = None,
        strict: bool = False,
        include_data_stats: bool = False,
        defer_sp: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
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
                    packed_cu_seqlens=packed_cu_seqlens,
                    packed_global_cu_seqlens=packed_global_cu_seqlens,
                    packed_global_physical_seq_len=packed_global_physical_seq_len,
                    tail_padding_length=tail_padding_length,
                    seq_len=seq_len,
                    ignore_index=ignore_index,
                    parallel_state=parallel_state,
                    sp_rank_override=sp_rank_override,
                    sp_size_override=sp_size_override,
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
            item = {
                "source_id": source_ids[i],
                "loss_sum": loss_sum,
                "token_count": token_count,
            }
            if include_data_stats:
                item["sample_count"] = torch.ones((), device=labels_slice.device, dtype=torch.int64)
                item["input_token_count"] = (
                    mask_slice.to(torch.bool).sum(dtype=torch.int64)
                    if mask_slice is not None
                    else torch.as_tensor(labels_slice.numel(), device=labels_slice.device, dtype=torch.int64)
                )
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
        packed_cu_seqlens: Any | None = None,
        packed_global_cu_seqlens: Any | None = None,
        packed_global_physical_seq_len: int | None = None,
        tail_padding_length: Any | None = None,
        parallel_state: ParallelState | None = None,
        sp_rank_override: int | None = None,
        sp_size_override: int | None = None,
        strict: bool = False,
        include_data_stats: bool = False,
        defer_reduce: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        if parallel_state is None:
            raise ValueError("parallel_state is required for SP channel-loss aggregation.")
        sp_group = parallel_state.sp_group
        sp_rank = int(parallel_state.sp_rank if sp_rank_override is None else sp_rank_override)
        if sp_rank < 0:
            sp_rank = dist.get_rank(sp_group) if sp_group is not None else 0
        sp_size = int(getattr(parallel_state, "sp_size", 1) if sp_size_override is None else sp_size_override)
        if sp_size < 1:
            raise ChannelLossMetadataError(f"Channel loss: invalid sequence-parallel size {sp_size}.")
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
            *,
            flat_indices: bool = False,
        ) -> None:
            if end < start:
                return
            flat_start = start if flat_indices else batch_idx * local_width + start
            flat_end = min(end if flat_indices else batch_idx * local_width + end, seq_len)
            labels_slice = labels_flat[flat_start:flat_end]
            loss_slice = per_token_loss[flat_start:flat_end]
            pos_slice = (
                positions_cpu.reshape(-1)[flat_start:flat_end]
                if flat_indices
                else positions_cpu[batch_idx, start:end].reshape(-1)[: labels_slice.numel()]
            )
            mask_slice = None
            if attention_mask_flat is not None:
                mask_slice = attention_mask_flat[flat_start:flat_end]
            valid = labels_slice != ignore_index
            token_count = valid.sum(dtype=torch.int64)
            loss_sum = loss_slice[valid].sum()
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
                segment["input_token_count"] = (
                    mask_slice.to(torch.bool).sum(dtype=torch.int64)
                    if mask_slice is not None
                    else torch.as_tensor(labels_slice.numel(), device=labels_slice.device, dtype=torch.int64)
                )
            segments.append(segment)

        headwise_cp = int(getattr(parallel_state, "cp_size", 1)) > 1
        points, points_error = _packed_cu_points(packed_cu_seqlens)
        if points_error:
            if strict:
                raise ChannelLossMetadataError(f"Channel loss: invalid packed CU metadata: {points_error}.")
            logger.warning_rank0(
                f"Channel loss: invalid packed CU metadata: {points_error}; skipping this micro-batch."
            )
            return []
        if points and points[-1] not in (seq_len, seq_len * sp_size):
            if strict:
                raise ChannelLossMetadataError(
                    "Channel loss: packed CU endpoint does not match the local or global width."
                )
            logger.warning_rank0("Channel loss: packed CU endpoint mismatch; skipping this micro-batch.")
            return []
        global_points, global_error = _packed_cu_points(packed_global_cu_seqlens)
        if global_error:
            if strict:
                raise ChannelLossMetadataError(f"Channel loss: invalid global packed CU metadata: {global_error}.")
            logger.warning_rank0(
                f"Channel loss: invalid global packed CU metadata: {global_error}; skipping this micro-batch."
            )
            return []
        if sp_size > 1 and global_points is None:
            if strict:
                raise ChannelLossMetadataError("Channel loss: SP source alignment requires global CU metadata.")
            logger.warning_rank0("Channel loss: SP lacks global CU metadata; skipping this micro-batch.")
            return []
        if sp_size > 1 and not _declared_tail_matches_sources(global_points, len(source_ids), tail_padding_length):
            if strict:
                raise ChannelLossMetadataError("Channel loss: global CU and declared tail do not match sources.")
            logger.warning_rank0("Channel loss: global CU/tail source mismatch; skipping this micro-batch.")
            return []
        if points and points[-1] != seq_len and global_points != points:
            if strict:
                raise ChannelLossMetadataError("Channel loss: global Ulysses CU metadata is missing or inconsistent.")
            logger.warning_rank0("Channel loss: global Ulysses CU metadata mismatch; skipping this micro-batch.")
            return []
        local_spans = list(zip(points, points[1:])) if points and points[-1] == seq_len else None
        if headwise_cp and sp_size > 1 and local_spans is None:
            if strict:
                raise ChannelLossMetadataError("Channel loss: CP requires rank-local packed CU metadata.")
            logger.warning_rank0("Channel loss: CP lacks rank-local packed CU metadata; skipping this micro-batch.")
            return []
        if (
            local_spans is None
            and sp_size > 1
            and (
                not _global_cu_positions_are_safe(
                    global_points, len(source_ids), seq_len * sp_size, tail_padding_length
                )
                or not _contiguous_starts_match_global(
                    global_points, positions_cpu, sp_rank, sp_size, tail_padding_length
                )
                or not _global_tail_is_padding(
                    global_points,
                    positions_cpu,
                    labels_flat,
                    sp_rank,
                    sp_size,
                    tail_padding_length,
                    ignore_index,
                )
            )
        ):
            if strict:
                raise ChannelLossMetadataError("Channel loss: global CU cannot prove position-based source alignment.")
            logger.warning_rank0("Channel loss: ambiguous position-based source alignment; skipping this micro-batch.")
            return []
        geometry_candidates = 1 if local_spans is not None and headwise_cp else 2
        tail_lengths = _as_flat_list(tail_padding_length)
        has_tail = len(tail_lengths) == 1 and isinstance(tail_lengths[0], Integral) and tail_lengths[0] > 0
        if local_spans is not None and global_points is not None:
            lengths = [end - start for start, end in local_spans]
            if headwise_cp or (
                packed_global_physical_seq_len is not None and global_points[-1] != packed_global_physical_seq_len
            ):
                multiple = 2 * sp_size
                expected_headwise = [
                    ((end - start + multiple - 1) // multiple) * 2
                    for start, end in zip(global_points, global_points[1:])
                ]
            else:
                expected_headwise = None
            if sp_size > 1 and global_points[-1] == seq_len * sp_size:
                expected_contiguous = _ulysses_local_span_lengths(
                    global_points, positions_2d.size(0), local_width, sp_rank, sp_size
                )
            else:
                expected_contiguous = None
            if sp_size == 1:
                expected_contiguous = [end - start for start, end in zip(global_points, global_points[1:])]
            headwise_valid = lengths == expected_headwise and _headwise_starts_match_positions(
                local_spans, positions_cpu.reshape(-1), sp_rank
            )
            contiguous_valid = lengths == expected_contiguous and (
                sp_size == 1
                or _contiguous_starts_match_global(global_points, positions_cpu, sp_rank, sp_size, tail_padding_length)
            )
            geometry_candidates = int(headwise_valid) | (int(contiguous_valid) << 1)
            if (
                packed_global_physical_seq_len is not None and packed_global_physical_seq_len != seq_len * sp_size
            ) or not geometry_candidates:
                if strict:
                    raise ChannelLossMetadataError("Channel loss: local packed CU disagrees with global geometry.")
                logger.warning_rank0("Channel loss: packed CU geometry mismatch; skipping this micro-batch.")
                return []
        if local_spans is not None and len(local_spans) == len(source_ids) + 1 and has_tail:
            valid_global = (
                global_points is not None
                and len(global_points) == len(local_spans) + 1
                and global_points[-1] - global_points[-2] == tail_lengths[0]
                and (global_points[-1] == seq_len * sp_size or packed_global_physical_seq_len == seq_len * sp_size)
            )
            if not valid_global:
                if strict:
                    raise ChannelLossMetadataError("Channel loss: packed tail does not match global CU metadata.")
                logger.warning_rank0("Channel loss: packed tail metadata mismatch; skipping this micro-batch.")
                return []
            tail_start, tail_end = local_spans[-1]
            tail_labels = labels_flat[tail_start:tail_end]
            tail_positions = positions_cpu.reshape(-1)[tail_start:tail_end]
            if not bool(tail_labels.eq(ignore_index).all()) or not bool(tail_positions.eq(0).all()):
                if strict:
                    raise ChannelLossMetadataError("Channel loss: packed tail contains non-padding tokens.")
                logger.warning_rank0(
                    "Channel loss: packed tail contains non-padding tokens; skipping this micro-batch."
                )
                return []
            local_spans = local_spans[:-1]
        if local_spans is not None:
            if not _validate_segment_count(len(local_spans), len(source_ids), strict, "SP packed segment count"):
                return []
            for source_idx, (start, end) in enumerate(local_spans):
                _partial(source_idx, int(start), int(end), sp_rank == 0, 0, flat_indices=True)
        for batch_idx, row_pos in enumerate(positions_cpu if local_spans is None else []):
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
            mode = "packed_cu" if local_spans is not None else "positions"
            layout = (mode, tuple(global_points or ()))
            return {
                "segments": segments,
                "source_ids": list(source_ids),
                "device": per_token_loss.device,
                "data": include_data_stats,
                "mode": mode,
                "geometry_candidates": geometry_candidates,
                "layout_digest": hashlib.sha256(repr(layout).encode()).hexdigest(),
            }

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
        self.computer._require_source_provenance = self.enabled
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
        model = getattr(self.trainer.model, "unwrapped_module", self.trainer.model)
        self.computer.install(model)
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
        channel_loss_source_repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            self._collect_step = False
            return
        if channel_loss_source_repeat < 1:
            raise ValueError(
                f"Channel loss channel_loss_source_repeat must be >= 1, got {channel_loss_source_repeat}."
            )

        self.computer._dpo_segments = channel_loss_source_repeat > 1
        self.computer._source_repeat = channel_loss_source_repeat
        provenances = [_pop_channel_loss_provenance(batch) for batch in micro_batches or []]
        self._collect_step = state.global_step % self.config.interval == 0
        if not self._collect_step:
            self.computer.begin_step([], [], [])
            if self.config.strict:
                missing_source_micro_steps = []
                for micro_step, micro_batch in enumerate(micro_batches or []):
                    source_ids, _, _, _, _, _, _, _ = self._extract_metadata(micro_batch)
                    if not source_ids:
                        missing_source_micro_steps.append(micro_step)
                self._raise_if_missing_source_ids(missing_source_micro_steps)
            return

        per_mb_source_ids: list[list[ChannelKey]] = []
        per_mb_position_ids: list[torch.Tensor | None] = []
        per_mb_attention_masks: list[torch.Tensor | None] = []
        per_mb_packed_cu_seqlens: list[Any | None] = []
        per_mb_packed_global_cu_seqlens: list[Any | None] = []
        per_mb_packed_global_physical_seq_lens: list[int | None] = []
        per_mb_tail_padding_lengths: list[Any | None] = []
        per_mb_source_provenances: list[dict[str, Any] | None] = []
        missing_source_micro_steps = []
        for micro_step, (micro_batch, provenance) in enumerate(zip(micro_batches or [], provenances, strict=True)):
            (
                source_ids,
                source_names,
                position_ids,
                attention_mask,
                packed_cu_seqlens,
                packed_global_cu_seqlens,
                packed_global_physical_seq_len,
                tail_padding_length,
            ) = self._extract_metadata(micro_batch)
            if self.config.strict and not source_ids:
                missing_source_micro_steps.append(micro_step)
            if provenance is None:
                source_ids, source_names = self._repeat_source_metadata(
                    source_ids, source_names, channel_loss_source_repeat
                )
            self.computer.record_source_names(source_ids, source_names)
            per_mb_source_ids.append(source_ids)
            per_mb_position_ids.append(position_ids)
            per_mb_attention_masks.append(attention_mask)
            per_mb_packed_cu_seqlens.append(packed_cu_seqlens)
            per_mb_packed_global_cu_seqlens.append(packed_global_cu_seqlens)
            per_mb_packed_global_physical_seq_lens.append(packed_global_physical_seq_len)
            per_mb_tail_padding_lengths.append(tail_padding_length)
            per_mb_source_provenances.append(provenance)

        self.computer.begin_step(
            per_mb_source_ids,
            per_mb_position_ids,
            per_mb_attention_masks,
            per_mb_packed_cu_seqlens,
            per_mb_packed_global_cu_seqlens,
            per_mb_packed_global_physical_seq_lens,
            per_mb_tail_padding_lengths,
            per_mb_source_provenances,
        )
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
        self.computer.before_micro_step(
            primary_labels=micro_batch.get("labels"), mtp_labels=micro_batch.get("mtp_labels")
        )

    def strip_model_inputs(self, micro_batch: dict[str, Any]) -> None:
        if not self.enabled or not isinstance(micro_batch, dict):
            return
        _pop_channel_loss_provenance(micro_batch)
        if self._collect_step:
            # BaseTrainer and DPO enter the micro-step before preforward moves
            # tensors. Refresh the labels here from the actual model inputs.
            self.computer._primary_labels = micro_batch.get("labels")
            self.computer._mtp_labels = micro_batch.get("mtp_labels")
        for key in self._strip_keys:
            micro_batch.pop(key, None)

    def on_micro_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        if self._collect_step:
            self.computer.after_micro_step()
            if self.config.strict and self._synchronize_missing_observation():
                raise ChannelLossMetadataError(
                    "Channel loss: sampled micro-step had missing or invalid observation before optimizer update "
                    f"(local errors: {self.computer.strict_observation_errors})."
                )

    @contextmanager
    def micro_step_context(self, state: TrainerState, micro_batch: dict[str, Any]) -> Iterator[None]:
        if not self._collect_step:
            yield
            return
        self.on_micro_step_begin(state, micro_batch=micro_batch)
        completed = False
        try:
            yield
            completed = True
        finally:
            if completed:
                self.on_micro_step_end(state)
            else:
                # Do not enter observer collectives while training unwinds.
                self.computer.after_micro_step()

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
        grad_norm: float | torch.Tensor,
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
        """Reduce compact integer counters without launching another CE observer."""

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
        # Reuse the core metric suffix so every source has one stable identity
        # across loss, token-count, and data-composition series.
        metric_names = self._render_metric_names(
            {source_id: (0.0, label_token_count) for source_id, (_, _, label_token_count) in totals.items()},
            source_names,
        )
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

    def _render_metric_names(
        self,
        totals: dict[ChannelKey, tuple[float, int]],
        source_names: dict[ChannelKey, str] | None,
    ) -> dict[ChannelKey, str]:
        self._register_sources(list(totals), source_names)
        rendered: dict[ChannelKey, str] = {}
        for source_id in totals:
            source_name = self._source_registry[source_id]
            display_name = (
                _sanitize_metric_fragment(source_name)
                if source_name is not None
                else self._resolve_source_name(source_id)
            )
            rendered[source_id] = f"{_stable_source_metric_fragment(source_id)}__{display_name}"
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
    ) -> tuple[
        list[ChannelKey],
        list[str],
        torch.Tensor | None,
        torch.Tensor | None,
        Any | None,
        Any | None,
        int | None,
        Any | None,
    ]:
        if isinstance(micro_batch, dict):
            source_ids = self._first_present_list(micro_batch, self.config.source_id_keys, _as_channel_key_list)
            source_names = self._first_present_list(micro_batch, self.config.source_name_keys, _as_str_list)
            position_ids = micro_batch.get("position_ids")
            # Headwise CP keeps the full attention mask for full-attention
            # kernels, while its router mask is rank-local and excludes the
            # per-sample CP padding. Channel-loss metadata must follow the
            # local labels/positions rather than the global full-attention
            # shape.
            labels = micro_batch.get("labels")
            global_attention_mask = micro_batch.get("attention_mask")
            packed_global_physical_seq_len = None
            router_attention_mask = micro_batch.get("router_attention_mask")
            if (
                isinstance(router_attention_mask, torch.Tensor)
                and isinstance(labels, torch.Tensor)
                and router_attention_mask.numel() == labels.numel()
            ):
                attention_mask = router_attention_mask
                if isinstance(global_attention_mask, torch.Tensor):
                    packed_global_physical_seq_len = int(global_attention_mask.numel())
            else:
                attention_mask = global_attention_mask
            packed_cu_seqlens = micro_batch.get("cu_seqlens_list_q")
            if packed_cu_seqlens is None:
                packed_cu_seqlens = micro_batch.get("cu_seq_lens_q")
            packed_global_cu_seqlens = micro_batch.get("linear_attn_cu_seqlens_list_q")
            if packed_global_cu_seqlens is None:
                packed_global_cu_seqlens = micro_batch.get("linear_attn_cu_seq_lens_q")
            tail_padding_length = micro_batch.get("tail_padding_length")
            return (
                source_ids,
                source_names,
                position_ids if isinstance(position_ids, torch.Tensor) else None,
                attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                packed_cu_seqlens,
                packed_global_cu_seqlens,
                packed_global_physical_seq_len,
                tail_padding_length,
            )

        if isinstance(micro_batch, (list, tuple)):
            source_ids: list[ChannelKey] = []
            source_names: list[str] = []
            for sample in micro_batch:
                if not isinstance(sample, dict):
                    continue
                source_ids.extend(self._first_present_list(sample, self.config.source_id_keys, _as_channel_key_list))
                source_names.extend(self._first_present_list(sample, self.config.source_name_keys, _as_str_list))
            return source_ids, source_names, None, None, None, None, None, None

        return [], [], None, None, None, None, None, None

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


def _supported_observer_kwargs(method: Callable[..., Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Pass extensions accepted by old or user-overridden observer signatures."""
    try:
        parameters = inspect.signature(method).parameters
    except Exception:  # Unsupported signatures are not a training failure.
        return {}
    return {
        name: value
        for name, value in extras.items()
        if name in parameters
        and parameters[name].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _unpack_partial_loss(
    loss_fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    while isinstance(loss_fn, partial):
        args = loss_fn.args + args
        kwargs = {**(loss_fn.keywords or {}), **kwargs}
        loss_fn = loss_fn.func
    return loss_fn, args, kwargs


def _bind_loss_call_args(
    loss_fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    loss_fn, args, kwargs = _unpack_partial_loss(loss_fn, args, kwargs)
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

        positional_names = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        known_names = set(_LOSS_CALL_POSITIONAL_KEYS) | {
            "hidden_states",
            "weights",
            "shift_labels",
            "logits_bias",
            "logits_bias_weight",
        }
        for index, (key, value) in enumerate(zip(_LOSS_CALL_POSITIONAL_KEYS, args)):
            if key in signature.parameters or (
                index < 3 and index < len(positional_names) and positional_names[index] not in known_names
            ):
                bound_args.setdefault(key, value)
        return bound_args
    except Exception as exc:
        raise ChannelLossMetadataError("cannot bind loss arguments") from exc


def _validate_source_provenance(
    provenance: dict[str, Any] | None,
    source_ids: list[ChannelKey],
    parallel_state: ParallelState,
    labels: torch.Tensor,
    ignore_index: int,
) -> dict[str, Any]:
    """Validate the producer's feature boundaries against this loss shard."""
    if not isinstance(provenance, dict) or "error" in provenance:
        raise ChannelLossMetadataError("Channel loss requires source boundaries from the packing collator.")
    points, error = _packed_cu_points(provenance.get("source_cu"))
    raw_ids = provenance.get("source_ids")
    if error or points is None or len(points) != len(source_ids) + 1 or not isinstance(raw_ids, list):
        raise ChannelLossMetadataError("Channel loss source boundary metadata is invalid.")
    normalized = [_as_channel_key_list(value) for value in raw_ids]
    if any(len(value) != 1 for value in normalized) or [value[0] for value in normalized] != source_ids:
        raise ChannelLossMetadataError("Channel loss source IDs do not match the packed features.")
    size = parallel_state.sp_size if parallel_state.sp_enabled else 1
    rank = parallel_state.sp_rank if parallel_state.sp_enabled else 0
    tail = provenance.get("tail")
    if (
        not isinstance(tail, Integral)
        or isinstance(tail, bool)
        or tail < 0
        or provenance.get("sp_size") != size
        or provenance.get("sp_rank") != rank
        or labels.ndim != 2
        or labels.size(0) != 1
        or points[-1] + tail != labels.numel() * size
    ):
        raise ChannelLossMetadataError("Channel loss source boundaries do not match the local loss shard.")
    mask = provenance.get("input_mask")
    if not isinstance(mask, torch.Tensor) or mask.device.type != "cpu" or mask.ndim != 1 or mask.numel() != points[-1]:
        raise ChannelLossMetadataError("Channel loss input mask does not match the packed features.")
    width = labels.numel()
    local_tail = max(0, (rank + 1) * width - max(rank * width, points[-1]))
    if local_tail and not bool(labels.reshape(-1)[-local_tail:].eq(ignore_index).all()):
        raise ChannelLossMetadataError("Channel loss packed tail contains supervised labels.")
    return {**provenance, "source_cu": points, "input_mask": mask.to(dtype=torch.bool)}


def _dpo_source_tail_mask(
    provenance: dict[str, Any], flash_cu: Any, labels: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Mask chosen/rejected tails using Flash boundaries, without modifying labels."""
    flash, error = _packed_cu_points(flash_cu)
    points = provenance["source_cu"]
    physical_width = points[-1] + provenance["tail"]
    if error or flash is None or flash[-1] != physical_width:
        raise ChannelLossMetadataError("DPO channel loss requires global Flash sequence boundaries.")
    endpoints = []
    for start, end in zip(points, points[1:]):
        within = [point for point in flash if start <= point <= end]
        if len(within) != 3 or within[0] != start or within[-1] != end or not (start < within[1] < end):
            raise ChannelLossMetadataError("DPO source must contain two nonempty Flash segments.")
        endpoints.extend(within[1:])
    width = labels.numel()
    start = provenance["sp_rank"] * width
    mask = torch.zeros_like(labels, dtype=torch.bool).reshape(-1)
    for end in endpoints:
        index = end - 1 - start
        if 0 <= index < width:
            mask[index] = True
    return mask.reshape_as(labels), tuple(flash)


def _aggregate_provenance(
    per_token_loss: torch.Tensor,
    labels: torch.Tensor,
    source_ids: list[ChannelKey],
    provenance: dict[str, Any],
    ignore_index: int,
    *,
    strict: bool,
    defer_sp: bool,
) -> list[dict[str, Any]] | dict[str, Any]:
    points = provenance["source_cu"]
    rank, size = provenance["sp_rank"], provenance["sp_size"]
    width = labels.numel()
    lo, hi = rank * width, (rank + 1) * width
    local_lengths = _ulysses_local_span_lengths(points, 1, width, rank, size)
    segments = []
    for index, start in enumerate(points[:-1]):
        first = min(max(start, lo), hi)
        last = first + local_lengths[index]
        local_start, local_end = first - lo, last - lo
        selected = labels[local_start:local_end] != ignore_index
        device = labels.device
        segments.append(
            {
                "batch_idx": index,
                "sp_rank": rank,
                "local_order": 0,
                "is_start": rank == 0,
                "has_explicit_padding_mask": torch.zeros((), dtype=torch.bool, device=device),
                "has_only_zero_positions": False,
                "loss_sum": per_token_loss[local_start:local_end][selected].sum(),
                "token_count": selected.sum(dtype=torch.int64),
                "sample_count": torch.tensor(int(rank == 0), dtype=torch.int64, device=device),
                "input_token_count": provenance["input_mask"][first:last].sum(dtype=torch.int64).to(device),
            }
        )
    if defer_sp and size > 1:
        mask = provenance["input_mask"].to(torch.uint8).contiguous().numpy().tobytes()
        layout = (
            tuple(points),
            provenance["tail"],
            size,
            hashlib.sha256(mask).hexdigest(),
            provenance.get("global_cu"),
            provenance.get("dpo_cu"),
        )
        return {
            "segments": segments,
            "source_ids": list(source_ids),
            "device": per_token_loss.device,
            "data": True,
            "mode": "source",
            "geometry_candidates": 2,
            "layout_digest": hashlib.sha256(repr(layout).encode()).hexdigest(),
        }
    return ChannelLossComputer._merge_sp_segments(segments, source_ids, strict=strict, include_data_stats=True)


def _headwise_starts_match_positions(spans: list[tuple[int, int]], positions: torch.Tensor, rank: int) -> bool:
    return rank != 0 or all(start == end or int(positions[start]) == 0 for start, end in spans)


def _global_cu_positions_are_safe(
    points: list[int] | None, source_count: int, global_width: int, tail_padding_length: Any | None
) -> bool:
    """Positions cannot identify empty sources or an unaccounted padding tail."""
    if points is None or points[-1] != global_width:
        return False
    tail = _as_flat_list(tail_padding_length)
    if len(tail) != 1 or not isinstance(tail[0], Integral) or isinstance(tail[0], bool) or tail[0] < 0:
        return False
    tail_length = int(tail[0])
    if len(points) != source_count + 1 + int(tail_length > 0):
        return False
    if tail_length and points[-1] - points[-2] != tail_length:
        return False
    return all(end > start for start, end in zip(points[:source_count], points[1 : source_count + 1]))


def _declared_tail_matches_sources(points: list[int], source_count: int, tail_padding_length: Any | None) -> bool:
    if tail_padding_length is None:
        return True
    tail = _as_flat_list(tail_padding_length)
    if len(tail) != 1 or not isinstance(tail[0], Integral) or isinstance(tail[0], bool) or tail[0] < 0:
        return False
    count = int(tail[0] > 0)
    return len(points) == source_count + 1 + count and (not count or points[-1] - points[-2] == tail[0])


def _contiguous_starts_match_global(
    points: list[int], positions: torch.Tensor, rank: int, sp_size: int, tail_padding_length: Any | None
) -> bool:
    """Reject CU boundaries inconsistent with local positions; this is not source provenance."""
    batch, width = positions.shape
    row_width = width * sp_size
    if not row_width or points[-1] != batch * row_width:
        return False
    expected = torch.zeros_like(positions, dtype=torch.bool)
    for point in points[:-1]:
        row, offset = divmod(point, row_width)
        local = offset - rank * width
        if row < batch and 0 <= local < width:
            expected[row, local] = True
    checked = torch.ones_like(expected)
    tail = _as_flat_list(tail_padding_length)
    if (
        len(points) > 1
        and len(tail) == 1
        and isinstance(tail[0], Integral)
        and not isinstance(tail[0], bool)
        and tail[0] > 0
        and points[-1] - points[-2] == tail[0]
    ):
        for row in range(batch):
            first_tail = max(0, min(width, points[-2] - row * row_width - rank * width))
            checked[row, first_tail:] = False
    return bool(torch.equal(positions.eq(0)[checked], expected[checked]))


def _global_tail_is_padding(
    points: list[int],
    positions: torch.Tensor,
    labels: torch.Tensor,
    rank: int,
    sp_size: int,
    tail_padding_length: Any | None,
    ignore_index: int,
) -> bool:
    tail = _as_flat_list(tail_padding_length)
    if not tail or tail[0] == 0:
        return True
    batch, width = positions.shape
    local_tail = _ulysses_local_span_lengths(points, batch, width, rank, sp_size)[-1]
    if not local_tail:
        return True
    return bool(labels[-local_tail:].eq(ignore_index).all()) and bool(positions.reshape(-1)[-local_tail:].eq(0).all())


def _ulysses_local_span_lengths(points: list[int], batch: int, width: int, rank: int, sp_size: int) -> list[int]:
    """Intersect global packed spans with this rank's contiguous per-row shards."""
    row_width = width * sp_size
    return [
        sum(
            max(0, min(end, row * row_width + (rank + 1) * width) - max(start, row * row_width + rank * width))
            for row in range(batch)
        )
        for start, end in zip(points, points[1:])
    ]


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


def _packed_cu_points(value: Any | None) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            return None, "expected a one-dimensional int32/int64 tensor"
        raw_points = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        raw_points = list(value)
    else:
        return None, f"expected a tensor/list/tuple, got {type(value).__name__}"

    points: list[int] = []
    for point in raw_points:
        if isinstance(point, bool) or not isinstance(point, Integral):
            return None, "all CU points must be integers"
        points.append(int(point))
    if not points or points[0] != 0:
        return None, "CU points must start at zero"
    if any(end < start for start, end in zip(points, points[1:])):
        return None, "CU points must be non-decreasing"
    return points, None


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
