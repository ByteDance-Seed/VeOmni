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

import asyncio
import hashlib
import inspect
import re
from collections import defaultdict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from threading import Lock, get_ident
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...data.data_collator import CHANNEL_LOSS_PROVENANCE
from ...distributed.parallel_state import get_parallel_state
from ...utils.constants import IGNORE_INDEX
from ...utils.device import get_device_id, get_device_type
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


class ChannelLossMetadataError(ValueError):
    """Raised when channel metadata cannot be aligned with packed segments."""


def _collective_device(group: Any) -> torch.device:
    if dist.get_backend(group) == "gloo":
        return torch.device("cpu")
    return torch.device(get_device_type(), get_device_id())


def _capture_owner() -> tuple[int, Any]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return get_ident(), task


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
        self._require_source_provenance = False
        self._dpo_segments = False
        self._dpo_attention_cu: Any | None = None
        self._result: list[dict[str, Any]] | None = None
        self._pending_sp: list[dict[str, Any]] = []
        self._capture_errors: list[str] = []
        self._capture_attempts = 0
        self._capture_lock = Lock()
        self._capture_owner: tuple[int, Any] | None = None
        self._per_mb_source_ids: list[list[ChannelKey]] = []
        self._per_mb_position_ids: list[torch.Tensor | None] = []
        self._per_mb_attention_masks: list[torch.Tensor | None] = []
        self._per_mb_packed_cu_seqlens: list[Any | None] = []
        self._per_mb_packed_global_cu_seqlens: list[Any | None] = []
        self._per_mb_packed_global_physical_seq_lens: list[int | None] = []
        self._per_mb_tail_padding_lengths: list[Any | None] = []
        self._per_mb_source_provenances: list[dict[str, Any] | None] = []
        self._micro_step = 0
        self.step_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor]] = {}
        self.source_names: dict[ChannelKey, str] = {}
        self.strict = False

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
        if _ACTIVE_CHANNEL_LOSS_COMPUTER.get() is not self or self._capture_owner is None:
            return False
        if self._capture_owner != _capture_owner():
            raise RuntimeError("Concurrent capture on the same ChannelLossComputer is unsupported.")
        return True

    @contextmanager
    def capture(self) -> Iterator[None]:
        nested = self.capture_active
        owner = _capture_owner()
        if not nested:
            if not self._capture_lock.acquire(blocking=False):
                raise RuntimeError("Concurrent capture on the same ChannelLossComputer is unsupported.")
            self._capture_owner = owner
            self._pending_sp, self._capture_errors, self._capture_attempts = [], [], 0
        previous_result = self._result
        pending_count, error_count, attempts = len(self._pending_sp), len(self._capture_errors), self._capture_attempts
        token = _ACTIVE_CHANNEL_LOSS_COMPUTER.set(self)
        completed = False
        try:
            yield
            completed = True
        finally:
            _ACTIVE_CHANNEL_LOSS_COMPUTER.reset(token)
            if not completed:
                self._result = previous_result
                del self._pending_sp[pending_count:]
                del self._capture_errors[error_count:]
                self._capture_attempts = attempts
            if not nested:
                try:
                    if completed:
                        self._finish_capture()
                finally:
                    self._pending_sp, self._capture_errors, self._capture_attempts = [], [], 0
                    self._capture_owner = None
                    self._capture_lock.release()

    def _finish_capture(self) -> None:
        """Agree within the SP group before entering any observer reduction."""
        group = self.parallel_state.sp_group if self.parallel_state.sp_enabled else None
        world = self.parallel_state.sp_size if group is not None else 1
        if self._source_ids and self._capture_attempts == 0:
            self._capture_errors.append("no causal-loss observation was recorded")
        if self._capture_attempts != len(self._pending_sp) and self.parallel_state.sp_enabled:
            self._capture_errors.append("some causal-loss observations failed")
        if group is not None and world > 1:
            signature = (
                tuple((type(value).__name__, repr(value)) for value in self._source_ids),
                tuple(observation["layout"] for observation in self._pending_sp),
            )
            digest = int.from_bytes(hashlib.sha256(repr(signature).encode()).digest()[:8], "little", signed=True)
            device = _collective_device(group)
            status = torch.tensor(
                [int(bool(self._capture_errors)), self._capture_attempts, digest], dtype=torch.int64, device=device
            )
            statuses = [torch.empty_like(status) for _ in range(world)]
            dist.all_gather(statuses, status, group=group)
            values = torch.stack(statuses).cpu().tolist()
            if any(value[0] for value in values) or any(value[1:] != values[0][1:] for value in values):
                self._capture_errors.append("SP ranks disagree on observer status, source order or packed layout")
            if not self._capture_errors and self._pending_sp:
                candidates = torch.tensor(
                    [observation["geometry_candidates"] for observation in self._pending_sp],
                    dtype=torch.int64,
                    device=device,
                )
                gathered = [torch.empty_like(candidates) for _ in range(world)]
                dist.all_gather(gathered, candidates, group=group)
                rows = torch.stack(gathered).cpu().tolist()
                for index in range(len(self._pending_sp)):
                    common = 3
                    for row in rows:
                        common &= row[index]
                    if not common:
                        self._capture_errors.append("SP ranks do not share a valid packed shard geometry")
                        break
        # Strict mode must stop every data-parallel stripe before any of them
        # can enter the next training collective. The default non-strict path
        # does not add a world-wide synchronization.
        if self.strict and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            failed = torch.tensor(
                [int(bool(self._capture_errors))], dtype=torch.int32, device=_collective_device(dist.group.WORLD)
            )
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item() and not self._capture_errors:
                self._capture_errors.append("channel-loss observation failed on another rank")
        if self._capture_errors:
            self._result = None
            message = "Channel loss: " + "; ".join(self._capture_errors)
            if self.strict:
                raise ChannelLossMetadataError(message)
            logger.warning_rank0(message + "; skipping this micro-batch.")
            return
        if self._pending_sp:
            combined = []
            for pending in self._pending_sp:
                reduced = self._reduce_sp(
                    pending["segments"], self._source_ids, pending["device"], self.parallel_state, self.strict
                )
                if not reduced:
                    if self.strict:
                        raise ChannelLossMetadataError("Channel loss: SP reduction produced no source-aligned result.")
                    logger.warning_rank0("Channel loss: SP reduction produced no result; skipping this micro-batch.")
                    return
                combined.extend(reduced)
            self._result = (self._result or []) + combined

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
        self._per_mb_packed_cu_seqlens = per_mb_packed_cu_seqlens or [None] * len(per_mb_source_ids)
        self._per_mb_packed_global_cu_seqlens = per_mb_packed_global_cu_seqlens or [None] * len(per_mb_source_ids)
        self._per_mb_packed_global_physical_seq_lens = per_mb_packed_global_physical_seq_lens or [None] * len(
            per_mb_source_ids
        )
        self._per_mb_tail_padding_lengths = per_mb_tail_padding_lengths or [None] * len(per_mb_source_ids)
        self._per_mb_source_provenances = per_mb_source_provenances or [None] * len(per_mb_source_ids)
        self._micro_step = 0
        self.step_totals = {}

    def before_micro_step(self) -> None:
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

    def after_micro_step(self) -> None:
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
        self._source_ids = []
        self._position_ids = None
        self._attention_mask = None
        self._packed_cu_seqlens = None
        self._packed_global_cu_seqlens = None
        self._packed_global_physical_seq_len = None
        self._tail_padding_length = None
        self._source_provenance = None
        self._result = None
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
        self._capture_attempts += 1
        from ...ops.kernels.loss.cross_entropy_loss.standard import chunk_loss

        capture = {}
        token = chunk_loss._CAPTURE.set(capture)
        try:
            main_loss = self._original_loss_fn(*args, **kwargs)
            reused_per_token_loss = capture.get("per_token")
        finally:
            chunk_loss._CAPTURE.reset(token)
        if capture.get("error"):
            self._capture_errors.append(f"chunk-loss observation failed: {capture['error']}")
            return main_loss

        call_args = _bind_loss_call_args(self._original_loss_fn, args, kwargs)
        logits = call_args.get("logits")
        labels = call_args.get("labels")
        if labels is None:
            labels = call_args.get("shift_labels")
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
                reused_per_token_loss=reused_per_token_loss,
                logits_bias=call_args.get("logits_bias"),
                logits_bias_weight=call_args.get("logits_bias_weight"),
            )
        elif self.capture_active and self._source_ids:
            self._capture_errors.append("loss call did not provide labels")

        return main_loss

    def observe_dpo_log_probs(self, log_probs: torch.Tensor, labels: torch.Tensor, attention_cu: Any) -> None:
        """Observe expanded policy log-probs, after any model-side row compaction."""
        if not self._dpo_segments or not self._source_ids or not self.capture_active:
            return
        self._capture_attempts += 1
        try:
            self._dpo_attention_cu = attention_cu
            self.compute_side_channel(
                logits=None,
                labels=labels,
                vocab_size=None,
                hidden_states=None,
                weights=None,
                reused_per_token_loss=-log_probs.detach(),
            )
        except Exception as error:
            self._capture_errors.append(f"DPO log-prob observation failed: {type(error).__name__}: {error}")

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
                reused_per_token_loss=reused_per_token_loss,
                logits_bias=logits_bias,
                logits_bias_weight=logits_bias_weight,
            )
            if not result:
                raise ChannelLossMetadataError("no source-aligned per-token loss observation was produced")
            if isinstance(result, dict):
                self._pending_sp.append(result)
            elif self.capture_active:
                self._result = (self._result or []) + result
            else:
                self._result = result
        except ChannelLossMetadataError as error:
            if self.capture_active:
                self._capture_errors.append(str(error))
                return
            if self.strict:
                raise
            logger.warning_rank0("Channel loss: metadata mismatch; skipping this micro-batch.", exc_info=True)
            self._result = None
        except Exception as error:
            if self.capture_active:
                self._capture_errors.append(f"{type(error).__name__}: {error}")
                return
            logger.warning_rank0("Channel loss: per-token CE failed; skipping this micro-batch.", exc_info=True)
            self._result = None

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
        reused_per_token_loss: torch.Tensor | None = None,
        logits_bias: torch.Tensor | None = None,
        logits_bias_weight: torch.Tensor | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        sp_enabled = bool(self.parallel_state.sp_enabled)
        # Match ``ForCausalLMLoss``: a provided ``shift_labels`` wins, including under SP.
        if shift_labels is not None:
            effective_labels = shift_labels
        elif not sp_enabled:
            effective_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        else:
            effective_labels = labels

        provenance = self._source_provenance
        if provenance is not None or self._require_source_provenance:
            provenance = _validate_source_provenance(
                provenance, self._source_ids, self.parallel_state, effective_labels, ignore_index
            )
        if provenance is not None and sp_enabled and self._packed_global_cu_seqlens is not None:
            global_cu, error = _packed_cu_points(self._packed_global_cu_seqlens)
            if error or global_cu is None:
                raise ChannelLossMetadataError("Channel loss global attention boundaries are invalid.")
            provenance = {**provenance, "global_cu": tuple(global_cu)}
        if self._dpo_segments:
            if provenance is None:
                raise ChannelLossMetadataError("DPO channel loss requires trusted source boundaries.")
            tail_mask, dpo_cu = _dpo_source_tail_mask(provenance, self._dpo_attention_cu, effective_labels)
            effective_labels = effective_labels.masked_fill(tail_mask, ignore_index)
            provenance = {**provenance, "dpo_cu": dpo_cu}

        labels_flat = effective_labels.reshape(-1)
        attention_mask_flat = _position_aligned_attention_mask_flat(
            attention_mask=attention_mask,
            labels=labels,
            effective_labels=effective_labels,
        )
        if reused_per_token_loss is not None:
            if reused_per_token_loss.numel() == effective_labels.numel():
                per_token_loss = reused_per_token_loss.reshape_as(effective_labels)
            elif (
                not sp_enabled
                and reused_per_token_loss.dim() == effective_labels.dim()
                and reused_per_token_loss.shape[:-1] == effective_labels.shape[:-1]
                and reused_per_token_loss.size(-1) + 1 == effective_labels.size(-1)
            ):
                per_token_loss = F.pad(reused_per_token_loss, (0, 1))
            else:
                raise ChannelLossMetadataError("Channel loss: chunk-loss observation and labels do not align.")
            per_token_loss = per_token_loss.reshape(-1)
        else:
            per_token_loss = self._per_token_ce(
                logits, labels_flat, vocab_size, hidden_states, weights, ignore_index, logits_bias, logits_bias_weight
            )
        if per_token_loss is None:
            return []
        labels_flat = labels_flat.to(per_token_loss.device)
        if attention_mask_flat is not None:
            attention_mask_flat = attention_mask_flat.to(per_token_loss.device)

        if provenance is not None:
            return _aggregate_provenance(
                per_token_loss,
                labels_flat,
                self._source_ids,
                provenance,
                self.parallel_state,
                ignore_index,
                strict=self.strict,
                defer_sp=self.capture_active,
            )

        return self._aggregate_by_source(
            per_token_loss=per_token_loss,
            labels_flat=labels_flat,
            attention_mask_flat=attention_mask_flat,
            source_ids=self._source_ids,
            position_ids=self._position_ids,
            ignore_index=ignore_index,
            packed_cu_seqlens=self._packed_cu_seqlens,
            packed_global_cu_seqlens=self._packed_global_cu_seqlens,
            packed_global_physical_seq_len=self._packed_global_physical_seq_len,
            tail_padding_length=self._tail_padding_length,
            sp_enabled=sp_enabled,
            parallel_state=self.parallel_state,
            strict=self.strict,
            defer_sp=self.capture_active,
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
            return self._eager_chunked(hs, w, labels_flat.to(hs.device), ignore_index, bias, logits_bias_weight)

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
                chunk_logits = chunk_logits + (
                    F.linear(bias, logits_bias_weight.detach()).float()
                    if logits_bias_weight is not None
                    else bias.float()
                )
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
        strict: bool = False,
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
                    seq_len=seq_len,
                    ignore_index=ignore_index,
                    packed_cu_seqlens=packed_cu_seqlens,
                    packed_global_cu_seqlens=packed_global_cu_seqlens,
                    packed_global_physical_seq_len=packed_global_physical_seq_len,
                    tail_padding_length=tail_padding_length,
                    parallel_state=parallel_state,
                    strict=strict,
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
            valid = labels_slice != ignore_index
            token_count = valid.sum()
            loss_sum = loss_slice[valid].sum()
            channel_loss.append(
                {
                    "source_id": source_ids[i],
                    "loss_sum": loss_sum,
                    "token_count": token_count,
                }
            )
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
        strict: bool = False,
        defer_reduce: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        if parallel_state is None:
            raise ValueError("parallel_state is required for SP channel-loss aggregation.")
        sp_group = parallel_state.sp_group
        sp_rank = dist.get_rank(sp_group) if sp_group is not None else 0
        # CP producers can use headwise padded or contiguous sequence shards.
        headwise_cp = int(getattr(parallel_state, "cp_size", 1)) > 1
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
            if end < start or (end == start and not flat_indices):
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
            segments.append(
                {
                    "batch_idx": batch_idx,
                    "sp_rank": sp_rank,
                    "local_order": local_order,
                    "is_start": is_start,
                    "has_explicit_padding_mask": has_explicit_padding_mask,
                    "has_only_zero_positions": has_only_zero_positions,
                    "loss_sum": loss_sum,
                    "token_count": token_count,
                }
            )

        points, points_error = _packed_cu_points(packed_cu_seqlens)
        if points_error:
            if strict:
                raise ChannelLossMetadataError(f"Channel loss: invalid packed CU metadata: {points_error}.")
            logger.warning_rank0(
                f"Channel loss: invalid packed CU metadata: {points_error}; skipping this micro-batch."
            )
            return []
        if points and points[-1] not in (seq_len, seq_len * parallel_state.sp_size):
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
        if parallel_state.sp_size > 1 and global_points is None:
            if strict:
                raise ChannelLossMetadataError("Channel loss: SP source alignment requires global CU metadata.")
            logger.warning_rank0("Channel loss: SP lacks global CU metadata; skipping this micro-batch.")
            return []
        if parallel_state.sp_size > 1 and not _declared_tail_matches_sources(
            global_points, len(source_ids), tail_padding_length
        ):
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
        if headwise_cp and parallel_state.sp_size > 1 and local_spans is None:
            if strict:
                raise ChannelLossMetadataError("Channel loss: CP requires rank-local packed CU metadata.")
            logger.warning_rank0("Channel loss: CP lacks rank-local packed CU metadata; skipping this micro-batch.")
            return []
        if (
            local_spans is None
            and parallel_state.sp_size > 1
            and (
                not _global_cu_positions_are_safe(
                    global_points, len(source_ids), seq_len * parallel_state.sp_size, tail_padding_length
                )
                or not _contiguous_starts_match_global(
                    global_points, positions_cpu, sp_rank, parallel_state.sp_size, tail_padding_length
                )
                or not _global_tail_is_padding(
                    global_points,
                    positions_cpu,
                    labels_flat,
                    sp_rank,
                    parallel_state.sp_size,
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
                multiple = 2 * parallel_state.sp_size
                expected_headwise = [
                    ((end - start + multiple - 1) // multiple) * 2
                    for start, end in zip(global_points, global_points[1:])
                ]
            else:
                expected_headwise = None
            if parallel_state.sp_size > 1 and global_points[-1] == seq_len * parallel_state.sp_size:
                expected_contiguous = _ulysses_local_span_lengths(
                    global_points, positions_2d.size(0), local_width, sp_rank, parallel_state.sp_size
                )
            else:
                expected_contiguous = None
            if parallel_state.sp_size == 1:
                expected_contiguous = [end - start for start, end in zip(global_points, global_points[1:])]
            headwise_valid = lengths == expected_headwise and _headwise_starts_match_positions(
                local_spans, positions_cpu.reshape(-1), sp_rank
            )
            contiguous_valid = lengths == expected_contiguous and (
                parallel_state.sp_size == 1
                or _contiguous_starts_match_global(
                    global_points, positions_cpu, sp_rank, parallel_state.sp_size, tail_padding_length
                )
            )
            geometry_candidates = int(headwise_valid) | (int(contiguous_valid) << 1)
            if (
                packed_global_physical_seq_len is not None
                and packed_global_physical_seq_len != seq_len * parallel_state.sp_size
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
                and (
                    global_points[-1] == seq_len * parallel_state.sp_size
                    or packed_global_physical_seq_len == seq_len * parallel_state.sp_size
                )
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
            return {
                "segments": segments,
                "device": per_token_loss.device,
                "geometry_candidates": geometry_candidates,
                "layout": (
                    "cu",
                    tuple(points) if headwise_cp and global_points is None else (),
                    tuple(global_points or ()),
                )
                if local_spans is not None
                else ("positions", tuple(global_points or ())),
            }

        reduced = ChannelLossComputer._reduce_sp(
            segments,
            source_ids,
            device=per_token_loss.device,
            parallel_state=parallel_state,
            strict=strict,
        )
        return [
            {
                "source_id": item["source_id"],
                "loss_sum": torch.as_tensor(item["loss_sum"], device=per_token_loss.device, dtype=torch.float32),
                "token_count": torch.as_tensor(item["token_count"], device=per_token_loss.device, dtype=torch.int64),
            }
            for item in reduced
        ]

    @staticmethod
    def _reduce_sp(
        local_segments: list[dict[str, Any]],
        source_ids: list[ChannelKey],
        device: torch.device,
        parallel_state: ParallelState | None = None,
        strict: bool = False,
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
                    local_integer_stats = torch.stack(
                        [local_token_counts, local_explicit_padding_flags, local_zero_position_flags], dim=1
                    )
                else:
                    local_loss_sums = torch.empty(0, dtype=torch.float32, device=device)
                    local_integer_stats = torch.empty((0, 3), dtype=torch.int64, device=device)

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
                        all_segments.append(
                            {
                                **metadata,
                                "loss_sum": gathered_loss_sums[rank_idx][segment_idx],
                                "token_count": integer_stats[0],
                                "has_explicit_padding_mask": integer_stats[1].to(torch.bool),
                                "has_only_zero_positions": integer_stats[2].to(torch.bool),
                            }
                        )

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
                result.append(
                    {
                        "source_id": source_ids[doc_idx],
                        "loss_sum": segment["loss_sum"],
                        "token_count": segment["token_count"],
                    }
                )
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
        self.computer._require_source_provenance = self.enabled
        self._strip_keys = set(self.config.source_id_keys)
        self._strip_keys.update(self.config.source_name_keys)
        self._strip_keys.update(self.config.extra_strip_keys)
        self.computer.strict = bool(self.config.strict)
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
        source_repeat: int = 1,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            self._collect_step = False
            return
        if source_repeat < 1:
            raise ValueError(f"Channel loss source_repeat must be >= 1, got {source_repeat}.")

        self.computer._dpo_segments = source_repeat > 1
        provenances = [
            batch.pop(CHANNEL_LOSS_PROVENANCE, None) if isinstance(batch, dict) else None
            for batch in micro_batches or []
        ]

        self._collect_step = state.global_step % self.config.interval == 0
        if not self._collect_step:
            self.computer.begin_step([], [], [])
            if self.config.strict:
                missing = any(not self._extract_metadata(batch)[0] for batch in micro_batches or [])
                self._require_source_ids([] if missing else [True])
            return

        per_mb_source_ids: list[list[ChannelKey]] = []
        per_mb_position_ids: list[torch.Tensor | None] = []
        per_mb_attention_masks: list[torch.Tensor | None] = []
        per_mb_packed_cu_seqlens: list[Any | None] = []
        per_mb_packed_global_cu_seqlens: list[Any | None] = []
        per_mb_packed_global_physical_seq_lens: list[int | None] = []
        per_mb_tail_padding_lengths: list[Any | None] = []
        per_mb_source_provenances: list[dict[str, Any] | None] = []
        missing = False
        for micro_batch, provenance in zip(micro_batches or [], provenances, strict=True):
            (
                source_ids,
                source_names,
                position_ids,
                attention_mask,
                packed_cu,
                global_cu,
                global_width,
                tail_padding,
            ) = self._extract_metadata(micro_batch)
            if self.config.strict:
                missing = missing or not source_ids
            if provenance is None:
                source_ids, source_names = self._repeat_source_metadata(source_ids, source_names, source_repeat)
            self.computer.record_source_names(source_ids, source_names)
            per_mb_source_ids.append(source_ids)
            per_mb_position_ids.append(position_ids)
            per_mb_attention_masks.append(attention_mask)
            per_mb_packed_cu_seqlens.append(packed_cu)
            per_mb_packed_global_cu_seqlens.append(global_cu)
            per_mb_packed_global_physical_seq_lens.append(global_width)
            per_mb_tail_padding_lengths.append(tail_padding)
            per_mb_source_provenances.append(provenance)

        if self.config.strict:
            self._require_source_ids([] if missing else [True])
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

    def _require_source_ids(self, source_ids: list[ChannelKey]) -> None:
        missing = not bool(source_ids)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor([int(missing)], dtype=torch.int32, device=_collective_device(dist.group.WORLD))
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            missing = bool(flag.item())
        if not missing:
            return
        raise ValueError(
            "train.channel_loss.enable=True but no configured source ID was found in a micro-batch. "
            f"Checked keys: {self.config.source_id_keys}."
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
        micro_batch.pop(CHANNEL_LOSS_PROVENANCE, None)
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
        grad_norm: float | torch.Tensor,
        **kwargs: Any,
    ) -> None:
        if not self._collect_step:
            return

        totals, source_names = self._reduce_step_totals(self.computer.step_totals)
        if not totals:
            return

        metrics = self._build_metrics(totals, source_names)
        if not metrics:
            return

        if getattr(self.trainer, "step_train_metrics", None) is None:
            self.trainer.step_train_metrics = {}
        if getattr(self.trainer, "step_env_metrics", None) is None:
            self.trainer.step_env_metrics = {}

        self.trainer.step_train_metrics.update(metrics)
        self.trainer.step_env_metrics.update(metrics)

    def _reduce_step_totals(
        self,
        local_totals: dict[ChannelKey, tuple[torch.Tensor, torch.Tensor]],
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
                    raise ChannelLossMetadataError(msg)
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
        registered_names = self._register_sources(source_ids, synchronized_names)
        return totals, registered_names

    def _register_sources(
        self,
        source_ids: list[ChannelKey],
        source_names: dict[ChannelKey, str] | None = None,
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
                raise ChannelLossMetadataError(msg)
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
            attention_mask = micro_batch.get("attention_mask")
            global_width = None
            router_mask, labels = micro_batch.get("router_attention_mask"), micro_batch.get("labels")
            if (
                isinstance(router_mask, torch.Tensor)
                and isinstance(labels, torch.Tensor)
                and router_mask.numel() == labels.numel()
            ):
                if isinstance(attention_mask, torch.Tensor):
                    global_width = attention_mask.numel()
                attention_mask = router_mask
            packed_cu = micro_batch.get("cu_seqlens_list_q")
            if packed_cu is None:
                packed_cu = micro_batch.get("cu_seq_lens_q")
            global_cu = micro_batch.get("linear_attn_cu_seqlens_list_q")
            if global_cu is None:
                global_cu = micro_batch.get("linear_attn_cu_seq_lens_q")
            return (
                source_ids,
                source_names,
                position_ids if isinstance(position_ids, torch.Tensor) else None,
                attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                packed_cu,
                global_cu,
                global_width,
                micro_batch.get("tail_padding_length"),
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
    parallel_state: ParallelState,
    ignore_index: int,
    *,
    strict: bool,
    defer_sp: bool,
) -> list[dict[str, Any]] | dict[str, Any]:
    points = provenance["source_cu"]
    rank, size = provenance["sp_rank"], provenance["sp_size"]
    width = labels.numel()
    lo, hi = rank * width, (rank + 1) * width
    segments = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        first, last = max(start, lo), min(end, hi)
        first = min(max(first, lo), hi)
        last = max(first, min(max(last, lo), hi))
        local_start, local_end = first - lo, last - lo
        selected = labels[local_start:local_end] != ignore_index
        zero = torch.zeros((), dtype=torch.bool, device=labels.device)
        segments.append(
            {
                "batch_idx": index,
                "sp_rank": rank,
                "local_order": 0,
                "is_start": rank == 0,
                "has_explicit_padding_mask": zero,
                "has_only_zero_positions": False,
                "loss_sum": per_token_loss[local_start:local_end][selected].sum(),
                "token_count": selected.sum(dtype=torch.int64),
            }
        )
    mask_digest = hashlib.sha256(provenance["input_mask"].to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()
    if defer_sp and size > 1:
        return {
            "segments": segments,
            "device": per_token_loss.device,
            "geometry_candidates": 2,
            "layout": (
                "source",
                tuple(points),
                provenance["tail"],
                size,
                mask_digest,
                provenance.get("global_cu"),
                provenance.get("dpo_cu"),
            ),
        }
    return ChannelLossComputer._reduce_sp(segments, source_ids, per_token_loss.device, parallel_state, strict)


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


def _packed_cu_points(value: Any | None) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            return None, "expected a one-dimensional int32/int64 tensor"
        points = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        points = list(value)
    else:
        return None, f"expected a tensor/list/tuple, got {type(value).__name__}"
    if not points or any(not isinstance(point, Integral) or isinstance(point, bool) for point in points):
        return None, "CU points must be a nonempty list of integers"
    if points[0] != 0 or any(end < start for start, end in zip(points, points[1:])):
        return None, "CU points must start at zero and be nondecreasing"
    return [int(point) for point in points], None


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
