# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import contextlib
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import wraps
from threading import Event as ThreadEvent
from threading import Lock
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import (
    _allowed_determinism_checks_to_fns,
    _checkpoint_hook,
    _CheckpointFrame,
    _get_autocast_kwargs,
    _get_device_module,
    _infer_device_type,
    _NoopSaveInputs,
    _recomputation_hook,
    _StopRecomputationError,
    detach_variable,
    get_device_states,
    noop_context_fn,
    set_device_states,
)

from ..utils import logging
from ..utils.device import IS_NPU_AVAILABLE
from .parallel_state import get_parallel_state


logger = logging.get_logger(__name__)


class _AsyncReplayLauncher:
    """Run one complete replay after a single non-blocking start signal."""

    def __init__(self, replay_once: Any, executor: ThreadPoolExecutor) -> None:
        self._replay_once = replay_once
        self._lock = Lock()
        self._started = ThreadEvent()
        self._completed = ThreadEvent()
        self.started = False
        self.done = False
        self.result = None
        self._error: Optional[BaseException] = None
        self._future = executor.submit(self._run)

    def start(self) -> None:
        with self._lock:
            if self.started:
                raise RuntimeError("ILP one-shot replay was started twice.")
            self.started = True
        self._started.set()

    def wait(self) -> Any:
        if not self.started:
            raise RuntimeError("ILP one-shot replay was joined before start.")
        with torch.autograd.profiler.record_function("ilp::launcher::wait::complete"):
            if not self._completed.wait(timeout=120):
                raise TimeoutError("ILP timed out waiting for one-shot replay completion.")
        if self._error is not None:
            raise self._error
        return self.result

    def release(self) -> None:
        if not self.done:
            raise RuntimeError("ILP one-shot replay launcher was released before completion.")
        with self._lock:
            self._replay_once = None
            self.result = None
            self._error = None
            self._future = None

    def _run(self) -> None:
        replay_once = None
        try:
            if not self._started.wait(timeout=120):
                raise TimeoutError("ILP launcher timed out waiting for one-shot start admission.")
            replay_once = self._replay_once
            self._replay_once = None
            self.result = replay_once()
        except BaseException as error:
            self._error = error
        finally:
            # Drop captured replay tensors before waking the joining autograd
            # thread. The persistent executor outlives every individual graph.
            replay_once = None
            self._replay_once = None
            self.done = True
            self._completed.set()


@dataclass
class _ResourceTicket:
    index: int
    owner: str
    predecessor: Optional["_ResourceTicket"] = None
    ready: ThreadEvent = field(default_factory=ThreadEvent)
    done: Any = None
    acquired: bool = False
    released: bool = False


class _DeviceEventFifo:
    """Serialize one physical resource while keeping waits on NPU streams."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        self._lock = Lock()
        self._tail: Optional[_ResourceTicket] = None
        self._next_index = 0

    def reserve(self, owner: str) -> _ResourceTicket:
        with self._lock:
            ticket = _ResourceTicket(
                index=self._next_index,
                owner=owner,
                predecessor=self._tail,
            )
            self._next_index += 1
            self._tail = ticket
        return ticket

    def acquire(self, ticket: _ResourceTicket) -> None:
        with self._lock:
            if ticket.acquired:
                raise RuntimeError(f"ILP {self.resource} ticket {ticket.index} was acquired twice.")
            ticket.acquired = True
        predecessor = ticket.predecessor
        if predecessor is None:
            return
        with torch.autograd.profiler.record_function(
            f"ilp::fifo::{self.resource}::{ticket.index}::{ticket.owner}::wait"
        ):
            if not predecessor.ready.wait(timeout=120):
                raise TimeoutError(
                    f"ILP timed out waiting for {self.resource} ticket {predecessor.index} ({predecessor.owner})."
                )
            if predecessor.done is None:
                raise RuntimeError(
                    f"ILP {self.resource} ticket {predecessor.index} was released without a device event."
                )
            torch.npu.current_stream().wait_event(predecessor.done)

    def release(self, ticket: _ResourceTicket) -> None:
        with self._lock:
            if not ticket.acquired:
                raise RuntimeError(f"ILP {self.resource} ticket {ticket.index} was released before acquire.")
            if ticket.released:
                raise RuntimeError(f"ILP {self.resource} ticket {ticket.index} was released twice.")
            ticket.released = True
        done = torch.npu.Event()
        done.record(torch.npu.current_stream())
        ticket.done = done
        ticket.ready.set()

    def release_completed(self) -> None:
        with self._lock:
            ticket = self._tail
            while ticket is not None:
                if not ticket.released:
                    raise RuntimeError(
                        f"ILP {self.resource} ticket {ticket.index} ({ticket.owner}) remained pending at replay join."
                    )
                ticket = ticket.predecessor
            self._tail = None


@dataclass
class _NativeBackwardSchedule:
    layer_id: int
    backward_experts_module: Optional[Any] = None
    gmm_count: int = 2
    backward_index: int = 0
    replay_index: int = 0
    lock: Any = field(default_factory=Lock)
    hccl_fifo: _DeviceEventFifo = field(default_factory=lambda: _DeviceEventFifo("hccl"))
    collective_tickets: dict[tuple[str, str], _ResourceTicket] = field(default_factory=dict)
    communication_active: bool = False
    replay_launcher: Optional[_AsyncReplayLauncher] = None
    replay_prefetch_active: bool = False
    replay_all_gather_ready: ThreadEvent = field(default_factory=ThreadEvent)
    replay_all_gather_done: Any = None
    replay_attention_ready: ThreadEvent = field(default_factory=ThreadEvent)
    replay_attention_done: Any = None
    replay_dispatch_ready: ThreadEvent = field(default_factory=ThreadEvent)
    replay_dispatch_done: Any = None
    backward_experts_ready: ThreadEvent = field(default_factory=ThreadEvent)
    backward_experts_done: Any = None
    replay_experts_ready: ThreadEvent = field(default_factory=ThreadEvent)
    replay_experts_done: Any = None
    backward_attention_ready: ThreadEvent = field(default_factory=ThreadEvent)
    backward_attention_done: Any = None
    current_reduce_scatter_ready: ThreadEvent = field(default_factory=ThreadEvent)
    current_reduce_scatter_done: Any = None
    device_notify_factory: Optional[Any] = None
    device_notify_dependencies: Optional[frozenset[str]] = None
    device_notifies: dict[str, Any] = field(default_factory=dict)
    native_stream_handle: Optional[int] = None
    replay_stream_handle: Optional[int] = None
    backward_experts_resharded: bool = False

    def attach_replay_launcher(self, launcher: _AsyncReplayLauncher) -> None:
        if self.replay_launcher is not None:
            raise RuntimeError(f"ILP B{self.layer_id} already owns an asynchronous replay launcher.")
        self.replay_launcher = launcher

    def start_replay(self) -> None:
        if self.replay_launcher is None:
            return
        with torch.autograd.profiler.record_function(
            f"ilp::launcher::B{self.layer_id}::Fprime{self.layer_id - 1}::start"
        ):
            self.replay_launcher.start()

    def wait_replay(self) -> Any:
        if self.replay_launcher is None:
            return None
        return self.replay_launcher.wait()

    def release_replay(self, launcher: _AsyncReplayLauncher) -> None:
        if self.replay_launcher is not launcher:
            raise RuntimeError(f"ILP B{self.layer_id} released a launcher it does not own.")
        self.replay_launcher = None
        self.collective_tickets.clear()
        self.hccl_fifo.release_completed()
        self.communication_active = False
        self.replay_all_gather_done = None
        self.replay_attention_done = None
        self.replay_dispatch_done = None
        self.backward_experts_done = None
        self.replay_experts_done = None
        self.backward_attention_done = None
        self.current_reduce_scatter_done = None
        self.backward_experts_module = None

    def close_device_notifies(self) -> None:
        for notify in self.device_notifies.values():
            notify.close()
        self.device_notifies.clear()

    def bind_streams(self, native_stream: Any, replay_stream: Any) -> None:
        self.native_stream_handle = int(native_stream.npu_stream)
        self.replay_stream_handle = int(replay_stream.npu_stream)
        if (
            self.native_stream_handle != self.replay_stream_handle
            and self.device_notify_factory is not None
            and not self.device_notifies
        ):
            names = {
                "replay_attention": f"Fprime{self.layer_id - 1}:attention-to-B{self.layer_id}:combine",
                "replay_dispatch": f"Fprime{self.layer_id - 1}:dispatch-to-B{self.layer_id}:dispatch",
                "backward_experts": f"B{self.layer_id}:experts-to-Fprime{self.layer_id - 1}:experts",
                "replay_experts": f"Fprime{self.layer_id - 1}:experts-to-B{self.layer_id}:dispatch",
                "backward_attention": f"B{self.layer_id}:attention-to-Fprime{self.layer_id - 1}:combine",
                "current_reduce_scatter": (f"B{self.layer_id}:reduce-scatter-to-Fprime{self.layer_id - 1}:all-gather"),
            }
            self.device_notifies = {
                dependency: self.device_notify_factory(name)
                for dependency, name in names.items()
                if self.device_notify_dependencies is None or dependency in self.device_notify_dependencies
            }

    def on_phase(self, phase: str) -> None:
        with torch.autograd.profiler.record_function(f"ilp::B{self.layer_id}::phase::{phase}"):
            if not self.communication_active:
                return
            if phase == "combine":
                self._wait_for_dependency(
                    "replay_attention",
                    self.replay_attention_ready,
                    lambda: self.replay_attention_done,
                    f"Fprime{self.layer_id - 1} replay attention",
                )
            elif phase == "experts":
                self.reshard_backward_experts()
                self.publish_backward_experts_done()
                self._wait_for_dependency(
                    "replay_dispatch",
                    self.replay_dispatch_ready,
                    lambda: self.replay_dispatch_done,
                    f"Fprime{self.layer_id - 1} replay dispatch materialization",
                )
            elif phase == "dispatch":
                self._wait_for_dependency(
                    "replay_experts",
                    self.replay_experts_ready,
                    lambda: self.replay_experts_done,
                    f"Fprime{self.layer_id - 1} replay experts",
                )
            elif phase == "attention":
                self.publish_backward_attention_done()

    def reshard_backward_experts(self) -> None:
        if self.backward_experts_resharded or self.backward_experts_module is None:
            return
        with torch.autograd.profiler.record_function(f"ilp::B{self.layer_id}::experts_reshard"):
            self.backward_experts_module.reshard()
        self.backward_experts_resharded = True

    def mark_backward_started(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function(f"ilp::joint::B{self.layer_id}::backward_start"):
            pass
        return grad

    def mark_backward_done(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function(f"ilp::joint::B{self.layer_id}::backward_done"):
            pass
        return grad

    def before_replay_phase(self, phase: str) -> None:
        if phase not in ("attention", "dispatch", "experts", "combine"):
            raise ValueError(f"Unsupported ILP replay phase: {phase}.")
        with torch.autograd.profiler.record_function(f"ilp::joint::Fprime{self.layer_id - 1}::{phase}::launcher"):
            if phase == "experts":
                self._wait_for_dependency(
                    "backward_experts",
                    self.backward_experts_ready,
                    lambda: self.backward_experts_done,
                    f"B{self.layer_id} backward experts",
                )
            elif phase == "combine":
                self._wait_for_dependency(
                    "backward_attention",
                    self.backward_attention_ready,
                    lambda: self.backward_attention_done,
                    f"B{self.layer_id} backward attention",
                )

    def after_replay_phase(self, phase: str) -> None:
        if phase not in ("attention", "dispatch", "experts", "combine"):
            raise ValueError(f"Unsupported completed ILP replay phase: {phase}.")
        with torch.autograd.profiler.record_function(f"ilp::joint::Fprime{self.layer_id - 1}::{phase}::done"):
            if phase == "attention":
                self.publish_replay_attention_done()
            elif phase == "experts":
                self.publish_replay_experts_done()

    def publish_replay_attention_done(self, done: Any = None) -> None:
        if self._record_dependency("replay_attention"):
            return
        with self.lock:
            if self.replay_attention_ready.is_set():
                return
            if done is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            self.replay_attention_done = done
            self.replay_attention_ready.set()

    def publish_replay_dispatch_done(self, done: Any = None) -> None:
        if self._record_dependency("replay_dispatch"):
            return
        with self.lock:
            if self.replay_dispatch_ready.is_set():
                return
            if done is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            self.replay_dispatch_done = done
            self.replay_dispatch_ready.set()

    def publish_backward_experts_done(self, done: Any = None) -> None:
        if self._record_dependency("backward_experts"):
            return
        with self.lock:
            if self.backward_experts_ready.is_set():
                return
            if done is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            self.backward_experts_done = done
            self.backward_experts_ready.set()

    def publish_replay_experts_done(self, done: Any = None) -> None:
        if self._record_dependency("replay_experts"):
            return
        with self.lock:
            if self.replay_experts_ready.is_set():
                return
            if done is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            self.replay_experts_done = done
            self.replay_experts_ready.set()

    def publish_backward_attention_done(self, done: Any = None) -> None:
        if self._record_dependency("backward_attention"):
            return
        with self.lock:
            if self.backward_attention_ready.is_set():
                return
            if done is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            self.backward_attention_done = done
            self.backward_attention_ready.set()

    def activate_communication_tickets(self) -> None:
        self.communication_active = True

    def begin_replay_prefetch(self) -> None:
        self.replay_prefetch_active = True

    def end_replay_prefetch(self) -> None:
        self.replay_prefetch_active = False

    def fsdp_role(self, layer_id: int) -> Optional[str]:
        if layer_id == self.layer_id:
            return "current"
        if layer_id == self.layer_id - 1:
            return "replay"
        return None

    def before_fsdp_collective(self, role: Optional[str], phase: str) -> tuple[str, str, bool] | None:
        if not self.communication_active:
            return None
        if role is None:
            return None
        is_replay_prefetch = role == "replay" and phase == "all_gather" and self.replay_prefetch_active
        if role == "current" and phase == "reduce_scatter":
            key = ("backward", "reduce_scatter")
            resource_ticket = self.collective_tickets.get(key)
            if resource_ticket is None:
                resource_ticket = self.hccl_fifo.reserve(f"B{self.layer_id}:reduce_scatter")
                self.collective_tickets[key] = resource_ticket
            self.hccl_fifo.acquire(resource_ticket)
        if role == "replay" and phase == "all_gather" and not is_replay_prefetch:
            with torch.autograd.profiler.record_function(
                f"ilp::fsdp::F{self.layer_id - 1}::all_gather::wait_B{self.layer_id}_reduce_scatter"
            ):
                self._wait_for_dependency(
                    "current_reduce_scatter",
                    self.current_reduce_scatter_ready,
                    lambda: self.current_reduce_scatter_done,
                    f"B{self.layer_id} FSDP reduce-scatter",
                )
        return role, phase, is_replay_prefetch

    def after_fsdp_collective(self, ticket: tuple[str, str, bool] | None) -> None:
        if ticket is None:
            return
        role, phase, is_replay_prefetch = ticket
        if role == "replay" and phase == "all_gather" and is_replay_prefetch:
            done = torch.npu.Event()
            done.record(torch.npu.current_stream())
            self.replay_all_gather_done = done
            self.replay_all_gather_ready.set()
        elif role == "current" and phase == "reduce_scatter":
            resource_ticket = self.collective_tickets.pop(("backward", "reduce_scatter"), None)
            if resource_ticket is None:
                done = torch.npu.Event()
                done.record(torch.npu.current_stream())
            else:
                self.hccl_fifo.release(resource_ticket)
                done = resource_ticket.done
            if not self._record_dependency("current_reduce_scatter"):
                self.current_reduce_scatter_done = done
                self.current_reduce_scatter_ready.set()
        elif role == "replay" and phase == "reduce_scatter":
            self.communication_active = False

    def before_current_forward_dispatch(self) -> None:
        if not self.communication_active:
            return
        if not self.replay_all_gather_ready.is_set():
            with torch.autograd.profiler.record_function(
                f"ilp::fsdp::F{self.layer_id}::dispatch::F{self.layer_id - 1}_all_gather_not_needed"
            ):
                pass
            return
        with torch.autograd.profiler.record_function(
            f"ilp::fsdp::F{self.layer_id}::dispatch::wait_F{self.layer_id - 1}_all_gather"
        ):
            all_gather_done = self._wait_for_event(
                self.replay_all_gather_ready,
                lambda: self.replay_all_gather_done,
                f"F{self.layer_id - 1} replay FSDP all-gather",
            )
            torch.npu.current_stream().wait_event(all_gather_done)

    def _prepare_complementary_collective_order(self) -> None:
        owners = (
            (("backward", "combine"), f"B{self.layer_id}:combine"),
            (("replay", "dispatch"), f"Fprime{self.layer_id - 1}:dispatch"),
            (("backward", "dispatch"), f"B{self.layer_id}:dispatch"),
            (("replay", "combine"), f"Fprime{self.layer_id - 1}:combine"),
            (("backward", "reduce_scatter"), f"B{self.layer_id}:reduce_scatter"),
        )
        for key, owner in owners:
            self.collective_tickets[key] = self.hccl_fifo.reserve(owner)
        with torch.autograd.profiler.record_function("ilp::admission::four_stage_complementary"):
            pass

    def on_collective(self, phase: str, boundary: str) -> None:
        with torch.autograd.profiler.record_function(f"ilp::comm::B{self.layer_id}::{phase}::{boundary}"):
            if not self.communication_active:
                return
            key = ("backward", phase)
            if boundary == "before":
                if phase == "combine":
                    self._prepare_complementary_collective_order()
                ticket = self.collective_tickets.get(key)
                if ticket is None:
                    ticket = self.hccl_fifo.reserve(f"B{self.layer_id}:{phase}")
                    self.collective_tickets[key] = ticket
                self.hccl_fifo.acquire(ticket)
            elif boundary == "after":
                ticket = self.collective_tickets.pop(key)
                self.hccl_fifo.release(ticket)
                if phase == "combine":
                    self.start_replay()
            else:
                raise ValueError(f"Unsupported ILP collective boundary: {boundary}.")

    def before_replay_collective(self, phase: str) -> None:
        if not self.communication_active:
            return
        if phase not in ("dispatch", "combine"):
            raise ValueError(f"Unsupported ILP replay collective phase: {phase}.")
        key = ("replay", phase)
        ticket = self.collective_tickets[key]
        self.hccl_fifo.acquire(ticket)

    def after_replay_collective(self, phase: str) -> None:
        if not self.communication_active:
            return
        ticket = self.collective_tickets.pop(("replay", phase))
        self.hccl_fifo.release(ticket)
        if phase not in ("dispatch", "combine"):
            raise ValueError(f"Unsupported completed ILP replay collective phase: {phase}.")

    def before_backward_gmm(self) -> int:
        with self.lock:
            ticket = self.backward_index
            self.backward_index += 1
        self._check_ticket(ticket, "backward")
        return ticket

    def after_backward_gmm(self, ticket: int) -> None:
        self._check_ticket(ticket, "backward")
        if not self.communication_active:
            return
        if ticket == self.gmm_count - 1:
            with torch.autograd.profiler.record_function(f"ilp::B{self.layer_id}::experts_complete"):
                pass

    def before_replay_gmm(self) -> int:
        with self.lock:
            ticket = self.replay_index
            self.replay_index += 1
        self._check_ticket(ticket, "replay")
        return ticket

    def after_replay_gmm(self, ticket: int) -> None:
        self._check_ticket(ticket, "replay")

    def _check_ticket(self, ticket: int, owner: str) -> None:
        if ticket >= self.gmm_count:
            raise RuntimeError(
                f"ILP expected {self.gmm_count} {owner} GMM calls for B{self.layer_id}, got index {ticket}."
            )

    def _wait_for_dependency(
        self,
        dependency: str,
        ready: ThreadEvent,
        event_getter: Any,
        name: str,
    ) -> None:
        notify = self._device_notify(dependency)
        if notify is not None:
            notify.wait(torch.npu.current_stream())
            return
        event = self._wait_for_event(ready, event_getter, name)
        torch.npu.current_stream().wait_event(event)

    def _record_dependency(self, dependency: str) -> bool:
        notify = self._device_notify(dependency)
        if notify is None:
            return False
        notify.record(torch.npu.current_stream())
        return True

    def _device_notify(self, dependency: str) -> Any:
        if self.native_stream_handle is not None and self.native_stream_handle == self.replay_stream_handle:
            return None
        return self.device_notifies.get(dependency)

    @staticmethod
    def _wait_for_event(ready: ThreadEvent, event_getter: Any, name: str) -> Any:
        if not ready.wait(timeout=120):
            raise TimeoutError(f"ILP timed out waiting for {name} event ticket.")
        event = event_getter()
        if event is None:
            raise RuntimeError(f"ILP {name} event ticket was signaled without an event.")
        return event


class _TicketedFSDPComm:
    def __init__(self, inner: Any, schedule_getter: Any, layer_id: int, phase: str) -> None:
        self._inner = inner
        self._schedule_getter = schedule_getter
        self._layer_id = layer_id
        self._phase = phase

    def allocate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._inner.allocate(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        schedule = self._schedule_getter()
        role = schedule.fsdp_role(self._layer_id) if schedule is not None else None
        ticket = schedule.before_fsdp_collective(role, self._phase) if schedule is not None else None
        marker = (
            torch.autograd.profiler.record_function(f"ilp::fsdp::{role}::{self._phase}")
            if ticket is not None
            else contextlib.nullcontext()
        )
        with marker:
            result = self._inner(*args, **kwargs)
        if schedule is not None:
            schedule.after_fsdp_collective(ticket)
        return result


@dataclass
class _ReplayCheckpointContext:
    preserve_rng_state: bool
    device: str
    device_autocast_kwargs: dict[str, Any]
    cpu_autocast_kwargs: dict[str, Any]
    fwd_cpu_state: Any = None
    had_device_in_fwd: bool = False
    fwd_devices: Any = None
    fwd_device_states: Any = None


class ReplayState(Enum):
    EMPTY = auto()
    ATTN_READY = auto()
    DISPATCH_PENDING = auto()
    DISPATCH_READY = auto()
    EXPERT_READY = auto()
    COMBINE_PENDING = auto()
    GRAPH_READY = auto()
    CONSUMED = auto()


@dataclass(frozen=True)
class ReplayMemorySnapshot:
    allocated_bytes: int
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class ReplayMemoryDecision:
    allowed: bool
    resumed: bool
    reason: str
    snapshot: ReplayMemorySnapshot
    estimated_bytes: int
    slots: int


class ReplayMemoryBudget:
    def __init__(
        self,
        budget_bytes: int = 0,
        reserve_bytes: int = 0,
        safety_factor: float = 1.1,
        probe: Optional[Any] = None,
        consensus: Optional[Any] = None,
    ) -> None:
        self.budget_bytes = budget_bytes
        self.reserve_bytes = reserve_bytes
        self.safety_factor = safety_factor
        self._probe = probe or self._device_probe
        self._consensus = consensus
        self.estimated_replay_bytes = 0
        self.paused = False

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0 or self.reserve_bytes > 0

    @staticmethod
    def _device_probe() -> ReplayMemorySnapshot:
        free_bytes, total_bytes = torch.npu.mem_get_info()
        return ReplayMemorySnapshot(
            allocated_bytes=int(torch.npu.memory_allocated()),
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
        )

    def observe_replay_bytes(self, replay_bytes: int) -> None:
        self.estimated_replay_bytes = max(self.estimated_replay_bytes, max(0, replay_bytes))

    def decide(self) -> ReplayMemoryDecision:
        return self.plan_slots(1)

    def plan_slots(self, max_slots: int) -> ReplayMemoryDecision:
        snapshot = self._probe()
        estimated_bytes = self.estimated_replay_bytes
        if self._consensus is not None:
            snapshot, estimated_bytes = self._consensus(snapshot, estimated_bytes)
        required_bytes = int(estimated_bytes * self.safety_factor)

        reason = "within_budget"
        if required_bytes == 0:
            slots = min(max_slots, 1)
            if self.budget_bytes > 0 and snapshot.allocated_bytes >= self.budget_bytes:
                slots = 0
                reason = "allocated_budget"
            elif self.reserve_bytes > 0 and snapshot.free_bytes <= self.reserve_bytes:
                slots = 0
                reason = "free_reserve"
        else:
            slots = max_slots
            if self.budget_bytes > 0:
                budget_margin = max(0, self.budget_bytes - snapshot.allocated_bytes)
                slots = min(slots, budget_margin // required_bytes)
                if slots == 0:
                    reason = "allocated_budget"
            if self.reserve_bytes > 0:
                reserve_margin = max(0, snapshot.free_bytes - self.reserve_bytes)
                slots = min(slots, reserve_margin // required_bytes)
                if slots == 0 and reason == "within_budget":
                    reason = "free_reserve"

        allowed = slots > 0
        resumed = allowed and self.paused
        self.paused = not allowed
        return ReplayMemoryDecision(
            allowed=allowed,
            resumed=resumed,
            reason=reason,
            snapshot=snapshot,
            estimated_bytes=required_bytes,
            slots=int(slots),
        )


@dataclass
class ReplayFrame:
    layer_id: int
    layer: torch.nn.Module
    run_function: Any
    checkpoint_ctx: Any
    state: ReplayState = ReplayState.EMPTY
    detached_inputs: Optional[tuple[Any, ...]] = None
    attention_hidden: Optional[torch.Tensor] = None
    moe_input: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None
    selected_experts: Optional[torch.Tensor] = None
    dispatch_plan: Any = None
    dispatch_input: Any = None
    dispatch_state: Any = None
    dispatched_input: Optional[torch.Tensor] = None
    expert_output: Optional[torch.Tensor] = None
    combine_state: Any = None
    replay_output: Optional[torch.Tensor] = None
    fsdp_unshard_handle: Any = None
    memory_before_prepare: int = 0
    execution_stream: Any = None
    execution_group: Any = None
    native_checkpoint_frame: Any = None
    backward_schedule: Optional[_NativeBackwardSchedule] = None
    native_graph_task_id: Optional[int] = None
    inner_recompute_context: Any = None
    replay_cpu_state: Any = None
    replay_devices: Any = None
    replay_device_states: Any = None

    def materialize_inputs(self) -> tuple[Any, ...]:
        if self.detached_inputs is not None:
            return self.detached_inputs
        if self.native_checkpoint_frame is None:
            raise RuntimeError(f"ILP replay layer {self.layer_id} has no native checkpoint frame.")
        input_saver = self.native_checkpoint_frame.input_saver.grad_fn
        _, *inputs = input_saver.get_args(input_saver.saved_tensors)
        self.detached_inputs = detach_variable(tuple(inputs))
        return self.detached_inputs

    def release_after_dispatch(self) -> None:
        self.selected_experts = None
        self.dispatch_plan = None
        self.dispatch_input = None
        if self.dispatch_state is not None:
            self.dispatch_state.output = None

    def release_after_experts(self) -> None:
        self.dispatched_input = None

    def release_after_combine(self) -> None:
        self.attention_hidden = None
        self.moe_input = None
        self.routing_weights = None
        self.dispatch_state = None
        self.expert_output = None
        self.combine_state = None

    def release_after_recompute_capture(self) -> None:
        self.replay_output = None

    def record_recomputed_tensors(self, stream: Any) -> None:
        if self.native_checkpoint_frame is None or self.native_graph_task_id is None:
            raise RuntimeError(f"ILP replay layer {self.layer_id} has no completed checkpoint graph.")
        recomputed = self.native_checkpoint_frame.recomputed.get(self.native_graph_task_id)
        if recomputed is None:
            raise RuntimeError(f"ILP replay layer {self.layer_id} has no saved recomputed tensors.")
        for tensor in list(recomputed.values()):
            tensor.record_stream(stream)

    def release_sidecar_graph(self) -> None:
        self.attention_hidden = None
        self.moe_input = None
        self.routing_weights = None
        self.selected_experts = None
        self.dispatch_plan = None
        self.dispatch_input = None
        self.dispatch_state = None
        self.dispatched_input = None
        self.expert_output = None
        self.combine_state = None
        self.replay_output = None
        self.fsdp_unshard_handle = None
        self.memory_before_prepare = 0

    def release_cached_graph(self) -> None:
        # Break the native-frame -> recompute closure -> ReplayFrame cycle only
        # after this layer's native backward has consumed the replay graph.
        self.release_sidecar_graph()
        self.detached_inputs = None
        self.native_checkpoint_frame = None
        self.backward_schedule = None
        self.native_graph_task_id = None
        self.inner_recompute_context = None
        self.checkpoint_ctx = None
        self.replay_cpu_state = None
        self.replay_devices = None
        self.replay_device_states = None

    @contextlib.contextmanager
    def recompute_context(self):
        ctx = self.checkpoint_ctx
        rng_devices = ctx.fwd_devices if ctx.preserve_rng_state and ctx.had_device_in_fwd else []
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=ctx.preserve_rng_state,
            device_type=ctx.device,
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            device_autocast = (
                torch.amp.autocast(device_type=ctx.device, **ctx.device_autocast_kwargs)
                if torch.amp.is_autocast_available(ctx.device)
                else contextlib.nullcontext()
            )
            inner_context = self.inner_recompute_context or contextlib.nullcontext()
            with (
                torch.enable_grad(),
                device_autocast,
                torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs),
                inner_context,
            ):
                yield

    @contextlib.contextmanager
    def phased_recompute_context(self):
        """Restore replay RNG per phase without leaking its state into B(current)."""
        ctx = self.checkpoint_ctx
        rng_devices = ctx.fwd_devices if ctx.preserve_rng_state and ctx.had_device_in_fwd else []
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=ctx.preserve_rng_state,
            device_type=ctx.device,
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(self.replay_cpu_state if self.replay_cpu_state is not None else ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    if self.replay_device_states is None:
                        set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
                    else:
                        set_device_states(self.replay_devices, self.replay_device_states)
            device_autocast = (
                torch.amp.autocast(device_type=ctx.device, **ctx.device_autocast_kwargs)
                if torch.amp.is_autocast_available(ctx.device)
                else contextlib.nullcontext()
            )
            inner_context = self.inner_recompute_context or contextlib.nullcontext()
            with (
                torch.enable_grad(),
                device_autocast,
                torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs),
                inner_context,
            ):
                yield
            if ctx.preserve_rng_state:
                self.replay_cpu_state = torch.get_rng_state()
                if ctx.had_device_in_fwd:
                    self.replay_devices, self.replay_device_states = get_device_states(*self.materialize_inputs())


class InterLayerReplayController:
    """Rolling Qwen3-MoE replay scheduler for the NPU piercing run."""

    def __init__(
        self,
        current_layer: int,
        window_size: int = 1,
        memory_budget_gb: float = 0.0,
        memory_reserve_gb: float = 0.0,
        memory_safety_factor: float = 1.1,
        memory_retry_steps: int = 4,
        memory_probe: Optional[Any] = None,
        primary_ep_group: Optional[Any] = None,
        alternate_ep_group: Optional[Any] = None,
        strict: bool = True,
        layer_modules: Optional[dict[int, torch.nn.Module]] = None,
    ) -> None:
        self.current_layer = current_layer
        self.window_size = window_size
        self.bottom_layer = current_layer - window_size
        self.strict = strict
        self._layer_modules = layer_modules or {}
        self._frames: dict[int, ReplayFrame] = {}
        self._pending_dispatch_plans: dict[int, Any] = {}
        self._primary_ep_group = primary_ep_group
        self._alternate_ep_group = alternate_ep_group
        self._primary_stream = None
        self._alternate_stream = None
        self._device_index = None
        self._launcher_executor: Optional[ThreadPoolExecutor] = None
        self._native_sidecar_launcher: Optional[_AsyncReplayLauncher] = None
        self._native_sidecar_frame: Optional[ReplayFrame] = None
        self._native_sidecar_schedule: Optional[_NativeBackwardSchedule] = None
        self._native_schedule: Optional[_NativeBackwardSchedule] = None
        self._completion_tail = None
        self._retired_device_schedules: list[tuple[Any, _NativeBackwardSchedule]] = []
        self._device_notify_factory = None
        self._device_notify_dependencies = None
        if os.environ.get("VEOMNI_ILP_DEVICE_NOTIFY", "0") == "1":
            from .device_notify import DeviceNotify

            self._device_notify_factory = DeviceNotify
            dependency_spec = os.environ.get("VEOMNI_ILP_DEVICE_NOTIFY_DEPENDENCIES", "all")
            if dependency_spec == "none":
                self._device_notify_dependencies = frozenset()
            elif dependency_spec != "all":
                self._device_notify_dependencies = frozenset(dependency_spec.split(","))
        gib = 1024**3
        self._memory_group = None
        if (
            memory_probe is None
            and (memory_budget_gb > 0 or memory_reserve_gb > 0)
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            self._memory_group = dist.new_group(backend="gloo")
        self.memory_budget = ReplayMemoryBudget(
            budget_bytes=int(memory_budget_gb * gib),
            reserve_bytes=int(memory_reserve_gb * gib),
            safety_factor=memory_safety_factor,
            probe=memory_probe,
            consensus=self._distributed_memory_consensus if memory_probe is None else None,
        )
        self.memory_retry_steps = memory_retry_steps
        self._memory_retry_remaining = 0
        self._memory_slots = 0

    def capture_dispatch_plan(self, layer_id: int, plan: Any) -> None:
        self._pending_dispatch_plans[layer_id] = plan

    def checkpoint_current(self, checkpoint_func: Any, function: Any, *args: Any, **kwargs: Any) -> Any:
        from ..ops.kernels.attention.backward_boundary import attention_backward_phase_callback
        from ..ops.kernels.moe._kernels.kernel.npu_group_gemm import gmm_backward_interleave
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_launch_callback
        from .moe.comm import moe_backward_collective_callback, moe_backward_phase_callback

        self._reap_device_notifies()
        schedule = self._new_backward_schedule(self.current_layer)
        self._native_schedule = schedule
        checkpoint_keywords = dict(getattr(checkpoint_func, "keywords", None) or {})
        native_context_fn = kwargs.pop("context_fn", checkpoint_keywords.get("context_fn"))
        if native_context_fn is None:
            native_context_fn = noop_context_fn
        sidecar_launched = False

        def context_fn() -> tuple[Any, Any]:
            return native_context_fn()

        def launch_sidecar_at_dispatch() -> None:
            nonlocal sidecar_launched
            if torch._C._current_graph_task_id() < 0 or sidecar_launched:
                return
            sidecar_launched = True
            if self._launch_native_current_sidecar(schedule):
                schedule.before_current_forward_dispatch()

        def run_with_phase_callback(*run_args: Any, **run_kwargs: Any) -> Any:
            with (
                attention_backward_phase_callback(schedule.on_phase),
                moe_backward_phase_callback(schedule.on_phase),
                moe_backward_collective_callback(schedule.on_collective),
                gmm_backward_interleave(schedule),
                npu_ep_dispatch_launch_callback(launch_sidecar_at_dispatch),
            ):
                return function(*run_args, **run_kwargs)

        output = checkpoint_func(run_with_phase_callback, *args, context_fn=context_fn, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("ILP native current checkpoint requires a decoder layer to return one Tensor.")
        output.register_hook(schedule.mark_backward_started)
        if not args or not isinstance(args[0], torch.Tensor) or not args[0].requires_grad:
            raise RuntimeError("ILP native current checkpoint requires a grad-enabled hidden-state input.")
        args[0].register_hook(schedule.mark_backward_done)
        return output

    def checkpoint_native_replay(
        self,
        checkpoint_func: Any,
        layer_id: int,
        layer: torch.nn.Module,
        function: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from ..ops.kernels.attention.backward_boundary import attention_backward_phase_callback
        from ..ops.kernels.moe._kernels.kernel.npu_group_gemm import gmm_backward_interleave
        from .moe.comm import moe_backward_collective_callback, moe_backward_phase_callback

        # Repeated control collectives can interfere with the rolling FSDP tail;
        # lower pairs retain the deterministic replay-first policy.
        schedule = self._new_backward_schedule(layer_id)
        checkpoint_keywords = dict(getattr(checkpoint_func, "keywords", None) or {})
        kwargs.pop("use_reentrant", None)
        preserve_rng_state = bool(
            kwargs.pop("preserve_rng_state", checkpoint_keywords.get("preserve_rng_state", True))
        )
        context_fn = kwargs.pop("context_fn", checkpoint_keywords.get("context_fn", noop_context_fn))
        if context_fn is None:
            context_fn = noop_context_fn
        determinism_check = kwargs.pop("determinism_check", checkpoint_keywords.get("determinism_check", "default"))
        early_stop = bool(kwargs.pop("early_stop", checkpoint_keywords.get("early_stop", True)))
        debug = bool(kwargs.pop("debug", checkpoint_keywords.get("debug", False)))
        if kwargs:
            raise TypeError(f"Unexpected native replay checkpoint kwargs: {sorted(kwargs)}")
        if debug:
            raise ValueError("ILP native replay checkpoint does not support checkpoint debug mode.")
        if determinism_check not in _allowed_determinism_checks_to_fns:
            raise ValueError(f"Unsupported checkpoint determinism check: {determinism_check}")

        device = _infer_device_type(*args)
        device_module = _get_device_module(device)
        device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device)
        forward_context, recompute_context = context_fn()
        checkpoint_ctx = _ReplayCheckpointContext(
            preserve_rng_state=preserve_rng_state,
            device=device,
            device_autocast_kwargs=device_autocast_kwargs,
            cpu_autocast_kwargs=cpu_autocast_kwargs,
        )
        if preserve_rng_state:
            checkpoint_ctx.fwd_cpu_state = torch.get_rng_state()
            if getattr(device_module, "_initialized", False):
                checkpoint_ctx.had_device_in_fwd = True
                checkpoint_ctx.fwd_devices, checkpoint_ctx.fwd_device_states = get_device_states(*args)

        replay_holder: list[ReplayFrame] = []

        def run_with_phase_callback(*run_args: Any, **run_kwargs: Any) -> Any:
            if layer_id == self.bottom_layer:
                return function(*run_args, **run_kwargs)
            with (
                attention_backward_phase_callback(schedule.on_phase),
                moe_backward_phase_callback(schedule.on_phase),
                moe_backward_collective_callback(schedule.on_collective),
                gmm_backward_interleave(schedule),
            ):
                return function(*run_args, **run_kwargs)

        def recompute_fn(*packed_inputs: Any) -> None:
            packed_kwargs, *run_args = packed_inputs
            replay = replay_holder[0]
            with replay.recompute_context():
                function(*run_args, **packed_kwargs)

        native_frame = _CheckpointFrame(
            recompute_fn,
            early_stop,
            None,
            _allowed_determinism_checks_to_fns[determinism_check],
        )
        dummy = torch.empty((0,), requires_grad=True)
        native_frame.input_saver = _NoopSaveInputs.apply(dummy, {}, *args)
        if native_frame.input_saver.grad_fn is None:
            return function(*args)

        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_plan_callback

        with (
            _checkpoint_hook(native_frame),
            forward_context,
            npu_ep_dispatch_plan_callback(lambda plan: self.capture_dispatch_plan(layer_id, plan)),
        ):
            output = run_with_phase_callback(*args)
        native_frame.forward_completed = True
        if not isinstance(output, torch.Tensor):
            raise TypeError("ILP native replay checkpoint requires a decoder layer to return one Tensor.")

        replay = ReplayFrame(
            layer_id=layer_id,
            layer=layer,
            run_function=function,
            checkpoint_ctx=checkpoint_ctx,
            native_checkpoint_frame=native_frame,
            backward_schedule=schedule,
            inner_recompute_context=recompute_context,
        )
        replay_holder.append(replay)
        self.register(replay)

        def join_sidecar(grad: torch.Tensor) -> torch.Tensor:
            self._join_native_current_sidecar(replay, torch._C._current_graph_task_id())
            if layer_id > self.bottom_layer:
                self._launch_native_current_sidecar(schedule)
            return grad

        def release_after_backward(grad: torch.Tensor) -> torch.Tensor:
            self._consume_frame(replay)
            return grad

        output.register_hook(join_sidecar)
        if not args or not isinstance(args[0], torch.Tensor) or not args[0].requires_grad:
            raise RuntimeError("ILP native replay checkpoint requires a grad-enabled hidden-state input.")
        args[0].register_hook(release_after_backward)
        return output

    def register(self, frame: ReplayFrame) -> None:
        existing = self._frames.get(frame.layer_id)
        if existing is not None and existing.state is not ReplayState.CONSUMED:
            raise RuntimeError(f"ILP frame for layer {frame.layer_id} was overwritten before backward completed.")
        self._frames[frame.layer_id] = frame
        frame.dispatch_plan = self._pending_dispatch_plans.pop(frame.layer_id, None)

    def _consume_frame(self, frame: ReplayFrame) -> None:
        if frame.state is ReplayState.CONSUMED:
            return
        frame.state = ReplayState.CONSUMED
        frame.release_cached_graph()
        if self._frames.get(frame.layer_id) is frame:
            self._frames.pop(frame.layer_id)

    def _prefetch_previous(
        self,
        layer_id: int,
        wait_for_stream: Any = None,
        schedule: Optional[_NativeBackwardSchedule] = None,
    ) -> Optional[ReplayFrame]:
        replay_layer = layer_id - 1
        if replay_layer < self.bottom_layer:
            return None
        if not self._memory_allows_replay(layer_id, replay_layer):
            return None
        replay = self._frames.get(replay_layer)
        if replay is None:
            if self.strict:
                raise RuntimeError(f"ILP layer {replay_layer} frame is unavailable when B{layer_id} starts.")
            return None
        self._assign_execution_domain(replay)
        if wait_for_stream is not None:
            replay.execution_stream.wait_stream(wait_for_stream)
        with torch.npu.stream(replay.execution_stream):
            with torch.autograd.profiler.record_function(f"ilp::Fprime{replay_layer}::fsdp_prefetch_launch"):
                if schedule is not None:
                    schedule.begin_replay_prefetch()
                try:
                    replay.fsdp_unshard_handle = replay.layer.unshard(async_op=True)
                finally:
                    if schedule is not None:
                        schedule.end_replay_prefetch()
        return replay

    def _launch_native_current_sidecar(self, schedule: _NativeBackwardSchedule) -> bool:
        if self._native_sidecar_launcher is not None:
            raise RuntimeError("ILP native-current sidecar was launched twice before its join.")

        self._wait_for_completion_tail()
        self._reap_device_notifies()
        self._native_schedule = schedule
        schedule.activate_communication_tickets()
        current_stream = torch.npu.current_stream()
        replay = self._prefetch_previous(schedule.layer_id, wait_for_stream=current_stream, schedule=schedule)
        if replay is None:
            schedule.communication_active = False
            return False
        schedule.bind_streams(current_stream, replay.execution_stream)
        graph_task_id = torch._C._current_graph_task_id()
        if graph_task_id < 0:
            raise RuntimeError("ILP native sidecar launched outside an autograd graph task.")
        replay.native_graph_task_id = graph_task_id
        assert self._launcher_executor is not None
        launcher = _AsyncReplayLauncher(
            lambda: self._run_async_sidecar_replay(replay, schedule),
            self._launcher_executor,
        )
        schedule.attach_replay_launcher(launcher)
        self._native_sidecar_frame = replay
        self._native_sidecar_launcher = launcher
        self._native_sidecar_schedule = schedule
        return True

    def install_fsdp_collective_tickets(self, module: Any, layer_id: int) -> None:
        state = module._get_fsdp_state()
        param_group = state._fsdp_param_group
        if param_group is None:
            raise RuntimeError(f"ILP cannot install layer {layer_id} FSDP tickets before fully_shard initialization.")

        def schedule_getter() -> Optional[_NativeBackwardSchedule]:
            return self._native_schedule

        module.set_custom_all_gather(
            _TicketedFSDPComm(param_group._all_gather_comm, schedule_getter, layer_id, "all_gather")
        )
        module.set_custom_reduce_scatter(
            _TicketedFSDPComm(param_group._reduce_scatter_comm, schedule_getter, layer_id, "reduce_scatter")
        )

    def _join_native_current_sidecar(self, frame: ReplayFrame, graph_task_id: Optional[int] = None) -> None:
        launcher = self._native_sidecar_launcher
        if launcher is None:
            return
        schedule = self._native_sidecar_schedule
        if schedule is None:
            raise RuntimeError("ILP native-current sidecar has no owning backward schedule.")
        if self._native_sidecar_frame is not frame:
            raise RuntimeError(
                f"ILP native-current sidecar expected layer {self._native_sidecar_frame.layer_id}, "
                f"but B{frame.layer_id} started."
            )
        if graph_task_id is not None and frame.native_graph_task_id != graph_task_id:
            raise RuntimeError(
                f"ILP native replay graph task changed from {frame.native_graph_task_id} to {graph_task_id}."
            )

        notify_done = None
        try:
            with torch.autograd.profiler.record_function(
                f"ilp::fork_join::B{frame.layer_id + 1}::Fprime{frame.layer_id}::native"
            ):
                replay_done = launcher.wait()
                if replay_done is None:
                    raise RuntimeError(f"ILP Fprime{frame.layer_id} completed without a device event.")
                current_stream = torch.npu.current_stream()
                current_done = torch.npu.Event()
                current_done.record(current_stream)
                frame.execution_stream.wait_event(current_done)
                current_stream.wait_event(replay_done)
                frame.record_recomputed_tensors(current_stream)
                notify_done = torch.npu.Event()
                notify_done.record(current_stream)
                self._completion_tail = notify_done
        finally:
            if launcher.done:
                launcher.release()
                schedule.release_replay(launcher)
                self._native_sidecar_launcher = None
                self._native_sidecar_frame = None
                self._native_sidecar_schedule = None
                if self._native_schedule is schedule:
                    self._native_schedule = None
                frame.release_sidecar_graph()
                if notify_done is not None and schedule.device_notifies:
                    self._retired_device_schedules.append((notify_done, schedule))

    def _new_backward_schedule(self, layer_id: int) -> _NativeBackwardSchedule:
        layer = self._layer_modules.get(layer_id)
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        return _NativeBackwardSchedule(
            layer_id,
            backward_experts_module=experts,
            device_notify_factory=self._device_notify_factory,
            device_notify_dependencies=self._device_notify_dependencies,
        )

    def _reap_device_notifies(self) -> None:
        pending = []
        for done, schedule in self._retired_device_schedules:
            if done.query():
                schedule.close_device_notifies()
            else:
                pending.append((done, schedule))
        self._retired_device_schedules = pending

    def _wait_for_completion_tail(self) -> None:
        tail = self._completion_tail
        if tail is None:
            return
        with torch.autograd.profiler.record_function("ilp::admission::completion_backpressure"):
            if not tail.query():
                tail.synchronize()
        self._completion_tail = None

    def _memory_allows_replay(self, layer_id: int, replay_layer: int) -> bool:
        if not self.memory_budget.enabled:
            return True
        if layer_id != self.current_layer:
            if self._memory_slots > 0:
                self._memory_slots -= 1
                return True
            with torch.autograd.profiler.record_function(f"ilp::pause_deferred::B{layer_id}::Fprime{replay_layer}"):
                pass
            return False
        if self.memory_budget.paused and self._memory_retry_remaining > 0:
            self._memory_retry_remaining -= 1
            with torch.autograd.profiler.record_function(f"ilp::pause_deferred::B{layer_id}::Fprime{replay_layer}"):
                pass
            return False
        with torch.autograd.profiler.record_function(f"ilp::memory_check::B{layer_id}::Fprime{replay_layer}"):
            decision = self.memory_budget.plan_slots(self.window_size)
        self._memory_slots = max(0, decision.slots - 1)
        gib = 1024**3
        action = "resume" if decision.resumed else ("schedule" if decision.allowed else "pause")
        logger.info_rank0(
            f"ILP memory {action} at B{layer_id}/F'{replay_layer}: reason={decision.reason}, "
            f"allocated={decision.snapshot.allocated_bytes / gib:.2f}GB, "
            f"free={decision.snapshot.free_bytes / gib:.2f}GB, "
            f"estimate={decision.estimated_bytes / gib:.2f}GB, slots={decision.slots}."
        )
        if not decision.allowed:
            with torch.autograd.profiler.record_function(f"ilp::pause::B{layer_id}::Fprime{replay_layer}"):
                pass
        elif decision.resumed:
            with torch.autograd.profiler.record_function(f"ilp::resume::B{layer_id}::Fprime{replay_layer}"):
                pass
        self._memory_retry_remaining = 0 if decision.allowed else self.memory_retry_steps - 1
        return decision.allowed

    def _distributed_memory_consensus(
        self,
        snapshot: ReplayMemorySnapshot,
        estimated_bytes: int,
    ) -> tuple[ReplayMemorySnapshot, int]:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return snapshot, estimated_bytes
        values = torch.tensor(
            [snapshot.allocated_bytes, -snapshot.free_bytes, estimated_bytes],
            dtype=torch.int64,
            device="cpu" if self._memory_group is not None else torch.device("npu", torch.npu.current_device()),
        )
        group = self._memory_group if self._memory_group is not None else get_parallel_state().ep_group
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        allocated_bytes, negative_free_bytes, estimated_bytes = values.tolist()
        return (
            ReplayMemorySnapshot(
                allocated_bytes=int(allocated_bytes),
                free_bytes=-int(negative_free_bytes),
                total_bytes=snapshot.total_bytes,
            ),
            int(estimated_bytes),
        )

    def _execution_domain_index(self, layer_id: int) -> int:
        return (self.current_layer - layer_id) % 2

    def _assign_execution_domain(self, frame: ReplayFrame) -> None:
        if frame.execution_stream is not None:
            return
        if self._primary_stream is None:
            self._device_index = torch.npu.current_device()
            self._primary_stream = torch.npu.current_stream()
            self._alternate_stream = torch.npu.Stream(device=self._device_index)
            self._launcher_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ilp-launcher")
        frame.execution_stream = self._alternate_stream
        if self._execution_domain_index(frame.layer_id) == 0:
            frame.execution_group = self._primary_ep_group
        else:
            frame.execution_group = self._alternate_ep_group

    def _run_async_sidecar_replay(
        self,
        replay: ReplayFrame,
        schedule: _NativeBackwardSchedule,
    ) -> Any:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_prepare

        assert self._device_index is not None and replay.execution_stream is not None
        torch.npu.set_device(self._device_index)
        stream = replay.execution_stream
        native_frame = replay.native_checkpoint_frame
        with torch.npu.stream(stream):
            for value in replay.materialize_inputs():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
            if self.memory_budget.enabled:
                replay.memory_before_prepare = int(torch.npu.memory_allocated())

        try:
            with self._async_replay_phase(replay, "attention"):
                schedule.before_replay_phase("attention")
                with torch.autograd.profiler.record_function(f"ilp::Fprime{replay.layer_id}::attention"):
                    self._prepare_replay_attention(replay)
                schedule.after_replay_phase("attention")

            with self._async_replay_phase(replay, "dispatch"):
                if replay.dispatch_plan is None:
                    assert replay.selected_experts is not None
                    experts = replay.layer.mlp.experts
                    with torch.autograd.profiler.record_function(f"ilp::Fprime{replay.layer_id}::dispatch_prepare"):
                        replay.dispatch_plan = npu_ep_dispatch_prepare(
                            replay.selected_experts,
                            experts.num_experts,
                            replay.execution_group,
                        )
                elif replay.dispatch_plan.ep_group is not replay.execution_group:
                    replay.dispatch_plan = replace(replay.dispatch_plan, ep_group=replay.execution_group)
                self._prepare_dispatch_input(replay)
                schedule.before_replay_phase("dispatch")
                schedule.before_replay_collective("dispatch")
                self._launch_dispatch(replay)
                self._wait_dispatch_collective(replay, schedule)
                schedule.after_replay_phase("dispatch")

            with self._async_replay_phase(replay, "experts"):
                schedule.before_replay_phase("experts")
                self._wait_dispatch(replay)
                schedule.publish_replay_dispatch_done()
                from ..ops.kernels.moe._kernels.kernel.npu_group_gemm import gmm_replay_forward_interleave

                with gmm_replay_forward_interleave(schedule):
                    self._run_experts(replay)
                schedule.after_replay_phase("experts")

            with self._async_replay_phase(replay, "combine"):
                schedule.before_replay_phase("combine")
                schedule.before_replay_collective("combine")
                self._launch_combine(replay)
                from ..ops.kernels.moe.npu_group_gemm import npu_ep_combine_wait

                with (
                    torch.enable_grad(),
                    torch.autograd.profiler.record_function(f"ilp::Fprime{replay.layer_id}::combine_wait"),
                ):
                    assert replay.attention_hidden is not None and replay.moe_input is not None
                    try:
                        moe_output = npu_ep_combine_wait(replay.combine_state).reshape_as(replay.moe_input)
                    finally:
                        schedule.after_replay_collective("combine")
                    replay.replay_output = replay.attention_hidden + moe_output
                    replay.release_after_combine()
                    replay.state = ReplayState.GRAPH_READY
                    schedule.after_replay_phase("combine")
        except _StopRecomputationError:
            if native_frame is None:
                raise

        if native_frame is not None:
            graph_task_id = replay.native_graph_task_id
            native_frame.is_recomputed[graph_task_id] = True
            native_frame.check_recomputed_tensors_match(graph_task_id)
            replay.state = ReplayState.GRAPH_READY
            replay.release_after_recompute_capture()
        with torch.npu.stream(stream):
            done = torch.npu.Event()
            done.record(stream)
        return done

    @contextlib.contextmanager
    def _async_replay_phase(self, replay: ReplayFrame, phase: str):
        native_frame = replay.native_checkpoint_frame
        if native_frame is not None:
            if replay.native_graph_task_id is None:
                raise RuntimeError("ILP native replay has no autograd graph task id.")
            recomputation_hook = _recomputation_hook(
                weakref.ref(native_frame),
                replay.native_graph_task_id,
            )
        else:
            recomputation_hook = contextlib.nullcontext()
        with (
            torch.npu.stream(replay.execution_stream),
            recomputation_hook,
            replay.phased_recompute_context(),
            torch.autograd.profiler.record_function(f"ilp::Fprime{replay.layer_id}::sidecar::{phase}"),
        ):
            yield

    def _prepare_replay_attention(self, frame: ReplayFrame) -> None:
        from ..ops.kernels.attention.backward_boundary import attention_backward_phase_callback

        # Consume the normal replay-layer backward unshard launched before F'(current).
        with torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::fsdp_prefetch_wait"):
            if frame.fsdp_unshard_handle is None:
                frame.layer.unshard(async_op=False)
            else:
                frame.fsdp_unshard_handle.wait()
                frame.fsdp_unshard_handle = None
        # Keep this graph until B(layer) so early replay does not run attention twice.
        callback = frame.backward_schedule.on_phase if frame.backward_schedule is not None else None
        with torch.enable_grad(), attention_backward_phase_callback(callback):
            frame.attention_hidden = self._qwen3_attention(frame)
            frame.moe_input = frame.layer.post_attention_layernorm(frame.attention_hidden)
        frame.routing_weights, frame.selected_experts = self._route(
            frame.layer.mlp.gate,
            frame.moe_input,
        )
        frame.state = ReplayState.ATTN_READY

    @staticmethod
    def _qwen3_attention(frame: ReplayFrame) -> torch.Tensor:
        inputs = frame.materialize_inputs()
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("ILP Qwen3-MoE piercing path expects hidden_states as the only positional layer input.")
        hidden_states = inputs[0]
        kwargs = dict(getattr(frame.run_function, "keywords", None) or {})
        return frame.layer.forward_attention(hidden_states, **kwargs)

    @staticmethod
    def _route(gate: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, gate.hidden_dim)
        logits = F.linear(flat, gate.weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(probabilities, gate.top_k, dim=-1)
        if gate.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(logits.dtype), selected_experts

    @staticmethod
    def _prepare_dispatch_input(frame: ReplayFrame) -> None:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_input_prepare

        assert frame.moe_input is not None and frame.selected_experts is not None
        flat_input = frame.moe_input.reshape(-1, frame.moe_input.shape[-1])
        with (
            torch.enable_grad(),
            torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::dispatch_input_prepare"),
        ):
            frame.dispatch_input = npu_ep_dispatch_input_prepare(
                flat_input,
                frame.selected_experts,
                frame.dispatch_plan,
            )

    @staticmethod
    def _launch_dispatch(frame: ReplayFrame) -> None:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_async

        assert frame.moe_input is not None and frame.selected_experts is not None
        flat_input = frame.moe_input.reshape(-1, frame.moe_input.shape[-1])
        experts = frame.layer.mlp.experts
        with (
            torch.enable_grad(),
            torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::dispatch_launch"),
        ):
            frame.dispatch_state = npu_ep_dispatch_async(
                flat_input,
                frame.selected_experts,
                experts.num_experts,
                frame.execution_group,
                plan=frame.dispatch_plan,
                prepared_input=frame.dispatch_input,
            )
        frame.state = ReplayState.DISPATCH_PENDING

    @staticmethod
    def _wait_dispatch_collective(
        frame: ReplayFrame,
        schedule: _NativeBackwardSchedule,
    ) -> None:
        with torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::dispatch_collective_wait"):
            frame.dispatch_state.handle.wait()
        schedule.after_replay_collective("dispatch")

    @staticmethod
    def _wait_dispatch(
        frame: ReplayFrame,
        schedule: Optional[_NativeBackwardSchedule] = None,
    ) -> None:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_dispatch_wait

        with (
            torch.enable_grad(),
            torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::dispatch_wait"),
        ):
            frame.dispatched_input = npu_ep_dispatch_wait(frame.dispatch_state, synchronize=False)
        if schedule is not None:
            schedule.after_replay_collective("dispatch")
        frame.release_after_dispatch()
        frame.state = ReplayState.DISPATCH_READY

    def _run_experts(self, frame: ReplayFrame) -> None:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_expert_forward

        experts = frame.layer.mlp.experts
        with torch.enable_grad(), torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::experts"):
            experts.unshard(async_op=False)
            if frame.dispatched_input is None:
                raise RuntimeError(f"ILP Fprime{frame.layer_id} experts started before replay dispatch completed.")
            frame.expert_output = npu_ep_expert_forward(
                frame.dispatched_input,
                frame.dispatch_state,
                experts.down_proj,
                experts.gate_up_proj,
            )
        frame.release_after_experts()
        frame.state = ReplayState.EXPERT_READY

    def _launch_combine(self, frame: ReplayFrame) -> None:
        from ..ops.kernels.moe.npu_group_gemm import npu_ep_combine_async

        assert frame.expert_output is not None and frame.routing_weights is not None
        with (
            torch.enable_grad(),
            torch.autograd.profiler.record_function(f"ilp::Fprime{frame.layer_id}::combine_launch"),
        ):
            frame.combine_state = npu_ep_combine_async(
                frame.expert_output,
                frame.routing_weights,
                frame.dispatch_state,
            )
        if self.memory_budget.enabled and frame.memory_before_prepare:
            self.memory_budget.observe_replay_bytes(int(torch.npu.memory_allocated()) - frame.memory_before_prepare)
        frame.state = ReplayState.COMBINE_PENDING


def _resolve_replay_window(
    layer_count: int,
    current_layer: int,
    window_size: int,
) -> tuple[int, int]:
    if layer_count < 2:
        raise ValueError("Inter-layer replay requires at least two decoder layers.")
    current_layer = layer_count - 1 if current_layer < 0 else current_layer
    window_size = current_layer if window_size == 0 else window_size
    if window_size < 1:
        raise ValueError("Inter-layer replay window_size must be at least 1 after resolution.")
    bottom_layer = current_layer - window_size
    if current_layer >= layer_count or bottom_layer < 0:
        raise ValueError(
            f"Inter-layer replay window [{bottom_layer}, {current_layer}] is invalid for {layer_count} decoder layers."
        )
    return current_layer, window_size


def _rank_lists_for_mesh_dim(device_mesh: Any, mesh_dim_name: str) -> list[list[int]]:
    mesh_dim_names = tuple(device_mesh.mesh_dim_names)
    if mesh_dim_name not in mesh_dim_names:
        raise ValueError(f"DeviceMesh does not contain dimension {mesh_dim_name!r}: {mesh_dim_names}.")
    mesh_dim = mesh_dim_names.index(mesh_dim_name)
    rank_rows = device_mesh.mesh.movedim(mesh_dim, -1).reshape(-1, device_mesh.mesh.shape[mesh_dim])
    return [list(map(int, row.tolist())) for row in rank_rows]


def _create_duplicate_ep_group(parallel_state: Any) -> Any:
    primary_ep_group = parallel_state.ep_group
    current_rank = dist.get_rank()
    selected_group = None
    backend = dist.get_backend(primary_ep_group)
    ep_rank_lists = _rank_lists_for_mesh_dim(parallel_state.extra_parallel_fsdp_device_mesh["ep"], "ep")

    # All ranks must create process groups in identical order. Each rank keeps
    # only the duplicate communicator for its own EP subgroup.
    for ranks in ep_rank_lists:
        group = dist.new_group(ranks=ranks, backend=backend)
        if current_rank in ranks:
            selected_group = group

    if selected_group is None:
        raise RuntimeError(f"Rank {current_rank} does not belong to any expert-parallel subgroup.")
    return selected_group


def apply_inter_layer_replay(model: torch.nn.Module, config: Any) -> torch.nn.Module:
    if config is None or not getattr(config, "enable", False):
        return model
    if not IS_NPU_AVAILABLE:
        raise RuntimeError("Inter-layer replay currently requires Ascend NPU and torch_npu.")
    if not dist.is_initialized():
        raise RuntimeError("Inter-layer replay requires initialized torch.distributed.")

    parallel_state = get_parallel_state()
    if not parallel_state.ep_enabled:
        raise RuntimeError("Inter-layer replay piercing currently requires expert parallelism.")
    world_size = dist.get_world_size()
    if parallel_state.ep_size > world_size or world_size % parallel_state.ep_size != 0:
        raise RuntimeError(
            f"Inter-layer replay requires ep_size to divide world_size, got "
            f"ep_size={parallel_state.ep_size}, world_size={world_size}."
        )

    layers = [module for module in model.modules() if module.__class__.__name__.endswith("Qwen3MoeDecoderLayer")]
    current_layer, window_size = _resolve_replay_window(
        len(layers),
        int(getattr(config, "current_layer", -1)),
        int(getattr(config, "window_size", 0)),
    )
    bottom_layer = current_layer - window_size
    primary_ep_group = parallel_state.ep_group
    alternate_ep_group = _create_duplicate_ep_group(parallel_state)
    controller = InterLayerReplayController(
        current_layer=current_layer,
        window_size=window_size,
        memory_budget_gb=float(getattr(config, "memory_budget_gb", 0.0)),
        memory_reserve_gb=float(getattr(config, "memory_reserve_gb", 0.0)),
        memory_safety_factor=float(getattr(config, "memory_safety_factor", 1.1)),
        memory_retry_steps=int(getattr(config, "memory_retry_steps", 4)),
        primary_ep_group=primary_ep_group,
        alternate_ep_group=alternate_ep_group,
        strict=bool(getattr(config, "strict", True)),
        layer_modules={layer_id: layers[layer_id] for layer_id in range(bottom_layer, current_layer + 1)},
    )
    for layer_id in range(bottom_layer, current_layer + 1):
        controller.install_fsdp_collective_tickets(layers[layer_id], layer_id)

    for layer_id in range(bottom_layer + 1, current_layer + 1):
        layers[layer_id].set_modules_to_backward_prefetch([])
    logger.info_rank0(
        f"Disable native FSDP backward prefetch for layers [{bottom_layer + 1}, {current_layer}]; "
        "the rolling replay sidecars own lower-layer unshards."
    )

    for layer_id in range(bottom_layer, current_layer + 1):
        layer = layers[layer_id]
        if not getattr(layer, "gradient_checkpointing", False):
            raise RuntimeError(f"Inter-layer replay requires gradient checkpointing on layer {layer_id}.")
        if not hasattr(layer.mlp, "experts") or not hasattr(layer.mlp, "gate"):
            raise TypeError(f"Inter-layer replay layer {layer_id} is not a Qwen3 sparse MoE layer.")
        if not getattr(layer, "_fsdp_modules", None):
            raise RuntimeError(f"Inter-layer replay must be applied after FSDP2 wraps layer {layer_id}.")

        checkpoint_func = layer._gradient_checkpointing_func

        @wraps(checkpoint_func)
        def wrapped_checkpoint_func(
            function,
            *args,
            _layer=layer,
            _layer_id=layer_id,
            _native_checkpoint_func=checkpoint_func,
            **kwargs,
        ):
            if _layer_id == controller.current_layer:
                with torch.autograd.profiler.record_function(f"ilp::native_checkpoint::F{_layer_id}"):
                    return controller.checkpoint_current(_native_checkpoint_func, function, *args, **kwargs)
            if _layer_id < controller.current_layer:
                with torch.autograd.profiler.record_function(f"ilp::native_replay_checkpoint::F{_layer_id}"):
                    return controller.checkpoint_native_replay(
                        _native_checkpoint_func,
                        _layer_id,
                        _layer,
                        function,
                        *args,
                        **kwargs,
                    )
            raise RuntimeError(f"ILP checkpoint wrapper received out-of-window layer {_layer_id}.")

        layer._gradient_checkpointing_func = wrapped_checkpoint_func
        logger.info_rank0(f"Enable inter-layer replay checkpoint wrapper for Qwen3-MoE layer {layer_id}.")

    model._inter_layer_replay_controller = controller
    logger.info_rank0(
        f"Inter-layer replay window enabled for layers [{bottom_layer}, {current_layer}], "
        f"window_size={window_size}, completion_backpressure=true, ep_size={parallel_state.ep_size}, "
        f"ep_fsdp_size={parallel_state.ep_fsdp_size}."
    )
    return model
