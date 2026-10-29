# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Checkpoint-input offload through fixed pinned staging and pageable backing.

Only the fixed staging rings are pinned. Autograd owns immutable pageable
snapshots, with a separate live-byte admission limit. One optional reverse-order
prefetch has its own credit; another restore credit is reserved for demand.
Consumer-owned tensors and allocator caches are outside the staging bound.
"""

import math
import queue
import threading
import time
import weakref
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import torch
from torch.autograd.graph import saved_tensors_hooks

from ..utils.device import get_device_type, get_torch_device
from .async_offload import get_offload_modules


@dataclass(frozen=True)
class StagingConfig:
    buffer_bytes: int = 192 * 1024**2
    pageable_limit_bytes: int = 8 * 1024**3
    slots_per_direction: int = 2
    prefetch_depth: int = 1
    wait_timeout_seconds: float = 120.0
    pipeline_transfers: bool = False

    def __post_init__(self):
        if self.buffer_bytes <= 0 or self.pageable_limit_bytes < self.buffer_bytes:
            raise ValueError("Staging and pageable budgets must be positive and fit one buffer")
        if self.slots_per_direction != 2 or self.prefetch_depth not in (0, 1):
            raise ValueError("Bounded activation offload requires two slots and at most one prefetch")
        if not math.isfinite(self.wait_timeout_seconds) or self.wait_timeout_seconds <= 0:
            raise ValueError("Invalid staging deadline")


class _UncertainTransfer(RuntimeError):
    """A copy may be queued without a published completion event.

    The caller must quarantine the payload, not just the exception traceback.
    In particular, a failed H2D submission must retain its newly created value.
    """

    def __init__(self, error, payload):
        super().__init__(f"{type(error).__name__}: {error}")
        self.payload = payload


class _NativeTransfers:
    """Accelerator operations kept separate from the CPU scheduling tests."""

    def __init__(self):
        self.api = get_torch_device()
        self.device = torch.device(get_device_type(), self.api.current_device())
        self.d2h_stream = self.api.Stream(device=self.device)
        self.h2d_stream = self.api.Stream(device=self.device)

    def worker_setup(self):
        self.api.set_device(self.device)

    def pinned(self, size):
        # Fail closed: silently falling back to pageable storage would invalidate
        # both the overlap hypothesis and the fixed native pinned allocation set.
        value = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        if not value.is_pinned():
            raise RuntimeError("Pinned activation staging allocation was not pinned")
        return value

    def snapshot(self, value):
        # Preserve save-on-CPU snapshot semantics even if the caller mutates a
        # view later. The clone precedes later writes on the producer stream.
        value = value.detach().clone(memory_format=torch.contiguous_format)
        producer = self.api.Event()
        producer.record(self.api.current_stream())
        return value, producer

    def d2h(self, source, target, producer):
        submitted, event = False, None
        try:
            with self.api.stream(self.d2h_stream):
                self.d2h_stream.wait_event(producer)
                # A failing copy_ can have partially queued work already.
                submitted = True
                target.copy_(source, non_blocking=True)
                event = self.api.Event()
                event.record(self.d2h_stream)
            return event
        except BaseException as error:
            if submitted:
                raise _UncertainTransfer(error, (source, target, producer, event)) from error
            raise

    def h2d(self, source, device):
        submitted, value, event = False, None, None
        try:
            with self.api.stream(self.h2d_stream):
                value = torch.empty_like(source, device=device)
                submitted = True
                value.copy_(source, non_blocking=True)
                event = self.api.Event()
                event.record(self.h2d_stream)
            return value, event
        except BaseException as error:
            if submitted:
                raise _UncertainTransfer(error, (source, value, event)) from error
            raise

    def consume(self, value, event):
        stream = self.api.current_stream()
        stream.wait_event(event)
        value.record_stream(stream)
        return value

    def accepts(self, value):
        return value.device == self.device


class _Ticket:
    def __init__(self, manager, ordinal, block, source, backing, producer, nbytes):
        self.manager = manager
        self.ordinal = ordinal
        self.block = block
        self.source = source
        self.backing = backing
        self.producer = producer
        self.nbytes = nbytes
        self.host_ready = threading.Event()
        self.restore_ready = threading.Event()
        self.lease = _TicketLease(nbytes)
        self.lock = self.lease.lock
        self.restore = None
        self.restore_event = None
        self.restore_requested = False
        self.prefetched = False
        self.device = source.device
        self.shape = source.shape
        self.dtype = source.dtype
        weakref.finalize(self, manager._release_ticket, self.lease)


class _TicketLease:
    """Finalizer state must not retain the ticket or any activation tensor."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.lock = threading.Lock()
        self.restore_credit = False
        self.prefetch_credit = False


class BoundedActivationManager:
    """One selected checkpoint schedule with separately bounded payload owners."""

    def __init__(self, config, *, transfers=None):
        self.config = config
        self.transfers = transfers
        self.lock = threading.Lock()
        self.failure = None
        self.closed = False
        self.cleanup_done = threading.Event()
        self.pageable_bytes = 0
        self.live_tickets = 0
        self.stats = Counter()
        self.items = weakref.WeakValueDictionary()
        self.ordinal = 0
        self.d2h_free = queue.Queue(maxsize=2)
        self.h2d_free = queue.Queue(maxsize=2)
        self.d2h_work = queue.Queue(maxsize=2)
        self.h2d_work = queue.PriorityQueue(maxsize=2)
        self.d2h_pending = queue.Queue(maxsize=2)
        self.h2d_pending = queue.Queue(maxsize=2)
        self.d2h_submission_done = threading.Event()
        self.h2d_submission_done = threading.Event()
        self.restore_credits = threading.BoundedSemaphore(2)
        self.prefetch_credit = threading.BoundedSemaphore(1)
        self.workers = []
        self.owner_released = False
        self.cleanup_scheduled = False
        self.quarantined = []

    @property
    def pinned_capacity_bytes(self):
        return 2 * self.config.slots_per_direction * self.config.buffer_bytes

    def _healthy(self):
        with self.lock:
            failure, closed = self.failure, self.closed
        if failure is not None:
            raise RuntimeError("Bounded activation worker failed") from failure
        if closed:
            raise RuntimeError("Bounded activation manager is closed")

    def _fail(self, error):
        with self.lock:
            if self.failure is None:
                # Traceback frames can retain the failed ticket, source and
                # quarantined pinned slot indefinitely. Keep diagnostics only.
                self.failure = RuntimeError(f"{type(error).__name__}: {error}")
            cleanup = self.owner_released
        if cleanup:
            self._schedule_close()

    def _wait(self, operation):
        deadline = time.monotonic() + self.config.wait_timeout_seconds
        while True:
            self._healthy()
            value = operation(min(0.1, max(0, deadline - time.monotonic())))
            if value is not None:
                self._healthy()
                return value
            if time.monotonic() >= deadline:
                error = TimeoutError("Bounded activation transfer deadline exceeded")
                self._fail(error)
                raise error

    def _get(self, work):
        def get(timeout):
            try:
                return work.get(timeout=timeout)
            except queue.Empty:
                return None

        return self._wait(get)

    def _event(self, event):
        self._wait(lambda timeout: True if event.wait(timeout) else None)

    def _credit(self):
        self._wait(lambda timeout: True if self.restore_credits.acquire(timeout=timeout) else None)

    def _start(self):
        if self.transfers is None:
            self.transfers = _NativeTransfers()
        if self.workers:
            return
        try:
            # Publish only a complete ring. A partial allocation failure must
            # never retry into half-full queues without running workers.
            d2h = [self.transfers.pinned(self.config.buffer_bytes) for _ in range(self.config.slots_per_direction)]
            h2d = [self.transfers.pinned(self.config.buffer_bytes) for _ in range(self.config.slots_per_direction)]
        except BaseException as error:
            self._fail(error)
            raise
        for slot in d2h:
            self.d2h_free.put_nowait(slot)
        for slot in h2d:
            self.h2d_free.put_nowait(slot)
        targets = [(self._d2h_worker, "activation-d2h"), (self._h2d_worker, "activation-h2d")]
        if self.config.pipeline_transfers:
            targets += [
                (self._d2h_completion_worker, "activation-d2h-complete"),
                (self._h2d_completion_worker, "activation-h2d-complete"),
            ]
        for target, name in targets:
            worker = threading.Thread(target=target, name=name, daemon=True)
            self.workers.append(worker)
            worker.start()

    @staticmethod
    def _view(buffer, ticket):
        return buffer[: ticket.nbytes].view(ticket.dtype).view(ticket.shape)

    def _worker_loop(self, work, operation):
        try:
            self.transfers.worker_setup()
            while True:
                try:
                    item = work.get(timeout=0.1)
                except queue.Empty:
                    with self.lock:
                        if self.closed or self.failure is not None:
                            return
                    continue
                try:
                    operation(item)
                finally:
                    work.task_done()
                    del item
        except BaseException as error:
            self._fail(error)

    def _d2h_worker(self):
        def transfer(item):
            ticket, slot = item
            posted, event = False, None
            try:
                event = self.transfers.d2h(ticket.source, self._view(slot, ticket), ticket.producer)
                posted = True
                if self.config.pipeline_transfers:
                    # Completion retains the ticket, immutable source and slot.
                    # A second DMA can be posted while CPU copies the first.
                    self.d2h_pending.put_nowait((ticket, slot, event))
                else:
                    self._complete_d2h((ticket, slot, event))
            except BaseException as error:
                if posted or isinstance(error, _UncertainTransfer):
                    with self.lock:
                        self.quarantined.append((ticket, slot, event, *getattr(error, "payload", ())))
                self._fail(error)
                raise

        try:
            self._worker_loop(self.d2h_work, transfer)
        finally:
            self.d2h_submission_done.set()

    def _complete_d2h(self, item):
        ticket, slot, event = item
        started = time.perf_counter()
        event.synchronize()
        copied = time.perf_counter()
        ticket.backing.copy_(self._view(slot, ticket))
        finished = time.perf_counter()
        ticket.source = None
        ticket.producer = None
        with self.lock:
            self.stats["d2h_event_wait_seconds"] += copied - started
            self.stats["d2h_cpu_copy_seconds"] += finished - copied
            self.stats["d2h_completed"] += 1
            self.stats["d2h_inflight"] -= 1
        ticket.host_ready.set()
        self.d2h_free.put_nowait(slot)

    def _h2d_worker(self):
        def transfer(item):
            _priority, _ordinal, ticket = item
            self._event(ticket.host_ready)
            slot = self._get(self.h2d_free)
            posted, value, event = False, None, None
            try:
                started = time.perf_counter()
                self._view(slot, ticket).copy_(ticket.backing)
                copied = time.perf_counter()
                value, event = self.transfers.h2d(self._view(slot, ticket), ticket.device)
                posted = True
                with self.lock:
                    self.stats["h2d_cpu_copy_seconds"] += copied - started
                    self.stats["h2d_inflight"] += 1
                    self.stats["peak_h2d_inflight"] = max(self.stats["peak_h2d_inflight"], self.stats["h2d_inflight"])
                if not self.config.pipeline_transfers:
                    self._complete_h2d((ticket, slot, value, event))
                with ticket.lock:
                    ticket.restore, ticket.restore_event = value, event
                ticket.restore_ready.set()
                if self.config.pipeline_transfers:
                    # Publishing an event is enough for a device stream wait.
                    # Completion keeps pinned CPU data alive and quarantined
                    # until DMA finishes, even if autograd consumes the value.
                    self.h2d_pending.put_nowait((ticket, slot, value, event))
            except BaseException as error:
                if posted or isinstance(error, _UncertainTransfer):
                    with self.lock:
                        self.quarantined.append((ticket, slot, value, event, *getattr(error, "payload", ())))
                self._fail(error)
                raise

        try:
            self._worker_loop(self.h2d_work, transfer)
        finally:
            self.h2d_submission_done.set()

    def _complete_h2d(self, item):
        _ticket, slot, _value, event = item
        started = time.perf_counter()
        event.synchronize()
        with self.lock:
            self.stats["h2d_event_wait_seconds"] += time.perf_counter() - started
            self.stats["h2d_completed"] += 1
            self.stats["h2d_inflight"] -= 1
        self.h2d_free.put_nowait(slot)

    def _completion_loop(self, pending, submission_done, complete):
        # Shutdown cannot let a reaper exit before its submitter publishes its
        # last event. Drain already-posted DMA even after a different worker
        # failed, so pinned storage is never released while still in use.
        setup_failed = False
        try:
            self.transfers.worker_setup()
        except BaseException as error:
            self._fail(error)
            setup_failed = True
        while True:
            try:
                item = pending.get(timeout=0.1)
            except queue.Empty:
                if not submission_done.is_set():
                    continue
                # The final publication can race with the timed-out get. The
                # submitter sets done only after its last put, so this second
                # atomic get proves the queue is empty after publication ends.
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    break
            try:
                if setup_failed:
                    # A submitter may already have posted DMA before this
                    # thread failed to bind its device. Retain every payload;
                    # shutdown is not proof of device completion.
                    with self.lock:
                        self.quarantined.append(item)
                else:
                    complete(item)
            except BaseException as error:
                self._fail(error)
                # A failed device synchronize gives no completion guarantee.
                # Retain the payload until the manager itself is torn down.
                with self.lock:
                    self.quarantined.append(item)
            finally:
                pending.task_done()
                del item

    def _d2h_completion_worker(self):
        self._completion_loop(self.d2h_pending, self.d2h_submission_done, self._complete_d2h)

    def _h2d_completion_worker(self):
        self._completion_loop(self.h2d_pending, self.h2d_submission_done, self._complete_h2d)

    def _release_pageable(self, nbytes):
        with self.lock:
            self.pageable_bytes -= nbytes
            self.live_tickets -= 1

    def _release_ticket(self, lease):
        # Abandoned graphs can discard a completed, unconsumed prefetch. Its
        # credits must not strand the next microbatch. Workers retain tickets
        # until all source/pinned accesses have completed.
        with lease.lock:
            restore, prefetch = lease.restore_credit, lease.prefetch_credit
            lease.restore_credit = lease.prefetch_credit = False
        with self.lock:
            self.pageable_bytes -= lease.nbytes
            self.live_tickets -= 1
            if restore:
                self.stats["owned_restore_count"] -= 1
                self.stats["abandoned_restores"] += 1
            cleanup = self.owner_released and self.pageable_bytes == 0
        if restore:
            self.restore_credits.release()
        if prefetch:
            self.prefetch_credit.release()
        if cleanup:
            self._schedule_close()

    def release_owner(self):
        # A root can disappear while autograd still owns checkpoint closures
        # and saved tickets. Keep the backend alive until those tickets die.
        with self.lock:
            self.owner_released = True
            cleanup = self.pageable_bytes == 0 or self.failure is not None
        if cleanup:
            self._schedule_close()

    def _schedule_close(self):
        with self.lock:
            if self.cleanup_scheduled or self.closed:
                return
            self.cleanup_scheduled = True
        # The final ticket may be released on a transfer worker. Join workers
        # from a separate cleanup thread rather than trying to join ourselves.
        threading.Thread(target=self.close, name="activation-cleanup", daemon=True).start()

    def pack(self, value, block):
        self._healthy()
        nbytes = value.numel() * value.element_size()
        if not nbytes:
            return value
        if nbytes > self.config.buffer_bytes:
            raise RuntimeError("Checkpoint input exceeds the fixed activation staging buffer")
        self._start()
        started = time.perf_counter()
        slot = self._get(self.d2h_free)
        with self.lock:
            self.stats["pack_slot_wait_seconds"] += time.perf_counter() - started
        admitted = False
        try:
            with self.lock:
                if self.pageable_bytes + nbytes > self.config.pageable_limit_bytes:
                    raise RuntimeError("Live pageable activation budget exhausted")
                self.pageable_bytes += nbytes
                self.live_tickets += 1
                self.stats["peak_live_tickets"] = max(self.stats["peak_live_tickets"], self.live_tickets)
                admitted = True
                ordinal = self.ordinal
                self.ordinal += 1
                self.stats["peak_pageable_bytes"] = max(self.stats["peak_pageable_bytes"], self.pageable_bytes)
            source, producer = self.transfers.snapshot(value)
            backing = torch.empty_like(source, device="cpu", memory_format=torch.contiguous_format, pin_memory=False)
            ticket = _Ticket(self, ordinal, block, source, backing, producer, nbytes)
            admitted = False  # The ticket finalizer now owns the byte admission.
            with self.lock:
                self.items[ordinal] = ticket
                self.stats["pack_count"] += 1
                self.stats["d2h_inflight"] += 1
                self.stats["peak_d2h_inflight"] = max(self.stats["peak_d2h_inflight"], self.stats["d2h_inflight"])
            self.d2h_work.put_nowait((ticket, slot))
            return ticket
        except BaseException:
            if admitted:
                self._release_pageable(nbytes)
            self.d2h_free.put_nowait(slot)
            raise

    def _request_restore(self, ticket, *, prefetch):
        with ticket.lock:
            if ticket.restore_requested:
                return True
        if prefetch:
            if not self.prefetch_credit.acquire(blocking=False):
                return False
            if not self.restore_credits.acquire(blocking=False):
                self.prefetch_credit.release()
                return False
        else:
            self._credit()
        with ticket.lock:
            if ticket.restore_requested:
                self.restore_credits.release()
                if prefetch:
                    self.prefetch_credit.release()
                return True
            ticket.restore_requested = True
            ticket.prefetched = prefetch
            ticket.lease.restore_credit = True
            ticket.lease.prefetch_credit = prefetch
            ticket.restore_ready.clear()
        with self.lock:
            self.stats["owned_restore_count"] += 1
            self.stats["peak_owned_restore_count"] = max(
                self.stats["peak_owned_restore_count"], self.stats["owned_restore_count"]
            )
            self.stats["prefetch_requests" if prefetch else "demand_requests"] += 1
        self.h2d_work.put_nowait((1 if prefetch else 0, ticket.ordinal, ticket))
        return True

    def unpack(self, ticket):
        if isinstance(ticket, torch.Tensor):
            return ticket
        self._healthy()
        self._request_restore(ticket, prefetch=False)
        started = time.perf_counter()
        self._event(ticket.restore_ready)
        waited = time.perf_counter() - started
        with ticket.lock:
            value, event, prefetched = ticket.restore, ticket.restore_event, ticket.prefetched
        # Unlike prefetch_hits, this records whether DMA was complete at the
        # actual consumer boundary. query() never adds a device synchronization.
        try:
            ready = event.query() if hasattr(event, "query") else None
            value = self.transfers.consume(value, event)
        except BaseException as error:
            # Keep ownership and both credits on the ticket until its finalizer
            # can release them. The poisoned manager rejects subsequent work.
            self._fail(error)
            raise
        with ticket.lock:
            ticket.restore, ticket.restore_event = None, None
            ticket.restore_requested, ticket.prefetched = False, False
            ticket.lease.restore_credit = ticket.lease.prefetch_credit = False
        # Ownership moves to the autograd consumer. record_stream protects its
        # actual lifetime; this counter only bounds manager-owned restorations.
        with self.lock:
            self.stats["owned_restore_count"] -= 1
            self.stats["unpack_count"] += 1
            self.stats["prefetch_hits"] += int(prefetched)
            self.stats["unpack_host_wait_seconds"] += waited
            if ready is not None:
                self.stats["restore_dma_ready"] += int(ready)
                self.stats["restore_dma_pending"] += int(not ready)
        self.restore_credits.release()
        if prefetched:
            self.prefetch_credit.release()
        if self.config.prefetch_depth and ticket.block > 0:
            with self.lock:
                previous = self.items.get(ticket.ordinal - 1)
            if previous is not None and previous.block == ticket.block - 1:
                self._request_restore(previous, prefetch=True)
        return value

    def reset(self):
        self._healthy()
        # Autograd, not a model-global list, owns packed snapshots. Do not clear
        # live tickets or reuse their CPU data when a graph is retained.

    def drain(self):
        """Wait for completion bookkeeping at an explicit, quiescent audit boundary."""
        queues = (self.d2h_work, self.h2d_work, self.d2h_pending, self.h2d_pending)

        def completed(timeout):
            if all(work.unfinished_tasks == 0 for work in queues):
                return True
            time.sleep(min(timeout, 0.001))
            return None

        self._wait(completed)

    def snapshot_stats(self):
        with self.lock:
            result = dict(
                self.stats,
                live_pageable_bytes=self.pageable_bytes,
                live_tickets=self.live_tickets,
                pageable_limit_bytes=self.config.pageable_limit_bytes,
                pinned_capacity_bytes=self.pinned_capacity_bytes,
                prefetch_depth=self.config.prefetch_depth,
                pipeline_transfers=self.config.pipeline_transfers,
                failure=None if self.failure is None else repr(self.failure),
            )
        return result

    def close(self):
        with self.lock:
            self.closed = True
        for worker in self.workers:
            worker.join(timeout=self.config.wait_timeout_seconds)
        if any(worker.is_alive() for worker in self.workers):
            raise RuntimeError("Bounded activation workers did not finish")
        if self.quarantined or any(work.unfinished_tasks for work in (self.d2h_pending, self.h2d_pending)):
            raise RuntimeError("Activation DMA completion failed; payloads remain quarantined")
        # Drop manager-owned pinned slots and queued payload references after
        # every worker has quiesced. Consumer-owned tensors remain untouched.
        for work in (self.d2h_work, self.h2d_work, self.d2h_pending, self.h2d_pending, self.d2h_free, self.h2d_free):
            while True:
                try:
                    work.get_nowait()
                    work.task_done()
                except queue.Empty:
                    break
        self.cleanup_done.set()


def _uses_supported_full_checkpoint(module):
    from torch.utils.checkpoint import checkpoint

    from ..utils import recompute_utils

    fn = getattr(module, "_gradient_checkpointing_func", None)
    if isinstance(fn, partial) and fn.func is checkpoint:
        return fn.keywords.get("use_reentrant") is False
    binding = recompute_utils._block_bindings.get(module)
    if binding is None or binding.plan.decision is not recompute_utils.Decision.FULL:
        return False
    if binding.plan.checkpoint_kwargs.get("use_reentrant") is not False:
        return False
    return True


def apply_bounded_activation_offload(model, patterns, config):
    """Wrap only selected nonreentrant checkpoint-input boundaries, before FSDP."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    if not patterns:
        raise ValueError("Bounded activation offload requires explicit module patterns")
    for pattern in patterns:
        if "{*}" in pattern:
            if pattern.count("{*}") != 1 or not pattern.endswith(".{*}") or not all(pattern[:-4].split(".")):
                raise ValueError(f"Bounded activation offload requires a terminal wildcard: {pattern!r}")
        if not get_offload_modules(model, [pattern]):
            raise ValueError(f"Bounded activation offload pattern {pattern!r} matched no modules")
    modules = get_offload_modules(model, patterns)
    if len({id(module) for _name, module, _block, _depth in modules}) != len(modules):
        raise ValueError("Bounded activation offload selected a module more than once")
    for name, module, _block, _depth in modules:
        if not isinstance(module, GradientCheckpointingLayer):
            raise ValueError(f"Bounded activation offload module {name!r} is not a checkpoint boundary")
        if type(module).__call__ is not GradientCheckpointingLayer.__call__:
            raise ValueError(f"Bounded activation offload module {name!r} overrides the checkpoint boundary")
        if hasattr(module, "_veomni_offload_manager"):
            raise ValueError("Activation offload was already installed on a selected module")
    manager = BoundedActivationManager(config)
    weakref.finalize(model, manager.release_owner)
    for _name, module, block, _depth in modules:
        original_class = type(module)
        original_call = original_class.__call__

        def wrapped(self, *args, _original=original_call, _block=block, **kwargs):
            if not self.training or not torch.is_grad_enabled():
                return _original(self, *args, **kwargs)
            if not getattr(self, "gradient_checkpointing", False):
                raise RuntimeError("Bounded activation offload requires checkpointing on every selected module")
            if not args:
                raise RuntimeError(
                    "Bounded activation offload requires hidden_states as the first positional argument"
                )
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                raise ValueError("Selected checkpoint module must expose its hidden_states input")
            if hidden.numel() and not hidden.is_contiguous():
                raise RuntimeError("Bounded activation offload requires a contiguous hidden_states input")
            if not _uses_supported_full_checkpoint(self):
                raise RuntimeError("Bounded activation offload requires a supported full nonreentrant checkpoint")
            # Autograd retains the pack hook until backward. Capturing hidden
            # itself would keep the original device activation alive as well.
            hidden_ptr = hidden.data_ptr()
            packed = 0

            def pack(value):
                nonlocal packed
                if value.data_ptr() == hidden_ptr and manager.transfers.accepts(value):
                    if not value.is_contiguous():
                        raise RuntimeError("Bounded activation offload cannot preserve this checkpoint input layout")
                    packed += 1
                    return manager.pack(value, _block)
                return value

            if manager.transfers is None:
                manager.transfers = _NativeTransfers()
            with saved_tensors_hooks(pack, manager.unpack) if hidden.numel() else nullcontext():
                result = _original(self, *args, **kwargs)
            if hidden.numel() and packed == 0:
                raise RuntimeError("Bounded activation offload did not capture the checkpoint input")
            return result

        module._veomni_offload_manager = manager
        module.__class__ = type(original_class.__name__, (original_class,), {"__call__": wrapped})
    return manager
