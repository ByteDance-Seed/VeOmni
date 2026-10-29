"""Bounded ownership, backpressure, copy failures, and native GC gradients."""

import gc
import os
import threading
import time
import weakref
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.bounded_activation_offload import (
    BoundedActivationManager,
    StagingConfig,
    _NativeTransfers,
    _UncertainTransfer,
)
from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE, get_device_type, get_torch_device


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class CompletedEvent:
    def synchronize(self):
        return None

    def query(self):
        return True


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
@pytest.mark.parametrize("failure", ["copy", "event_create", "event_record"])
def test_native_partial_submission_exception_retains_unpublished_payload(monkeypatch, direction, failure):
    references, copied = [], []

    class CopyTarget:
        def copy_(self, source, **kwargs):
            copied.append(True)
            if failure == "copy":
                raise RuntimeError("copy may already be queued")

    class Event:
        def record(self, stream):
            if failure == "event_record":
                raise RuntimeError("completion record failed")

    def make_event():
        if failure == "event_create":
            raise RuntimeError("completion allocation failed")
        return Event()

    def allocate(*args, **kwargs):
        value = CopyTarget()
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(torch, "empty_like", allocate)
    transfers = _NativeTransfers.__new__(_NativeTransfers)
    transfers.api = SimpleNamespace(stream=lambda stream: nullcontext(), Event=make_event)
    transfers.d2h_stream = transfers.h2d_stream = SimpleNamespace(wait_event=lambda event: None)

    def submit():
        source = CopyTarget()
        references.append(weakref.ref(source))
        try:
            if direction == "d2h":
                target = allocate()
                transfers.d2h(source, target, Event())
            else:
                transfers.h2d(source, "cpu")
        except _UncertainTransfer as error:
            # A traceback would hide missing explicit ownership in this test.
            error.__traceback__ = error.__cause__ = error.__context__ = None
            return error
        raise AssertionError("An uncertain submission must not return success")

    error = submit()
    gc.collect()
    assert copied == [True] and error.payload
    assert all(reference() is not None for reference in references)
    del error
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_native_failure_before_copy_does_not_claim_pending_dma(monkeypatch, direction):
    def fail(*args, **kwargs):
        raise RuntimeError("failed before any copy")

    transfers = _NativeTransfers.__new__(_NativeTransfers)
    transfers.api = SimpleNamespace(stream=lambda stream: nullcontext())
    transfers.d2h_stream = transfers.h2d_stream = SimpleNamespace(wait_event=fail)
    monkeypatch.setattr(torch, "empty_like", fail)
    with pytest.raises(RuntimeError, match="before any copy") as captured:
        if direction == "d2h":
            transfers.d2h(object(), object(), object())
        else:
            transfers.h2d(object(), "cpu")
    assert not isinstance(captured.value, _UncertainTransfer)


def test_bounded_config_requires_explicit_capacity_and_checkpoint_targets():
    from veomni.arguments.arguments_types import OffloadConfig

    valid = dict(
        enable_async_activation=True, activation_offload_backend="bounded", activation_offload_modules=["layers.{*}"]
    )
    assert OffloadConfig(**valid).activation_offload_staging_buffer_mib == 192
    for field, value in (
        ("enable_async_activation", False),
        ("activation_offload_modules", []),
        ("activation_offload_staging_buffer_mib", 0),
        ("activation_offload_prefetch_depth", 2),
        ("activation_offload_pageable_limit_gb", float("nan")),
        ("activation_offload_pageable_limit_gb", 0.01),
    ):
        with pytest.raises(ValueError):
            OffloadConfig(**dict(valid, **{field: value}))


class CPUTransfers:
    """Actual CPU tensors, explicit test events, no accelerator initialization."""

    def __init__(self):
        self.allocations = 0
        self.wait_copy = None
        self.fail_d2h = False

    def worker_setup(self):
        return None

    def pinned(self, size):
        self.allocations += 1
        return torch.empty(size, dtype=torch.uint8)

    def accepts(self, value):
        return value.device.type == "cpu"

    def snapshot(self, value):
        return value.detach().clone(memory_format=torch.contiguous_format), CompletedEvent()

    def d2h(self, source, target, producer):
        if self.fail_d2h:
            raise RuntimeError("injected D2H failure")
        if self.wait_copy is not None and not self.wait_copy.wait(2):
            raise TimeoutError("test release was not signalled")
        producer.synchronize()
        target.copy_(source)
        return CompletedEvent()

    def h2d(self, source, device):
        return source.to(device, copy=True), CompletedEvent()

    def consume(self, value, event):
        event.synchronize()
        return value


def manager(*, prefetch=0, budget=8192, pipeline=False, transfers=None):
    return BoundedActivationManager(
        StagingConfig(
            buffer_bytes=1024,
            pageable_limit_bytes=budget,
            prefetch_depth=prefetch,
            wait_timeout_seconds=3,
            pipeline_transfers=pipeline,
        ),
        transfers=CPUTransfers() if transfers is None else transfers,
    )


def settle(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("CPU scheduler did not reach the required state")
        time.sleep(0.005)


@pytest.mark.parametrize("failure", ["snapshot", "backing"])
def test_snapshot_admission_counter_rolls_back_failed_pack(monkeypatch, failure):
    m = manager()
    try:
        with monkeypatch.context() as patcher:

            def fail(*args, **kwargs):
                raise RuntimeError("injected pack allocation failure")

            if failure == "snapshot":
                patcher.setattr(m.transfers, "snapshot", fail)
            else:
                patcher.setattr(torch, "empty_like", fail)
            with pytest.raises(RuntimeError, match="pack allocation"):
                m.pack(torch.ones(8), 0)
        stats = m.snapshot_stats()
        assert stats["live_tickets"] == stats["live_pageable_bytes"] == 0
        ticket = m.pack(torch.ones(8), 0)
        assert m.snapshot_stats()["live_tickets"] == 1
        m.unpack(ticket)
        m.drain()
        del ticket
        gc.collect()
        settle(lambda: m.snapshot_stats()["live_tickets"] == 0)
    finally:
        m.close()


def test_live_counter_includes_ticket_whose_finalizer_is_waiting():
    m = manager()
    waiting, proceed = threading.Event(), threading.Event()
    release = m._release_ticket
    failures = []

    def delayed(lease):
        waiting.set()
        if not proceed.wait(2):
            failures.append("test finalizer was not released")
        release(lease)

    m._release_ticket = delayed
    ticket = m.pack(torch.ones(8), 0)
    m.unpack(ticket)
    m.drain()
    owners = [ticket]
    del ticket
    worker = threading.Thread(target=lambda: owners.pop())
    try:
        worker.start()
        assert waiting.wait(2)
        assert m.snapshot_stats()["live_tickets"] == 1
        second = m.pack(torch.ones(8), 1)
        stats = m.snapshot_stats()
        assert stats["live_tickets"] == stats["peak_live_tickets"] == 2
        assert stats["live_pageable_bytes"] == 16 * torch.ones(1).element_size()
        proceed.set()
        worker.join(timeout=2)
        assert not worker.is_alive() and not failures
        assert m.snapshot_stats()["live_tickets"] == 1
        m.unpack(second)
        m.drain()
        del second
        gc.collect()
        settle(lambda: m.snapshot_stats()["live_tickets"] == 0)
    finally:
        proceed.set()
        worker.join(timeout=2)
        m.close()


@pytest.mark.parametrize("pipeline", [False, True])
def test_alias_snapshot_and_repeated_unpack_are_exact(pipeline):
    m = manager(pipeline=pipeline)
    try:
        source = torch.arange(128, dtype=torch.bfloat16).reshape(8, 16)
        view = source[:, ::2]
        expected = view.clone()
        ticket = m.pack(view, 0)
        source.add_(100)
        first, second = m.unpack(ticket), m.unpack(ticket)
        assert torch.equal(first, expected) and torch.equal(second, expected)
        assert m.snapshot_stats()["owned_restore_count"] == 0
        assert m.transfers.allocations == 4
        del ticket
        gc.collect()
        settle(lambda: m.snapshot_stats()["live_pageable_bytes"] == 0)
    finally:
        m.close()


@pytest.mark.parametrize("pipeline", [False, True])
def test_fixed_d2h_slots_apply_backpressure_before_more_snapshots(pipeline):
    transfers = CPUTransfers()
    transfers.wait_copy = threading.Event()
    m = manager(transfers=transfers, pipeline=pipeline)
    tickets, failures = [], []
    try:
        tickets.extend([m.pack(torch.ones(256), 0), m.pack(torch.ones(256), 1)])

        def third():
            try:
                tickets.append(m.pack(torch.ones(256), 2))
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=third)
        worker.start()
        time.sleep(0.05)
        assert len(tickets) == 2
        assert m.snapshot_stats()["live_pageable_bytes"] == 2048
        transfers.wait_copy.set()
        worker.join(timeout=2)
        assert not worker.is_alive() and not failures and len(tickets) == 3
        assert m.snapshot_stats()["peak_d2h_inflight"] <= 2
        for ticket in tickets:
            assert torch.equal(m.unpack(ticket), torch.ones(256))
    finally:
        transfers.wait_copy.set()
        m.close()


def test_pageable_budget_fails_instead_of_waiting_for_backward():
    m = manager(budget=2048)
    try:
        first, second = m.pack(torch.ones(256), 0), m.pack(torch.ones(256), 1)
        m._event(second.host_ready)
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="pageable activation budget"):
            m.pack(torch.ones(256), 2)
        assert time.monotonic() - before < 1
        assert m.snapshot_stats()["live_pageable_bytes"] == 2048
        assert torch.equal(m.unpack(first), torch.ones(256))
    finally:
        m.close()


def test_oversize_input_fails_before_allocating_pinned_buffers():
    m = manager()
    with pytest.raises(RuntimeError, match="exceeds"):
        m.pack(torch.ones(1025, dtype=torch.uint8), 0)
    assert m.transfers.allocations == 0
    m.close()


@pytest.mark.parametrize("pipeline", [False, True])
def test_worker_exception_is_sticky_and_unblocks_consumer(pipeline):
    transfers = CPUTransfers()
    transfers.fail_d2h = True
    m = manager(transfers=transfers, pipeline=pipeline)
    try:
        ticket = m.pack(torch.ones(32), 0)
        settle(lambda: m.snapshot_stats()["failure"] is not None)
        with pytest.raises(RuntimeError, match="worker failed"):
            m.unpack(ticket)
        with pytest.raises(RuntimeError, match="worker failed"):
            m.pack(torch.ones(32), 1)
    finally:
        m.close()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_unpublished_dma_payload_is_quarantined_after_partial_submission(direction, pipeline):
    transfers = CPUTransfers()
    references = []

    def d2h(source, target, producer):
        target.copy_(source)
        references.extend((weakref.ref(source), weakref.ref(target)))
        raise _UncertainTransfer(RuntimeError("injected event publication failure"), (source, target))

    def h2d(source, device):
        value = source.clone()
        references.extend((weakref.ref(source), weakref.ref(value)))
        raise _UncertainTransfer(RuntimeError("injected event publication failure"), (source, value))

    setattr(transfers, direction, d2h if direction == "d2h" else h2d)
    m = manager(transfers=transfers, pipeline=pipeline)
    ticket = m.pack(torch.ones(32), 0)
    if direction == "h2d":
        with pytest.raises(RuntimeError, match="worker failed"):
            m.unpack(ticket)
    settle(lambda: m.snapshot_stats()["failure"] is not None)
    with pytest.raises(RuntimeError, match="remain quarantined"):
        m.close()
    del ticket
    gc.collect()
    assert m.quarantined and all(reference() is not None for reference in references)
    assert not m.cleanup_done.is_set()
    assert (m.d2h_free if direction == "d2h" else m.h2d_free).qsize() == 1
    with pytest.raises(RuntimeError, match="worker failed"):
        m.pack(torch.ones(32), 1)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_posted_dma_cannot_report_clean_after_completion_or_publication_failure(direction, pipeline):
    import queue

    class FailedEvent:
        def synchronize(self):
            raise RuntimeError("injected completion failure")

    transfers = CPUTransfers()
    if direction == "d2h":
        transfers.d2h = lambda *args: FailedEvent() if not pipeline else CompletedEvent()
    else:
        transfers.h2d = lambda source, device: (source.clone(), FailedEvent() if not pipeline else CompletedEvent())
    m = manager(transfers=transfers, pipeline=pipeline)
    if pipeline:
        pending = m.d2h_pending if direction == "d2h" else m.h2d_pending

        def fail_publication(item):
            raise queue.Full("injected pending publication failure")

        pending.put_nowait = fail_publication
    ticket = m.pack(torch.ones(32), 0)
    if direction == "h2d":
        m._request_restore(ticket, prefetch=False)
    settle(lambda: m.snapshot_stats()["failure"] is not None)
    with pytest.raises(RuntimeError, match="remain quarantined"):
        m.close()
    assert m.quarantined[0][0] is ticket and not m.cleanup_done.is_set()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_close_racing_partial_submission_keeps_payload_quarantined(direction, pipeline):
    submitted, release = threading.Event(), threading.Event()
    references, failures = [], []
    transfers = CPUTransfers()

    def d2h(source, target, producer):
        target.copy_(source)
        references.extend((weakref.ref(source), weakref.ref(target)))
        submitted.set()
        assert release.wait(2)
        raise _UncertainTransfer(RuntimeError("late D2H publication failure"), (source, target))

    def h2d(source, device):
        value = source.clone()
        references.extend((weakref.ref(source), weakref.ref(value)))
        submitted.set()
        assert release.wait(2)
        raise _UncertainTransfer(RuntimeError("late H2D publication failure"), (source, value))

    setattr(transfers, direction, d2h if direction == "d2h" else h2d)
    m = manager(transfers=transfers, pipeline=pipeline)
    ticket = m.pack(torch.ones(32), 0)
    if direction == "h2d":
        m._request_restore(ticket, prefetch=False)
    assert submitted.wait(2)

    def close():
        try:
            m.close()
        except RuntimeError as error:
            failures.append(str(error))

    closer = threading.Thread(target=close)
    closer.start()
    settle(lambda: m.closed)
    assert closer.is_alive() and all(reference() is not None for reference in references)
    release.set()
    closer.join(timeout=3)
    assert not closer.is_alive() and len(failures) == 1 and "remain quarantined" in failures[0]
    del ticket
    gc.collect()
    assert not m.cleanup_done.is_set() and all(reference() is not None for reference in references)


def test_pipeline_posts_second_dma_before_first_completion_and_keeps_slots_owned():
    release = threading.Event()
    posted = []

    class PendingEvent:
        def synchronize(self):
            assert release.wait(2), "DMA release was not signalled"

        def query(self):
            return release.is_set()

    transfers = CPUTransfers()

    def d2h(source, target, producer):
        target.copy_(source)
        posted.append(weakref.ref(source))
        return PendingEvent()

    transfers.d2h = d2h
    m = manager(transfers=transfers, pipeline=True)
    try:
        tickets = [m.pack(torch.full((256,), float(i)), i) for i in range(2)]
        settle(lambda: len(posted) == 2)
        assert all(ref() is not None for ref in posted)
        assert m.d2h_free.empty() and not any(t.host_ready.is_set() for t in tickets)
        assert m.transfers.allocations == 4
        release.set()
        for i, ticket in enumerate(tickets):
            assert torch.equal(m.unpack(ticket), torch.full((256,), float(i)))
        settle(lambda: all(ref() is None for ref in posted))
    finally:
        release.set()
        m.close()


def test_pipeline_publishes_restore_event_without_reusing_its_pinned_source():
    release = threading.Event()
    posted = []
    transfers = CPUTransfers()

    class PendingEvent:
        def synchronize(self):
            assert release.wait(2), "H2D release was not signalled"

        def query(self):
            return release.is_set()

    def h2d(source, device):
        posted.append(source)
        return source.clone(), PendingEvent()

    transfers.h2d = h2d
    m = manager(transfers=transfers, pipeline=True)
    try:
        tickets = [m.pack(torch.full((32,), float(i)), i) for i in range(2)]
        for ticket in tickets:
            m._request_restore(ticket, prefetch=False)
            m._event(ticket.restore_ready)
        assert len(posted) == 2 and m.h2d_free.empty()
        assert m.snapshot_stats()["h2d_inflight"] == 2
        assert all(ticket.restore_event.query() is False for ticket in tickets)
        # Publication never means a host buffer can be overwritten already.
        assert torch.equal(posted[0], torch.zeros(32)) and torch.equal(posted[1], torch.ones(32))
        release.set()
        for i, ticket in enumerate(tickets):
            assert torch.equal(m.unpack(ticket), torch.full((32,), float(i)))
        m.h2d_pending.join()
        assert m.h2d_free.qsize() == 2 and m.snapshot_stats()["h2d_inflight"] == 0
    finally:
        release.set()
        m.close()


def test_pipeline_completion_failure_poisoning_quarantines_payload():
    transfers = CPUTransfers()

    class FailedEvent:
        def synchronize(self):
            raise RuntimeError("injected completion failure")

    def d2h(source, target, producer):
        return FailedEvent()

    transfers.d2h = d2h
    m = manager(transfers=transfers, pipeline=True)
    ticket = m.pack(torch.ones(32), 0)
    settle(lambda: m.snapshot_stats()["failure"] is not None)
    with pytest.raises(RuntimeError, match="worker failed"):
        m.unpack(ticket)
    with pytest.raises(RuntimeError, match="remain quarantined"):
        m.close()
    assert len(m.quarantined) == 1 and m.quarantined[0][0] is ticket
    assert not m.cleanup_done.is_set()


def test_pipeline_completion_setup_failure_retains_already_posted_dma():
    transfers = CPUTransfers()
    submitted = threading.Event()
    snapshots, slots = [], []
    allocate = transfers.pinned

    def pinned(size):
        value = allocate(size)
        slots.append(weakref.ref(value))
        return value

    def setup():
        if threading.current_thread().name == "activation-d2h-complete":
            assert submitted.wait(2)
            raise RuntimeError("injected completion device setup failure")

    class IncompleteEvent:
        def synchronize(self):
            raise AssertionError("The failed device context must not consume DMA")

    def d2h(source, target, producer):
        snapshots.append(weakref.ref(source))
        submitted.set()
        return IncompleteEvent()

    transfers.worker_setup, transfers.pinned, transfers.d2h = setup, pinned, d2h
    m = manager(transfers=transfers, pipeline=True)
    ticket = m.pack(torch.ones(32), 0)
    settle(lambda: m.snapshot_stats()["failure"] is not None)
    with pytest.raises(RuntimeError, match="remain quarantined"):
        m.close()
    del ticket
    gc.collect()
    assert all(ref() is not None for ref in snapshots + slots)
    assert len(m.quarantined) == 1 and not m.cleanup_done.is_set()


def test_completion_rechecks_pending_after_final_publication_races_with_timeout():
    import queue

    m = manager(pipeline=True)
    done = threading.Event()
    completed = []

    class FinalPublicationQueue(queue.Queue):
        published = False

        def get(self, block=True, timeout=None):
            if timeout is not None and not self.published:
                self.published = True
                self.put_nowait("last DMA")
                done.set()
                raise queue.Empty
            return super().get(block=block, timeout=timeout)

    pending = FinalPublicationQueue()
    m._completion_loop(pending, done, completed.append)
    assert completed == ["last DMA"] and pending.unfinished_tasks == 0
    m.close()


def test_partial_pinned_allocation_failure_is_sticky_without_publishing_a_ring():
    transfers = CPUTransfers()
    allocate = transfers.pinned

    def fail_third(size):
        if transfers.allocations == 2:
            raise RuntimeError("injected pin allocation failure")
        return allocate(size)

    transfers.pinned = fail_third
    m = manager(transfers=transfers)
    try:
        with pytest.raises(RuntimeError, match="pin allocation failure"):
            m.pack(torch.ones(16), 0)
        assert m.d2h_free.empty() and m.h2d_free.empty() and not m.workers
        before = time.monotonic()
        with pytest.raises(RuntimeError, match="worker failed"):
            m.pack(torch.ones(16), 0)
        assert time.monotonic() - before < 1
    finally:
        m.close()


def test_consumer_failure_preserves_lease_until_ticket_is_released():
    transfers = CPUTransfers()

    def fail_consume(value, event):
        raise RuntimeError("injected consumer failure")

    transfers.consume = fail_consume
    m = manager(transfers=transfers)
    try:
        ticket = m.pack(torch.ones(16), 0)
        with pytest.raises(RuntimeError, match="consumer failure"):
            m.unpack(ticket)
        assert ticket.lease.restore_credit
        assert m.snapshot_stats()["owned_restore_count"] == 1
    finally:
        m.close()


def test_failed_owner_teardown_does_not_retain_traceback_payloads():
    transfers = CPUTransfers()
    transfers.fail_d2h = True
    allocate = transfers.pinned
    slots = []

    def observe(size):
        result = allocate(size)
        slots.append(weakref.ref(result))
        return result

    transfers.pinned = observe
    m = manager(transfers=transfers)
    ticket = m.pack(torch.ones(16), 0)
    settle(lambda: m.snapshot_stats()["failure"] is not None)
    del ticket
    m.release_owner()
    gc.collect()
    assert m.cleanup_done.wait(3)
    settle(lambda: m.snapshot_stats()["live_pageable_bytes"] == 0)
    assert all(slot() is None for slot in slots)


def test_one_prefetch_leaves_demand_credit_and_abandoned_graph_releases_it():
    m = manager(prefetch=1)
    try:
        first, second, third = [m.pack(torch.full((32,), float(i)), i) for i in range(3)]
        assert torch.equal(m.unpack(second), torch.ones(32))
        m._event(first.restore_ready)
        assert not m._request_restore(third, prefetch=True)
        assert torch.equal(m.unpack(third), torch.full((32,), 2.0))
        # Dropping a graph with an unconsumed prefetch cannot strand a credit.
        m.h2d_work.join()
        del first, second, third
        gc.collect()
        settle(lambda: m.snapshot_stats()["owned_restore_count"] == 0)
        next_ticket = m.pack(torch.zeros(32), 0)
        assert torch.equal(m.unpack(next_ticket), torch.zeros(32))
        assert m.snapshot_stats()["peak_owned_restore_count"] <= 2
        assert m.transfers.allocations == 4
    finally:
        m.close()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("prefetch", [0, 1])
def test_ring_and_pageable_capacity_do_not_grow_across_microbatches(prefetch, pipeline):
    m = manager(prefetch=prefetch, pipeline=pipeline)
    try:
        for _ in range(12):
            tickets = [m.pack(torch.full((32,), float(i)), i) for i in range(4)]
            for index in reversed(range(4)):
                actual = m.unpack(tickets[index])
                assert torch.equal(actual, torch.full((32,), float(index)))
            m.h2d_work.join()
            m.h2d_pending.join()
            del tickets
            gc.collect()
            settle(lambda: m.snapshot_stats()["live_pageable_bytes"] == 0)
        assert m.transfers.allocations == 4
        assert m.snapshot_stats()["peak_d2h_inflight"] <= 2
        assert m.snapshot_stats()["peak_owned_restore_count"] <= 2
        if prefetch:
            assert m.snapshot_stats()["prefetch_hits"] > 0
    finally:
        m.close()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("prefetch", [0, 1])
def test_checkpoint_hook_values_and_gradients_on_cpu(monkeypatch, prefetch, pipeline):
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_layers import GradientCheckpointingLayer

    import veomni.distributed.bounded_activation_offload as staging

    monkeypatch.setattr(staging, "_NativeTransfers", CPUTransfers)

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(16, 16)
            self.input_refs = []
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def forward(self, hidden_states):
            self.input_refs.append(weakref.ref(hidden_states))
            return hidden_states + self.projection(hidden_states).tanh()

    def make():
        return torch.nn.Sequential(Block(), Block(), Block()).train()

    torch.manual_seed(7)
    reference, candidate = make(), make()
    candidate.load_state_dict(reference.state_dict())
    config = StagingConfig(
        buffer_bytes=1024,
        pageable_limit_bytes=8192,
        prefetch_depth=prefetch,
        pipeline_transfers=pipeline,
    )
    m = staging.apply_bounded_activation_offload(candidate, ["0", "1", "2"], config)
    try:
        a = torch.randn(2, 4, 16, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        hidden = b + 0
        y, z = reference(a + 0), candidate(hidden)
        # The snapshot must tolerate later input mutation, without the pack
        # hook retaining the original activation until backward.
        with torch.no_grad():
            hidden.add_(100)
        del hidden
        # Only the independent snapshot may be retained by in-flight DMA.
        gc.collect()
        assert all(ref() is None for block in candidate for ref in block.input_refs)
        y.square().mean().backward()
        z.square().mean().backward()
        assert torch.equal(y, z) and torch.equal(a.grad, b.grad)
        assert all(torch.equal(p.grad, q.grad) for p, q in zip(reference.parameters(), candidate.parameters()))
        assert m.snapshot_stats()["pack_count"] == 3
        assert m.snapshot_stats()["unpack_count"] == 3
        if prefetch:
            assert m.snapshot_stats()["prefetch_hits"] == 2
    finally:
        m.close()


def test_root_collection_does_not_close_live_checkpoint_graph(monkeypatch):
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_layers import GradientCheckpointingLayer

    import veomni.distributed.bounded_activation_offload as staging

    monkeypatch.setattr(staging, "_NativeTransfers", CPUTransfers)

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(16, 16)
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def forward(self, hidden_states):
            return self.projection(hidden_states).tanh()

    model = torch.nn.Sequential(Block(), Block()).train()
    m = staging.apply_bounded_activation_offload(
        model, ["0", "1"], StagingConfig(buffer_bytes=1024, pageable_limit_bytes=8192)
    )
    root = weakref.ref(model)
    x = torch.randn(2, 16, requires_grad=True)
    y = model(x)
    del model
    gc.collect()
    assert root() is None and not m.closed and m.owner_released
    y.sum().backward()
    assert x.grad is not None
    del y
    gc.collect()
    assert m.cleanup_done.wait(3)
    assert m.d2h_free.empty() and m.h2d_free.empty()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("nondefault_stream", [False, True])
@pytest.mark.skipif(not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE), reason="Requires native accelerator streams")
@pytest.mark.launch_plan(provider=LaunchSpec(device="accelerator", nproc=1))
def test_native_checkpoint_gradients_and_streams(prefetch, nondefault_stream, pipeline):
    native_checkpoint_case(prefetch, nondefault_stream, pipeline=pipeline)


def native_checkpoint_case(prefetch, nondefault_stream, *, pipeline=False):
    """Check native BF16 checkpoint gradients and producer/consumer stream ordering."""
    from contextlib import nullcontext

    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_layers import GradientCheckpointingLayer

    from veomni.distributed.bounded_activation_offload import apply_bounded_activation_offload

    if not (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE):
        raise RuntimeError("Native activation gate requires an accelerator")
    api = get_torch_device()
    api.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device(get_device_type(), api.current_device())

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(64, 64, bias=False, device=device, dtype=torch.bfloat16)
            self.input_refs = []
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

        def forward(self, hidden_states):
            self.input_refs.append(weakref.ref(hidden_states))
            return hidden_states + self.projection(hidden_states).tanh() * 0.125

    torch.manual_seed(19)
    reference = torch.nn.Sequential(Block(), Block(), Block()).train()
    candidate = torch.nn.Sequential(Block(), Block(), Block()).train()
    candidate.load_state_dict(reference.state_dict())
    m = apply_bounded_activation_offload(
        candidate,
        ["0", "1", "2"],
        StagingConfig(
            buffer_bytes=2 * 1024**2,
            pageable_limit_bytes=16 * 1024**2,
            prefetch_depth=prefetch,
            pipeline_transfers=pipeline,
        ),
    )
    try:
        stream = api.Stream(device=device) if nondefault_stream else None
        if stream is not None:
            stream.wait_stream(api.current_stream())
        with api.stream(stream) if stream is not None else nullcontext():
            a = torch.arange(2 * 8 * 64, device=device, dtype=torch.float32).reshape(2, 8, 64) / 8192
            a = a.to(torch.bfloat16).requires_grad_()
            b = a.detach().clone().requires_grad_()
            hidden = b + 0
            y, z = reference(a + 0), candidate(hidden)
            with torch.no_grad():
                hidden.add_(100)
            del hidden
            # Original Tensor lifetime must not depend on DMA completion.
            gc.collect()
            assert all(ref() is None for block in candidate for ref in block.input_refs)
            y.float().square().mean().backward()
            z.float().square().mean().backward()
        api.synchronize()
        assert torch.equal(y, z) and torch.equal(a.grad, b.grad)
        assert all(torch.equal(p.grad, q.grad) for p, q in zip(reference.parameters(), candidate.parameters()))
        stats = m.snapshot_stats()
        assert stats["pack_count"] == stats["unpack_count"] == 3
        assert stats["peak_d2h_inflight"] <= 2 and stats["peak_owned_restore_count"] <= 2
        assert stats["pinned_capacity_bytes"] == 8 * 1024**2
        if prefetch:
            assert stats["prefetch_hits"] == 2
        # Preserve alias snapshot semantics on the actual producer stream.
        with api.stream(stream) if stream is not None else nullcontext():
            source = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
            expected = source[:, ::2].clone()
            ticket = m.pack(source[:, ::2], 0)
            source.add_(100)
            restored, repeated = m.unpack(ticket), m.unpack(ticket)
        api.synchronize()
        assert torch.equal(restored, expected) and torch.equal(repeated, expected)
        return dict(
            prefetch=prefetch,
            nondefault_stream=nondefault_stream,
            original_inputs_released=True,
            bitwise_equal=True,
            stats=stats,
        )
    finally:
        m.close()


@pytest.mark.parametrize("layers", [0, 12])
def test_bounded_runtime_rejects_partial_or_disabled_recomputation(layers):
    from veomni.models.model_runtime import VeOmniModelRuntime

    offload = SimpleNamespace(enable_async_activation=True, activation_offload_backend="bounded")
    gc_config = SimpleNamespace(enable=True, enable_reentrant=False, recompute_layers=layers)
    runtime = SimpleNamespace(
        args=SimpleNamespace(accelerator=SimpleNamespace(offload_config=offload, gradient_checkpointing=gc_config))
    )
    with pytest.raises(ValueError, match="requires recompute_layers=-1"):
        VeOmniModelRuntime._apply_async_activation_offload(runtime)


def test_bounded_runtime_accepts_full_nonreentrant_recomputation(monkeypatch):
    import veomni.distributed.bounded_activation_offload as staging
    from veomni.models.model_runtime import VeOmniModelRuntime

    calls = []
    monkeypatch.setattr(staging, "apply_bounded_activation_offload", lambda *args: calls.append(args))
    offload = SimpleNamespace(
        enable_async_activation=True,
        activation_offload_backend="bounded",
        activation_offload_modules=["layers.{*}"],
        activation_offload_staging_buffer_mib=1,
        activation_offload_pageable_limit_gb=1,
        activation_offload_prefetch_depth=1,
        activation_offload_pipeline_transfers=False,
    )
    gc_config = SimpleNamespace(enable=True, enable_reentrant=False, recompute_layers=-1)
    runtime = SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        args=SimpleNamespace(accelerator=SimpleNamespace(offload_config=offload, gradient_checkpointing=gc_config)),
    )
    VeOmniModelRuntime._apply_async_activation_offload(runtime)
    assert len(calls) == 1 and calls[0][0] is runtime.model


def test_bounded_offload_rejects_each_unmatched_pattern():
    from transformers.modeling_layers import GradientCheckpointingLayer

    from veomni.distributed.bounded_activation_offload import StagingConfig, apply_bounded_activation_offload

    model = torch.nn.Sequential(GradientCheckpointingLayer())
    with pytest.raises(ValueError, match="does_not_exist.*matched no modules"):
        apply_bounded_activation_offload(model, ["0", "does_not_exist"], StagingConfig())


def test_bounded_offload_rejects_non_checkpoint_container():
    from veomni.distributed.bounded_activation_offload import StagingConfig, apply_bounded_activation_offload

    model = torch.nn.Sequential(torch.nn.Identity())
    model[0].gradient_checkpointing = True
    with pytest.raises(ValueError, match="not a checkpoint boundary"):
        apply_bounded_activation_offload(model, ["0"], StagingConfig())


def test_bounded_offload_rejects_repeated_wildcard_target():
    from transformers.modeling_layers import GradientCheckpointingLayer

    from veomni.distributed.bounded_activation_offload import StagingConfig, apply_bounded_activation_offload

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([GradientCheckpointingLayer()])
    with pytest.raises(ValueError, match="selected a module more than once"):
        apply_bounded_activation_offload(model, ["layers.{*}", "layers.{*}"], StagingConfig())


@pytest.mark.parametrize("pattern", ["layers.{*}.does_not_exist", "layers.{*}.{*}", "layers..{*}", ".{*}"])
def test_bounded_offload_rejects_nonterminal_wildcard(pattern):
    from transformers.modeling_layers import GradientCheckpointingLayer

    from veomni.distributed.bounded_activation_offload import StagingConfig, apply_bounded_activation_offload

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([GradientCheckpointingLayer()])
    with pytest.raises(ValueError, match="requires a terminal wildcard"):
        apply_bounded_activation_offload(model, [pattern], StagingConfig())


def test_bounded_offload_rejects_noncontiguous_checkpoint_input(monkeypatch):
    from transformers.modeling_layers import GradientCheckpointingLayer

    import veomni.distributed.bounded_activation_offload as staging

    class Block(GradientCheckpointingLayer):
        gradient_checkpointing = True

        def forward(self, hidden_states):
            return hidden_states.tanh()

    monkeypatch.setattr(staging, "_NativeTransfers", CPUTransfers)
    model = torch.nn.Sequential(Block()).train()
    manager = staging.apply_bounded_activation_offload(
        model, ["0"], staging.StagingConfig(buffer_bytes=1024, pageable_limit_bytes=8192)
    )
    try:
        with pytest.raises(RuntimeError, match="contiguous hidden_states"):
            model(torch.randn(4, 8)[:, ::2].requires_grad_())
    finally:
        manager.close()


def test_bounded_offload_rejects_keyword_only_checkpoint_input(monkeypatch):
    from transformers.modeling_layers import GradientCheckpointingLayer

    import veomni.distributed.bounded_activation_offload as staging

    class Block(GradientCheckpointingLayer):
        gradient_checkpointing = True

        def forward(self, hidden_states):
            return hidden_states.sigmoid()

    monkeypatch.setattr(staging, "_NativeTransfers", CPUTransfers)
    model = torch.nn.Sequential(Block()).train()
    manager = staging.apply_bounded_activation_offload(model, ["0"], staging.StagingConfig())
    try:
        with pytest.raises(RuntimeError, match="first positional argument"):
            model[0](hidden_states=torch.randn(4, requires_grad=True))
    finally:
        manager.close()


def test_bounded_offload_rejects_checkpoint_boundary_that_does_not_capture_input(monkeypatch):
    from transformers.modeling_layers import GradientCheckpointingLayer

    import veomni.distributed.bounded_activation_offload as staging

    class Bypass(GradientCheckpointingLayer):
        gradient_checkpointing = True

        def __call__(self, hidden_states):
            return hidden_states.sin()

    monkeypatch.setattr(staging, "_NativeTransfers", CPUTransfers)
    model = torch.nn.Sequential(Bypass()).train()
    with pytest.raises(ValueError, match="overrides the checkpoint boundary"):
        staging.apply_bounded_activation_offload(model, ["0"], staging.StagingConfig())
