import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from threading import Event, get_ident
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

import veomni.distributed.inter_layer_replay as inter_layer_replay
from veomni.distributed.inter_layer_replay import (
    InterLayerReplayController,
    ReplayFrame,
    ReplayMemoryBudget,
    ReplayMemorySnapshot,
    ReplayState,
    _DeviceEventFifo,
    _NativeBackwardSchedule,
    _create_duplicate_ep_group,
    _rank_lists_for_mesh_dim,
    _resolve_replay_window,
    _TicketedFSDPComm,
)


def test_async_replay_launcher_runs_once_after_one_shot_start() -> None:
    owner = get_ident()
    entered = Event()
    release = Event()
    observed = []

    def replay_once():
        entered.set()
        release.wait(timeout=1)
        observed.append(get_ident())
        return "done"

    with ThreadPoolExecutor(max_workers=1) as executor:
        launcher = inter_layer_replay._AsyncReplayLauncher(replay_once, executor)

        assert not entered.wait(timeout=0.01)
        launcher.start()

        assert entered.wait(timeout=1)
        assert observed == []
        release.set()
        assert launcher.wait() == "done"

    assert launcher.done
    assert len(observed) == 1
    assert observed[0] != owner


def test_async_replay_launcher_releases_completed_callable_immediately() -> None:
    class ReplayPayload:
        pass

    def build_launcher(executor):
        payload = ReplayPayload()
        payload_ref = weakref.ref(payload)
        launcher = inter_layer_replay._AsyncReplayLauncher(
            lambda retained=payload: "done",
            executor,
        )
        return launcher, payload_ref

    with ThreadPoolExecutor(max_workers=1) as executor:
        launcher, payload_ref = build_launcher(executor)
        launcher.start()
        assert launcher.wait() == "done"

        # The persistent worker may retain the launcher, but a completed launch
        # must not keep the replay graph captured by its callable alive.
        assert payload_ref() is None


def test_native_sidecar_join_releases_launcher_schedule_and_frame_references(monkeypatch) -> None:
    class FakeEvent:
        def record(self, stream) -> None:
            pass

    class FakeStream:
        def wait_event(self, event) -> None:
            pass

    class FakeTensor:
        def record_stream(self, stream) -> None:
            pass

    class FakeLauncher:
        done = True
        released = False

        def wait(self):
            return FakeEvent()

        def release(self) -> None:
            self.released = True

    current_stream = FakeStream()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: current_stream),
    )
    controller = InterLayerReplayController(current_layer=7)
    schedule = _NativeBackwardSchedule(layer_id=7)
    launcher = FakeLauncher()
    schedule.replay_launcher = launcher
    frame = ReplayFrame(
        layer_id=6,
        layer=torch.nn.Identity(),
        run_function=lambda value: value,
        checkpoint_ctx=object(),
        attention_hidden=torch.randn(4),
        execution_stream=FakeStream(),
        native_checkpoint_frame=SimpleNamespace(recomputed={11: {object(): FakeTensor()}}),
        native_graph_task_id=11,
    )
    controller._native_schedule = schedule
    controller._native_sidecar_schedule = schedule
    controller._native_sidecar_launcher = launcher
    controller._native_sidecar_frame = frame

    controller._join_native_current_sidecar(frame, graph_task_id=11)

    assert launcher.released
    assert schedule.replay_launcher is None
    assert frame.attention_hidden is None
    assert controller._native_schedule is None
    assert controller._native_sidecar_schedule is None
    assert controller._native_sidecar_launcher is None
    assert controller._native_sidecar_frame is None


def test_sidecar_launch_applies_completion_backpressure_before_prefetch(monkeypatch) -> None:
    events = []
    controller = InterLayerReplayController(current_layer=23, window_size=23)
    controller._wait_for_completion_tail = lambda: events.append("completion_backpressure")
    controller._prefetch_previous = lambda *args, **kwargs: events.append("prefetch")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: object()))

    launched = controller._launch_native_current_sidecar(_NativeBackwardSchedule(layer_id=23))

    assert not launched
    assert events == ["completion_backpressure", "prefetch"]


def test_completion_backpressure_waits_only_for_an_unfinished_pair() -> None:
    class FakeEvent:
        def __init__(self, complete: bool) -> None:
            self.complete = complete
            self.queries = 0
            self.synchronizations = 0

        def query(self) -> bool:
            self.queries += 1
            return self.complete

        def synchronize(self) -> None:
            self.synchronizations += 1
            self.complete = True

    controller = InterLayerReplayController(current_layer=47, window_size=47)
    unfinished = FakeEvent(complete=False)
    controller._completion_tail = unfinished

    controller._wait_for_completion_tail()

    assert unfinished.queries == 1
    assert unfinished.synchronizations == 1
    assert controller._completion_tail is None

    completed = FakeEvent(complete=True)
    controller._completion_tail = completed
    controller._wait_for_completion_tail()

    assert completed.queries == 1
    assert completed.synchronizations == 0
    assert controller._completion_tail is None


def test_resource_fifo_hands_off_ownership_with_device_events(monkeypatch) -> None:
    ownership = []

    class FakeEvent:
        def __init__(self) -> None:
            self.owner = None

        def record(self, stream) -> None:
            self.owner = stream.name
            ownership.append(f"record:{stream.name}")

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            ownership.append(f"wait:{self.name}:{event.owner}")

    streams = iter((FakeStream("B"), FakeStream("F")))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: next(streams)),
    )
    fifo = _DeviceEventFifo("hccl")
    backward = fifo.reserve("B7:combine")
    replay = fifo.reserve("Fprime6:dispatch")

    fifo.acquire(backward)
    fifo.release(backward)
    fifo.acquire(replay)

    assert ownership == ["record:B", "wait:F:B"]


def test_backward_gmm_does_not_wait_when_replay_is_paused() -> None:
    schedule = _NativeBackwardSchedule(layer_id=7)

    for expected in range(2):
        ticket = schedule.before_backward_gmm()
        schedule.after_backward_gmm(ticket)
        assert ticket == expected


def test_replay_experts_wait_for_completed_backward_experts(monkeypatch) -> None:
    waits = []
    done = object()
    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.backward_experts_done = done
    schedule.backward_experts_ready.set()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=waits.append)),
    )

    schedule.before_replay_phase("experts")

    assert waits == [done]


def test_backward_experts_wait_for_published_replay_attention_event(monkeypatch) -> None:
    waits = []
    done = object()
    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    schedule.replay_attention_done = done
    schedule.replay_attention_ready.set()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=waits.append)),
    )

    schedule.on_phase("combine")

    assert waits == [done]


def test_completed_replay_attention_publishes_device_event(monkeypatch) -> None:
    records = []
    stream = object()

    class FakeEvent:
        def record(self, current_stream) -> None:
            records.append(current_stream)

    schedule = _NativeBackwardSchedule(layer_id=7)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream),
    )

    schedule.after_replay_phase("attention")

    assert schedule.replay_attention_ready.is_set()
    assert isinstance(schedule.replay_attention_done, FakeEvent)
    assert records == [stream]


def test_attention_backward_event_releases_replay_combine(monkeypatch) -> None:
    records = []
    waits = []
    stream = SimpleNamespace(wait_event=waits.append)

    class FakeEvent:
        def record(self, current_stream) -> None:
            records.append(current_stream)

    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=FakeEvent,
            current_stream=lambda: stream,
        ),
    )

    schedule.on_phase("attention")
    schedule.before_replay_phase("combine")

    assert schedule.backward_attention_ready.is_set()
    assert records == [stream]
    assert waits == [schedule.backward_attention_done]


def test_device_notify_enqueues_stage_wait_before_producer_record(monkeypatch) -> None:
    operations = []

    class FakeNotify:
        def __init__(self, name) -> None:
            self.name = name

        def wait(self, stream) -> None:
            operations.append(("wait", self.name, stream.name))

        def record(self, stream) -> None:
            operations.append(("record", self.name, stream.name))

        def close(self) -> None:
            operations.append(("close", self.name))

    stream = SimpleNamespace(name="consumer", npu_stream=11)
    producer_stream = SimpleNamespace(name="producer", npu_stream=22)
    schedule = _NativeBackwardSchedule(layer_id=7, device_notify_factory=FakeNotify)
    schedule.activate_communication_tickets()
    schedule.bind_streams(stream, producer_stream)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: stream))

    schedule.on_phase("combine")
    stream = producer_stream
    schedule.after_replay_phase("attention")

    dependency = "Fprime6:attention-to-B7:combine"
    assert operations == [
        ("wait", dependency, "consumer"),
        ("record", dependency, "producer"),
    ]


def test_same_stream_dependency_falls_back_to_recorded_event(monkeypatch) -> None:
    notify_operations = []
    event_operations = []

    class FakeNotify:
        def __init__(self, name) -> None:
            self.name = name

        def wait(self, stream) -> None:
            notify_operations.append("wait")

        def record(self, stream) -> None:
            notify_operations.append("record")

        def close(self) -> None:
            pass

    class FakeEvent:
        def record(self, stream) -> None:
            event_operations.append(("record", stream.npu_stream))

    stream = SimpleNamespace(
        npu_stream=17,
        wait_event=lambda event: event_operations.append(("wait", event)),
    )
    schedule = _NativeBackwardSchedule(layer_id=7, device_notify_factory=FakeNotify)
    schedule.activate_communication_tickets()
    schedule.bind_streams(stream, stream)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream),
    )

    schedule.after_replay_phase("attention")
    schedule.on_phase("combine")

    assert notify_operations == []
    assert event_operations == [
        ("record", 17),
        ("wait", schedule.replay_attention_done),
    ]


def test_completed_device_event_reaps_notify_without_device_sync() -> None:
    closed = []

    class FakeNotify:
        def __init__(self, name) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    schedule = _NativeBackwardSchedule(layer_id=7, device_notify_factory=FakeNotify)
    schedule.activate_communication_tickets()
    controller = InterLayerReplayController(current_layer=7)
    schedule.bind_streams(
        SimpleNamespace(npu_stream=1),
        SimpleNamespace(npu_stream=2),
    )
    controller._retired_device_schedules.append((SimpleNamespace(query=lambda: True), schedule))

    controller._reap_device_notifies()

    assert len(closed) == 6
    assert controller._retired_device_schedules == []


def test_completed_backward_experts_publish_device_event(monkeypatch) -> None:
    records = []
    waits = []
    stream = SimpleNamespace(wait_event=waits.append)

    class FakeEvent:
        def record(self, current_stream) -> None:
            records.append(current_stream)

    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    replay_dispatch_done = object()
    schedule.replay_dispatch_done = replay_dispatch_done
    schedule.replay_dispatch_ready.set()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream),
    )

    schedule.on_phase("experts")

    assert schedule.backward_experts_ready.is_set()
    assert isinstance(schedule.backward_experts_done, FakeEvent)
    assert records == [stream]
    assert waits == [replay_dispatch_done]


def test_completed_replay_experts_publish_device_event(monkeypatch) -> None:
    records = []
    stream = object()

    class FakeEvent:
        def record(self, current_stream) -> None:
            records.append(current_stream)

    schedule = _NativeBackwardSchedule(layer_id=7)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream),
    )

    schedule.after_replay_phase("experts")

    assert schedule.replay_experts_ready.is_set()
    assert isinstance(schedule.replay_experts_done, FakeEvent)
    assert records == [stream]


def test_backward_attention_waits_for_published_replay_experts_event(monkeypatch) -> None:
    waits = []
    done = object()
    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    schedule.replay_experts_done = done
    schedule.replay_experts_ready.set()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=waits.append)),
    )

    schedule.on_phase("dispatch")

    assert waits == [done]


def test_one_shot_hccl_fifo_uses_deterministic_interleaved_order(monkeypatch) -> None:
    class FakeEvent:
        def record(self, stream) -> None:
            pass

        def query(self) -> bool:
            return True

    class FakeStream:
        def wait_event(self, event) -> None:
            pass

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=FakeEvent, current_stream=FakeStream),
    )
    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    ownership = []

    class FakeLauncher:
        result = object()

        def start(self) -> None:
            ownership.append("start")

        def wait(self):
            ownership.append("join")
            return self.result

    schedule.replay_launcher = FakeLauncher()
    ownership.append("B-combine")
    schedule.on_collective("combine", "before")
    schedule.on_collective("combine", "after")
    assert schedule.collective_tickets[("replay", "dispatch")].index == 1
    assert schedule.collective_tickets[("backward", "dispatch")].index == 2
    assert schedule.collective_tickets[("replay", "combine")].index == 3
    assert schedule.collective_tickets[("backward", "reduce_scatter")].index == 4
    ownership.append("B-experts")
    first_gmm = schedule.before_backward_gmm()
    schedule.after_backward_gmm(first_gmm)
    second_gmm = schedule.before_backward_gmm()
    schedule.after_backward_gmm(second_gmm)
    schedule.replay_dispatch_done = object()
    schedule.replay_dispatch_ready.set()
    schedule.on_phase("experts")
    schedule.before_replay_collective("dispatch")
    ownership.append("F-dispatch")
    schedule.after_replay_collective("dispatch")
    schedule.before_replay_phase("experts")
    ownership.append("F-experts")
    schedule.on_collective("dispatch", "before")
    ownership.append("B-dispatch")
    schedule.on_collective("dispatch", "after")
    schedule.before_replay_collective("combine")
    ownership.append("F-combine")
    schedule.after_replay_collective("combine")
    reduce_scatter_ticket = schedule.before_fsdp_collective("current", "reduce_scatter")
    ownership.append("B-reduce-scatter")
    schedule.after_fsdp_collective(reduce_scatter_ticket)
    schedule.wait_replay()

    assert ownership.index("B-combine") < ownership.index("start")
    assert ownership.index("start") < ownership.index("F-dispatch")
    assert ownership.index("F-dispatch") < ownership.index("B-dispatch")
    assert ownership.index("B-dispatch") < ownership.index("F-combine")
    assert ownership.index("F-combine") < ownership.index("B-reduce-scatter")
    assert ownership.index("B-experts") < ownership.index("F-experts")
    assert ownership.index("F-dispatch") < ownership.index("F-combine")
    assert ownership[-1] == "join"


def test_fsdp_tickets_order_replay_all_gather_and_current_reduce_scatter(monkeypatch) -> None:
    ownership = []

    class FakeEvent:
        def record(self, stream) -> None:
            ownership.append("event")

    class FakeStream:
        def wait_event(self, event) -> None:
            ownership.append("wait")

    class FakeComm:
        def __init__(self, name: str) -> None:
            self.name = name

        def allocate(self, *args, **kwargs):
            return self.name

        def __call__(self, *args, **kwargs):
            ownership.append(self.name)
            return self.name

    stream = FakeStream()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=FakeEvent, current_stream=lambda: stream))
    schedule = _NativeBackwardSchedule(layer_id=7)
    schedule.activate_communication_tickets()
    replay_all_gather = _TicketedFSDPComm(FakeComm("F-all-gather"), lambda: schedule, 6, "all_gather")
    current_reduce_scatter = _TicketedFSDPComm(FakeComm("B-reduce-scatter"), lambda: schedule, 7, "reduce_scatter")

    schedule.begin_replay_prefetch()
    assert replay_all_gather() == "F-all-gather"
    schedule.end_replay_prefetch()
    schedule.before_current_forward_dispatch()
    assert current_reduce_scatter() == "B-reduce-scatter"
    assert replay_all_gather() == "F-all-gather"

    assert ownership == [
        "F-all-gather",
        "event",
        "wait",
        "B-reduce-scatter",
        "event",
        "wait",
        "F-all-gather",
    ]


def test_fsdp_ticket_roles_follow_the_rolling_backward_schedule() -> None:
    schedule = _NativeBackwardSchedule(layer_id=7)

    assert schedule.fsdp_role(7) == "current"
    assert schedule.fsdp_role(6) == "replay"
    assert schedule.fsdp_role(5) is None

    schedule = _NativeBackwardSchedule(layer_id=6)

    assert schedule.fsdp_role(7) is None
    assert schedule.fsdp_role(6) == "current"
    assert schedule.fsdp_role(5) == "replay"


def test_replay_memory_budget_pauses_and_resumes() -> None:
    gib = 1024**3
    snapshots = iter(
        (
            ReplayMemorySnapshot(allocated_bytes=9 * gib, free_bytes=7 * gib, total_bytes=16 * gib),
            ReplayMemorySnapshot(allocated_bytes=6 * gib, free_bytes=10 * gib, total_bytes=16 * gib),
        )
    )
    budget = ReplayMemoryBudget(
        budget_bytes=8 * gib,
        probe=lambda: next(snapshots),
    )

    paused = budget.decide()
    resumed = budget.decide()

    assert not paused.allowed
    assert paused.reason == "allocated_budget"
    assert resumed.allowed
    assert resumed.resumed


def test_replay_memory_budget_uses_observed_estimate_and_reserve() -> None:
    gib = 1024**3
    budget = ReplayMemoryBudget(
        reserve_bytes=4 * gib,
        safety_factor=1.5,
        probe=lambda: ReplayMemorySnapshot(
            allocated_bytes=8 * gib,
            free_bytes=5 * gib,
            total_bytes=16 * gib,
        ),
    )
    budget.observe_replay_bytes(1 * gib)

    decision = budget.decide()

    assert not decision.allowed
    assert decision.reason == "free_reserve"
    assert decision.estimated_bytes == int(1.5 * gib)


def test_replay_memory_budget_plans_a_bounded_step_window() -> None:
    gib = 1024**3
    budget = ReplayMemoryBudget(
        budget_bytes=8 * gib,
        probe=lambda: ReplayMemorySnapshot(
            allocated_bytes=5 * gib,
            free_bytes=11 * gib,
            total_bytes=16 * gib,
        ),
    )
    budget.observe_replay_bytes(1 * gib)

    decision = budget.plan_slots(7)

    assert decision.allowed
    assert decision.slots == 2


def test_paused_controller_retries_only_at_top_boundary() -> None:
    probes = 0

    def probe() -> ReplayMemorySnapshot:
        nonlocal probes
        probes += 1
        return ReplayMemorySnapshot(allocated_bytes=2, free_bytes=1, total_bytes=3)

    controller = InterLayerReplayController(
        current_layer=7,
        window_size=7,
        memory_budget_gb=1 / (1024**3),
        memory_retry_steps=2,
        memory_probe=probe,
    )

    assert not controller._memory_allows_replay(7, 6)
    assert not controller._memory_allows_replay(6, 5)
    assert probes == 1

    assert not controller._memory_allows_replay(7, 6)
    assert probes == 1

    assert not controller._memory_allows_replay(7, 6)
    assert probes == 2


def test_rolling_replay_memory_slots_pause_and_resume() -> None:
    snapshots = iter(
        (
            ReplayMemorySnapshot(allocated_bytes=2, free_bytes=1, total_bytes=3),
            ReplayMemorySnapshot(allocated_bytes=0, free_bytes=3, total_bytes=3),
        )
    )
    controller = InterLayerReplayController(
        current_layer=7,
        window_size=3,
        memory_budget_gb=1 / (1024**3),
        memory_retry_steps=1,
        memory_probe=lambda: next(snapshots),
    )

    assert not controller._memory_allows_replay(7, 6)
    assert controller._memory_allows_replay(7, 6)


def test_execution_communicators_alternate_across_the_replay_window() -> None:
    controller = InterLayerReplayController(current_layer=7, window_size=4)

    assert [controller._execution_domain_index(layer_id) for layer_id in range(7, 2, -1)] == [0, 1, 0, 1, 0]


def test_all_layer_window_covers_every_backward_replay_boundary() -> None:
    controller = InterLayerReplayController(current_layer=7, window_size=7)

    assert controller.bottom_layer == 0
    assert [(layer_id, layer_id - 1) for layer_id in range(controller.current_layer, controller.bottom_layer, -1)] == [
        (7, 6),
        (6, 5),
        (5, 4),
        (4, 3),
        (3, 2),
        (2, 1),
        (1, 0),
    ]


def test_arbitrary_depth_window_admits_every_replay_boundary() -> None:
    controller = InterLayerReplayController(current_layer=47, window_size=47)

    assert [(layer_id, layer_id - 1) for layer_id in range(controller.current_layer, controller.bottom_layer, -1)] == [
        (layer_id, layer_id - 1) for layer_id in range(47, 0, -1)
    ]


def test_auto_window_resolution_is_independent_of_model_depth() -> None:
    assert _resolve_replay_window(8, -1, 0) == (7, 7)
    assert _resolve_replay_window(24, -1, 0) == (23, 23)
    assert _resolve_replay_window(48, -1, 0) == (47, 47)
    assert _resolve_replay_window(48, 31, 8) == (31, 8)


def test_replays_share_the_sidecar_stream_and_alternate_ep_communicators() -> None:
    primary_group = object()
    replay_group = object()
    controller = InterLayerReplayController(
        current_layer=7,
        primary_ep_group=primary_group,
        alternate_ep_group=replay_group,
    )
    controller._primary_stream = object()
    controller._alternate_stream = object()
    replay6 = ReplayFrame(6, object(), lambda value: value, None)
    replay5 = ReplayFrame(5, object(), lambda value: value, None)

    controller._assign_execution_domain(replay6)
    controller._assign_execution_domain(replay5)

    assert replay6.execution_stream is controller._alternate_stream
    assert replay5.execution_stream is controller._alternate_stream
    assert replay6.execution_group is replay_group
    assert replay5.execution_group is primary_group


def test_rank_lists_for_mesh_dim_groups_ranks_by_ep_axis() -> None:
    mesh = SimpleNamespace(
        mesh=torch.arange(16).reshape(2, 2, 4),
        mesh_dim_names=("ep_replicate", "ep_fsdp", "ep"),
    )

    assert _rank_lists_for_mesh_dim(mesh, "ep") == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_duplicate_ep_group_uses_current_rank_subgroup(monkeypatch) -> None:
    primary_group = object()
    created_groups = []
    mesh = SimpleNamespace(
        mesh=torch.arange(16).reshape(4, 4),
        mesh_dim_names=("ep_fsdp", "ep"),
    )
    parallel_state = SimpleNamespace(
        ep_group=primary_group,
        extra_parallel_fsdp_device_mesh={"ep": mesh},
    )

    def fake_new_group(ranks, backend):
        group = {"ranks": tuple(ranks), "backend": backend}
        created_groups.append(group)
        return group

    monkeypatch.setattr(inter_layer_replay.dist, "get_rank", lambda: 10)
    monkeypatch.setattr(inter_layer_replay.dist, "get_backend", lambda group: "hccl")
    monkeypatch.setattr(inter_layer_replay.dist, "new_group", fake_new_group)

    selected = _create_duplicate_ep_group(parallel_state)

    assert [group["ranks"] for group in created_groups] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    ]
    assert selected == {"ranks": (8, 9, 10, 11), "backend": "hccl"}


def test_recomputed_tensor_storage_is_handed_to_the_backward_stream() -> None:
    class SavedTensor:
        def __init__(self) -> None:
            self.streams = []

        def record_stream(self, stream) -> None:
            self.streams.append(stream)

    stream = object()
    saved = SavedTensor()
    checkpoint_frame = SimpleNamespace(recomputed={42: {object(): saved}})
    replay = ReplayFrame(6, object(), lambda value: value, None)
    replay.native_checkpoint_frame = checkpoint_frame
    replay.native_graph_task_id = 42

    replay.record_recomputed_tensors(stream)

    assert saved.streams == [stream]


def test_native_current_checkpoint_launches_sidecar_at_recompute_dispatch(monkeypatch) -> None:
    from veomni.ops.kernels.moe import npu_group_gemm

    events = []
    calls = 0

    @contextmanager
    def dispatch_launch(callback):
        callback()
        yield

    monkeypatch.setattr(npu_group_gemm, "npu_ep_dispatch_launch_callback", dispatch_launch)

    @contextmanager
    def record_context(name: str):
        events.append(f"{name}:enter")
        try:
            yield
        finally:
            events.append(f"{name}:exit")

    def native_context_fn():
        return record_context("forward_context"), record_context("recompute_context")

    def run_function(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        events.append(f"function:{calls}")
        return torch.sin(value)

    controller = InterLayerReplayController(current_layer=7)
    controller._launch_native_current_sidecar = lambda schedule: events.append("sidecar")
    checkpoint_func = partial(checkpoint, use_reentrant=False, context_fn=native_context_fn)
    value = torch.randn(4, requires_grad=True)
    value.register_hook(lambda grad: events.append("input_backward"))

    output = controller.checkpoint_current(checkpoint_func, run_function, value)
    output.sum().backward()

    assert calls == 2
    assert value.grad is not None
    assert controller._frames == {}
    assert events.index("sidecar") < events.index("function:2")
    assert events.index("recompute_context:enter") < events.index("sidecar")


def test_native_replay_checkpoint_keeps_backward_in_native_autograd() -> None:
    calls = 0

    def run_function(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.sin(value).square()

    controller = InterLayerReplayController(current_layer=7, window_size=1)
    value = torch.randn(4, requires_grad=True)
    output = controller.checkpoint_native_replay(
        partial(checkpoint, use_reentrant=False),
        6,
        torch.nn.Identity(),
        run_function,
        value,
    )

    output.sum().backward()

    assert calls == 2
    assert value.grad is not None


def test_expert_backward_boundary_reshards_before_replay_experts_are_released() -> None:
    events = []
    experts = SimpleNamespace(reshard=lambda: events.append("reshard"))
    schedule = _NativeBackwardSchedule(layer_id=7, backward_experts_module=experts)
    schedule.communication_active = True
    schedule.publish_backward_experts_done = lambda: events.append("publish")
    schedule._wait_for_dependency = lambda *args: events.append("wait_dispatch")

    schedule.on_phase("experts")

    assert events == ["reshard", "publish", "wait_dispatch"]


def test_controller_binds_expert_fsdp_module_to_backward_schedule() -> None:
    experts = SimpleNamespace(reshard=lambda: None)
    layer = SimpleNamespace(mlp=SimpleNamespace(experts=experts))
    controller = InterLayerReplayController(
        current_layer=7,
        layer_modules={7: layer},
    )

    schedule = controller._new_backward_schedule(7)

    assert schedule.backward_experts_module is experts


def test_replay_backward_release_breaks_native_checkpoint_reference_cycle() -> None:
    def build_replay_cycle():
        frame = ReplayFrame(
            layer_id=6,
            layer=torch.nn.Identity(),
            run_function=lambda value: value,
            checkpoint_ctx=object(),
            detached_inputs=(torch.randn(4, requires_grad=True),),
            inner_recompute_context=object(),
        )
        replay_holder = [frame]
        frame.native_checkpoint_frame = SimpleNamespace(recompute_fn=lambda: replay_holder[0])
        frame.native_graph_task_id = 1

        frame.release_cached_graph()

        assert frame.detached_inputs is None
        assert frame.native_checkpoint_frame is None
        assert frame.native_graph_task_id is None
        assert frame.inner_recompute_context is None
        assert frame.checkpoint_ctx is None
        return weakref.ref(frame)

    frame_ref = build_replay_cycle()

    # Refcounting should reclaim the frame immediately; this must not rely on
    # the trainer's infrequent cyclic garbage collection.
    assert frame_ref() is None


def test_replay_frame_releases_phase_local_storage_at_last_consumer() -> None:
    frame = ReplayFrame(
        layer_id=6,
        layer=torch.nn.Identity(),
        run_function=lambda value: value,
        checkpoint_ctx=object(),
        attention_hidden=object(),
        moe_input=object(),
        routing_weights=object(),
        selected_experts=object(),
        dispatch_plan=object(),
        dispatch_input=object(),
        dispatch_state=SimpleNamespace(output=object()),
        dispatched_input=object(),
        expert_output=object(),
        combine_state=object(),
        replay_output=object(),
    )

    frame.release_after_dispatch()
    assert frame.selected_experts is None
    assert frame.dispatch_plan is None
    assert frame.dispatch_input is None
    assert frame.dispatch_state.output is None

    frame.release_after_experts()
    assert frame.dispatched_input is None

    frame.release_after_combine()
    assert frame.routing_weights is None
    assert frame.dispatch_state is None
    assert frame.expert_output is None
    assert frame.combine_state is None
    assert frame.attention_hidden is None
    assert frame.moe_input is None

    frame.release_after_recompute_capture()
    assert frame.replay_output is None


def test_native_replay_checkpoint_launches_the_next_rolling_sidecar() -> None:
    events = []
    controller = InterLayerReplayController(current_layer=7, window_size=2)
    controller._join_native_current_sidecar = lambda frame, graph_task_id: events.append(f"join:{frame.layer_id}")
    controller._launch_native_current_sidecar = lambda schedule: events.append(f"launch:{schedule.layer_id}")
    value = torch.randn(4, requires_grad=True)
    output = controller.checkpoint_native_replay(
        partial(checkpoint, use_reentrant=False),
        6,
        torch.nn.Identity(),
        lambda tensor: torch.sin(tensor).square(),
        value,
    )

    output.sum().backward()

    assert events == ["join:6", "launch:6"]
    assert value.grad is not None


def test_replay_attention_builds_a_connected_graph_without_a_second_forward() -> None:
    class ToyGate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = 4
            self.top_k = 1
            self.norm_topk_prob = False
            self.weight = torch.nn.Parameter(torch.ones(2, 4))

    class ToyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = torch.nn.Linear(4, 4, bias=False)
            self.post_attention_layernorm = torch.nn.Linear(4, 4, bias=False)
            self.mlp = SimpleNamespace(gate=ToyGate())
            self.attention_calls = 0

        def unshard(self, async_op: bool = False) -> None:
            assert not async_op

        def forward_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.attention_calls += 1
            return self.attention(hidden_states)

    layer = ToyLayer()
    hidden_states = torch.randn(2, 4, requires_grad=True)
    frame = ReplayFrame(
        layer_id=6,
        layer=layer,
        run_function=lambda value: value,
        checkpoint_ctx=None,
        detached_inputs=(hidden_states,),
    )
    controller = InterLayerReplayController(current_layer=7)

    controller._prepare_replay_attention(frame)
    assert frame.state is ReplayState.ATTN_READY
    assert frame.attention_hidden is not None and frame.attention_hidden.grad_fn is not None
    assert frame.moe_input is not None and frame.moe_input.grad_fn is not None

    assert layer.attention_calls == 1
