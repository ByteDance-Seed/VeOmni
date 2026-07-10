import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from veomni.data.data_collator import MainCollator
from veomni.distributed import parallel_state as parallel_state_module
from veomni.distributed import torch_parallelize
from veomni.distributed.parallel_state import (
    ParallelState,
    bind_model_parallel_state,
    build_parallel_state,
    clear_parallel_state_cache,
    get_model_parallel_state,
    get_parallel_state,
    resolve_model_parallel_state,
    use_model_parallel_state,
    use_parallel_state,
)
from veomni.distributed.sequence_parallel import comm
from veomni.distributed.sequence_parallel import loss as sequence_parallel_loss


class _Model:
    pass


@pytest.fixture(autouse=True)
def _reset_parallel_state(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "_PARALLEL_STATE", None)
    token = parallel_state_module._CURRENT_PARALLEL_STATE.set(None)
    yield
    parallel_state_module._CURRENT_PARALLEL_STATE.reset(token)


def test_model_owned_state_selects_each_model_independently():
    policy_state = ParallelState(async_enabled=False)
    drafter_state = ParallelState(async_enabled=True)
    policy = bind_model_parallel_state(_Model(), policy_state)
    drafter = bind_model_parallel_state(_Model(), drafter_state)

    with use_model_parallel_state(policy):
        assert get_parallel_state() is policy_state
        with use_model_parallel_state(drafter):
            assert get_parallel_state() is drafter_state
        assert get_parallel_state() is policy_state

    assert get_model_parallel_state(policy) is policy_state
    assert get_model_parallel_state(drafter) is drafter_state


def test_model_context_restores_after_exception():
    default_state = ParallelState(async_enabled=False)
    model_state = ParallelState(async_enabled=True)
    parallel_state_module._PARALLEL_STATE = default_state
    model = bind_model_parallel_state(_Model(), model_state)

    with pytest.raises(RuntimeError, match="boom"):
        with use_model_parallel_state(model):
            assert get_parallel_state() is model_state
            raise RuntimeError("boom")

    assert get_parallel_state() is default_state


def test_resolve_prefers_explicit_then_model_then_context():
    context_state = ParallelState(async_enabled=False)
    model_state = ParallelState(async_enabled=True)
    explicit_state = ParallelState(dp_mode="ddp")
    model = bind_model_parallel_state(_Model(), model_state)

    with use_parallel_state(context_state):
        assert resolve_model_parallel_state(model) is model_state
        assert resolve_model_parallel_state(model, explicit_state) is explicit_state
        assert resolve_model_parallel_state(_Model()) is context_state


def test_unbound_model_lookup_fails_loudly():
    with pytest.raises(ValueError, match="no bound ParallelState"):
        get_model_parallel_state(_Model())


def test_context_state_does_not_leak_to_worker_thread():
    default_state = ParallelState(async_enabled=False)
    model_state = ParallelState(async_enabled=True)
    parallel_state_module._PARALLEL_STATE = default_state

    with use_parallel_state(model_state), ThreadPoolExecutor(max_workers=1) as executor:
        assert get_parallel_state() is model_state
        assert executor.submit(get_parallel_state).result() is default_state


def test_build_parallelize_model_binds_explicit_state(monkeypatch):
    model = _Model()
    model_state = ParallelState(async_enabled=True)
    observed = []

    def _fake_build(model, **_kwargs):
        observed.append(get_parallel_state())
        return model

    monkeypatch.setattr(torch_parallelize, "_build_parallelize_model", _fake_build)
    result = torch_parallelize.build_parallelize_model(model, parallel_state=model_state)

    assert result is model
    assert observed == [model_state]
    assert get_model_parallel_state(model) is model_state


def test_model_context_takes_precedence_over_legacy_group_override(monkeypatch):
    legacy_group = object()
    model_group = object()
    monkeypatch.setattr(comm, "_ULYSSES_SEQUENCE_PARALLEL_GROUP", {"default": legacy_group})
    model = bind_model_parallel_state(_Model(), SimpleNamespace(ulysses_group=model_group))

    with use_model_parallel_state(model):
        assert comm.get_ulysses_sequence_parallel_group() is model_group


def test_worker_collator_drops_process_group_backed_state_before_pickle():
    parallel_state = SimpleNamespace(sp_enabled=True, sp_size=2, sp_rank=1)
    collator = MainCollator(parallel_state=parallel_state)

    assert collator.parallel_state is None
    assert all(getattr(stage, "parallel_state", None) is None for stage in collator.preforward_pipeline)
    pickle.dumps(collator)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed required")
def test_parallel_state_cache_is_scoped_to_default_process_group_lifetime():
    with tempfile.TemporaryDirectory() as tmpdir:
        first_init = Path(tmpdir) / "first"
        second_init = Path(tmpdir) / "second"
        try:
            dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{first_init}")
            first_state = build_parallel_state(device_type="cpu")
            dist.destroy_process_group()

            dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{second_init}")
            second_state = build_parallel_state(device_type="cpu")
            assert second_state is not first_state
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            clear_parallel_state_cache()


def test_reduce_loss_backward_uses_world_size_captured_during_forward(monkeypatch):
    group = object()
    monkeypatch.setattr(sequence_parallel_loss, "get_unified_sequence_parallel_group", lambda: group)
    monkeypatch.setattr(sequence_parallel_loss.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sequence_parallel_loss.dist, "get_world_size", lambda observed: 2 if observed is group else 99)

    value = torch.tensor(3.0, requires_grad=True)
    reduced = sequence_parallel_loss.ReduceLoss.apply(value, torch.tensor(1.0))

    # Backward must not resolve group/world-size again after the model context
    # that owned the forward has ended.
    monkeypatch.setattr(sequence_parallel_loss.dist, "get_world_size", lambda _group: 99)
    reduced.backward()

    torch.testing.assert_close(value.grad, torch.tensor(2.0))
