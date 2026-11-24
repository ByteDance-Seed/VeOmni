from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from veomni.trainer.callbacks import EnvironMeterCallback, TrainerState


def _callback(parallel_state):
    # The test owns only the train-begin boundary.  Constructing EnvironMeter here
    # would require a real default process group and test unrelated initialization.
    callback = object.__new__(EnvironMeterCallback)
    callback.trainer = SimpleNamespace()
    callback.parallel_state = parallel_state
    return callback


def test_environment_meter_initializes_its_collectives_before_training():
    ep_fsdp_group = object()
    ep_group = object()
    ep_fsdp_mesh = {"ep_fsdp": SimpleNamespace(get_group=Mock(return_value=ep_fsdp_group))}
    parallel_state = SimpleNamespace(
        is_initialized=True,
        dp_group=object(),
        fsdp_group=object(),
        extra_parallel_names=("ep", "disabled"),
        extra_parallel_fsdp_device_mesh={"ep": ep_fsdp_mesh, "disabled": None},
        extra_parallel_enabled=lambda name: name == "ep",
        extra_parallel_group=Mock(return_value=ep_group),
    )
    callback = _callback(parallel_state)

    with patch("veomni.trainer.callbacks.trace_callback.all_reduce", Mock()) as reduce:
        callback.on_train_begin(TrainerState())

    assert reduce.call_args_list == [
        call([0.0, 0.0, 0.0], op="sum", group=parallel_state.dp_group),
        call([0.0], op="sum", group=parallel_state.fsdp_group),
        call([0.0], op="sum", group=ep_fsdp_group),
        call([0.0], op="sum", group=ep_group),
    ]
    parallel_state.extra_parallel_group.assert_called_once_with("ep")


def test_environment_meter_skips_collective_initialization_without_distributed():
    callback = _callback(SimpleNamespace(is_initialized=False))

    with patch("veomni.trainer.callbacks.trace_callback.all_reduce", Mock()) as reduce:
        callback.on_train_begin(TrainerState())

    reduce.assert_not_called()
