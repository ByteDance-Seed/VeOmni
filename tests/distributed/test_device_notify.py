from types import SimpleNamespace

import pytest

from veomni.distributed.device_notify import DeviceNotify


class FakeRuntime:
    def __init__(self) -> None:
        self.operations = []

    def create(self, name):
        self.operations.append(("create", name))
        return object()

    def wait_and_reset(self, handle, stream_handle, timeout_ms, name):
        self.operations.append(("wait", name, stream_handle, timeout_ms))

    def record(self, handle, stream_handle, name):
        self.operations.append(("record", name, stream_handle))

    def destroy(self, handle, name):
        self.operations.append(("destroy", name))


def test_device_notify_supports_wait_before_record() -> None:
    runtime = FakeRuntime()
    notify = DeviceNotify("B7-to-F6", runtime=runtime, timeout_ms=321)

    notify.wait(SimpleNamespace(npu_stream=11))
    notify.record(SimpleNamespace(npu_stream=22))
    notify.close()

    assert runtime.operations == [
        ("create", "B7-to-F6"),
        ("wait", "B7-to-F6", 11, 321),
        ("record", "B7-to-F6", 22),
        ("destroy", "B7-to-F6"),
    ]


@pytest.mark.parametrize("operation", ["wait", "record"])
def test_device_notify_rejects_duplicate_operations(operation) -> None:
    runtime = FakeRuntime()
    notify = DeviceNotify("one-shot", runtime=runtime)
    stream = SimpleNamespace(npu_stream=1)

    getattr(notify, operation)(stream)

    with pytest.raises(RuntimeError, match=f"was {operation}ed twice"):
        getattr(notify, operation)(stream)


def test_device_notify_close_is_idempotent() -> None:
    runtime = FakeRuntime()
    notify = DeviceNotify("close", runtime=runtime)

    notify.close()
    notify.close()

    assert runtime.operations.count(("destroy", "close")) == 1
