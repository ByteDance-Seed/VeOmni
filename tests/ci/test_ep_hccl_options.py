"""Expert communication options are explicit, validated and accelerator-local."""

import sys
from types import SimpleNamespace

import pytest

from veomni.distributed.hccl_options import ep_backend_override, validate_ep_hccl_options


@pytest.mark.parametrize("size,mode", [(0, None), (-1, None), (True, None), (1.5, None), (None, "invalid")])
def test_invalid_options_fail(size, mode):
    """Bad options fail before constructing any distributed communicator."""
    with pytest.raises(ValueError):
        validate_ep_hccl_options(size, mode)


def test_default_is_device_independent():
    """Existing CPU, GPU and NPU configurations need no TorchNPU import."""
    for device in ("cpu", "cuda", "npu"):
        assert ep_backend_override(device, None, None) == {}
    with pytest.raises(ValueError, match="require device_type=npu"):
        ep_backend_override("cuda", 1280, "host")


@pytest.mark.parametrize("mode,code", [("host", 1), ("device", 2), (None, None)])
def test_only_ep_gets_requested_hccl_options(monkeypatch, mode, code):
    """The public DeviceMesh override carries no rank or FSDP topology changes."""
    fake = SimpleNamespace(
        _C=SimpleNamespace(
            _distributed_c10d=SimpleNamespace(ProcessGroupHCCL=SimpleNamespace(Options=SimpleNamespace))
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    override = ep_backend_override("npu", 1280, mode)
    assert list(override) == ["ep"]
    backend, options = override["ep"]
    assert backend == "hccl"
    assert options.hccl_config["hccl_buffer_size"] == 1280
    assert options.hccl_config.get("hccl_op_expansion_mode") == code
