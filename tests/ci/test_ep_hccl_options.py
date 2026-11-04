"""Expert communication options are explicit, validated and accelerator-local."""

import sys
from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.hccl_options import ep_backend_override, validate_ep_hccl_options
from veomni.utils.device import IS_NPU_AVAILABLE


@pytest.mark.parametrize(
    "options",
    [
        {"npu_ep_hccl_buffer_mib": 32},
        {"npu_ep_hccl_expansion_mode": "host"},
        {"npu_ep_hccl_buffer_mib": 32, "npu_ep_hccl_expansion_mode": "host"},
    ],
)
def test_config_rejects_hccl_options_when_effective_ep_is_disabled(monkeypatch, options):
    from veomni.arguments.arguments_types import AcceleratorConfig

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="require an enabled ep dimension"):
        AcceleratorConfig(ep_size=1, **options)
    # Restored extra-parallel metadata determines the mesh, even if ep_size differs.
    with pytest.raises(ValueError, match="require an enabled ep dimension"):
        AcceleratorConfig(
            ep_size=2,
            extra_parallel_names=["ep"],
            extra_parallel_sizes=[1],
            extra_parallel_placement_innermost=[False],
            **options,
        )


def test_config_accepts_hccl_options_for_effective_ep_and_default_without_ep(monkeypatch):
    from dataclasses import asdict

    from veomni.arguments.arguments_types import AcceleratorConfig, MixedPrecisionConfig
    from veomni.arguments.parser import _instantiate_recursive

    monkeypatch.setenv("WORLD_SIZE", "2")
    assert AcceleratorConfig(ep_size=1).ep_size == 1
    config = AcceleratorConfig(ep_size=2, npu_ep_hccl_buffer_mib=32)
    assert dict(zip(config.extra_parallel_names, config.extra_parallel_sizes))["ep"] == 2
    roundtrip = _instantiate_recursive(AcceleratorConfig, asdict(config))
    assert roundtrip.extra_parallel_names.count("ep") == 1
    assert dict(zip(roundtrip.extra_parallel_names, roundtrip.extra_parallel_sizes))["ep"] == 2
    assert isinstance(roundtrip.fsdp_config.mixed_precision, MixedPrecisionConfig)
    restored = AcceleratorConfig(
        ep_size=1,
        extra_parallel_names=["ep"],
        extra_parallel_sizes=[2],
        extra_parallel_placement_innermost=[False],
        npu_ep_hccl_expansion_mode="host",
    )
    assert dict(zip(restored.extra_parallel_names, restored.extra_parallel_sizes))["ep"] == 2


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


def _native_ep_mesh_worker(rank, endpoint):
    """Observe real mesh construction, then exercise its actual HCCL communicators."""
    from datetime import timedelta
    from unittest.mock import patch

    import torch
    import torch.distributed as dist

    from veomni.distributed import parallel_state as ps
    from veomni.utils.device import get_torch_device

    get_torch_device().set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group("hccl", store=endpoint.connect(), rank=rank, world_size=4, timeout=timedelta(seconds=180))
    original = ps.init_device_mesh
    observed = []

    def record_mesh(*args, **kwargs):
        """Record arguments without substituting the native mesh or backend."""
        snapshot = kwargs.copy()
        if kwargs.get("backend_override") is not None:
            # init_device_mesh consumes this mapping with pop(). Keep the
            # options objects, but snapshot the mapping before native setup.
            snapshot["backend_override"] = dict(kwargs["backend_override"])
        observed.append(snapshot)
        return original(*args, **kwargs)

    try:
        with patch.object(ps, "init_device_mesh", record_mesh):
            state = ps._init_parallel_state(
                dp_size=4,
                dp_shard_size=4,
                device_type="npu",
                name="ep-native",
                extra_parallel_sizes=(2, 2),
                extra_parallel_names=("ep", "ple"),
                extra_parallel_placement_innermost=(False, False),
                npu_ep_hccl_buffer_mib=32,
                npu_ep_hccl_expansion_mode="host",
            )
        configured = [call for call in observed if call.get("backend_override")]
        assert len(configured) == 1
        assert "ep" in configured[0]["mesh_dim_names"]
        override = configured[0]["backend_override"]
        assert list(override) == ["ep"]
        assert override["ep"][0] == "hccl"
        assert override["ep"][1].hccl_config["hccl_buffer_size"] == 32
        assert override["ep"][1].hccl_config["hccl_op_expansion_mode"] == 1
        for call in observed:
            if "ep" not in call["mesh_dim_names"]:
                assert not call.get("backend_override")
        expected_parallel_ranks = [rank // 2 * 2, rank // 2 * 2 + 1]
        for label, group, expected_ranks in (
            ("ep", state.ep_group, expected_parallel_ranks),
            ("ple", state.extra_parallel_group("ple"), expected_parallel_ranks),
            ("fsdp", state.fsdp_group, [0, 1, 2, 3]),
        ):
            ranks = dist.get_process_group_ranks(group)
            assert ranks == expected_ranks, f"Unexpected {label} members on rank {rank}: {ranks}"
            value = torch.tensor(float(rank + 1), device=device)
            dist.all_reduce(value, group=group)
            assert value.item() == sum(r + 1 for r in expected_ranks)
    finally:
        ps.clear_parallel_state()
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4))
@pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Requires native Ascend")
def test_native_ep_hccl_mesh_and_collectives():
    """Verify EP-only options through _init_parallel_state and real HCCL all-reduces."""
    from tests.tools.launch_utils import spawn_with_rendezvous
    from veomni.utils.device import IS_NPU_AVAILABLE, get_torch_device

    if not IS_NPU_AVAILABLE or get_torch_device().device_count() < 4:
        pytest.skip("Requires four native Ascend devices")
    spawn_with_rendezvous(_native_ep_mesh_worker, nprocs=4)
