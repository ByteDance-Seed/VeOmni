import copy
import gc
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from veomni.arguments import FSDPConfig, MixedPrecisionConfig
from veomni.arguments.arguments_types import validate_reduce_scatter_transport
from veomni.arguments.parser import _instantiate_recursive
from veomni.distributed import torch_parallelize
from veomni.distributed.fsdp2 import reduce_scatter as reduce_scatter_module
from veomni.distributed.fsdp2.reduce_scatter import (
    FP32ReduceScatterWithLowPrecisionTransport,
    ReduceScatterTransportPolicy,
    register_fp32_reduce_scatter_with_low_precision_transport,
)
from veomni.distributed.torch_parallelize import _configure_fsdp_gradient_reduction
from veomni.utils import device as device_utils
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("reduction_scale", [1.0, 0.5, 1.0 / 3.0])
def test_low_precision_transport_accumulates_and_outputs_fp32(monkeypatch, dtype, reduction_scale):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist,
        "all_to_all_single",
        lambda output, input, group, async_op: output.copy_(input),
    )

    input_tensor = torch.tensor([256.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    output_tensor = torch.empty(2, dtype=torch.float32)
    comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale)

    result = comm(output_tensor, input_tensor, object(), dist.ReduceOp.SUM)

    assert result is None
    expected = input_tensor.to(dtype).view(2, -1).float().sum(dim=0).mul(reduction_scale)
    torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)


def test_fp16_transport_obeys_fp16_finite_range(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist,
        "all_to_all_single",
        lambda output, input, group, async_op: output.copy_(input),
    )

    input_tensor = torch.tensor([70000.0, 0.0], dtype=torch.float32)
    fp16_output = torch.empty(1, dtype=torch.float32)
    bf16_output = torch.empty(1, dtype=torch.float32)

    FP32ReduceScatterWithLowPrecisionTransport(torch.float16, reduction_scale=1.0)(
        fp16_output, input_tensor, object(), dist.ReduceOp.SUM
    )
    FP32ReduceScatterWithLowPrecisionTransport(torch.bfloat16, reduction_scale=1.0)(
        bf16_output, input_tensor, object(), dist.ReduceOp.SUM
    )

    assert torch.isinf(fp16_output).all()
    assert torch.isfinite(bf16_output).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_rejects_async(dtype):
    with pytest.raises(NotImplementedError, match="async_op=True"):
        FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=0.5)(
            torch.empty(2),
            torch.empty(4),
            object(),
            dist.ReduceOp.SUM,
            async_op=True,
        )


def test_non_fp32_reduction_buffers_are_rejected():
    with pytest.raises(TypeError, match="requires FP32"):
        FP32ReduceScatterWithLowPrecisionTransport(torch.bfloat16, reduction_scale=0.5)(
            torch.empty(2, dtype=torch.bfloat16),
            torch.empty(4, dtype=torch.bfloat16),
            object(),
            dist.ReduceOp.SUM,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_validates_contract(monkeypatch, dtype):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=0.5)

    with pytest.raises(ValueError, match="one equally sized output shard"):
        comm(torch.empty(3), torch.empty(4), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="requires SUM"):
        comm(
            torch.empty(2),
            torch.empty(4),
            object(),
            dist.ReduceOp.AVG,
        )


def test_registers_only_selected_fsdp_modules_and_moves_scaling_into_hook(monkeypatch):
    class FakeFSDPModule:
        def __init__(self) -> None:
            self.comms = []
            self.gradient_divide_factors = []
            self.force_sum_reductions = []

        def set_gradient_divide_factor(self, factor) -> None:
            self.gradient_divide_factors.append(factor)

        def set_force_sum_reduction_for_comms(self, enable) -> None:
            self.force_sum_reductions.append(enable)

        def set_custom_reduce_scatter(self, comm) -> None:
            self.comms.append(comm)

    class FakeModel:
        def __init__(self) -> None:
            self.fsdp1 = FakeFSDPModule()
            self.unwrapped = object()
            self.fsdp2 = FakeFSDPModule()

        def modules(self):
            return [self, self.fsdp1, self.unwrapped, self.fsdp2]

    monkeypatch.setattr(reduce_scatter_module, "FSDPModule", FakeFSDPModule)
    model = FakeModel()

    count = register_fp32_reduce_scatter_with_low_precision_transport(
        model,
        transport_dtype=torch.bfloat16,
        reduction_scales={model.fsdp1: 0.25},
    )

    assert count == 1
    assert len(model.fsdp1.comms) == 1
    assert model.fsdp1.comms[0]._transport_dtype == torch.bfloat16
    assert model.fsdp1.comms[0]._reduction_scale == 0.25
    assert model.fsdp1.gradient_divide_factors == [1.0]
    assert model.fsdp1.force_sum_reductions == [True]
    assert model.fsdp2.comms == []
    assert model.fsdp2.gradient_divide_factors == []
    assert model.fsdp2.force_sum_reductions == []


@pytest.mark.parametrize("use_low_precision_transport", [False, True])
def test_fsdp_gradient_scaling_uses_custom_path_only_when_needed(use_low_precision_transport):
    class FakeFSDPModule:
        def __init__(self) -> None:
            self.gradient_divide_factors = []

        def set_gradient_divide_factor(self, factor) -> None:
            self.gradient_divide_factors.append(factor)

    module = FakeFSDPModule()
    reduction_scales = {}
    _configure_fsdp_gradient_reduction(
        module,
        gradient_divide_factor=8.0,
        use_low_precision_transport=use_low_precision_transport,
        transport_reduction_scales=reduction_scales,
    )

    if not use_low_precision_transport:
        assert module.gradient_divide_factors == [8.0]
        assert reduction_scales == {}
    else:
        assert module.gradient_divide_factors == []
        assert reduction_scales == {module: 0.125}


class _FakeReductionMesh:
    def __init__(self, shard_group, replica_group=None):
        self.groups = {"dp_shard_sp": shard_group}
        if replica_group is not None:
            self.groups = {"dp_replicate": replica_group, **self.groups}
        self.mesh_dim_names = tuple(self.groups)

    def get_group(self, name):
        return self.groups[name]


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("9B1E4328-5347-4C5D-8E18-492833332CA1\n", "9b1e4328-5347-4c5d-8e18-492833332ca1"),
        ("", None),
        ("not-a-node-id", None),
        ("00000000-0000-0000-0000-000000000000", None),
        ("ffffffff-ffff-ffff-ffff-ffffffffffff", None),
    ],
)
def test_node_identity_requires_valid_kernel_boot_id(monkeypatch, contents, expected):
    def read_text(path):
        assert str(path) == "/proc/sys/kernel/random/boot_id"
        return contents

    monkeypatch.setattr(Path, "read_text", read_text)
    assert reduce_scatter_module._get_node_id() == expected


def test_unreadable_node_identity_does_not_trust_environment(monkeypatch):
    def read_text(path):
        raise PermissionError("kernel identity unavailable")

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setenv("HOSTNAME", "apparently-shared-node")
    monkeypatch.setenv("NODE_NAME", "apparently-shared-node")
    assert reduce_scatter_module._get_node_id() is None


@pytest.mark.parametrize(
    ("node_ids", "expected"),
    [(["node-a", "node-a"], True), (["node-a", "node-b"], False), (["node-a", None], False)],
)
def test_transport_policy_uses_actual_group_and_caches_decision(monkeypatch, node_ids, expected):
    group = object()
    calls = []
    identity_reads = []

    def get_node_id():
        identity_reads.append(True)
        return node_ids[0]

    def gather(output, value, *, group):
        calls.append((group, value))
        output[:] = node_ids

    monkeypatch.setattr(reduce_scatter_module, "_get_node_id", get_node_id)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    # Group members need not be contiguous global ranks.
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [1, 7])
    monkeypatch.setattr(dist, "all_gather_object", gather)
    policy = ReduceScatterTransportPolicy()
    assert policy.can_use(_FakeReductionMesh(group)) is expected
    assert policy.can_use(_FakeReductionMesh(group)) is expected
    assert calls == [(group, node_ids[0])]
    assert identity_reads == [True]


@pytest.mark.parametrize("has_replica_group", [False, True])
def test_singleton_transport_policy_does_not_communicate(monkeypatch, has_replica_group):
    def fail_gather(*args, **kwargs):
        raise AssertionError("singleton shard groups do not need topology communication")

    monkeypatch.setattr(reduce_scatter_module, "_get_node_id", lambda: None)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(dist, "all_gather_object", fail_gather)
    mesh = _FakeReductionMesh(object(), object() if has_replica_group else None)
    assert not ReduceScatterTransportPolicy().can_use(mesh)


def test_transport_policy_does_not_hide_collective_failures(monkeypatch):
    def fail_gather(*args, **kwargs):
        raise RuntimeError("topology collective failed")

    monkeypatch.setattr(reduce_scatter_module, "_get_node_id", lambda: None)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_gather_object", fail_gather)
    with pytest.raises(RuntimeError, match="topology collective failed"):
        ReduceScatterTransportPolicy().can_use(_FakeReductionMesh(object()))


@pytest.mark.parametrize("local_ids", [["node-a", "node-a"], ["node-a", "node-b"], [None, None]])
@pytest.mark.parametrize("peer_status", ["node_local", "cross_node", "unknown"])
@pytest.mark.parametrize("shard_rank", [0, 1])
def test_hsdp_replica_consensus_runs_even_for_rejected_shard(monkeypatch, caplog, local_ids, peer_status, shard_rank):
    # VeOmni's logger does not propagate to pytest's root capture handler.
    monkeypatch.setattr(reduce_scatter_module.logger, "handlers", [caplog.handler])
    caplog.set_level("WARNING", logger=reduce_scatter_module.__name__)
    shard_group, replica_group = object(), object()
    calls = []

    def gather(output, value, *, group):
        calls.append(group)
        output[:] = local_ids if group is shard_group else [value, peer_status]

    monkeypatch.setattr(reduce_scatter_module, "_get_node_id", lambda: local_ids[0])
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: shard_rank)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [1, 7])
    monkeypatch.setattr(dist, "all_gather_object", gather)
    policy = ReduceScatterTransportPolicy()
    mesh = _FakeReductionMesh(shard_group, replica_group)
    expected = local_ids == ["node-a", "node-a"] and peer_status == "node_local"
    assert policy.can_use(mesh) is expected
    assert policy.can_use(mesh) is expected
    assert calls == [shard_group, replica_group]
    warnings = [record for record in caplog.records if record.name == reduce_scatter_module.__name__]
    if expected or shard_rank != 0:
        assert not warnings
    else:
        assert len(warnings) == 1
        assert warnings[0].levelname == "WARNING"
        message = warnings[0].getMessage()
        assert "Using native ReduceScatter for shard group [1, 7]" in message
        reason = "node identity is unavailable" if None in local_ids or peer_status == "unknown" else "spans nodes"
        assert reason in message


def test_hsdp_decision_cache_includes_replica_group(monkeypatch):
    shard_group, replica_a, replica_b = object(), object(), object()
    calls = []

    def gather(output, value, *, group):
        calls.append(group)
        output[:] = [value, "cross_node" if group is replica_b else value]

    monkeypatch.setattr(reduce_scatter_module, "_get_node_id", lambda: "node-a")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    policy = ReduceScatterTransportPolicy()
    assert policy.can_use(_FakeReductionMesh(shard_group))
    assert policy.can_use(_FakeReductionMesh(shard_group, replica_a))
    assert not policy.can_use(_FakeReductionMesh(shard_group, replica_b))
    assert policy.can_use(_FakeReductionMesh(shard_group, replica_a))
    assert calls == [shard_group, replica_a, replica_b]


@pytest.mark.parametrize("comm_dtype", [None, "bfloat16", "float16", "float32"])
def test_reduce_scatter_comm_dtype_configuration_reaches_builder(monkeypatch, comm_dtype):
    config = _instantiate_recursive(
        FSDPConfig,
        {
            "reduce_scatter_comm_dtype": comm_dtype,
            "mixed_precision": {"param_dtype": comm_dtype or "bfloat16", "reduce_dtype": "float32"},
        },
    )
    calls = []

    def parallelize(**kwargs):
        calls.append(kwargs)
        return kwargs["model"]

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", parallelize)
    model = nn.Linear(2, 2)
    assert (
        torch_parallelize.build_parallelize_model(
            model,
            mixed_precision=config.mixed_precision,
            reduce_scatter_comm_dtype=config.reduce_scatter_comm_dtype,
            enable_gradient_checkpointing=False,
        )
        is model
    )
    assert len(calls) == 1
    assert calls[0]["reduce_scatter_comm_dtype"] == comm_dtype


def test_reduce_scatter_transport_config_and_native_fallback():
    with pytest.raises(ValueError, match="fsdp_mode='fsdp2'"):
        FSDPConfig(
            fsdp_mode="ddp",
            reduce_scatter_comm_dtype="bfloat16",
        )
    for reduce_dtype, transport_dtype in (("bfloat16", "float16"), ("float16", "bfloat16")):
        with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
            FSDPConfig(
                mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
                reduce_scatter_comm_dtype=transport_dtype,
            )
    with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
        FSDPConfig(
            mixed_precision=MixedPrecisionConfig(enable=False, reduce_dtype="float32"),
            reduce_scatter_comm_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="must be one of"):
        FSDPConfig(reduce_scatter_comm_dtype="float8")

    for transport_dtype in ("bfloat16", "float16"):
        config = FSDPConfig(
            mixed_precision=MixedPrecisionConfig(param_dtype=transport_dtype),
            reduce_scatter_comm_dtype=transport_dtype,
        )
        assert config.reduce_scatter_comm_dtype == transport_dtype

    for reduce_dtype in ("bfloat16", "float16", "float32"):
        config = FSDPConfig(
            mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
            reduce_scatter_comm_dtype=reduce_dtype,
        )
        assert config.reduce_scatter_comm_dtype == reduce_dtype

    native_config = FSDPConfig(
        fsdp_mode="ddp",
        mixed_precision=MixedPrecisionConfig(enable=False, reduce_dtype="float32"),
        reduce_scatter_comm_dtype="float32",
    )
    assert native_config.reduce_scatter_comm_dtype == native_config.mixed_precision.reduce_dtype


@pytest.mark.parametrize("param_dtype", [None, "bfloat16", "float16", "float32"])
@pytest.mark.parametrize("reduce_dtype", [None, "bfloat16", "float16", "float32"])
@pytest.mark.parametrize("enable", [False, True])
def test_native_transport_does_not_constrain_parameter_precision(param_dtype, reduce_dtype, enable):
    mixed_precision = MixedPrecisionConfig(enable=enable, param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    for transport_dtype in (None, reduce_dtype):
        config = FSDPConfig(
            fsdp_mode="ddp",
            mixed_precision=mixed_precision,
            reduce_scatter_comm_dtype=transport_dtype,
        )
        assert config.reduce_scatter_comm_dtype == transport_dtype
        assert not validate_reduce_scatter_transport(transport_dtype, mixed_precision)


@pytest.mark.parametrize(
    ("param_dtype", "transport_dtype"),
    [
        ("float16", "bfloat16"),
        ("bfloat16", "float16"),
        ("float32", "bfloat16"),
        ("float32", "float16"),
        (None, "bfloat16"),
        (None, "float16"),
    ],
)
def test_mismatched_parameter_and_transport_precision_is_rejected(monkeypatch, param_dtype, transport_dtype):
    mixed_precision = MixedPrecisionConfig(param_dtype=param_dtype, reduce_dtype="float32")
    with pytest.raises(ValueError, match="requires.*to match.*param_dtype") as config_error:
        FSDPConfig(mixed_precision=mixed_precision, reduce_scatter_comm_dtype=transport_dtype)

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", object)

    def fail_backend_check():
        raise AssertionError("invalid precision must be rejected before backend checks")

    monkeypatch.setattr(torch_parallelize, "get_device_type", fail_backend_check)
    with pytest.raises(ValueError) as direct_error:
        torch_parallelize.parallelize_model_fsdp2(
            nn.Linear(2, 2),
            mixed_precision=mixed_precision,
            reduce_scatter_comm_dtype=transport_dtype,
        )
    assert str(direct_error.value) == str(config_error.value)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matching_transport_round_trip_preserves_all_finite_16bit_values(dtype):
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    values = values[torch.isfinite(values)]
    assert torch.equal(values.float().to(dtype).view(torch.int16), values.view(torch.int16))


@pytest.mark.parametrize(
    ("reduce_dtype", "transport_dtype"),
    [("bfloat16", "float16"), ("float16", "bfloat16")],
)
def test_parallelize_rejects_cross_16bit_transport_before_backend_check(
    monkeypatch,
    reduce_dtype,
    transport_dtype,
):
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", object)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")

    with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
        torch_parallelize.parallelize_model_fsdp2(
            nn.Linear(2, 2),
            mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
            reduce_scatter_comm_dtype=transport_dtype,
        )


@pytest.mark.parametrize("transport_dtype", [None, "bfloat16", "float16", "float32"])
@pytest.mark.parametrize("enable", [False, True])
def test_native_transport_does_not_register_custom_collective(monkeypatch, transport_dtype, enable):
    class ParallelState:
        any_extra_parallel_enabled = False
        extra_parallel_names = []
        fsdp_mesh = None

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: ParallelState())
    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)

    def fail_registration(*args, **kwargs):
        raise AssertionError("matching dtypes must not register a custom ReduceScatter")

    monkeypatch.setattr(torch_parallelize, "ReduceScatterTransportPolicy", fail_registration)
    monkeypatch.setattr(
        torch_parallelize,
        "register_fp32_reduce_scatter_with_low_precision_transport",
        fail_registration,
    )

    model = nn.Linear(2, 2)
    result = torch_parallelize.parallelize_model_fsdp2(
        model,
        mixed_precision=MixedPrecisionConfig(
            enable=enable, param_dtype="float32", reduce_dtype=transport_dtype or "float32"
        ),
        reduce_scatter_comm_dtype=transport_dtype,
        init_device="meta",
    )

    assert result is model


@pytest.mark.parametrize(
    ("mesh_dim_names", "sizes"),
    [
        (("dp_shard",), {"dp_shard": 4}),
        (("dp_replicate", "dp_shard"), {"dp_replicate": 2, "dp_shard": 2}),
    ],
)
@pytest.mark.parametrize("transport_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("node_local", [False, True])
def test_parallelize_registers_transport_only_for_eligible_fsdp_and_hsdp(
    monkeypatch, mesh_dim_names, sizes, transport_dtype, node_local
):
    class FakeMesh:
        def __init__(self):
            self.mesh_dim_names = mesh_dim_names

        def size(self):
            result = 1
            for size in sizes.values():
                result *= size
            return result

    class ParallelState:
        any_extra_parallel_enabled = False
        extra_parallel_names = []
        fsdp_mesh = FakeMesh()

    registration_calls = []
    topology_calls = []

    class FakeTransportPolicy:
        def can_use(self, mesh):
            topology_calls.append(mesh)
            return node_local

    def record_registration(model, *, transport_dtype, reduction_scales):
        registration_calls.append((model, transport_dtype, dict(reduction_scales)))
        return len(reduction_scales)

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: ParallelState())
    monkeypatch.setattr(torch_parallelize, "ReduceScatterTransportPolicy", FakeTransportPolicy)
    monkeypatch.setattr(device_utils, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch_parallelize,
        "register_fp32_reduce_scatter_with_low_precision_transport",
        record_registration,
    )

    model = nn.Linear(2, 2)
    result = torch_parallelize.parallelize_model_fsdp2(
        model,
        mixed_precision=MixedPrecisionConfig(
            enable=True,
            param_dtype=transport_dtype,
            reduce_dtype="float32",
        ),
        reduce_scatter_comm_dtype=transport_dtype,
        init_device="meta",
    )

    assert result is model
    assert registration_calls == [(model, getattr(torch, transport_dtype), {model: 0.25} if node_local else {})]
    assert topology_calls == [ParallelState.fsdp_mesh]


@pytest.mark.parametrize("dense_eligible", [False, True])
@pytest.mark.parametrize("expert_eligible", [False, True])
def test_parallelize_checks_dense_and_expert_meshes_independently(monkeypatch, dense_eligible, expert_eligible):
    class FakeMesh:
        mesh_dim_names = ("ep_fsdp", "ep")

        def __init__(self, shard_mesh=None):
            self.shard_mesh = shard_mesh

        def __getitem__(self, names):
            assert names == ("ep_fsdp",)
            return self.shard_mesh

        def size(self):
            return 4

    dense_mesh, expert_mesh = FakeMesh(), FakeMesh()

    class ParallelState:
        any_extra_parallel_enabled = True
        extra_parallel_names = ["ep"]
        fsdp_mesh = dense_mesh
        extra_parallel_fsdp_device_mesh = {"ep": FakeMesh(expert_mesh)}

        def extra_parallel_enabled(self, name):
            return True

        def extra_parallel_gradient_divide_factor(self, name):
            return 8.0

    class ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.Linear(2, 2)
            self.high_precision = nn.LayerNorm(2)

    model = nn.Module()
    model.decoder = ToyDecoder()
    model._no_split_modules = ["ToyDecoder"]
    model.get_ignore_modules_in_mixed_precision = lambda: (nn.LayerNorm,)
    factors = []
    model.decoder.experts.set_gradient_divide_factor = factors.append

    class FakePlan:
        extra_parallel_plan = {"ep": {}}

        def apply(self, model, meshes):
            return {}

        def get_extra_parallel_fsdp_no_shard_info(self, model, name):
            return {"decoder.experts": model.decoder.experts}

    topology_calls = []

    class FakeTransportPolicy:
        def can_use(self, mesh):
            topology_calls.append(mesh)
            assert mesh in (dense_mesh, expert_mesh)
            return dense_eligible if mesh is dense_mesh else expert_eligible

    registration_calls = []
    wrapping_calls = {}

    def record_registration(model, *, transport_dtype, reduction_scales):
        registration_calls.append(dict(reduction_scales))
        return len(reduction_scales)

    def record_wrap(module, **kwargs):
        wrapping_calls[module] = kwargs

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", ParallelState)
    monkeypatch.setattr(torch_parallelize, "get_runtime_parallel_plan", lambda model: FakePlan())
    monkeypatch.setattr(torch_parallelize, "ReduceScatterTransportPolicy", FakeTransportPolicy)
    monkeypatch.setattr(device_utils, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(torch_parallelize, "fully_shard", record_wrap)
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch_parallelize, "register_fp32_reduce_scatter_with_low_precision_transport", record_registration
    )

    assert (
        torch_parallelize.parallelize_model_fsdp2(
            model,
            mixed_precision=MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
            reduce_scatter_comm_dtype="bfloat16",
            init_device="meta",
        )
        is model
    )
    expected_scales = {model.decoder: 0.25, model: 0.25} if dense_eligible else {}
    if expert_eligible:
        expected_scales[model.decoder.experts] = 0.125
    assert registration_calls == [expected_scales]
    assert factors == ([] if expert_eligible else [8.0])
    assert topology_calls == [dense_mesh, expert_mesh]
    assert wrapping_calls[model.decoder.experts]["mesh"] is expert_mesh
    assert "mp_policy" not in wrapping_calls[model.decoder.high_precision]
    assert model.decoder.high_precision not in expected_scales


def _run_transport_policy_gloo(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    original_get_node_id = reduce_scatter_module._get_node_id
    original_gather = dist.all_gather_object
    try:
        from torch.distributed.device_mesh import DeviceMesh

        # Synthetic node identities isolate placement/consensus logic; all workers run locally.
        # Noncontiguous shard rows ensure rank arithmetic cannot substitute for actual groups.
        mesh = DeviceMesh("cpu", [[0, 2], [1, 3]], mesh_dim_names=("replica", "shard"))
        for identities, expected in (
            (["node-a", "node-b", "node-a", "node-b"], True),
            (["node-a", "node-b", "node-a", "node-c"], False),
            (["node-a", "node-b", "node-a", None], False),
        ):
            calls = []

            def recording_gather(output, value, *, group, calls=calls):
                calls.append(group)
                return original_gather(output, value, group=group)

            reduce_scatter_module._get_node_id = lambda identities=identities: identities[rank]
            dist.all_gather_object = recording_gather
            policy = ReduceScatterTransportPolicy()
            assert policy.can_use(mesh) is expected
            assert policy.can_use(mesh) is expected
            assert calls == [mesh.get_group("shard"), mesh.get_group("replica")]
            decisions = [None] * 4
            original_gather(decisions, policy.can_use(mesh))
            assert decisions == [expected] * 4
    finally:
        reduce_scatter_module._get_node_id = original_get_node_id
        dist.all_gather_object = original_gather
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the CPU Gloo backend")
def test_transport_policy_gloo_noncontiguous_hsdp_consensus(tmp_path):
    mp.spawn(_run_transport_policy_gloo, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)


def _run_reduce_scatter_nccl() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(get_device_type(), int(os.environ.get("LOCAL_RANK", rank)))
    shard_numel = 4096
    values = torch.arange(world_size * shard_numel, device=device, dtype=torch.float32)
    for dtype in (torch.bfloat16, torch.float16):
        input_tensor = (values.remainder(31) + rank * 0.25).to(dtype).float()
        for scale in (1.0, 1.0 / world_size, 1.0 / 3.0):
            reference = torch.empty(shard_numel, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(reference, input_tensor, group=dist.group.WORLD, op=dist.ReduceOp.SUM)
            expected = reference.mul(scale)

            output = torch.empty(shard_numel, device=device, dtype=torch.float32)
            comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=scale)
            comm(output, input_tensor, dist.group.WORLD, dist.ReduceOp.SUM)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_fsdp2_optimizer_step() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(get_device_type(), int(os.environ.get("LOCAL_RANK", rank)))
    mesh = init_device_mesh(get_device_type(), (world_size,), mesh_dim_names=("dp_shard",))

    class RecordingReduceScatter(FP32ReduceScatterWithLowPrecisionTransport):
        def __init__(self, transport_dtype, reduction_scale):
            super().__init__(transport_dtype, reduction_scale)
            self.calls = []

        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            self.calls.append((input_tensor.dtype, op))
            return super().__call__(output_tensor, input_tensor, group, op, async_op)

    for transport_dtype in (torch.bfloat16, torch.float16):
        for gradient_divide_factor in (float(world_size), 3.0):
            torch.manual_seed(1234)
            baseline = nn.Linear(32, 16, bias=False, device=device)
            custom = copy.deepcopy(baseline)
            baseline_policy = MixedPrecisionPolicy(param_dtype=transport_dtype, reduce_dtype=torch.float32)
            custom_policy = MixedPrecisionPolicy(param_dtype=transport_dtype, reduce_dtype=torch.float32)
            fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
            fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
            baseline.set_gradient_divide_factor(gradient_divide_factor)
            custom.set_gradient_divide_factor(1.0)
            custom.set_force_sum_reduction_for_comms(True)
            comm = RecordingReduceScatter(transport_dtype, 1.0 / gradient_divide_factor)
            custom.set_custom_reduce_scatter(comm)

            torch.manual_seed(9000 + rank)
            inputs = torch.randn(8, 32, device=device, dtype=transport_dtype)
            baseline(inputs).float().square().sum().backward()
            custom(inputs).float().square().sum().backward()

            baseline_grad = baseline.weight.grad.to_local()
            custom_grad = custom.weight.grad.to_local()
            assert baseline_grad.dtype == torch.float32
            assert custom_grad.dtype == torch.float32
            assert comm.calls
            assert all(dtype == torch.float32 for dtype, _ in comm.calls)
            assert all(op == dist.ReduceOp.SUM for _, op in comm.calls)
            torch.testing.assert_close(custom_grad, baseline_grad, rtol=5e-6, atol=5e-6)

            baseline_optim = torch.optim.SGD(baseline.parameters(), lr=1e-3)
            custom_optim = torch.optim.SGD(custom.parameters(), lr=1e-3)
            baseline_optim.step()
            custom_optim.step()
            assert torch.isfinite(custom.weight.to_local()).all()
            del baseline_optim, custom_optim, baseline_grad, custom_grad, inputs, comm, baseline, custom
            gc.collect()
            dist.barrier(device_ids=[device.index])

    baseline = nn.Linear(4, 1, bias=False, device=device)
    custom = copy.deepcopy(baseline)
    baseline_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    custom_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
    fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
    custom.set_gradient_divide_factor(1.0)
    custom.set_force_sum_reduction_for_comms(True)
    comm = RecordingReduceScatter(torch.float16, 1.0 / world_size)
    custom.set_custom_reduce_scatter(comm)

    inputs = torch.full((1, 4), 65504.0, device=device, dtype=torch.float16)
    baseline(inputs).sum().backward()
    custom(inputs).sum().backward()
    baseline_grad = baseline.weight.grad.to_local()
    custom_grad = custom.weight.grad.to_local()
    assert torch.isfinite(custom_grad).all()
    torch.testing.assert_close(custom_grad, baseline_grad, rtol=0, atol=0)

    hsdp_mesh = init_device_mesh(
        get_device_type(),
        (2, 2),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    torch.manual_seed(5678)
    baseline = nn.Linear(32, 16, bias=False, device=device)
    custom = copy.deepcopy(baseline)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(baseline, mesh=hsdp_mesh, mp_policy=policy)
    fully_shard(custom, mesh=hsdp_mesh, mp_policy=policy)
    custom.set_gradient_divide_factor(1.0)
    custom.set_force_sum_reduction_for_comms(True)
    comm = RecordingReduceScatter(torch.bfloat16, 1.0 / world_size)
    custom.set_custom_reduce_scatter(comm)

    torch.manual_seed(12000 + rank)
    inputs = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
    baseline(inputs).float().square().sum().backward()
    all_reduce_dtypes = []
    original_all_reduce = dist.all_reduce

    def recording_all_reduce(tensor, *args, **kwargs):
        all_reduce_dtypes.append(tensor.dtype)
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = recording_all_reduce
    try:
        custom(inputs).float().square().sum().backward()
    finally:
        dist.all_reduce = original_all_reduce

    assert all_reduce_dtypes == [torch.float32]
    torch.testing.assert_close(custom.weight.grad.to_local(), baseline.weight.grad.to_local(), rtol=5e-6, atol=5e-6)
    _run_transport_policy_fsdp2_gradients(mesh, hsdp_mesh, device)


def _run_transport_policy_fsdp2_gradients(fsdp_mesh, hsdp_mesh, device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    original_get_node_id = reduce_scatter_module._get_node_id
    original_comm_class = reduce_scatter_module.FP32ReduceScatterWithLowPrecisionTransport
    calls = []

    class RecordingReduceScatter(original_comm_class):
        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            calls.append((input_tensor.dtype, op))
            return super().__call__(output_tensor, input_tensor, group, op, async_op)

    boot_ids = [None] * world_size
    dist.all_gather_object(boot_ids, original_get_node_id())
    real_node_local = None not in boot_ids and len(set(boot_ids)) == 1
    # Synthetic identities exercise cross-node fallback on single-node GPU CI;
    # this verifies NCCL integration and gradient scaling, not multi-node performance.
    cases = (
        (fsdp_mesh, None, real_node_local),
        (fsdp_mesh, ["node-a"] * 4, True),
        (fsdp_mesh, ["node-a", "node-a", "node-b", "node-b"], False),
        (fsdp_mesh, ["node-a", "node-a", "node-a", None], False),
        (hsdp_mesh, ["node-a", "node-a", "node-b", "node-b"], True),
        (hsdp_mesh, ["node-a", "node-a", "node-b", "node-c"], False),
        (hsdp_mesh, ["node-a", "node-a", "node-b", None], False),
    )
    try:
        reduce_scatter_module.FP32ReduceScatterWithLowPrecisionTransport = RecordingReduceScatter
        for mesh, identities, expected in cases:
            reduce_scatter_module._get_node_id = (
                original_get_node_id if identities is None else lambda identities=identities: identities[rank]
            )
            topology_policy = ReduceScatterTransportPolicy()
            enabled = topology_policy.can_use(mesh)
            assert enabled is expected
            assert topology_policy.can_use(mesh) is expected
            decisions = [None] * world_size
            dist.all_gather_object(decisions, enabled)
            assert decisions == [expected] * world_size

            torch.manual_seed(137)
            baseline = nn.Linear(16, 8, bias=False, device=device)
            candidate = copy.deepcopy(baseline)
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            fully_shard(baseline, mesh=mesh, mp_policy=mp_policy)
            fully_shard(candidate, mesh=mesh, mp_policy=mp_policy)
            registered = register_fp32_reduce_scatter_with_low_precision_transport(
                candidate,
                transport_dtype=torch.bfloat16,
                reduction_scales={candidate: 1.0 / mesh.size()} if enabled else {},
            )
            assert registered == int(expected)

            torch.manual_seed(700 + rank)
            inputs = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
            calls.clear()
            baseline(inputs).float().square().sum().backward()
            candidate(inputs).float().square().sum().backward()
            assert bool(calls) is expected
            assert all(dtype == torch.float32 and op == dist.ReduceOp.SUM for dtype, op in calls)
            baseline_grad = baseline.weight.grad.to_local()
            candidate_grad = candidate.weight.grad.to_local()
            assert baseline_grad.dtype == candidate_grad.dtype == torch.float32
            torch.testing.assert_close(candidate_grad, baseline_grad, rtol=5e-6, atol=5e-6)
            del baseline, candidate, baseline_grad, candidate_grad, inputs
            gc.collect()
            dist.barrier(device_ids=[device.index])
    finally:
        reduce_scatter_module._get_node_id = original_get_node_id
        reduce_scatter_module.FP32ReduceScatterWithLowPrecisionTransport = original_comm_class


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_nccl():
    from ..tools.launch_utils import torchrun

    torchrun(_run_reduce_scatter_nccl, world_size=4)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_fsdp2_step():
    from ..tools.launch_utils import torchrun

    torchrun(_run_fsdp2_optimizer_step, world_size=4)
