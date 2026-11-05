"""Optimizer checkpoint keys must preserve their full parameter namespace."""

import pytest
import torch
from torch.distributed.tensor import Replicate

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint import dcp_checkpointer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("action", ["drop", "restore"])
@pytest.mark.parametrize(
    "key, expected, scalar",
    [
        ("state.embed_tokens.weight.exp_avg", "embed_tokens.weight", False),
        ("state.decoder.embed_tokens.weight.exp_avg", "decoder.embed_tokens.weight", False),
        ("state.decoder2.embed_tokens.weight.exp_avg_sq", "decoder2.embed_tokens.weight", False),
        ("state.decoder.embed_tokens.weight.step", None, True),
        ("state.decoder.embed_tokens.weight_extra.exp_avg", None, False),
        ("state.unrelated.decoder.embed_tokens.weight.exp_avg", None, False),
        ("param_groups.decoder.embed_tokens.weight.lr", None, False),
        ("decoder.embed_tokens.weight.exp_avg", None, False),
    ],
)
def test_optimizer_spec_matching_uses_the_full_namespace(monkeypatch, action, key, expected, scalar):
    selected = []

    class Spec:
        def __init__(self, name):
            self.name = name

        @property
        def placement(self):
            selected.append(self.name)
            return Replicate()

    specs = {
        name: Spec(name)
        for name in ("embed_tokens.weight", "decoder.embed_tokens.weight", "decoder2.embed_tokens.weight")
    }
    value = torch.ones(()) if scalar else torch.ones(2, 3)
    state = {key: value, "param_groups.decoder.embed_tokens.weight.amsgrad": False}
    monkeypatch.setattr(dcp_checkpointer, "_validate_extra_parallel_meshes", lambda _: None)
    result = dcp_checkpointer._apply_extra_parallel_dim(state, specs, None, action, key_match="substring")
    assert selected == ([] if expected is None else [expected])
    assert result is state and result[key] is value
    assert result["param_groups.decoder.embed_tokens.weight.amsgrad"] is False


@pytest.mark.parametrize("action", ["drop", "restore"])
def test_optimizer_spec_matching_still_rejects_ambiguous_specs(monkeypatch, action):
    monkeypatch.setattr(dcp_checkpointer, "_validate_extra_parallel_meshes", lambda _: None)
    with pytest.raises(AssertionError, match="Ambiguous ExtraParallel spec match"):
        dcp_checkpointer._apply_extra_parallel_dim(
            {"state.embedding.weight.exp_avg": torch.ones(2, 3)},
            dict.fromkeys(("embedding", "embedding.weight")),
            None,
            action,
            key_match="substring",
        )


def test_plain_optimizer_state_with_multiple_fsdp_ranks_is_rejected(monkeypatch):
    from types import SimpleNamespace

    from torch.distributed.tensor import Shard

    class FakeMesh:
        mesh_dim_names = ("ep_fsdp", "ep")

        def __getitem__(self, _):
            return SimpleNamespace(size=lambda: 2)

    monkeypatch.setattr(dcp_checkpointer, "_validate_extra_parallel_meshes", lambda _: None)
    spec = SimpleNamespace(placement=Shard(0), para_fsdp_mesh=FakeMesh())
    state = {"state.experts.weight.exp_avg": torch.ones(2, 3)}
    with pytest.raises(RuntimeError, match="no proven replica layout"):
        dcp_checkpointer._apply_extra_parallel_dim(
            state, {"experts.weight": spec}, None, "restore", key_match="substring"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_same_axis_offloaded_shards_keep_cpu_storage_and_strided_layout(
    dtype, noncontiguous, requires_grad, grad_enabled
):
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.placement_types import _StridedShard

    class FakeMesh:
        def __init__(self, shape):
            self.device_type = "xpu"
            self.mesh = torch.zeros(shape, dtype=torch.int64)
            self.ndim = len(shape)
            self.mesh_dim_names = tuple(f"d{i}" for i in range(self.ndim))

        def size(self, dim=None):
            return self.mesh.numel() if dim is None else self.mesh.size(dim)

        def get_coordinate(self):
            return [0] * self.ndim

    full, sub = FakeMesh((2, 2)), FakeMesh((2,))
    if noncontiguous:
        local = torch.arange(24, dtype=dtype).reshape(3, 8)[:, ::2]
    else:
        local = torch.arange(12, dtype=dtype).reshape(3, 4)
    local.requires_grad_(requires_grad)
    with torch.set_grad_enabled(grad_enabled):
        original = dcp_checkpointer._from_local_preserve_device(local, sub, [Shard(0)])
        restored = dcp_checkpointer.restore_extra_parallel_dim(original, full, sub, ep_shard_dim=-2, fsdp_shard_dim=0)
        dropped = dcp_checkpointer.drop_extra_parallel_dim(restored, sub, fsdp_shard_dim=-2)
    assert restored.shape == (12, 4)
    assert restored.placements == (_StridedShard(0, split_factor=2), Shard(0))
    assert dropped.shape == (6, 4)
    assert dropped.placements == (Shard(0),)
    for value in (original, restored, dropped):
        assert value.dtype == dtype
        assert value.to_local().device.type == "cpu"
        assert value.to_local().data_ptr() == local.data_ptr()
        assert value.to_local().stride() == local.stride()
        assert value.stride() == local.stride()
        assert value.requires_grad == requires_grad
        assert value._local_tensor.requires_grad == requires_grad


def _plain_extra_parallel_roundtrip(rank, root):
    from datetime import timedelta

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("ep_fsdp", "ep"))
        submesh = mesh["ep_fsdp"]
        local = torch.full((2, 3), float(rank + 1))
        saved = dcp_checkpointer.restore_extra_parallel_dim(local, mesh, submesh, ep_shard_dim=0)
        expected = torch.cat([torch.ones((2, 3)), torch.full((2, 3), 2.0)])
        torch.testing.assert_close(saved.full_tensor(), expected, rtol=0, atol=0)
        dcp.save({"expert": saved}, checkpoint_id=f"{root}/checkpoint")
        restored = dcp_checkpointer.restore_extra_parallel_dim(torch.zeros_like(local), mesh, submesh, ep_shard_dim=0)
        dcp.load({"expert": restored}, checkpoint_id=f"{root}/checkpoint")
        dropped = dcp_checkpointer.drop_extra_parallel_dim(restored, submesh)
        assert not isinstance(dropped, torch.distributed.tensor.DTensor)
        torch.testing.assert_close(dropped, local, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_plain_extra_parallel_values_survive_dcp_roundtrip(tmp_path):
    torch.multiprocessing.spawn(_plain_extra_parallel_roundtrip, args=(str(tmp_path),), nprocs=2, join=True)


def _asymmetric_optimizer_preflight(rank, root, case):
    from datetime import timedelta
    from pathlib import Path
    from types import SimpleNamespace

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.tensor import Shard

    class FakeMesh:
        mesh_dim_names = ("ep_fsdp", "ep")

        def __getitem__(self, _):
            return SimpleNamespace(size=lambda: 2)

    class FakeOptimizer:
        called = False

        @property
        def _is_multi_optimizer(self):
            if case == "type_property_error" and rank == 0:
                raise ValueError("injected optimizer type lookup error")
            return not (case.startswith("wrong_type") and rank == 0)

        def state_dict(self):
            self.called = True
            if case in ("wrong_type_collective", "type_property_error"):
                dist.barrier()
            return {"state.experts.weight.exp_avg": torch.ones(2, 3)} if rank == 0 else {}

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/init", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        dcp_checkpointer._validate_extra_parallel_meshes = lambda _: None
        spec = SimpleNamespace(placement=Shard(0), para_fsdp_mesh=FakeMesh())
        model = SimpleNamespace(_fqn2spec_info={"experts.weight": spec})
        parallel = SimpleNamespace(dp_mode="fsdp2")
        leaf = FakeOptimizer()
        optimizer = dcp_checkpointer.OptimizerState(model, leaf, parallel_state=parallel)
        checkpoint = Path(root) / "rejected"
        if rank == 0 and case.startswith("wrong_type"):
            message = "not a MultiOptimizer"
        elif rank == 0 and case == "type_property_error":
            message = "optimizer type lookup error"
        elif case.startswith("wrong_type") or case == "type_property_error":
            message = "validating ExtraParallel optimizer type"
        else:
            message = "no proven replica layout" if rank == 0 else "preparing ExtraParallel optimizer checkpoint state"
        with pytest.raises((RuntimeError, ValueError), match=message):
            dcp.save({"optimizer": optimizer}, checkpoint_id=checkpoint)
        if case.startswith("wrong_type") or case == "type_property_error":
            assert not leaf.called
        assert not (checkpoint / ".metadata").exists()
        value = torch.ones(())
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
@pytest.mark.parametrize("case", ["plain_state", "wrong_type", "wrong_type_collective", "type_property_error"])
def test_asymmetric_optimizer_preflight_rejects_every_rank(tmp_path, case):
    torch.multiprocessing.spawn(_asymmetric_optimizer_preflight, args=(str(tmp_path), case), nprocs=2, join=True)
