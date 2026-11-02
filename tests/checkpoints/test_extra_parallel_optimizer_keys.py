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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_axis_offloaded_shards_keep_cpu_storage_and_strided_layout(dtype):
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
    local = torch.arange(12, dtype=dtype).reshape(3, 4)
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
