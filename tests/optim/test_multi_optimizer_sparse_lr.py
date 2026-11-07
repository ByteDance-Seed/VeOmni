"""An unused optimizer leaf still has a scheduler-controlled learning rate."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build():
    model = torch.nn.Linear(2, 2)
    weight = torch.optim.AdamW([model.weight], lr=0.01)
    bias = torch.optim.AdamW([model.bias], lr=0.01)
    optimizer = MultiOptimizer(model, {"weight": weight, "bias": bias}, ["weight", "bias"])
    return model, weight, bias, optimizer


def test_sparse_leaf_preserves_learning_rate_without_creating_state():
    model, weight, bias, optimizer = _build()
    model.weight.grad = torch.ones_like(model.weight)
    weight.step()
    weight.zero_grad()
    bias.param_groups[0]["lr"] = 0.0025
    result = optimizer._sparse_state_dict()
    assert result["param_groups.bias.lr"] == 0.0025
    assert not bias.state


def test_sparse_leaf_uses_optimizer_state_dict_post_hook():
    _, _, bias, optimizer = _build()

    def change_serialized_lr(_optimizer, state):
        state["param_groups"][0]["lr"] = 0.037
        return state

    bias.register_state_dict_post_hook(change_serialized_lr)
    result = optimizer._sparse_state_dict()
    assert result["param_groups.bias.lr"] == 0.037
    assert bias.param_groups[0]["lr"] == 0.01
    assert not bias.state


def test_sparse_leaf_preserves_tensor_learning_rate():
    model = torch.nn.Linear(2, 2)
    learning_rate = torch.tensor(0.0025)
    leaf = torch.optim.AdamW([model.bias], lr=learning_rate)
    optimizer = MultiOptimizer(model, {"bias": leaf}, ["bias"])
    result = optimizer._sparse_state_dict()
    torch.testing.assert_close(result["param_groups.bias.lr"], learning_rate, rtol=0, atol=0)
    assert not leaf.state


def test_sparse_leaf_respects_post_hook_parameter_reordering():
    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW([{"params": [model.weight], "lr": 0.01}, {"params": [model.bias], "lr": 0.02}])
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    def swap_serialized_groups(_optimizer, state):
        first, second = state["param_groups"]
        first["params"], second["params"] = second["params"], first["params"]
        return state

    leaf.register_state_dict_post_hook(swap_serialized_groups)
    result = optimizer._sparse_state_dict()
    assert result["param_groups.weight.lr"] == 0.02
    assert result["param_groups.bias.lr"] == 0.01
    assert not leaf.state


@pytest.mark.parametrize("patched", [False, True])
def test_sparse_leaf_preserves_custom_optimizer_state_dict(patched):
    calls = []

    class CustomAdamW(torch.optim.AdamW):
        def state_dict(self):
            calls.append(True)
            return super().state_dict()

    model = torch.nn.Linear(2, 2)
    leaf = CustomAdamW(model.parameters(), lr=0.0025)
    if patched:
        from torch.distributed.checkpoint.state_dict import _patch_optimizer_state_dict

        _patch_optimizer_state_dict(model, optimizers=(leaf,))
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    result = optimizer._sparse_state_dict()
    assert calls == [True]
    assert result["param_groups.weight.lr"] == 0.0025
    assert not leaf.state


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("target_patched", [False, True])
def test_pytorch_patched_adamw_dcp_resume_preserves_next_update(tmp_path, initialized, target_patched):
    from torch.distributed.checkpoint.state_dict import _patch_optimizer_state_dict

    model = torch.nn.Linear(2, 2)
    leaf = torch.optim.AdamW(model.parameters(), lr=0.0025)
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    _patch_optimizer_state_dict(model, optimizers=(leaf,))
    if initialized:
        for param in model.parameters():
            param.grad = torch.full_like(param, 0.25)
        optimizer.step()
        optimizer.zero_grad()
    before = [param.detach().clone() for param in model.parameters()]
    before_states = [
        {key: value.detach().clone() for key, value in leaf.state.get(param, {}).items()}
        for param in model.parameters()
    ]
    parallel = SimpleNamespace(dp_mode="ddp")
    checkpoint = str(tmp_path / "patched")
    DistributedCheckpointer.save(
        checkpoint, {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    for actual, expected, expected_state in zip(model.parameters(), before, before_states, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert leaf.state.get(actual, {}).keys() == expected_state.keys()
        for key, value in expected_state.items():
            torch.testing.assert_close(leaf.state[actual][key], value, rtol=0, atol=0)

    restored_model = torch.nn.Linear(2, 2)
    restored_leaf = torch.optim.AdamW(restored_model.parameters(), lr=0.9)
    if target_patched:
        _patch_optimizer_state_dict(restored_model, optimizers=(restored_leaf,))
    restored = MultiOptimizer(restored_model, {"leaf": restored_leaf}, ["leaf"])
    DistributedCheckpointer.load(
        checkpoint,
        {"model": restored_model, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert bool(restored_leaf.state) is initialized
    assert restored_leaf.param_groups[0]["lr"] == 0.0025
    for current, target in zip(model.parameters(), restored_model.parameters(), strict=True):
        current.grad = torch.ones_like(current)
        target.grad = torch.ones_like(target)
    optimizer.step()
    restored.step()
    for current, target in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(current, target, rtol=0, atol=0)
        for key in leaf.state[current]:
            torch.testing.assert_close(leaf.state[current][key], restored_leaf.state[target][key], rtol=0, atol=0)


def test_sparse_leaf_learning_rate_survives_dcp_roundtrip(tmp_path):
    model, weight, bias, optimizer = _build()
    model.weight.grad = torch.ones_like(model.weight)
    weight.step()
    weight.zero_grad()
    bias.param_groups[0]["lr"] = 0.0025
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        str(tmp_path),
        {"model": model, "optimizer": optimizer, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert not bias.state
    restored_model, _, restored_bias, restored = _build()
    DistributedCheckpointer.load(
        str(tmp_path),
        {"model": restored_model, "optimizer": restored, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    assert restored_bias.param_groups[0]["lr"] == 0.0025


def _fsdp2_empty_leaf_worker(rank, checkpoint, endpoint):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    from veomni.distributed.parallel_state import _init_parallel_state, get_parallel_state

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _init_parallel_state(dp_size=2, dp_mode="fsdp2", device_type="cpu")
        parallel = get_parallel_state()

        def build(seed):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
            for layer in model:
                fully_shard(layer, mesh=parallel.dp_shard_mesh)
            fully_shard(model, mesh=parallel.dp_shard_mesh)
            first = torch.optim.AdamW(model[0].parameters(), lr=0.01)
            second = torch.optim.AdamW(model[1].parameters(), lr=0.01)
            return model, first, second, MultiOptimizer(model, {"first": first, "second": second}, ["first", "second"])

        from veomni.optim.lr_scheduler import build_lr_scheduler

        model, first, second, optimizer = build(42)
        scheduler = build_lr_scheduler(optimizer, train_steps=4, lr=0.01, lr_decay_style="linear", lr_warmup_ratio=0)
        model(torch.ones(2, 4)).sum().backward()
        for param in model[1].parameters():
            param.grad = None
        first.step()
        first.zero_grad()
        for _ in range(3):
            second.step()
            scheduler.step()
        assert second.param_groups[0]["lr"] == 0.0025
        assert first.state
        assert not second.state
        source_shards = [param.to_local().detach().clone() for param in model.parameters()]
        DistributedCheckpointer.save(
            checkpoint,
            {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {"lr_scheduler": scheduler.state_dict()},
                "_veomni_scheduler_object": scheduler,
            },
            parallel_state=parallel,
        )
        assert not second.state
        for expected, param in zip(source_shards, model.parameters(), strict=True):
            torch.testing.assert_close(expected, param.to_local(), rtol=0, atol=0)

        restored_model, _, restored_second, restored = build(123)
        assert any(
            not torch.equal(saved, param.to_local())
            for saved, param in zip(source_shards, restored_model.parameters(), strict=True)
        )
        restored_scheduler = build_lr_scheduler(
            restored, train_steps=4, lr=0.01, lr_decay_style="linear", lr_warmup_ratio=0
        )
        recovered = {
            "model": restored_model,
            "optimizer": restored,
            "extra_state": {},
            "_veomni_scheduler_object": restored_scheduler,
        }
        DistributedCheckpointer.load(checkpoint, recovered, parallel_state=parallel)
        restored_scheduler.load_state_dict(recovered["extra_state"]["lr_scheduler"])
        assert restored_second.param_groups[0]["lr"] == 0.0025
        for expected, param in zip(source_shards, restored_model.parameters(), strict=True):
            torch.testing.assert_close(expected, param.to_local(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_sparse_leaf_learning_rate_survives_two_rank_fsdp2_dcp(tmp_path):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_fsdp2_empty_leaf_worker, args=(str(tmp_path / "ckpt"),), nprocs=2)
