"""Regression tests for VLM-style optimizer param groups + DCP resume."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.optim.optimizer import (
    build_optimizer,
    filter_empty_param_groups,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.backbone(x)


def test_filter_empty_param_groups_drops_frozen_vit_group():
    p = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [], "lr": 1e-6},
        {"params": [p], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [p]


def test_filter_empty_param_groups_drops_frozen_only_group():
    """A group whose params are all requires_grad=False has no optimizer state
    and must be dropped (otherwise the same KeyError surfaces on resume)."""
    frozen = torch.nn.Parameter(torch.randn(2, 2), requires_grad=False)
    trainable = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [frozen], "lr": 1e-6},
        {"params": [frozen, trainable], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [trainable]  # frozen dropped, trainable kept


def test_build_optimizer_skips_empty_vlm_param_groups():
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 2e-5


def test_vlm_style_optimizer_dcp_roundtrip_step_succeeds():
    """Reproduces production freeze_vit resume: save/load must keep betas."""
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )

    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()

    optim_sd = get_optimizer_state_dict(
        model=model,
        optimizers=opt,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )

    model2 = _TinyModel()
    opt2 = build_optimizer(
        model2,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model2.parameters()), "lr": 2e-5},
        ],
    )
    set_optimizer_state_dict(
        model=model2,
        optimizers=opt2,
        optim_state_dict=optim_sd,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )

    for group in opt2.param_groups:
        assert "betas" in group

    loss2 = model2(torch.randn(2, 4)).sum()
    loss2.backward()
    opt2.step()


@pytest.mark.parametrize("extra_parallel", [False, True])
def test_qwen38_vlm_groups_resolve_ple_decay_before_planning(monkeypatch, extra_parallel):
    """VLM supplies per-tower learning rates without explicit decay fields."""
    import veomni.optim.optimizer as factory

    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
    model.lm_head = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    monkeypatch.setattr(factory, "_should_build_extra_parallel_aware", lambda _: extra_parallel)
    if extra_parallel:
        # Exercise the exact factory hand-off without needing a device mesh.
        def build_extra(model, lr, betas, eps, decay, fused, kind, groups, *args, cpu_offload_kwargs):
            assert kind == "qwen38_muon"
            return factory.Qwen38MuonAdamW(groups, lr, betas, eps, decay, **cpu_offload_kwargs)

        monkeypatch.setattr(factory, "build_extra_parallel_fsdp2_optimizer", build_extra)

    def build():
        return build_optimizer(
            model,
            optimizer_type="qwen38_muon",
            lr=2e-5,
            weight_decay=0.1,
            no_decay_params=["ngram_embedding"],
            param_groups=[{"params": list(model.parameters()), "lr": 1e-5}],
            cpu_offload_param_patterns=["ngram_embedding"],
        )

    opt = build()
    groups = {id(p): g for g in opt.param_groups for p in g["params"]}
    ple = model.ngram_embedding.weight
    assert groups[id(ple)]["weight_decay"] == 0.0
    assert groups[id(ple)]["qwen38_ple_lookup"] is True
    assert groups[id(model.lm_head.weight)]["weight_decay"] == 0.1
    assert all(g["lr"] == 1e-5 for g in opt.param_groups)
    reference = torch.nn.Parameter(ple.detach().float().clone())
    adam = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    for _ in range(2):
        ple.grad = torch.randn_like(ple)
        reference.grad = ple.grad.float()
        model.lm_head.weight.grad = torch.randn_like(model.lm_head.weight)
        opt.step()
        adam.step()
        torch.testing.assert_close(opt.state[ple]["master_weight"], reference)
    saved = copy.deepcopy(opt.state_dict())
    resumed = build()
    resumed.load_state_dict(saved)
    assert torch.equal(resumed.state[ple]["master_weight"], opt.state[ple]["master_weight"])


def test_qwen38_explicit_ple_decay_still_rejected():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4)
    with pytest.raises(ValueError, match="explicit zero weight decay"):
        build_optimizer(
            model,
            optimizer_type="qwen38_muon",
            no_decay_params=["ngram_embedding"],
            cpu_offload_param_patterns=["ngram_embedding"],
            param_groups=[{"params": list(model.parameters()), "weight_decay": 0.1}],
        )


def test_qwen38_extra_parallel_does_not_override_explicit_nonzero_ple_decay(monkeypatch):
    import veomni.optim.optimizer as factory

    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(7, 4)
    monkeypatch.setattr(factory, "_should_build_extra_parallel_aware", lambda _: True)
    monkeypatch.setattr(factory, "get_parallel_state", lambda: SimpleNamespace(extra_parallel_names=["ep"]))
    with pytest.raises(ValueError, match="zero weight decay"):
        build_optimizer(
            model,
            optimizer_type="qwen38_muon",
            no_decay_params=["ngram_embedding"],
            cpu_offload_param_patterns=["ngram_embedding"],
            param_groups=[{"params": list(model.parameters()), "weight_decay": 0.1}],
        )


def test_extra_parallel_respects_explicit_and_default_decay(monkeypatch):
    import veomni.optim.optimizer as factory

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.LayerNorm(4))
    monkeypatch.setattr(factory, "get_parallel_state", lambda: SimpleNamespace(extra_parallel_names=["ep"]))
    optimizer = factory.build_extra_parallel_fsdp2_optimizer(
        model,
        weight_decay=0.1,
        no_decay_modules=["LayerNorm"],
        param_groups=[
            {"params": [model[0].weight, model[1].weight], "weight_decay": 0.3},
            {"params": [model[1].bias]},
        ],
    )
    groups = {
        id(param): group
        for leaf in optimizer.optimizers_dict.values()
        for group in leaf.param_groups
        for param in group["params"]
    }
    assert groups[id(model[0].weight)]["weight_decay"] == 0.3
    assert groups[id(model[1].weight)]["weight_decay"] == 0.3
    assert groups[id(model[1].bias)]["weight_decay"] == 0.0


def test_extra_parallel_explicit_decay_resumes_legacy_dcp_groups(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    import veomni.optim.optimizer as factory

    def make_model():
        return torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.LayerNorm(4))

    source_model = make_model()
    source_leaf = torch.optim.AdamW(
        [
            {"params": [source_model[0].weight], "weight_decay": 0.1},
            {"params": [source_model[1].weight], "weight_decay": 0.0},
        ],
        lr=1e-2,
    )
    source = factory.MultiOptimizer(source_model, {"non_extra_parallel": source_leaf}, ["non_extra_parallel"])
    for param in (source_model[0].weight, source_model[1].weight):
        param.grad = torch.full_like(param, 0.5)
    source.step()
    dcp.save(
        {"model": source_model.state_dict(), "optimizer": source.state_dict(for_load=False)}, checkpoint_id=tmp_path
    )

    target_model = make_model()
    monkeypatch.setattr(factory, "get_parallel_state", lambda: SimpleNamespace(extra_parallel_names=["ep"]))
    target = factory.build_extra_parallel_fsdp2_optimizer(
        target_model,
        lr=1e-2,
        weight_decay=0.1,
        no_decay_modules=["LayerNorm"],
        param_groups=[{"params": [target_model[0].weight, target_model[1].weight], "weight_decay": 0.3}],
    )
    target_state = target.state_dict()
    dcp.load({"model": target_model.state_dict(), "optimizer": target_state}, checkpoint_id=tmp_path)
    target.load_state_dict(target_state)
    assert [group["weight_decay"] for group in target.optimizers_dict["non_extra_parallel"].param_groups] == [0.1, 0.0]
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(source_param, target_param, rtol=0, atol=0)
    for param in (source_model[0].weight, source_model[1].weight, target_model[0].weight, target_model[1].weight):
        param.grad = torch.zeros_like(param)
    source.step()
    target.step()
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters(), strict=True):
        torch.testing.assert_close(source_param, target_param, rtol=0, atol=0)


def _qwen38_ple_dtensor_decay_worker(rank, rendezvous):
    from unittest.mock import patch

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    import veomni.optim.optimizer as factory

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep_fsdp",))
        model = torch.nn.Module()
        model.ngram_embedding = torch.nn.Embedding(8, 4)
        full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        model.ngram_embedding.weight = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
        model.lm_head = torch.nn.Linear(4, 8, bias=False)
        ple = model.ngram_embedding.weight
        reference = torch.nn.Parameter(ple.to_local().detach().clone())
        adam = torch.optim.AdamW([reference], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
        with (
            patch.object(factory, "_should_build_extra_parallel_aware", return_value=True),
            patch.object(factory, "get_parallel_state", return_value=SimpleNamespace(extra_parallel_names=["ep"])),
        ):
            optimizer = build_optimizer(
                model,
                optimizer_type="qwen38_muon",
                lr=1e-3,
                weight_decay=0.1,
                cpu_offload_param_patterns=["ngram_embedding"],
                param_groups=[
                    {"params": [ple], "weight_decay": 0.0},
                    {"params": [model.lm_head.weight], "weight_decay": 0.1},
                ],
            )
        assert optimizer.optimizers_dict["ep"].param_groups[0]["weight_decay"] == 0.0
        assert optimizer.optimizers_dict["non_extra_parallel"].param_groups[0]["weight_decay"] == 0.1
        for _ in range(2):
            ple.grad = torch.full_like(ple, 0.5)
            model.lm_head.weight.grad = torch.full_like(model.lm_head.weight, 0.25)
            reference.grad = torch.full_like(reference, 0.5)
            optimizer.step()
            adam.step()
        torch.testing.assert_close(optimizer.optimizers_dict["ep"].state[ple]["master_weight"].to_local(), reference)
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
def test_qwen38_dtensor_ple_decay_survives_extra_parallel_split(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _qwen38_ple_dtensor_decay_worker,
        args=(str(tmp_path / "rdzv"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )
