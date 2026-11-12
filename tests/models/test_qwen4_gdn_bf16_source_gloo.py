"""Distributed agreement and rank-zero-only HF loading for Qwen4 GDN."""

import json
import os
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.test_qwen4_gdn_precision import ToyQwen4
from veomni.arguments import MixedPrecisionConfig
from veomni.distributed import torch_parallelize
from veomni.distributed.parallel_state import _init_parallel_state, use_parallel_state
from veomni.distributed.torch_parallelize import _hf_loading_models
from veomni.models import qwen4_gdn_precision
from veomni.models.checkpoint.weights import load_model_weights, rank0_load_and_broadcast_weights
from veomni.models.qwen4_gdn_precision import (
    prepare_qwen4_gdn_precision,
    qwen4_bf16_hf_source_allowed,
    qwen4_gdn_bf16_source_policy,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


def test_hf_loading_models_respects_per_child_skip():
    root = torch.nn.Module()
    root.add_module("qwen", ToyQwen4())
    root.add_module("other", torch.nn.Linear(1, 1))
    assert _hf_loading_models(root, None, False, None) == ()
    assert _hf_loading_models(root, "model", True, None) == ()
    assert _hf_loading_models(root, "model", False, None) == (root,)
    paths = {"qwen": "weights", "other": "weights", "missing": None}
    assert _hf_loading_models(root, paths, True, {"qwen": False}) == (root.qwen,)
    assert _hf_loading_models(root, paths, False, {"qwen": True}) == (root.other,)


@pytest.mark.parametrize("broadcast", [False, True])
def test_fsdp_preflight_uses_only_loading_children_and_no_subgroup_broadcast_check(monkeypatch, broadcast):
    root = torch.nn.Module()
    root.add_module("qwen", ToyQwen4())
    root.add_module("other", torch.nn.Linear(1, 1))
    group = object()
    state = SimpleNamespace(
        any_extra_parallel_enabled=False, extra_parallel_names=[], fsdp_mesh=None, fsdp_group=group
    )
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)
    calls = []

    @contextmanager
    def record_policy(models, group):
        calls.append((models, group))
        yield

    monkeypatch.setattr(qwen4_gdn_precision, "qwen4_gdn_bf16_source_policy", record_policy)
    torch_parallelize.parallelize_model_fsdp2(
        root,
        weights_path={"qwen": "weights", "other": "weights"},
        should_skip_hf_weight_load=True,
        module_skip_hf_weight_load={"qwen": False},
        broadcast_model_weights_from_rank0=broadcast,
        mixed_precision=MixedPrecisionConfig(enable=False),
        init_device="meta",
    )
    assert calls == ([] if broadcast else [((root.qwen,), group)])


def _policy_worker(rank, rendezvous, output_dir, mode):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        enabled = mode == "enabled" or (mode.endswith("mismatch") and rank == 0)
        os.environ["VEOMNI_QWEN4_ALLOW_BF16_GDN_SOURCE"] = "1" if enabled else "0"
        if mode == "composed-mismatch":
            model = torch.nn.Module()
            model.config = SimpleNamespace(model_type="omni")
            model.add_module("qwen", ToyQwen4())
            model = (model.qwen,)
        elif mode == "composed-wrapper-mismatch":
            model = torch.nn.Module()
            model.config = SimpleNamespace(model_type="omni")
            model.add_module("qwen", ToyQwen4())
        elif mode == "no-gdn-mismatch":
            model = SimpleNamespace(
                config=SimpleNamespace(model_type="qwen4_exp_text", layer_types=["full_attention"])
            )
        elif mode == "other-model-mismatch":
            model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
        else:
            model = SimpleNamespace(config=SimpleNamespace(model_type="qwen4_exp"))
        error = None
        try:
            with qwen4_gdn_bf16_source_policy(model, group=dist.group.WORLD):
                expected = enabled and mode not in (
                    "composed-wrapper-mismatch",
                    "no-gdn-mismatch",
                    "other-model-mismatch",
                )
                assert qwen4_bf16_hf_source_allowed() is expected
        except RuntimeError as exc:
            error = str(exc)
        assert not qwen4_bf16_hf_source_allowed()
        with open(os.path.join(output_dir, f"{mode}-{rank}.json"), "w") as f:
            json.dump({"error": error}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "mode",
    [
        "enabled",
        "disabled",
        "mismatch",
        "composed-mismatch",
        "composed-wrapper-mismatch",
        "no-gdn-mismatch",
        "other-model-mismatch",
    ],
)
def test_qwen4_bf16_source_policy_agrees_on_all_ranks(tmp_path, mode):
    mp.spawn(_policy_worker, args=(str(tmp_path / f"{mode}-rdzv"), str(tmp_path), mode), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"{mode}-{rank}.json").read_text()) for rank in range(2)]
    if mode in ("mismatch", "composed-mismatch"):
        assert all("policy differs across loading ranks" in row["error"] for row in rows)
    else:
        assert all(row["error"] is None for row in rows)


def _rank0_only_worker(rank, rendezvous, source_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        if rank == 0:
            os.environ["VEOMNI_QWEN4_ALLOW_BF16_GDN_SOURCE"] = "1"
            model = ToyQwen4()
            prepare_qwen4_gdn_precision(model)
            with pytest.raises(RuntimeError, match="requires FP32 source and target"):
                load_model_weights(model, source_dir, init_device="cpu")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_qwen4_bf16_rank0_only_hf_load_does_not_require_collective(tmp_path):
    save_file(
        {
            "linear_attn.A_log": torch.full((4,), 0.125, dtype=torch.bfloat16),
            "linear_attn.dt_bias": torch.full((4,), 0.125, dtype=torch.bfloat16),
            "linear_attn.proj.weight": torch.ones(4, 4, dtype=torch.bfloat16),
            "linear_attn.proj.bias": torch.zeros(4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    mp.spawn(_rank0_only_worker, args=(str(tmp_path / "load-rdzv"), str(tmp_path)), nprocs=2, join=True)


def _broadcast_mismatch_worker(rank, rendezvous, source_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        os.environ["VEOMNI_QWEN4_ALLOW_BF16_GDN_SOURCE"] = "1" if rank == 0 else "0"
        model = ToyQwen4()
        prepare_qwen4_gdn_precision(model)
        with pytest.raises(RuntimeError, match="policy differs across loading ranks"):
            rank0_load_and_broadcast_weights(model, source_dir, init_device="cpu")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_qwen4_bf16_rank0_broadcast_rejects_policy_mismatch(tmp_path):
    mp.spawn(_broadcast_mismatch_worker, args=(str(tmp_path / "broadcast-rdzv"), str(tmp_path)), nprocs=2, join=True)


def _broadcast_enabled_worker(rank, rendezvous, source_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        import veomni.models.checkpoint.weights as weights

        os.environ["VEOMNI_QWEN4_ALLOW_BF16_GDN_SOURCE"] = "1"
        weights.get_parallel_state = lambda: SimpleNamespace(global_rank=rank)
        model = ToyQwen4()
        prepare_qwen4_gdn_precision(model)
        rank0_load_and_broadcast_weights(model, source_dir, init_device="cpu")
        assert model.linear_attn.A_log.dtype == torch.float32
        assert torch.equal(model.linear_attn.A_log, torch.full((4,), 0.125))
        assert not qwen4_bf16_hf_source_allowed()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_qwen4_bf16_rank0_broadcast_loads_exact_values(tmp_path):
    save_file(
        {
            "linear_attn.A_log": torch.full((4,), 0.125, dtype=torch.bfloat16),
            "linear_attn.dt_bias": torch.full((4,), 0.125, dtype=torch.bfloat16),
            "linear_attn.proj.weight": torch.ones(4, 4, dtype=torch.bfloat16),
            "linear_attn.proj.bias": torch.zeros(4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    mp.spawn(_broadcast_enabled_worker, args=(str(tmp_path / "broadcast-ok-rdzv"), str(tmp_path)), nprocs=2, join=True)


class Qwen4ExpTextGatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.empty(4, device="meta", dtype=torch.bfloat16))
        self.dt_bias = torch.nn.Parameter(torch.empty(4, device="meta", dtype=torch.bfloat16))
        self.proj = torch.nn.Linear(4, 4, device="meta", dtype=torch.bfloat16)

    def forward(self, x):
        return self.proj(x).float() * self.A_log + self.dt_bias


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = Qwen4ExpTextGatedDeltaNet()

    def forward(self, x):
        return self.linear_attn(x)


class _FSDPGdnModel(torch.nn.Module):
    _no_split_modules = ["_Decoder"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen4_exp")
        self.layers = torch.nn.ModuleList([_Decoder()])

    def forward(self, x):
        return self.layers[0](x)


def _fsdp_load_worker(rank, rendezvous, source_dir, mismatch):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        os.environ["VEOMNI_QWEN4_ALLOW_BF16_GDN_SOURCE"] = "0" if mismatch and rank == 1 else "1"
        _init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cpu", name="base")
        with use_parallel_state("base"):
            model = _FSDPGdnModel()
            prepare_qwen4_gdn_precision(model)
            kwargs = dict(
                weights_path=source_dir, mixed_precision=MixedPrecisionConfig(enable=False), init_device="meta"
            )
            if mismatch:
                with pytest.raises(RuntimeError, match="policy differs across loading ranks"):
                    torch_parallelize.parallelize_model_fsdp2(model, **kwargs)
            else:
                model = torch_parallelize.parallelize_model_fsdp2(model, **kwargs)
                value = model.layers[0].linear_attn.A_log.full_tensor()
                assert value.dtype == torch.float32 and torch.equal(value, torch.full((4,), 0.125))
                model(torch.ones(2, 4, dtype=torch.bfloat16)).square().sum().backward()
                gradient = model.layers[0].linear_attn.A_log.grad.full_tensor()
                assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
                assert (gradient != 0).any()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
def test_qwen4_bf16_real_fsdp2_source_policy(tmp_path, mismatch):
    save_file(
        {
            "layers.0.linear_attn.A_log": torch.full((4,), 0.125, dtype=torch.bfloat16),
            "layers.0.linear_attn.dt_bias": torch.full((4,), 0.25, dtype=torch.bfloat16),
            "layers.0.linear_attn.proj.weight": torch.ones(4, 4, dtype=torch.bfloat16),
            "layers.0.linear_attn.proj.bias": torch.zeros(4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    mp.spawn(_fsdp_load_worker, args=(str(tmp_path / "fsdp-rdzv"), str(tmp_path), mismatch), nprocs=2, join=True)
