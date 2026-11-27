"""Combined Qwen4 FP32 GDN and proven MultiOptimizer DCP regression."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.test_qwen4_gdn_precision import ToyQwen4
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, read_qwen4_gdn_manifest
from veomni.checkpoint.layout import weights_dir
from veomni.models.qwen4_gdn_precision import prepare_qwen4_gdn_precision
from veomni.optim.master_fp32_adamw import FullFP32AdamW
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _make():
    model = ToyQwen4()
    prepare_qwen4_gdn_precision(model)
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.linear_attn.A_log.fill_(-0.12345679)
        model.linear_attn.dt_bias.fill_(0.07456543)
        model.linear_attn.proj.weight.fill_(0.25)
        model.linear_attn.proj.bias.zero_()
    leaf = FullFP32AdamW(model.parameters(), lr=0.001)
    return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])


def _update(model, optimizer, gradient=0.25):
    for param in model.parameters():
        param.grad = torch.full_like(param, gradient)
    optimizer.step()
    optimizer.zero_grad()


def test_fp32_gdn_and_proven_optimizer_save_restore_next_update(tmp_path):
    source, source_optim = _make()
    _update(source, source_optim)
    path = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(
        path, {"model": source, "optimizer": source_optim, "_veomni_scheduler_object": None}, parallel_state=parallel
    )
    manifest = read_qwen4_gdn_manifest(weights_dir(path))
    assert {"linear_attn.A_log", "linear_attn.dt_bias"}.issubset(manifest)
    restored, restored_optim = _make()
    DistributedCheckpointer.load(
        path,
        {"model": restored, "optimizer": restored_optim, "_veomni_scheduler_object": None},
        parallel_state=parallel,
    )
    for (name, original), (name2, loaded) in zip(source.named_parameters(), restored.named_parameters(), strict=True):
        assert name == name2
        if name.endswith(("A_log", "dt_bias")):
            assert original.dtype == loaded.dtype == torch.float32
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
        source_state = source_optim.optimizers_dict["leaf"].state[original]
        target_state = restored_optim.optimizers_dict["leaf"].state[loaded]
        assert set(source_state) == set(target_state) == {"step", "master_weight", "exp_avg", "exp_avg_sq"}
        for key in source_state:
            torch.testing.assert_close(source_state[key], target_state[key], rtol=0, atol=0)
            assert source_state[key].dtype == target_state[key].dtype == torch.float32
    _update(source, source_optim, gradient=-0.125)
    _update(restored, restored_optim, gradient=-0.125)
    for original, loaded in zip(source.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
