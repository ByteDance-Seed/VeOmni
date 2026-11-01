"""Lossless PLE file-backed moments with strict MultiOptimizer DCP."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
from veomni.optim import file_backed_state, qwen38_muon
from veomni.optim.optimizer import MultiOptimizer
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build(directory, file_mode):
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.Embedding(9, 5, dtype=torch.bfloat16)
    with torch.no_grad():
        model.ngram_embedding.weight.copy_(torch.arange(45).reshape(9, 5).float() / 20)
    groups = add_qwen38_plans(
        model, [{"params": [model.ngram_embedding.weight], "weight_decay": 0.0, "cpu_offload": True}]
    )
    kwargs = {"ple_moments_scratch_dir": str(directory), "ple_moments_world_profile": True} if file_mode else {}
    # CI may run on an overlay filesystem or with less than the production
    # 64-GiB disk reserve. Test arithmetic here; storage policy is tested in
    # test_file_backed_state.py without weakening the production default.
    with (
        patch.object(file_backed_state, "_filesystem_type", return_value="ext4"),
        patch.object(
            qwen38_muon,
            "FileBackedFP32State",
            lambda path: file_backed_state.FileBackedFP32State(path, minimum_free_bytes=0),
        ),
    ):
        leaf = Qwen38MuonAdamW(groups, lr=1e-3, **kwargs)
    return model, leaf, MultiOptimizer(model, {"ple": leaf}, ["ple"])


def _update(model, optimizer):
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.25)
    optimizer.step()
    optimizer.zero_grad()


def test_file_backed_ple_matches_fp32_and_strict_dcp_next_update(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    model, leaf, optimizer = _build(scratch, True)
    reference, ref_leaf, ref_optimizer = _build(scratch, False)
    for _ in range(2):
        _update(model, optimizer)
        _update(reference, ref_optimizer)
    param, ref_param = model.ngram_embedding.weight, reference.ngram_embedding.weight
    for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        actual, expected = leaf.state[param][key], ref_leaf.state[ref_param][key]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq"):
        assert leaf._ple_moment_storage.owns(leaf.state[param][key])
    parallel = SimpleNamespace(dp_mode="ddp")
    path = tmp_path / "checkpoint"
    DistributedCheckpointer.save(str(path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    target, target_leaf, restored = _build(scratch, True)
    DistributedCheckpointer.load(str(path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    target_param = target.ngram_embedding.weight
    for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(leaf.state[param][key], target_leaf.state[target_param][key], rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq"):
        assert target_leaf._ple_moment_storage.owns(target_leaf.state[target_param][key])
    _update(model, optimizer)
    _update(target, restored)
    torch.testing.assert_close(param, target_param, rtol=0, atol=0)
    for key in ("master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(leaf.state[param][key], target_leaf.state[target_param][key], rtol=0, atol=0)


@pytest.mark.parametrize("save_file,load_file", [(True, False), (False, True)])
def test_file_backed_provider_mismatch_fails_before_model_mutation(tmp_path, save_file, load_file):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source, _, optimizer = _build(scratch, save_file)
    _update(source, optimizer)
    path = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(path, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, _, target_optimizer = _build(scratch, load_file)
    with torch.no_grad():
        target.ngram_embedding.weight.fill_(-3)
    previous = target.ngram_embedding.weight.detach().clone()
    with pytest.raises((ValueError, RuntimeError), match="(?i)optimizer|identity|metadata|layout|recipe"):
        DistributedCheckpointer.load(path, {"model": target, "optimizer": target_optimizer}, parallel_state=parallel)
    torch.testing.assert_close(previous, target.ngram_embedding.weight, rtol=0, atol=0)


def test_file_backed_state_rejects_async_checkpoint(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    model, _, optimizer = _build(scratch, True)
    _update(model, optimizer)
    with pytest.raises((ValueError, RuntimeError), match="(?i)synchronous|asynchronous|async"):
        DistributedCheckpointer.save(
            str(tmp_path / "checkpoint"),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
            save_async=True,
        )


def _cold_worker(scratch_path, checkpoint_path, reference_path, save):
    scratch = Path(scratch_path)
    model, leaf, optimizer = _build(scratch, True)
    parallel = SimpleNamespace(dp_mode="ddp")
    if save:
        _update(model, optimizer)
        _update(model, optimizer)
        DistributedCheckpointer.save(
            checkpoint_path, {"model": model, "optimizer": optimizer}, parallel_state=parallel
        )
        _update(model, optimizer)
        param = model.ngram_embedding.weight
        torch.save({"weight": param.detach(), **leaf.state[param]}, reference_path)
        return
    DistributedCheckpointer.load(checkpoint_path, {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    _update(model, optimizer)
    expected = torch.load(reference_path, map_location="cpu", weights_only=True)
    param = model.ngram_embedding.weight
    torch.testing.assert_close(param, expected["weight"], rtol=0, atol=0)
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(leaf.state[param][name], expected[name], rtol=0, atol=0)


def test_file_backed_dcp_fresh_process_next_update(tmp_path):
    import multiprocessing

    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    args = (str(scratch), str(tmp_path / "checkpoint"), str(tmp_path / "reference.pt"))
    for save in (True, False):
        child = context.Process(target=_cold_worker, args=(*args, save))
        child.start()
        child.join(timeout=30)
        if child.is_alive():
            child.terminate()
            child.join()
            pytest.fail("cold checkpoint worker timed out")
        assert child.exitcode == 0, f"cold {'save' if save else 'load'} process failed: {child.exitcode}"


def test_file_backed_failed_load_poisons_optimizer(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source, _, optimizer = _build(scratch, True)
    _update(source, optimizer)
    path = str(tmp_path / "checkpoint")
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(path, {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, target_leaf, target_optimizer = _build(scratch, True)
    from veomni.checkpoint import dcp_checkpointer

    original_load = dcp_checkpointer.dcp.load

    def fail_optimizer_load(*args, **kwargs):
        if set(kwargs.get("state_dict", {})) == {"optimizer"}:
            raise OSError("injected optimizer DCP load failure")
        return original_load(*args, **kwargs)

    with patch.object(dcp_checkpointer.dcp, "load", side_effect=fail_optimizer_load):
        with pytest.raises(OSError, match="injected optimizer DCP load failure"):
            DistributedCheckpointer.load(
                path, {"model": target, "optimizer": target_optimizer}, parallel_state=parallel
            )
    assert target_leaf._ple_state_poisoned
    with pytest.raises(RuntimeError, match="checkpoint|state failed"):
        _update(target, target_optimizer)


def test_direct_file_backed_leaf_rejects_hooks_before_invocation(tmp_path):
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    model, leaf, _ = _build(scratch, True)
    model.ngram_embedding.weight.grad = torch.ones_like(model.ngram_embedding.weight)
    calls = []
    handle = register_optimizer_step_pre_hook(lambda *args: calls.append("global"))
    try:
        with pytest.raises(ValueError, match="step hooks"):
            leaf.step()
    finally:
        handle.remove()
    assert calls == []
    assert not leaf.state
