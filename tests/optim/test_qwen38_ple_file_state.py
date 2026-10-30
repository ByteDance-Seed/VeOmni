"""Precision and DCP coverage for optional file-backed FP32 PLE moments."""

import hashlib
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.planner import LoadItemType

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint import dcp_checkpointer as checkpoint_module
from veomni.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    OptimizerState,
    _FileBackedSavePlanner,
    _ModelStrictLoadPlanner,
)
from veomni.optim import file_backed_state, qwen38_muon
from veomni.optim.file_backed_state import FileBackedFP32State
from veomni.optim.qwen38_muon import Qwen38MuonAdamW, add_qwen38_plans


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture(autouse=True)
def small_test_disk_reserve(monkeypatch):
    monkeypatch.setattr(file_backed_state, "_filesystem_type", lambda root: "ext4")
    monkeypatch.setattr(
        qwen38_muon, "FileBackedFP32State", lambda root: FileBackedFP32State(root, minimum_free_bytes=0)
    )


def build(model, scratch=None):
    groups = add_qwen38_plans(model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}])
    return Qwen38MuonAdamW(
        groups, lr=1e-4, chunk_size=7, ple_moments_scratch_dir=scratch, ple_moments_world_profile=scratch is not None
    )


def make_model():
    model = torch.nn.Module()
    model.ngram_embedding = torch.nn.ModuleDict({"a": torch.nn.Embedding(11, 5, dtype=torch.bfloat16)})
    return model


def state_equal(a, b):
    for ap, bp in zip(a.param_groups, b.param_groups, strict=True):
        left, right = a.state[ap["params"][0]], b.state[bp["params"][0]]
        for key in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
            assert left[key].dtype == right[key].dtype == torch.float32
            assert torch.equal(left[key].view(torch.int32), right[key].view(torch.int32)), key


def update(model, opt, index):
    p = next(model.parameters())
    p.grad = (torch.arange(p.numel()).view(p.shape) * 0.003 + index * 0.001).bfloat16()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_file_state_preserves_adam_updates_bitwise(tmp_path):
    torch.manual_seed(7)
    reference_model = make_model()
    mapped_model = make_model()
    mapped_model.load_state_dict(reference_model.state_dict())
    reference, mapped = build(reference_model), build(mapped_model, str(tmp_path))
    for step in range(6):
        update(reference_model, reference, step)
        update(mapped_model, mapped, step)
        state_equal(reference, mapped)
        assert torch.equal(next(reference_model.parameters()), next(mapped_model.parameters()))
    p = next(mapped_model.parameters())
    assert mapped._ple_moment_storage.owns(mapped.state[p]["exp_avg"])
    assert mapped._ple_moment_storage.owns(mapped.state[p]["exp_avg_sq"])


def test_file_state_preserves_owner_local_adam_when_requests_alternate(tmp_path):
    models = []
    optimizers = []
    references = []
    reference_optimizers = []
    for _owner in range(2):
        pair = []
        pair_opt = []
        for mapped in (False, True):
            model = torch.nn.Module()
            model.ngram_embedding = torch.nn.Embedding(1, 1, dtype=torch.float32)
            model.ngram_embedding.weight.data.fill_(1)
            groups = add_qwen38_plans(
                model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
            )
            optimizer = Qwen38MuonAdamW(
                groups,
                lr=0.1,
                betas=(0.9, 0.95),
                ple_moments_scratch_dir=str(tmp_path) if mapped else None,
                ple_moments_world_profile=mapped,
            )
            pair.append(model)
            pair_opt.append(optimizer)
        reference = torch.nn.Parameter(torch.ones(1, 1))
        models.append(pair)
        optimizers.append(pair_opt)
        references.append(reference)
        reference_optimizers.append(torch.optim.Adam([reference], lr=0.1, betas=(0.9, 0.95)))

    for active in (0, 1, 0, 1):
        for owner in range(2):
            for model in models[owner]:
                model.ngram_embedding.weight.grad = torch.ones(1, 1) if owner == active else None
            references[owner].grad = torch.ones(1, 1) if owner == active else None
            for optimizer in optimizers[owner]:
                optimizer.step()
            reference_optimizers[owner].step()
            ordinary, mapped = models[owner]
            torch.testing.assert_close(ordinary.ngram_embedding.weight, references[owner], rtol=0, atol=1e-7)
            assert torch.equal(ordinary.ngram_embedding.weight, mapped.ngram_embedding.weight)
            left = optimizers[owner][0].state.get(ordinary.ngram_embedding.weight, {})
            right = optimizers[owner][1].state.get(mapped.ngram_embedding.weight, {})
            assert left.keys() == right.keys()
            for field in left:
                assert torch.equal(left[field], right[field]), field


def test_file_state_requires_explicit_world_profile(tmp_path):
    model = make_model()
    groups = add_qwen38_plans(model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}])
    with pytest.raises(ValueError, match="full-world optimizer profile"):
        Qwen38MuonAdamW(groups, ple_moments_scratch_dir=str(tmp_path))


@pytest.mark.parametrize("flatten", [False, True])
def test_empty_non_ple_file_optimizer_snapshot_never_steps(tmp_path, flatten):
    from veomni.optim.optimizer import optimizer_state_dict_for_checkpoint

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    groups = add_qwen38_plans(model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}])
    optimizer = Qwen38MuonAdamW(groups, lr=0.1, ple_moments_scratch_dir=str(tmp_path), ple_moments_world_profile=True)
    before = model.lm_head.weight.detach().clone()
    snapshot = optimizer_state_dict_for_checkpoint(model, optimizer, for_load=False, flatten=flatten)
    assert not optimizer.state
    assert torch.equal(model.lm_head.weight, before)
    if flatten:
        assert any(key.startswith("param_groups.") for key in snapshot)
        assert not any(key.startswith("state.") for key in snapshot)
    else:
        assert snapshot["state"] == {}
        assert snapshot["param_groups"]

    pristine_model = torch.nn.Module()
    pristine_model.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    pristine_model.lm_head.weight.data.copy_(before)
    pristine_groups = add_qwen38_plans(
        pristine_model, [{"params": list(pristine_model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
    )
    pristine = Qwen38MuonAdamW(pristine_groups, lr=0.1)
    gradient = torch.full_like(before, 0.25)
    model.lm_head.weight.grad = gradient.clone()
    pristine_model.lm_head.weight.grad = gradient.clone()
    optimizer.step()
    pristine.step()
    assert torch.equal(model.lm_head.weight, pristine_model.lm_head.weight)
    for name in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            optimizer.state[model.lm_head.weight][name], pristine.state[pristine_model.lm_head.weight][name]
        )


def test_empty_non_ple_file_leaf_keeps_recipe_with_active_ple(tmp_path):
    from veomni.optim.optimizer import MultiOptimizer

    model = make_model()
    model.lm_head = torch.nn.Linear(5, 5, bias=False, dtype=torch.bfloat16)
    ple_groups = add_qwen38_plans(
        model, [{"params": [model.ngram_embedding["a"].weight], "weight_decay": 0.0, "cpu_offload": True}]
    )
    dense_groups = add_qwen38_plans(
        model, [{"params": [model.lm_head.weight], "weight_decay": 0.0, "cpu_offload": True}]
    )
    options = {"ple_moments_scratch_dir": str(tmp_path), "ple_moments_world_profile": True}
    ple = Qwen38MuonAdamW(ple_groups, **options)
    dense = Qwen38MuonAdamW(dense_groups, **options)
    model.ngram_embedding["a"].weight.grad = torch.ones_like(model.ngram_embedding["a"].weight)
    ple.step()
    multi = MultiOptimizer(model, {"ple": ple, "dense": dense}, ["ple", "dense"])
    snapshot = multi._sparse_state_dict()
    assert not dense.state
    assert any(key.startswith("param_groups.lm_head.weight.") for key in snapshot)
    assert not any(key.startswith("state.lm_head.weight.") for key in snapshot)
    assert any(key.startswith("state.ngram_embedding.a.weight.") for key in snapshot)


def test_empty_snapshot_rejects_concurrent_optimizer_mutation(tmp_path):
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(3, 3, bias=False)
    groups = add_qwen38_plans(model, [{"params": list(model.parameters()), "weight_decay": 0.0}])
    optimizer = Qwen38MuonAdamW(groups, ple_moments_scratch_dir=str(tmp_path), ple_moments_world_profile=True)
    parameter = model.lm_head.weight
    with pytest.raises(RuntimeError, match="changed during"):
        with optimizer.checkpoint_state(load=False):
            optimizer.state[parameter]["step"] = torch.tensor(1.0)
    assert optimizer._ple_state_poisoned
    assert optimizer.state[parameter]["step"].item() == 1


def test_zero_state_file_checkpoint_restores_lazy_first_update(tmp_path):
    def create():
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        groups = add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        optimizer = Qwen38MuonAdamW(
            groups, lr=0.1, ple_moments_scratch_dir=str(tmp_path), ple_moments_world_profile=True
        )
        return model, optimizer

    torch.manual_seed(83)
    original, source = create()
    target, resumed = create()
    reference, pristine = create()
    reference.load_state_dict(original.state_dict())
    path = tmp_path / "zero-checkpoint"
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(path), {"model": original, "optimizer": source}, parallel_state=parallel)
    assert not source.state
    DistributedCheckpointer.load(str(path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert not resumed.state
    assert torch.equal(target.lm_head.weight, original.lm_head.weight)
    gradient = torch.full_like(target.lm_head.weight, 0.125)
    target.lm_head.weight.grad = gradient.clone()
    reference.lm_head.weight.grad = gradient.clone()
    resumed.step()
    pristine.step()
    assert torch.equal(target.lm_head.weight, reference.lm_head.weight)
    for field in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            resumed.state[target.lm_head.weight][field], pristine.state[reference.lm_head.weight][field]
        )


def test_zero_state_file_extra_parallel_checkpoint_keeps_lazy_state(tmp_path, monkeypatch):
    from veomni.optim.optimizer import MultiOptimizer

    monkeypatch.setattr(checkpoint_module, "_validate_extra_parallel_meshes", lambda state: None)

    def create():
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        model._fqn2spec_info = {}
        groups = add_qwen38_plans(
            model, [{"params": list(model.parameters()), "weight_decay": 0.0, "cpu_offload": True}]
        )
        leaf = Qwen38MuonAdamW(groups, lr=0.1, ple_moments_scratch_dir=str(tmp_path), ple_moments_world_profile=True)
        return model, leaf, MultiOptimizer(model, {"dense": leaf}, ["dense"])

    original, source_leaf, source = create()
    target, resumed_leaf, resumed = create()
    path = tmp_path / "zero-extra-checkpoint"
    parallel = SimpleNamespace(dp_mode="fsdp2")
    DistributedCheckpointer.save(str(path), {"model": original, "optimizer": source}, parallel_state=parallel)
    assert not source_leaf.state
    DistributedCheckpointer.load(str(path), {"model": target, "optimizer": resumed}, parallel_state=parallel)
    assert not resumed_leaf.state
    assert torch.equal(target.lm_head.weight, original.lm_head.weight)
    gradient = torch.full_like(target.lm_head.weight, 0.125)
    original.lm_head.weight.grad = gradient.clone()
    target.lm_head.weight.grad = gradient.clone()
    source.step()
    resumed.step()
    assert torch.equal(target.lm_head.weight, original.lm_head.weight)
    for field in ("step", "master_weight", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            resumed_leaf.state[target.lm_head.weight][field], source_leaf.state[original.lm_head.weight][field]
        )


def test_file_state_dcp_restore_and_next_update_bitwise(tmp_path):
    torch.manual_seed(11)
    source_model = make_model()
    source = build(source_model)
    for step in range(3):
        update(source_model, source, step)
    sd = OptimizerState(source_model, source, parallel_state=SimpleNamespace(dp_mode="ddp"))
    dcp.save({"optimizer": sd}, checkpoint_id=tmp_path / "dcp")
    resumed_model = make_model()
    resumed_model.load_state_dict(source_model.state_dict())
    resumed = build(resumed_model, str(tmp_path))
    target = OptimizerState(resumed_model, resumed, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)
    planner = _ModelStrictLoadPlanner(
        strict_model=False, strict_optimizer_dtype=True, strict_optimizer_recipe=True, optimizer=resumed
    )
    dcp.load({"optimizer": target}, checkpoint_id=tmp_path / "dcp", planner=planner)
    state_equal(source, resumed)
    update(source_model, source, 4)
    update(resumed_model, resumed, 4)
    state_equal(source, resumed)
    assert torch.equal(next(source_model.parameters()), next(resumed_model.parameters()))


def test_file_state_rejects_nonmapped_direct_resume(tmp_path):
    model = make_model()
    source = build(model)
    update(model, source, 1)
    other = make_model()
    other.load_state_dict(model.state_dict())
    resumed = build(other, str(tmp_path))
    with pytest.raises(RuntimeError, match="file-backed DCP target"):
        resumed.load_state_dict(source.state_dict())
    with pytest.raises(RuntimeError, match="fresh process"):
        resumed.step()
    with pytest.raises(RuntimeError, match="fresh process"):
        resumed.state_dict()


def test_file_state_mapped_save_restore_and_next_update(tmp_path):
    model = make_model()
    source = build(model, str(tmp_path))
    for i in range(2):
        update(model, source, i)
    state = OptimizerState(model, source, parallel_state=SimpleNamespace(dp_mode="ddp"))
    planner = _FileBackedSavePlanner(optimizers=(source,))
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path / "mapped-dcp", planner=planner)
    source.checkpoint_evict_file_backed_state()
    other = make_model()
    other.load_state_dict(model.state_dict())
    resumed = build(other, str(tmp_path))
    target = OptimizerState(other, resumed, parallel_state=SimpleNamespace(dp_mode="ddp"), load=True)
    loader = _ModelStrictLoadPlanner(
        strict_model=False, strict_optimizer_dtype=True, strict_optimizer_recipe=True, optimizer=resumed
    )
    dcp.load({"optimizer": target}, checkpoint_id=tmp_path / "mapped-dcp", planner=loader)
    state_equal(source, resumed)
    update(model, source, 3)
    update(other, resumed, 3)
    state_equal(source, resumed)


def test_file_backed_dcp_load_evicts_source_without_changing_update(tmp_path, monkeypatch):
    torch.manual_seed(19)
    source_model = make_model()
    source = build(source_model)
    update(source_model, source, 1)
    path = tmp_path / "source-split"
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(path), {"model": source_model, "optimizer": source}, parallel_state=parallel)
    checkpoint_files = list(path.rglob("*.distcp"))
    original_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files}
    seen = []
    original_evict = checkpoint_module._SourceCacheEvictor.evict

    def observe(self, item):
        seen.append((item.type, self.reader.storage_data[item.storage_index].length))
        return original_evict(self, item)

    monkeypatch.setattr(checkpoint_module._SourceCacheEvictor, "evict", observe)
    model = make_model()
    optimizer = build(model, str(tmp_path))
    DistributedCheckpointer.load(str(path), {"model": model, "optimizer": optimizer}, parallel_state=parallel)
    assert seen and all(length > 0 for _, length in seen)
    assert {kind for kind, _ in seen} == {LoadItemType.TENSOR, LoadItemType.BYTE_IO}
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files} == original_hashes
    state_equal(source, optimizer)
    update(source_model, source, 2)
    update(model, optimizer, 2)
    state_equal(source, optimizer)
    assert torch.equal(next(source_model.parameters()), next(model.parameters()))


def test_source_cache_advice_aligns_range_and_closes_fd(tmp_path, monkeypatch):
    path = tmp_path / "shard"
    path.write_bytes(b"a" * 8192)
    reader = SimpleNamespace(
        storage_data={"key": SimpleNamespace(relative_path="shard", offset=3, length=4099)},
        fs=SimpleNamespace(concat_path=lambda parent, name: parent / name),
        path=tmp_path,
    )
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, start, count, advice: calls.append((fd, start, count, advice)))
    with checkpoint_module._source_cache_evictor(reader, True) as evictor:
        evictor.evict(SimpleNamespace(storage_index="key"))
    assert len(calls) == 1
    fd, start, count, advice = calls[0]
    assert start == 0 and count == 8192 and advice == os.POSIX_FADV_DONTNEED
    with pytest.raises(OSError):
        os.fstat(fd)


def test_source_cache_advice_bounds_fds_under_low_rlimit(tmp_path):
    for index in range(64):
        (tmp_path / f"shard-{index}").write_bytes(b"a" * 4096)
    script = """
import resource, sys
from pathlib import Path
from types import SimpleNamespace
from veomni.checkpoint.dcp_checkpointer import _source_cache_evictor
root = Path(sys.argv[1])
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32 if soft == resource.RLIM_INFINITY else min(32, soft), hard))
data = {i: SimpleNamespace(relative_path=f"shard-{i}", offset=0, length=4096) for i in range(64)}
reader = SimpleNamespace(storage_data=data, fs=SimpleNamespace(concat_path=lambda parent, name: parent / name), path=root)
with _source_cache_evictor(reader, True) as evictor:
    for index in range(64):
        evictor.evict(SimpleNamespace(storage_index=index))
assert evictor.requests == 64
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_source_cache_advice_supports_symlink_shard(tmp_path):
    (tmp_path / "real").write_bytes(b"untouched")
    (tmp_path / "link").symlink_to("real")
    reader = SimpleNamespace(
        storage_data={"key": SimpleNamespace(relative_path="link", offset=0, length=9)},
        fs=SimpleNamespace(concat_path=lambda parent, name: parent / name),
        path=tmp_path,
    )
    with checkpoint_module._source_cache_evictor(reader, True) as evictor:
        evictor.evict(SimpleNamespace(storage_index="key"))
    assert (tmp_path / "real").read_bytes() == b"untouched"


def test_source_cache_close_attempts_all_fds_after_one_error(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / f"shard-{index}").write_bytes(b"a" * 4096)
    reader = SimpleNamespace(
        storage_data={i: SimpleNamespace(relative_path=f"shard-{i}", offset=0, length=4096) for i in range(3)},
        fs=SimpleNamespace(concat_path=lambda parent, name: parent / name),
        path=tmp_path,
    )
    original_close = os.close
    closed = []
    with pytest.raises(OSError, match="injected close error"):
        with checkpoint_module._source_cache_evictor(reader, True) as evictor:
            for index in range(3):
                evictor.evict(SimpleNamespace(storage_index=index))
            owned_fds = set(evictor._fds.values())

            def injected_close(fd):
                original_close(fd)
                if fd in owned_fds:
                    closed.append(fd)
                    if len(closed) == 1:
                        raise OSError("injected close error")

            monkeypatch.setattr(os, "close", injected_close)
    assert set(closed) == owned_fds


def test_source_cache_log_failure_closes_fds(tmp_path, monkeypatch):
    (tmp_path / "shard").write_bytes(b"a" * 4096)
    reader = SimpleNamespace(
        storage_data={"key": SimpleNamespace(relative_path="shard", offset=0, length=4096)},
        fs=SimpleNamespace(concat_path=lambda parent, name: parent / name),
        path=tmp_path,
    )

    def failed_log(*args, **kwargs):
        raise OSError("injected logging failure")

    monkeypatch.setattr(checkpoint_module.logger, "info_rank0", failed_log)
    with pytest.raises(OSError, match="injected logging failure"):
        with checkpoint_module._source_cache_evictor(reader, True) as evictor:
            evictor.evict(SimpleNamespace(storage_index="key"))
            fd = next(iter(evictor._fds.values()))
    with pytest.raises(OSError):
        os.fstat(fd)


def test_source_cache_close_waits_for_inflight_advice(tmp_path, monkeypatch):
    (tmp_path / "shard").write_bytes(b"a" * 4096)
    reader = SimpleNamespace(
        storage_data={"key": SimpleNamespace(relative_path="shard", offset=0, length=4096)},
        fs=SimpleNamespace(concat_path=lambda parent, name: parent / name),
        path=tmp_path,
    )
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def blocked_advice(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(os, "posix_fadvise", blocked_advice)
    evictor = checkpoint_module._SourceCacheEvictor(reader)
    worker = threading.Thread(target=evictor.evict, args=(SimpleNamespace(storage_index="key"),))
    closer = threading.Thread(target=lambda: (evictor.close(), closed.set()))
    worker.start()
    assert entered.wait(5)
    closer.start()
    assert not closed.wait(0.1)
    release.set()
    worker.join(5)
    closer.join(5)
    assert closed.is_set() and not worker.is_alive()
    with pytest.raises(RuntimeError, match="already closed"):
        evictor.evict(SimpleNamespace(storage_index="key"))


def test_file_state_rejects_async_dcp_before_writing(tmp_path):
    model = make_model()
    opt = build(model, str(tmp_path))
    destination = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="synchronous DCP"):
        DistributedCheckpointer.save(str(destination), {"model": model, "optimizer": opt}, save_async=True)
    assert not destination.exists()


def test_file_state_refuses_to_overwrite_completed_checkpoint(tmp_path):
    model = make_model()
    opt = build(model, str(tmp_path))
    marker = tmp_path / "existing" / "model" / "ckpt" / ".metadata"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"existing-checkpoint")
    with pytest.raises(FileExistsError, match="overwrite"):
        DistributedCheckpointer.save(
            str(tmp_path / "existing"),
            {"model": model, "optimizer": opt},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert marker.read_bytes() == b"existing-checkpoint"


def test_file_state_retry_uses_new_root_without_overwriting_incomplete_checkpoint(tmp_path, monkeypatch):
    import torch.distributed.checkpoint.filesystem as filesystem
    from torch.distributed.checkpoint.api import CheckpointException

    source_model = make_model()
    source = build(source_model)
    update(source_model, source, 1)
    parallel = SimpleNamespace(dp_mode="ddp")
    complete = tmp_path / "last-complete"
    DistributedCheckpointer.save(str(complete), {"model": source_model, "optimizer": source}, parallel_state=parallel)

    first_model = make_model()
    first = build(first_model, str(tmp_path))
    DistributedCheckpointer.load(str(complete), {"model": first_model, "optimizer": first}, parallel_state=parallel)
    update(first_model, first, 2)
    interrupted = tmp_path / "interrupted"
    original_write = filesystem._write_item

    def fail_after_one_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise OSError("injected failure after writing a DCP item")

    with monkeypatch.context() as patch:
        patch.setattr(filesystem, "_write_item", fail_after_one_write)
        with pytest.raises(CheckpointException, match="injected failure"):
            DistributedCheckpointer.save(
                str(interrupted), {"model": first_model, "optimizer": first}, parallel_state=parallel
            )
    assert not (interrupted / "model" / "ckpt" / ".metadata").exists()
    shards = list(interrupted.rglob("*.distcp"))
    assert shards
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in shards}

    retry_model = make_model()
    retry_optimizer = build(retry_model, str(tmp_path))
    DistributedCheckpointer.load(
        str(complete), {"model": retry_model, "optimizer": retry_optimizer}, parallel_state=parallel
    )
    update(retry_model, retry_optimizer, 2)
    with pytest.raises(FileExistsError, match="overwrite"):
        DistributedCheckpointer.save(
            str(interrupted), {"model": retry_model, "optimizer": retry_optimizer}, parallel_state=parallel
        )
    retry = tmp_path / "retry"
    DistributedCheckpointer.save(
        str(retry), {"model": retry_model, "optimizer": retry_optimizer}, parallel_state=parallel
    )
    resumed_model = make_model()
    resumed = build(resumed_model, str(tmp_path))
    DistributedCheckpointer.load(str(retry), {"model": resumed_model, "optimizer": resumed}, parallel_state=parallel)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in shards} == before
    state_equal(retry_optimizer, resumed)
    update(retry_model, retry_optimizer, 3)
    update(resumed_model, resumed, 3)
    state_equal(retry_optimizer, resumed)
    assert torch.equal(next(retry_model.parameters()), next(resumed_model.parameters()))


def test_file_mode_serializes_both_checkpoint_writers(tmp_path, monkeypatch):
    model = make_model()
    optimizer = build(model, str(tmp_path))
    calls = []

    def capture_save(*, state_dict, storage_writer, **kwargs):
        calls.append((next(iter(state_dict)), storage_writer.thread_count, storage_writer.release_page_cache))

    monkeypatch.setattr(checkpoint_module.dcp, "save", capture_save)
    DistributedCheckpointer.save(
        str(tmp_path / "checkpoint"),
        {"model": model, "optimizer": optimizer},
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    assert calls == [("model", 1, True), ("optimizer", 1, True)]


def test_file_mode_model_write_error_does_not_publish_checkpoint(tmp_path, monkeypatch):
    import torch.distributed.checkpoint.filesystem as filesystem
    from torch.distributed.checkpoint.api import CheckpointException

    model = make_model()
    optimizer = build(model, str(tmp_path))
    destination = tmp_path / "checkpoint"

    def fail_write(*args, **kwargs):
        raise OSError("injected model write failure")

    monkeypatch.setattr(filesystem, "_write_item", fail_write)
    with pytest.raises(CheckpointException, match="injected model write failure"):
        DistributedCheckpointer.save(
            str(destination),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not (destination / "model" / "ckpt" / ".metadata").exists()
    assert not (destination / "checkpoint_manifest.json").exists()


def test_file_mode_optimizer_cache_io_error_removes_model_marker(tmp_path, monkeypatch):
    from pathlib import Path

    from torch.distributed.checkpoint.api import CheckpointException

    from veomni.checkpoint.bounded_writer import BoundedFileSystemWriter

    model = make_model()
    optimizer = build(model, str(tmp_path))
    destination = tmp_path / "checkpoint"
    original = BoundedFileSystemWriter._advise_closed_file

    def fail_optimizer(self, path):
        if Path(path).parent.name == "optimizer":
            raise OSError("injected optimizer cache-advice I/O failure")
        return original(self, path)

    monkeypatch.setattr(BoundedFileSystemWriter, "_advise_closed_file", fail_optimizer)
    with pytest.raises(CheckpointException, match="injected optimizer cache-advice I/O failure"):
        DistributedCheckpointer.save(
            str(destination),
            {"model": model, "optimizer": optimizer},
            parallel_state=SimpleNamespace(dp_mode="ddp"),
        )
    assert not (destination / "model" / "ckpt" / ".metadata").exists()
    assert not (destination / "model" / "optimizer" / ".metadata").exists()
    assert not (destination / "checkpoint_manifest.json").exists()


def test_standard_adam_async_path_runs_both_save_slots(tmp_path, monkeypatch):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters())
    calls = []
    monkeypatch.setattr(checkpoint_module.dist, "new_group", lambda **kw: object())
    monkeypatch.setattr(DistributedCheckpointer, "_async_process_groups", {})
    monkeypatch.setattr(DistributedCheckpointer, "_save_futures", {})

    def fake_async_save(**kwargs):
        calls.append(kwargs)
        future = Future()
        future.set_result(None)
        return future

    monkeypatch.setattr(checkpoint_module.dcp, "async_save", fake_async_save)
    DistributedCheckpointer.save(
        str(tmp_path / "checkpoint"),
        {"model": model, "optimizer": opt},
        save_async=True,
        parallel_state=SimpleNamespace(dp_mode="ddp"),
    )
    assert len(calls) == 2
    assert {tuple(call["state_dict"]) for call in calls} == {("model",), ("optimizer",)}
    assert all(type(call["storage_writer"]) is checkpoint_module.FileSystemWriter for call in calls)


def test_standard_sync_save_keeps_original_barrier(monkeypatch):
    calls = []
    monkeypatch.setattr(checkpoint_module.dcp, "save", lambda **kw: None)
    monkeypatch.setattr(checkpoint_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(checkpoint_module, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpoint_module, "synchronize", lambda: None)
    DistributedCheckpointer.execute_save({}, SimpleNamespace(), save_async=False, file_mode_override=False)
    assert calls == ["barrier"]


def test_mapped_checkpoint_claim_excludes_other_process(tmp_path):
    claim = checkpoint_module._claim_file_checkpoint
    with claim(str(tmp_path), ""):
        locks = list(tmp_path.glob(".veomni-file-state-save-*.lock"))
        assert len(locks) == 1
        lock = locks[0]
        code = "import os,sys; os.open(sys.argv[1], os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)"
        competing = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True, check=False)
        assert competing.returncode != 0 and b"FileExistsError" in competing.stderr
    assert not lock.exists()
    with claim(str(tmp_path), ""):
        assert lock.exists()


def test_file_state_io_failure_invalidates_optimizer(tmp_path, monkeypatch):
    model = make_model()
    opt = build(model, str(tmp_path))
    update(model, opt, 1)

    def failed(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(opt._ple_moment_storage, "evict_range", failed)
    with pytest.raises(OSError, match="injected"):
        update(model, opt, 2)
    with pytest.raises(RuntimeError, match="fresh process"):
        opt.step()
    with pytest.raises(RuntimeError, match="fresh process"):
        opt.state_dict()


def test_checkpoint_failure_invalidates_all_optimizer_leaves(monkeypatch):
    class Leaf:
        _ple_moment_storage = object()
        _ple_file_mode_enabled = True

        def __init__(self, fail):
            self.fail = fail
            self.poisoned = False

        def checkpoint_evict_file_backed_state(self):
            if self.fail:
                raise OSError("injected flush error")

        def checkpoint_abort_file_backed_state(self):
            self.poisoned = True

    left, right = Leaf(False), Leaf(True)
    root = SimpleNamespace(optimizers_dict={"left": left, "right": right})
    state = {"optimizer": SimpleNamespace(optimizer=root)}
    writer = SimpleNamespace(thread_count=16, per_thread_copy_ahead=1)
    monkeypatch.setattr(checkpoint_module.dcp, "save", lambda **kwargs: None)
    with pytest.raises(OSError, match="injected"):
        DistributedCheckpointer.execute_save(state, writer, save_async=False)
    assert left.poisoned and right.poisoned
