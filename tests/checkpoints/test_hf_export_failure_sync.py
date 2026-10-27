"""A rank-zero HF export failure must not strand its peers."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec


def _run_export(rank, root, fail, backend):
    import veomni.utils.save_safetensor_utils as exports
    from veomni.utils import dist_utils

    root = Path(root)
    # A default Gloo group must use CPU even when the visible accelerator is
    # not CPU; meta makes an accidental device-based reduction fail on hosts
    # without an actual accelerator as well.
    dist_utils.get_device_type = lambda: "meta"
    dist.init_process_group(
        backend, init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        exports.synchronize = lambda: None
        if hasattr(exports, "should_use_distributed_hf_safetensor_save"):
            exports.should_use_distributed_hf_safetensor_save = lambda _: False
        else:
            exports.is_torch_version_greater_than = lambda _: False

        def legacy(*args, **kwargs):
            if fail:
                raise RuntimeError("injected export failure")

        exports._save_hf_safetensor_legacy = legacy
        try:
            exports.save_hf_safetensor(
                save_hf_safetensor_path=str(root / "hf"), ckpt_manager="legacy", is_rank_0=rank == 0
            )
            outcome = "success"
        except RuntimeError as exc:
            outcome = str(exc)
        (root / f"rank-{rank}.json").write_text(json.dumps({"outcome": outcome}))
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("backend", ["gloo", "cpu:gloo"])
def test_rank_zero_legacy_export_failure_reaches_all_ranks(tmp_path, fail, backend):
    mp.spawn(_run_export, args=(str(tmp_path), fail, backend), nprocs=2, join=True)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text())["outcome"] for rank in range(2)]
    if fail:
        assert "injected export failure" in rows[0]
        assert "failed on another rank" in rows[1]
    else:
        assert rows == ["success", "success"]
