"""Standalone CPU Gloo probe of exact-checkpoint preflight coordination."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.callbacks import global_state_callback as g
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.utils import dist_utils, helper


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))


def worker(rank, rendezvous):
    dist_utils.get_device_type = lambda: "cpu"
    helper.empty_cache = lambda: None
    g._has_file_profile = lambda _: True
    g._exact_checkpoint_contract = lambda *a, **k: SimpleNamespace(
        install=lambda: None, checkpoint_identity=lambda: "contract"
    )
    g._resume_identity = lambda _: "runtime"
    g._validate_exact_environ_meter = lambda *a: None
    g.validate_resumed_prefetch_is_drained = lambda *a: None
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        for scenario in (
            "matching",
            "different_bound",
            "different_save_steps",
            "bad_attestation",
            "missing_attestation",
            "epoch_boundary",
        ):
            stop = 3 if scenario == "different_bound" and rank == 1 else 2
            if scenario == "epoch_boundary":
                stop = 100
            verified = 3 if scenario == "bad_attestation" and rank == 1 else stop
            if scenario == "missing_attestation" and rank == 1:
                verified = None
            interval = 2 if scenario == "different_save_steps" and rank == 1 else 1
            calls = []
            checkpoint = SimpleNamespace(save_epochs=0, save_hf_weights=False, save_steps=interval)
            args = SimpleNamespace(
                train_steps=100,
                train=SimpleNamespace(num_train_epochs=1, checkpoint=checkpoint, stop_after_steps=stop),
            )
            trainer = SimpleNamespace(
                args=args,
                _veomni_exact_verified_stop_after_steps=verified,
                environ_meter=SimpleNamespace(state_dict=lambda: {}),
                save_model_assets=lambda calls=calls: calls.append("assets"),
                load=lambda calls=calls: calls.append("load"),
            )
            callback = CheckpointCallback.__new__(CheckpointCallback)
            callback.trainer = trainer
            failed = False
            try:
                callback.on_train_begin(SimpleNamespace(global_step=0, epoch=0))
            except (RuntimeError, ValueError) as exc:
                failed = True
                message = str(exc)
            assert failed == (scenario != "matching"), (rank, scenario, failed, calls)
            assert calls == (["assets", "load"] if scenario == "matching" else []), (rank, scenario, calls)
            dist.barrier()
            print("rank", rank, scenario, "agreed", "rejected " + message[:80] if failed else "accepted", flush=True)
    finally:
        dist.destroy_process_group()


def test_file_profile_preflight_fails_consistently_on_rank_disagreement(tmp_path):
    mp.start_processes(worker, args=(os.path.join(tmp_path, "rdzv"),), nprocs=2, start_method="fork")
