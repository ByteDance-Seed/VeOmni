"""``OmniEnvironMeterCallback`` reduces per-node losses the same way on every rank.

A node records a loss only when its batch produced one, so two ranks with
different modality mixes hand the callback different ``loss_dict`` keys. Two
CPU gloo ranks, so this runs anywhere.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.trainer.callbacks.omni_callbacks.environ_meter_callback import OmniEnvironMeterCallback


_LOSS_DICTS = [{"node_a": 1.0, "node_b": 2.0}, {"node_b": 4.0}]


def _rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    import veomni.utils.dist_utils as dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        trainer = SimpleNamespace(
            model=SimpleNamespace(lr_scheduler=SimpleNamespace(get_last_lr=lambda: [1e-4])),
            collect_step_metrics=lambda: {},
            environ_meter=SimpleNamespace(step=lambda delta_time, global_step, module_metrics: {}),
        )
        callback = OmniEnvironMeterCallback.__new__(OmniEnvironMeterCallback)
        callback.trainer = trainer
        callback.parallel_state = SimpleNamespace(fsdp_group=None)
        callback.start_time = time.time()

        callback.on_step_end(SimpleNamespace(global_step=1), loss=1.0, loss_dict=_LOSS_DICTS[rank], grad_norm=0.5)

        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(trainer.step_train_metrics, f)
    finally:
        dist.destroy_process_group()


def test_ranks_with_different_loss_nodes_reduce_the_same_metrics(tmp_path):
    mp.spawn(_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)

    metrics = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert metrics[0] == metrics[1]
    # node_a ran on rank 0 only: its loss is rank 0's, not halved by rank 1.
    assert metrics[0]["training/node_a"] == 1.0
    assert metrics[0]["training/node_b"] == 3.0
    assert metrics[0]["training/total_loss"] == 1.0
