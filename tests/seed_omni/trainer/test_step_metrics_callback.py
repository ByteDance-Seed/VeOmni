"""``OmniStepMetricsCallback`` reduces per-node losses and module metrics the same way on every rank.

A node records a loss only when its batch produced one, so two ranks with
different modality mixes hand the callback different ``loss_dict`` keys.
Metered modules report their own ``(theoretical_flops, seqlens)``, which
:class:`OmniEnvironMeter` rolls up over the DP group. Two CPU gloo ranks, so
this runs anywhere.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.trainer.callbacks.omni_callbacks.step_metrics_callback import OmniStepMetricsCallback


_LOSS_DICTS = [{"node_a": 1.0, "node_b": 2.0}, {"node_b": 4.0}]
_MODULE_METRICS = [{"text": (30.0, [3, 5])}, {"text": (10.0, [2])}]
_MICRO_BATCHES = [[{"conversation_list": [[], []]}], [{"conversation_list": [[]]}]]


def _rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    import veomni.trainer.callbacks.base as callback_base
    import veomni.utils.dist_utils as dist_utils
    import veomni.utils.omni_helper as omni_helper

    dist_utils.get_device_type = lambda: "cpu"
    callback_base.get_parallel_state = lambda: SimpleNamespace(fsdp_group=None, dp_group=None)
    omni_helper.get_device_flops = lambda: 100.0
    omni_helper.compute_device_memory_metrics = lambda: {}
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_batch_size=4, empty_cache_steps=0, gc_steps=0),
                data=SimpleNamespace(enable_multisource=False, train_path=""),
            ),
            train_dataloader=None,
            model=SimpleNamespace(
                lr_scheduler=SimpleNamespace(get_last_lr=lambda: [1e-4]),
                metric_meter_collect=lambda: _MODULE_METRICS[rank],
            ),
        )
        callback = OmniStepMetricsCallback(trainer)

        callback.on_step_begin(SimpleNamespace(), micro_batches=_MICRO_BATCHES[rank])
        callback.start_time -= 2.0
        callback.on_step_end(SimpleNamespace(global_step=1), loss=1.0, loss_dict=_LOSS_DICTS[rank], grad_norm=0.5)

        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump({"train": trainer.step_train_metrics, "env": trainer.step_env_metrics}, f)
    finally:
        dist.destroy_process_group()


def test_ranks_reduce_the_same_step_metrics(tmp_path):
    mp.spawn(_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)

    metrics = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    train = metrics[0]["train"]
    assert train == metrics[1]["train"]
    # node_a ran on rank 0 only: its loss is rank 0's, not halved by rank 1.
    assert train["training/node_a"] == 1.0
    assert train["training/node_b"] == 3.0
    assert train["training/total_loss"] == 1.0

    env = metrics[0]["env"]
    assert env.keys() == metrics[1]["env"].keys()
    assert {key: env[key] for key in train} == train
    assert env["consumed_chunk_num"] == 3
    assert env["trace/text/consume_tokens(M)"] == 10 / 1e6
    assert env["trace/text/avg_seq_len"] == 10 / 4
    assert abs(env["flops_achieved(T)"] - 40.0 / 2.0) < 1.0
    assert env["flops_promised(T)"] == 200.0
