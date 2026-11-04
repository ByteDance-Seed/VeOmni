"""Topology and shifted-label accounting without changing model inputs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.utils import helper


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_packed_lengths_keep_real_single_token_documents():
    batch = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.arange(4).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, 3, 4]),
        "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 4]),
        "packed_sequence_slices": [(0, 3), (3, 4)],
    }
    meter = _meter(SimpleNamespace(sp_enabled=False, dp_group=None))
    meter._record_token_stats(batch)
    assert meter._batch_token_stats[0][3] == 4
    # Existing dataloader accounting keeps its prior behavior.
    assert helper._compute_seqlens(batch) == [3]
    batch["tail_padding_length"] = torch.tensor(1)
    meter = _meter(SimpleNamespace(sp_enabled=False, dp_group=None))
    meter._record_token_stats(batch)
    assert meter._batch_token_stats[0][3] == 3


def _meter(state):
    meter = object.__new__(helper.EnvironMeter)
    meter.config = SimpleNamespace(condition_model_type=None)
    meter.parallel_state = state
    meter.global_batch_size = 2
    meter.world_size = 4
    meter.consume_tokens = meter.consume_chunks = 0
    meter.batch_seqlens = []
    meter.batch_ds_idx = []
    meter.images_seqlens = []
    meter._batch_token_stats = []
    meter._batch_token_stats_complete = True
    meter.enable_multisource = False
    meter.empty_cache_steps = meter.gc_steps = 0
    meter.supports_lora_flops = False
    meter.estimate_flops = lambda lengths, delta_time, **kwargs: (0.0, 1.0)
    return meter


def _sp_worker(rank, rendezvous, missing):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        # Two data-parallel replicas, each with two distinct sequence shards.
        sp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        state = SimpleNamespace(
            sp_enabled=True, sp_size=2, sp_group=sp_groups[rank // 2], dp_group=dp_groups[rank % 2]
        )
        meter = _meter(state)
        full_labels = torch.tensor([[-100, 1, 2, -100, 3, -100, -100, -100]])
        batch = {
            "input_ids": torch.arange(4).view(1, -1),
            "labels": full_labels[:, (rank % 2) * 4 : (rank % 2 + 1) * 4],
            "cu_seq_lens_q": torch.tensor([0, 2, 4]),
            "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 5, 8]),
            "packed_sequence_slices": [(0, 3), (3, 5), (5, 8)],
            "tail_padding_length": torch.tensor(3),
        }
        if missing and rank == 0:
            batch.pop("labels")
        meter.add(batch)
        assert meter.batch_seqlens == []  # Existing local-cu tracker behavior is unchanged.
        with (
            patch.object(helper, "compute_device_memory_metrics", return_value={}),
            patch.object(helper, "get_device_type", return_value="cpu"),
        ):
            metrics = meter.step(delta_time=2.0, global_step=1)
        if missing:
            assert not any(key.startswith("token_accounting/") for key in metrics)
            return
        assert metrics["token_accounting/physical_window_tokens"] == 16
        assert metrics["token_accounting/aligned_compute_tokens"] == 10
        assert metrics["token_accounting/source_input_tokens"] == 10
        assert metrics["token_accounting/loss_tokens"] == 6
        assert metrics["token_accounting/source_fill"] == pytest.approx(10 / 16)
        assert meter._batch_token_stats == []
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("missing", [False, True])
def test_token_meter_packed_dp2_sp2(tmp_path, missing):
    mp.start_processes(_sp_worker, args=(str(tmp_path / "rdzv"), missing), nprocs=4, join=True, start_method="fork")


def _real_collator_sp_worker(rank, rendezvous):
    from veomni.data import data_collator

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        dp_groups = [dist.new_group([member]) for member in range(4)]
        state = SimpleNamespace(
            sp_enabled=True, sp_size=4, sp_rank=rank, sp_group=dist.group.WORLD, dp_group=dp_groups[rank]
        )
        samples = [
            {"input_ids": torch.arange(length), "labels": torch.arange(length), "attention_mask": torch.ones(length)}
            for length in (5, 7)
        ]
        with patch.object(data_collator, "get_parallel_state", return_value=state):
            batch = data_collator.MainCollator(pad_to_length=16)(samples)
        meter = _meter(state)
        meter.add(batch)
        assert meter.batch_seqlens == [5, 7]
        with (
            patch.object(helper, "compute_device_memory_metrics", return_value={}),
            patch.object(helper, "get_device_type", return_value="cpu"),
        ):
            metrics = meter.step(delta_time=2.0, global_step=1)
        assert metrics["token_accounting/physical_window_tokens"] == 16
        assert metrics["token_accounting/aligned_compute_tokens"] == 12
        assert metrics["token_accounting/source_input_tokens"] == 12
        assert metrics["token_accounting/loss_tokens"] == 10
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
def test_real_collator_packed_sp4(tmp_path):
    mp.start_processes(
        _real_collator_sp_worker, args=(str(tmp_path / "rdzv-sp4"),), nprocs=4, join=True, start_method="fork"
    )


def _actual_mesh_worker(rank, rendezvous, use_cp):
    from veomni.data import data_collator
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        state = _init_parallel_state(
            dp_size=2,
            dp_shard_size=2,
            cp_size=2 if use_cp else 1,
            ulysses_size=1 if use_cp else 2,
            device_type="cpu",
        )
        lengths = (3, 1) if state.dp_rank == 0 else (5, 2)
        samples = [
            {
                "input_ids": torch.arange(n),
                "labels": torch.arange(n),
                "attention_mask": torch.ones(n, dtype=torch.long),
            }
            for n in lengths
        ]
        batch = data_collator.MainCollator(pad_to_length=8)(samples)
        meter = _meter(state)
        meter.add(batch)
        with (
            patch.object(helper, "compute_device_memory_metrics", return_value={}),
            patch.object(helper, "get_device_type", return_value="cpu"),
        ):
            metrics = meter.step(delta_time=2.0, global_step=1)
        names = ("physical_window_tokens", "aligned_compute_tokens", "source_input_tokens", "loss_tokens")
        assert tuple(metrics["token_accounting/" + name] for name in names) == (16, 11, 11, 7)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4, device="cpu"))
@pytest.mark.parametrize("use_cp", [False, True])
def test_actual_dp_sequence_mesh(tmp_path, use_cp):
    mp.start_processes(
        _actual_mesh_worker, args=(str(tmp_path / "rdzv-mesh"), use_cp), nprocs=4, join=True, start_method="fork"
    )


def test_unsharded_generic_shifted_label_count():
    state = SimpleNamespace(sp_enabled=False, dp_group=None)
    meter = _meter(state)
    meter._record_token_stats(
        {
            "input_ids": torch.arange(4).unsqueeze(0),
            "labels": torch.tensor([[4, 5, -100, 7]]),
            "attention_mask": torch.ones(1, 4),
        }
    )
    assert meter._batch_token_stats == [(4, 4, 2, 4)]


def test_ambiguous_input_is_not_reported_as_exact():
    meter = _meter(SimpleNamespace(sp_enabled=True, sp_size=2, dp_group=None))
    meter._record_token_stats({"input_ids": torch.arange(4), "labels": torch.arange(4)})
    assert not meter._batch_token_stats_complete
    assert meter._batch_token_stats == []

    meter = _meter(SimpleNamespace(sp_enabled=False, dp_group=None))
    meter._record_token_stats({"input_ids": torch.arange(4).unsqueeze(0), "labels": torch.tensor([1])})
    assert not meter._batch_token_stats_complete
    assert meter._batch_token_stats == []

    for metadata in ({}, {"cu_seq_lens_q": torch.tensor([0, 3, 4])}):
        meter = _meter(SimpleNamespace(sp_enabled=False, dp_group=None))
        meter._record_token_stats(
            {"input_ids": torch.arange(4).unsqueeze(0), "labels": torch.arange(4).unsqueeze(0), **metadata}
        )
        assert not meter._batch_token_stats_complete

    meter = _meter(SimpleNamespace(sp_enabled=True, sp_size=2, dp_group=None))
    meter._record_token_stats(
        {
            "input_ids": torch.arange(4).unsqueeze(0),
            "labels": torch.arange(4).unsqueeze(0),
            "linear_attn_cu_seq_lens_q": torch.tensor([0, 3, 4]),
            "packed_sequence_slices": [(0, 3), (3, 4)],
        }
    )
    assert not meter._batch_token_stats_complete

    meter = _meter(SimpleNamespace(sp_enabled=False, dp_group=None))
    meter._record_token_stats(
        {
            "input_ids": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(8).unsqueeze(0),
            "cu_seq_lens_q": torch.tensor([0, 5, 8]),
            "tail_padding_length": torch.tensor(2),
        }
    )
    assert not meter._batch_token_stats_complete
