"""Per-module metric meter: what a module reports, and how the meter rolls it up.

A module only ever produces time-independent quantities (its own token lengths
and its own theoretical FLOPs); MFU / tokens-per-second are computed once by
:class:`~veomni.utils.omni_helper.OmniEnvironMeter` from the single whole-graph
wall-clock. The roll-up tests stub out the collective / device probes so the
math is checked in-process, with no distributed init.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.models.seed_omni import MetricMeterMixin
from veomni.utils import helper, omni_helper
from veomni.utils.omni_helper import OmniEnvironMeter


class MeteredModule(MetricMeterMixin):
    """Minimal metered module: two tokens-worth of FLOPs per token."""

    def estimate_flops(self, seqlens: list[int]) -> float:
        return 2.0 * sum(seqlens)


def test_collect_drains_the_stash_and_scales_flops_by_the_step_tokens():
    module = MeteredModule()

    module.metric_meter_set_seqlens("forward", [4, 6])
    module.metric_meter_add("forward", data={})
    flops, seqlens = module.metric_meter_collect()

    assert seqlens == [4, 6]
    assert flops == 20.0


def test_add_accumulates_over_a_gradient_accumulation_step():
    """One ``add`` per micro-batch must sum over the whole global step."""
    module = MeteredModule()

    for length in (3, 5, 7):
        module.metric_meter_set_seqlens("forward", [length])
        module.metric_meter_add("forward", data={})

    flops, seqlens = module.metric_meter_collect()
    assert seqlens == [3, 5, 7]
    assert flops == 30.0


def test_collect_resets_so_the_next_step_starts_empty():
    module = MeteredModule()
    module.metric_meter_set_seqlens("forward", [8])
    module.metric_meter_add("forward", data={})
    module.metric_meter_collect()

    assert module.metric_meter_collect() == (0.0, [])


def test_a_call_site_that_stashed_nothing_contributes_nothing():
    """A module with no ``pre_forward`` for a call-site (e.g. a VQ ``decode``) adds no tokens."""
    module = MeteredModule()

    module.metric_meter_set_seqlens("encode", [9])
    module.metric_meter_add("decode", data={})

    assert module.metric_meter_collect() == (0.0, [])
    # The untouched `encode` stash is still there for its own call-site.
    assert module.metric_meter_token_lengths("encode", {}) == [9]


def test_the_stash_is_drained_once_per_call_site():
    """``metric_meter_add`` pops, so a re-read of the same stash cannot double-count."""
    module = MeteredModule()
    module.metric_meter_set_seqlens("forward", [10])

    module.metric_meter_add("forward", data={})
    module.metric_meter_add("forward", data={})

    assert module.metric_meter_collect() == (20.0, [10])


def test_a_metered_module_must_implement_estimate_flops():
    class NoFormula(MetricMeterMixin):
        pass

    with pytest.raises(NotImplementedError, match="NoFormula"):
        NoFormula().metric_meter_collect()


@pytest.fixture
def single_process_meter(monkeypatch):
    """``OmniEnvironMeter`` with the collective / device probes stubbed out.

    ``all_reduce`` over a one-rank DP group is the identity, so the roll-up math
    is exercised without a process group; device FLOPs are pinned so MFU is
    exactly checkable.
    """
    monkeypatch.setattr(omni_helper.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(omni_helper, "all_reduce", lambda values, **kwargs: list(values))
    monkeypatch.setattr(omni_helper, "get_device_flops", lambda: 100.0)
    monkeypatch.setattr(omni_helper, "compute_device_memory_metrics", dict)
    monkeypatch.setattr(omni_helper, "get_parallel_state", lambda: type("PS", (), {"dp_group": None})())
    return OmniEnvironMeter(global_batch_size=2, empty_cache_steps=0)


def test_step_sums_flops_across_modules_but_keeps_tokens_per_module(single_process_meter):
    """FLOPs are distinct compute per module (summable); tokens are not (would double-count)."""
    meter = single_process_meter
    meter.add({"conversation_list": [object(), object()]})

    metrics = meter.step(
        delta_time=2.0,
        global_step=1,
        module_metrics={"backbone": (60.0, [30, 30]), "vision": (40.0, [16])},
    )

    assert metrics["flops_achieved(T)"] == 50.0  # (60 + 40) / 2.0
    assert metrics["mfu"] == 0.5  # 50 achieved / (100 promised * 1 rank)
    assert metrics["consumed_chunk_num"] == 2  # real samples seen by add()
    assert metrics["trace/backbone/tokens_per_second(M)"] == 60 / 2.0 / 1e6
    assert metrics["trace/vision/tokens_per_second(M)"] == 16 / 2.0 / 1e6
    assert metrics["trace/backbone/avg_seq_len"] == 30.0  # 60 tokens / global_batch_size 2
    assert "tokens_per_second(M)" not in metrics  # no merged cross-module token total


def test_consumed_tokens_accumulate_per_module_across_steps(single_process_meter):
    meter = single_process_meter

    for _ in range(2):
        meter.add({"conversation_list": [object()]})
        meter.step(delta_time=1.0, global_step=1, module_metrics={"backbone": (1.0, [100])})

    assert meter.consume_tokens == {"backbone": 200}
    assert meter.consume_chunks == 2


def test_step_resets_the_per_step_sample_accumulator(single_process_meter):
    meter = single_process_meter
    meter.add({"conversation_list": [object(), object()]})
    meter.step(delta_time=1.0, global_step=1, module_metrics={"backbone": (1.0, [4])})

    metrics = meter.step(delta_time=1.0, global_step=2, module_metrics={"backbone": (1.0, [4])})
    assert metrics["consumed_chunk_num"] == 2  # unchanged: no samples added this step


def test_multisource_source_is_the_aligned_module_with_the_most_tokens():
    """Several modules can be per-sample aligned; the backbone's tokens are the superset."""
    tokens = {"text_encoder": 40.0, "backbone": 200.0, "vision": 500.0}
    aligned = {"text_encoder": 2.0, "backbone": 2.0, "vision": 1.0}  # vision aligned on one rank only
    assert OmniEnvironMeter._multisource_module(tokens, aligned, num_ranks=2) == "backbone"


def test_multisource_source_is_absent_unless_some_module_aligns_on_every_rank():
    assert OmniEnvironMeter._multisource_module({"vision": 16.0}, {"vision": 1.0}, num_ranks=2) is None
    assert OmniEnvironMeter._multisource_module({}, {}, num_ranks=2) is None


def test_multisource_source_ties_resolve_by_name():
    assert OmniEnvironMeter._multisource_module({"b": 4.0, "a": 4.0}, {"b": 1.0, "a": 1.0}, num_ranks=1) == "a"


def test_step_omits_flops_when_no_module_is_metered(single_process_meter):
    """Unmeasured FLOPs must not be logged as an MFU of zero."""
    metrics = single_process_meter.step(delta_time=1.0, global_step=1, module_metrics={})
    assert "mfu" not in metrics and "flops_achieved(T)" not in metrics
    assert "consumed_chunk_num" in metrics


def test_multisource_ds_idx_accepts_each_collator_shape():
    """`SeedOmniCollator` hands back a per-sample list, unlike the packed tensor shape."""
    assert helper._get_multisource_ds_idx({"ds_idx": torch.tensor([2, 5])}) == [2, 5]
    assert helper._get_multisource_ds_idx({"ds_idx": [2, 5]}) == [2, 5]
    assert helper._get_multisource_ds_idx({"ds_idx": 7}) == [7]


def test_device_memory_metrics_are_shared_by_both_meters(monkeypatch):
    """The keys `EnvironMeter` used to build inline now come from one helper."""
    fake_device = SimpleNamespace(
        max_memory_allocated=lambda: 2 * 1024**3,
        max_memory_reserved=lambda: 4 * 1024**3,
        memory_stats=lambda: {"num_alloc_retries": 3},
    )
    monkeypatch.setattr(helper, "get_torch_device", lambda: fake_device)
    monkeypatch.setattr(helper, "all_reduce", lambda values, **kwargs: values)
    monkeypatch.setattr(helper.psutil, "virtual_memory", lambda: SimpleNamespace(used=0, available=0, percent=0))

    metrics = helper.compute_device_memory_metrics()
    assert metrics["max_memory_allocated(GB)"] == 2.0
    assert metrics["max_memory_reserved(GB)"] == 4.0
    assert metrics["num_alloc_retries"] == 3


def test_host_memory_metrics_report_the_worst_rank(monkeypatch):
    """Rank 0 logs for every host: most used, highest usage, least available."""
    fake_device = SimpleNamespace(
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0, memory_stats=lambda: {"num_alloc_retries": 0}
    )
    other_rank = (0, 0, 0, 6 * 1024**3, 75.0, -2 * 1024**3)  # the busier host

    def _max_with_other_rank(values, op):
        assert op == "max"
        return [max(mine, theirs) for mine, theirs in zip(values, other_rank)]

    monkeypatch.setattr(helper, "get_torch_device", lambda: fake_device)
    monkeypatch.setattr(helper, "all_reduce", _max_with_other_rank)
    monkeypatch.setattr(
        helper.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(used=1 * 1024**3, available=7 * 1024**3, percent=12.5),
    )

    metrics = helper.compute_device_memory_metrics()
    assert metrics["cpu_used_memory(GB)"] == 6.0
    assert metrics["cpu_memory_usage(%)"] == 75.0
    assert metrics["cpu_available_memory(GB)"] == 2.0


def test_multisource_tracker_steps_even_when_no_module_aligns(single_process_meter, monkeypatch):
    """The tracker step is a DP collective: a rank must not skip it on local data."""
    calls = []

    class _Tracker:
        def __init__(self, **kwargs):
            pass

        def step(self, ds_idx, seqlens):
            calls.append((list(ds_idx), list(seqlens)))
            return {}

    monkeypatch.setattr(omni_helper, "MultiSourceInfoTracker", _Tracker)
    meter = OmniEnvironMeter(global_batch_size=2, enable_multisource=True, dataloader=object(), empty_cache_steps=0)
    meter.add({"conversation_list": [[], []], "ds_idx": [0, 1]})

    meter.step(delta_time=1.0, global_step=1, module_metrics={"vision": (1.0, [16])})

    assert calls == [([0, 1], [0, 0])]


def _two_rank_step_main(rank: int, rendezvous: str, out_dir: str) -> None:
    """Rank 1 lists its modules in another order, and only rank 0's vision happens to align."""
    import veomni.utils.dist_utils as dist_utils

    dist_utils.get_device_type = lambda: "cpu"
    omni_helper.get_device_flops = lambda: 100.0
    omni_helper.compute_device_memory_metrics = dict
    tracker_calls = []

    class _Tracker:
        def __init__(self, **kwargs):
            pass

        def step(self, ds_idx, seqlens):
            tracker_calls.append(list(seqlens))
            return {}

    omni_helper.MultiSourceInfoTracker = _Tracker
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        meter = OmniEnvironMeter(
            global_batch_size=4,
            enable_multisource=True,
            dataloader=object(),
            empty_cache_steps=0,
            parallel_state=SimpleNamespace(dp_group=None),
        )
        meter.add({"conversation_list": [[], []], "ds_idx": [rank, rank]})
        if rank == 0:
            module_metrics = {"backbone": (10.0, [30, 50]), "vision": (1.0, [8, 8])}
        else:
            module_metrics = {"vision": (2.0, [4]), "backbone": (20.0, [70, 90])}
        metrics = meter.step(delta_time=1.0, global_step=1, module_metrics=module_metrics)
        result = {
            "backbone_tokens": metrics["trace/backbone/consume_tokens(M)"] * 1e6,
            "vision_tokens": metrics["trace/vision/consume_tokens(M)"] * 1e6,
            "flops": metrics["flops_achieved(T)"],
            "tracker_seqlens": tracker_calls,
        }
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_step_reduces_by_module_name_and_agrees_on_the_multisource_module(tmp_path):
    mp.spawn(_two_rank_step_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)

    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    for rank, result in enumerate(results):
        assert result["backbone_tokens"] == pytest.approx(240.0), rank
        assert result["vision_tokens"] == pytest.approx(20.0), rank
        assert result["flops"] == pytest.approx(33.0), rank
    assert [result["tracker_seqlens"] for result in results] == [[[30, 50]], [[70, 90]]]
