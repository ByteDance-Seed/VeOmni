"""Per-module metric meter: what a module reports, and how the meter rolls it up.

A module only ever produces time-independent quantities (its own token lengths
and its own theoretical FLOPs); MFU / tokens-per-second are computed once by
:class:`~veomni.utils.omni_helper.OmniEnvironMeter` from the single whole-graph
wall-clock. The roll-up tests stub out the collective / device probes so the
math is checked in-process, with no distributed init.
"""

from __future__ import annotations

import pytest

from veomni.models.seed_omni import MetricMeterMixin
from veomni.utils import omni_helper
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


def test_multisource_seqlens_come_from_the_module_that_covers_the_whole_sample():
    """Several modules can be per-sample aligned; the backbone's tokens are the superset."""
    module_metrics = {
        "text_encoder": (1.0, [10, 10]),
        "backbone": (1.0, [40, 60]),
        "vision": (1.0, [16]),
    }
    assert OmniEnvironMeter._per_sample_seqlens(module_metrics, num_samples=2) == [40, 60]


def test_multisource_seqlens_are_absent_when_no_module_aligns_with_the_samples():
    assert OmniEnvironMeter._per_sample_seqlens({"vision": (1.0, [16])}, num_samples=2) is None
    assert OmniEnvironMeter._per_sample_seqlens({"backbone": (1.0, [4])}, num_samples=0) is None
