# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BucketIndexer + BucketBatchSampler + BucketScheduler as wired for HunyuanImage 3.

CPU-only. Locks the invariants the DCP-resumable, cross-rank-consistent bucket
stack depends on:
  - indexer fast-path (dataset columns) + slow-path (PIL fallback);
  - indexer fingerprint is deterministic and shifts on any policy / column
    change (protects resume);
  - each micro-batch is single-bucket (``same_bucket_batching`` core contract);
  - every DP rank picks the same bucket_id at the same ``(global_step,
    micro_step)`` via the pure ``BucketScheduler`` (no collective);
  - state_dict / load_state_dict reproduce subsequent batches (DCP resume);
  - DP ranks draw disjoint indices from the shared bucket;
  - global drop under ``drop_insufficient_bucket`` (default): every rank
    agrees, and a bucket that fits on some ranks but not others is dropped
    globally to avoid mid-run starvation;
  - scheduler weighted distribution matches configured weights;
  - ``restrict_to`` shrinks selection and shifts ``policy_hash`` (DCP manifest
    gate for drop-set changes).
"""

from __future__ import annotations

import numpy as np
import pytest

from veomni.models.transformers.hunyuan_image_3.resolution_policy import (
    ResolutionAnchorConfig,
    ResolutionPolicyConfig,
    build_hunyuan_image_3_bucket_batch_sampler,
    build_hunyuan_image_3_bucket_indexer,
    build_hunyuan_image_3_bucket_scheduler,
    build_resolution_policy,
)


# ============================================================
#  Fixtures
# ============================================================


def _make_policy_1024_only(ratio_indices: list[int] | None = None):
    """A resolution policy limited to the 1024 anchor."""
    anchors = [ResolutionAnchorConfig(base_size=1024, ratio_indices=ratio_indices or [])]
    return build_resolution_policy(ResolutionPolicyConfig(anchors=anchors))


class _FakeDataset:
    """Minimal map-style dataset with configurable columns.

    Emulates HF ``Dataset``'s ``column_names`` / ``__getitem__[str]`` API for
    the fast path, and per-index ``__getitem__[int]`` -> dict for the slow.
    """

    def __init__(self, rows: list[dict], *, expose_columns: bool = True):
        self._rows = rows
        self._expose_columns = expose_columns

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            if not self._expose_columns:
                raise KeyError(key)
            return [row[key] for row in self._rows]
        return self._rows[key]

    @property
    def column_names(self):
        if not self._expose_columns:
            return None
        cols: set[str] = set()
        for row in self._rows:
            cols.update(row.keys())
        return sorted(cols)


def _uniform_bucket_ids_for_policy(policy, n_per_bucket: int) -> np.ndarray:
    ids: list[int] = []
    for bucket in policy.buckets:
        ids.extend([bucket.bucket_id] * n_per_bucket)
    return np.asarray(ids, dtype=np.int32)


def _build_sampler(policy, **overrides):
    defaults = dict(dp_rank=0, dp_size=1, micro_batch_size=2, num_micro_batch=1, seed=0)
    defaults.update(overrides)
    return build_hunyuan_image_3_bucket_batch_sampler(policy, **defaults)


# ============================================================
#  BucketIndexer
# ============================================================


def test_bucket_indexer_fast_path_reads_columns_without_pil():
    policy = _make_policy_1024_only()
    # 1024 anchor at h/w = 1.0 -> the (1024, 1024) bucket; at 1.333 -> (1024, 768).
    rows = [
        {"width": 1024, "height": 1024},
        {"width": 768, "height": 1024},  # 1.333
        {"width": 1024, "height": 1024},
    ]
    dataset = _FakeDataset(rows, expose_columns=True)
    indexer = build_hunyuan_image_3_bucket_indexer(
        policy, default_base_size=1024, width_key="width", height_key="height"
    )

    bucket_ids = indexer.index(dataset)
    assert bucket_ids.dtype == np.int32
    assert bucket_ids.shape == (3,)
    assert bucket_ids[0] == bucket_ids[2]
    assert bucket_ids[0] != bucket_ids[1]


def test_bucket_indexer_slow_path_falls_back_to_pil():
    pil = pytest.importorskip("PIL.Image")
    from veomni.data.bucket.indexer import _reset_slow_path_warning_cache_for_tests

    _reset_slow_path_warning_cache_for_tests()

    policy = _make_policy_1024_only()
    rows = [
        {"image": pil.new("RGB", (1024, 1024)), "prompt": "a"},
        {"image": pil.new("RGB", (768, 1024)), "prompt": "b"},
    ]
    dataset = _FakeDataset(rows, expose_columns=False)
    indexer = build_hunyuan_image_3_bucket_indexer(
        policy, default_base_size=1024, width_key="width", height_key="height", image_key="image"
    )
    bucket_ids = indexer.index(dataset)
    assert bucket_ids.shape == (2,)
    assert bucket_ids[0] != bucket_ids[1]


def test_bucket_indexer_fingerprint_stable_and_sensitive():
    """Same inputs -> same fingerprint; policy or column key change -> different.

    Protects DCP resume: the sampler manifest keys off this fingerprint and
    rejects any resume where the sample-to-bucket mapping could have drifted.
    """
    rows = [{"width": 1024, "height": 1024}]
    dataset = _FakeDataset(rows)

    policy_1024 = _make_policy_1024_only()
    fp1 = build_hunyuan_image_3_bucket_indexer(policy_1024, default_base_size=1024).fingerprint(dataset)
    fp1_again = build_hunyuan_image_3_bucket_indexer(policy_1024, default_base_size=1024).fingerprint(dataset)
    assert fp1 == fp1_again

    policy_narrow = _make_policy_1024_only(ratio_indices=[0])
    fp_narrow = build_hunyuan_image_3_bucket_indexer(policy_narrow, default_base_size=1024).fingerprint(dataset)
    assert fp_narrow != fp1

    fp_renamed = build_hunyuan_image_3_bucket_indexer(policy_1024, default_base_size=1024, width_key="w").fingerprint(
        dataset
    )
    assert fp_renamed != fp1


# ============================================================
#  BucketBatchSampler
# ============================================================


def test_sampler_each_micro_batch_is_single_bucket():
    """Core contract: within one yielded batch, every ``mbs``-segment is single-bucket."""
    policy = _make_policy_1024_only(ratio_indices=[0, 1, 2])
    bucket_ids_arr = _uniform_bucket_ids_for_policy(policy, 32)
    scheduler = build_hunyuan_image_3_bucket_scheduler(policy)

    sampler = _build_sampler(
        policy,
        bucket_ids=bucket_ids_arr,
        micro_batch_size=4,
        num_micro_batch=3,
        seed=42,
        schedule_fn=scheduler.select_bucket_id,
    )

    it = iter(sampler)
    for _ in range(5):
        batch = next(it)
        assert len(batch) == 12  # mbs * num_micro_batch
        for m in range(3):
            seg = batch[m * 4 : (m + 1) * 4]
            seg_buckets = {bucket_ids_arr[i] for i in seg}
            assert len(seg_buckets) == 1, f"micro-batch {m} spans multiple buckets: {seg_buckets}"


def test_sampler_cross_rank_agreement_at_same_step():
    """Same ``(global_step, micro_step)`` across ranks -> same ``bucket_id``.

    The schedule is a pure function so no collective is needed. Each rank
    independently draws its own samples from that shared bucket.
    """
    policy = _make_policy_1024_only(ratio_indices=[0, 1])
    dp_size = 4
    bucket_ids_arr = _uniform_bucket_ids_for_policy(policy, 16 * dp_size)
    scheduler = build_hunyuan_image_3_bucket_scheduler(policy)

    samplers = [
        _build_sampler(
            policy,
            bucket_ids=bucket_ids_arr,
            dp_rank=r,
            dp_size=dp_size,
            num_micro_batch=2,
            seed=7,
            schedule_fn=scheduler.select_bucket_id,
        )
        for r in range(dp_size)
    ]
    iters = [iter(s) for s in samplers]

    for _ in range(6):
        batches = [next(it) for it in iters]
        for m in range(2):
            per_rank_buckets = []
            for rank_idx, batch in enumerate(batches):
                seg = batch[m * 2 : (m + 1) * 2]
                seg_buckets = {int(bucket_ids_arr[i]) for i in seg}
                assert len(seg_buckets) == 1, f"rank {rank_idx} micro {m} multi-bucket"
                per_rank_buckets.append(next(iter(seg_buckets)))
            assert len(set(per_rank_buckets)) == 1, f"cross-rank bucket mismatch at micro {m}: {per_rank_buckets}"


def test_sampler_state_dict_roundtrip_reproduces_subsequent_batches():
    """DCP resume gate: save + restore state -> identical index sequence going forward."""
    policy = _make_policy_1024_only(ratio_indices=[0, 1, 2])
    bucket_ids_arr = _uniform_bucket_ids_for_policy(policy, 24)
    scheduler = build_hunyuan_image_3_bucket_scheduler(policy)

    def _new_sampler():
        return _build_sampler(
            policy,
            bucket_ids=bucket_ids_arr,
            micro_batch_size=4,
            num_micro_batch=2,
            seed=123,
            schedule_fn=scheduler.select_bucket_id,
        )

    baseline = _new_sampler()
    it = iter(baseline)
    prefix = [next(it) for _ in range(3)]
    baseline_state = baseline.state_dict()
    baseline_tail = [next(it) for _ in range(5)]

    restored = _new_sampler()
    restored.load_state_dict(baseline_state)
    restored_tail = [next(iter(restored)) for _ in range(5)]

    assert restored_tail == baseline_tail, "restored sequence must match baseline tail"

    fresh = _new_sampler()
    fresh_prefix = [next(iter(fresh)) for _ in range(3)]
    assert fresh_prefix == prefix


def test_sampler_ranks_get_disjoint_indices():
    """Each DP rank's shard is disjoint (round-robin split of the full index)."""
    policy = _make_policy_1024_only(ratio_indices=[0])
    bucket_ids_arr = _uniform_bucket_ids_for_policy(policy, 16)
    only_bucket = policy.buckets[0].bucket_id

    per_rank_seen = []
    for r in range(2):
        sampler = _build_sampler(
            policy,
            bucket_ids=bucket_ids_arr,
            dp_rank=r,
            dp_size=2,
            seed=17,
            schedule_fn=lambda g, m: only_bucket,
        )
        seen: set[int] = set()
        it = iter(sampler)
        for _ in range(4):
            seen.update(next(it))
        per_rank_seen.append(seen)
    assert per_rank_seen[0].isdisjoint(per_rank_seen[1]), "DP ranks must draw disjoint sample indices"
    assert per_rank_seen[0] | per_rank_seen[1] == set(range(16))


def test_sampler_drop_decision_is_global_across_dp_ranks():
    """Regression: a bucket that fits on some ranks but starves others must be
    dropped GLOBALLY, otherwise the scheduler picking it desyncs the ranks.

    Setup: 16 samples split round-robin across dp_size=2 with a layout where
    bucket X lands entirely on rank 0 and Y entirely on rank 1. Naive per-rank
    check keeps both (each rank sees mbs=2 met in "its" bucket); the correct
    global check drops both.
    """
    policy = _make_policy_1024_only(ratio_indices=[0, 1])
    x, y = policy.buckets[0].bucket_id, policy.buckets[1].bucket_id
    bucket_ids_arr = np.array([x if i % 2 == 0 else y for i in range(16)], dtype=np.int32)

    for r in range(2):
        with pytest.raises(ValueError, match="dropped every configured bucket"):
            _build_sampler(
                policy,
                bucket_ids=bucket_ids_arr,
                dp_rank=r,
                dp_size=2,
                schedule_fn=lambda g, m: x,
            )


def test_sampler_default_policy_drops_undersized_bucket_silently():
    """Under the default ``drop_insufficient_bucket``, undersized buckets are
    silently dropped (no raise) and exposed via ``dropped_bucket_ids``."""
    policy = _make_policy_1024_only(ratio_indices=[0, 1])
    bucket_ids_arr = np.array([policy.buckets[0].bucket_id] * 8 + [policy.buckets[1].bucket_id] * 1, dtype=np.int32)
    sampler = _build_sampler(
        policy,
        bucket_ids=bucket_ids_arr,
        schedule_fn=lambda g, m: policy.buckets[0].bucket_id,
    )
    assert sampler.active_bucket_ids == frozenset({policy.buckets[0].bucket_id})
    assert sampler.dropped_bucket_ids == frozenset({policy.buckets[1].bucket_id})


# ============================================================
#  BucketScheduler (deterministic, weighted, restrict_to)
# ============================================================


def _scheduler(anchors=None, seed=0):
    config = ResolutionPolicyConfig(anchors=anchors or [], scheduler_seed=seed)
    return build_hunyuan_image_3_bucket_scheduler(build_resolution_policy(config))


def test_scheduler_weighted_distribution_tracks_configured_weights():
    """Anchor weights drive the per-step draw distribution (large N)."""
    anchors = [
        ResolutionAnchorConfig(base_size=256, ratio_indices=[0], weight=3.0),
        ResolutionAnchorConfig(base_size=512, ratio_indices=[0], weight=1.0),
    ]
    sched = _scheduler(anchors=anchors)
    ids = [sched.select_bucket_id(step, 0) for step in range(20000)]
    frac0 = sum(1 for i in ids if i == 0) / len(ids)
    assert 0.72 < frac0 < 0.78  # target 0.75


def test_scheduler_restrict_to_shrinks_selection_and_hash():
    """``restrict_to`` filters the weighted set and shifts ``policy_hash`` so
    the DCP manifest gate hard-fails on a resume across a drop-set change."""
    policy = _make_policy_1024_only(ratio_indices=[0, 1, 2])
    scheduler = build_hunyuan_image_3_bucket_scheduler(policy)
    original_hash = scheduler.policy_hash()
    original_num = scheduler.num_buckets
    keep = {policy.buckets[0].bucket_id}
    scheduler.restrict_to(keep)

    assert scheduler.num_buckets == 1 < original_num
    picks = {scheduler.select_bucket_id(g, m) for g in range(10) for m in range(3)}
    assert picks == keep
    assert scheduler.policy_hash() != original_hash
