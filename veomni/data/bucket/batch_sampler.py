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

"""Bucket-aware batch sampler for ``same_bucket_batching=True`` training.

Yields ``dataloader_batch_size`` indices per iteration, structured as
``num_micro_batch`` contiguous ``micro_batch_size`` segments that each land in
one bucket. Cross-rank sync uses the pure-function schedule callback -- every
DP rank at the same ``(global_step, micro_step)`` picks the same ``bucket_id``
without a collective. Per-rank per-bucket queues (round-robin sharded from the
indexer's ``bucket_ids``) are shuffled with a seeded permutation per epoch, so
DCP resume via ``state_dict()`` / ``load_state_dict()`` replays the sequence
exactly.

This is a main-process ``BatchSampler`` whose state lives in the main process;
workers just load the indices the sampler hands them, so
``dataloader.num_workers > 0`` is fine.
"""

from __future__ import annotations

from typing import Callable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch.utils.data as torch_data

from ...utils import helper


logger = helper.create_logger(__name__)


class BucketBatchSampler(torch_data.Sampler):
    """Same-bucket batch sampler with deterministic cross-rank agreement.

    Args:
        bucket_ids: ``np.ndarray[int32]`` from
            :meth:`~veomni.data.bucket.indexer.BucketIndexer.index`.
        all_bucket_ids: Every bucket id the resolution policy exposes. Drives
            the per-bucket queue construction and the global drop check.
        dp_rank / dp_size: Data-parallel rank identity. The sampler shards the
            per-bucket indices as ``bucket_indices[dp_rank::dp_size]``.
        micro_batch_size: Samples per micro-batch (each same-bucket).
        num_micro_batch: Micro-batches per optimizer step; each iteration of
            ``__iter__`` yields ``num_micro_batch * micro_batch_size`` indices.
        seed: Base seed for the per-(rank, bucket, epoch) shuffle.
        schedule_fn: ``(global_step, micro_step) -> bucket_id``. Passing
            :meth:`~veomni.data.bucket.scheduler.BucketScheduler.select_bucket_id`
            reuses the pure-function schedule so cross-rank agreement holds
            without a collective.
        bucket_labels: Optional ``bucket_id -> human label`` for log messages
            (e.g. ``"1024x768 base=1024 ratio_index=3"``).
        insufficient_bucket_policy: Handling when a bucket cannot serve ``mbs``
            samples on **any** DP rank.

            * ``"drop_insufficient_bucket"`` (default): silently drop such
              buckets from both the sampler queues and (via the trainer's
              wiring) the scheduler, so bucket-id whitelisting is not
              required for datasets whose actual aspect-ratio distribution
              only covers a subset of the policy's table. Each dropped bucket
              is announced with a WARNING that includes per-rank sample counts
              + lost-sample total, so operators cannot miss a large silent
              skip. The drop decision is GLOBAL: iterate every rank's shard
              locally (``bucket_ids`` is the full dataset array on every
              rank, ``dp_size`` is known, so no collective needed) and drop
              the bucket if **any** rank falls below ``mbs`` -- otherwise the
              synchronized scheduler would still pick it and the low-count
              rank(s) would starve.
            * ``"deterministic_replacement"`` is reserved for a future
              extension (raises ``NotImplementedError`` at ``__init__``).
    """

    def __init__(
        self,
        *,
        bucket_ids: np.ndarray,
        all_bucket_ids: Sequence[int],
        dp_rank: int,
        dp_size: int,
        micro_batch_size: int,
        num_micro_batch: int,
        seed: int,
        schedule_fn: Callable[[int, int], int],
        bucket_labels: Mapping[int, str] | None = None,
        insufficient_bucket_policy: Literal[
            "drop_insufficient_bucket", "deterministic_replacement"
        ] = "drop_insufficient_bucket",
    ) -> None:
        # Validate scalars first so error messages point at the config problem
        # rather than a downstream ``np.ndarray`` conversion.
        for name, value in (
            ("dp_rank", dp_rank),
            ("dp_size", dp_size),
            ("micro_batch_size", micro_batch_size),
            ("num_micro_batch", num_micro_batch),
            ("seed", seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
        if dp_size <= 0:
            raise ValueError("dp_size must be positive.")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"dp_rank {dp_rank} out of range [0, {dp_size}).")
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if num_micro_batch <= 0:
            raise ValueError("num_micro_batch must be positive.")
        if not callable(schedule_fn):
            raise TypeError("schedule_fn must be a callable (global_step, micro_step) -> bucket_id.")
        if insufficient_bucket_policy != "drop_insufficient_bucket":
            raise NotImplementedError(
                f"insufficient_bucket_policy={insufficient_bucket_policy!r} is not implemented yet; "
                "use 'drop_insufficient_bucket' (default)."
            )

        self._mbs = int(micro_batch_size)
        self._num_micro_batch = int(num_micro_batch)
        self._seed = int(seed)
        self._dp_rank = int(dp_rank)
        self._dp_size = int(dp_size)
        self._schedule = schedule_fn
        self._labels: dict[int, str] = {int(k): str(v) for k, v in (bucket_labels or {}).items()}

        bucket_ids = np.asarray(bucket_ids, dtype=np.int64)
        if bucket_ids.ndim != 1:
            raise ValueError(f"bucket_ids must be a 1-D array, got shape {bucket_ids.shape}.")
        policy_bucket_ids = [int(bucket_id) for bucket_id in all_bucket_ids]
        if not policy_bucket_ids:
            raise ValueError("all_bucket_ids must be a non-empty sequence.")

        # Per-DP-rank shard of the full dataset index.
        all_indices = np.arange(bucket_ids.shape[0], dtype=np.int64)
        rank_indices = all_indices[dp_rank::dp_size]
        rank_bucket_ids = bucket_ids[rank_indices]

        self._per_bucket_indices: dict[int, np.ndarray] = {}
        dropped_bucket_ids: list[int] = []
        total_dropped_samples = 0
        # GLOBAL drop decision -- every rank computes the same drop set locally
        # because it iterates every shard, not just its own. Without this
        # cross-rank pass the sampler would desync: e.g. mbs=2, bucket X has 5
        # samples all landing on rank 0's shard; rank 0 keeps X, rank 1 drops
        # X -> scheduler.restrict_to receives a different active set on each
        # rank -> the synchronized picker crashes mid-training. bucket_ids is
        # the full dataset array on every rank, so this loop is deterministic
        # and requires no collective.
        for bucket_id in policy_bucket_ids:
            per_rank_counts: list[int] = []
            for r in range(dp_size):
                r_indices = all_indices[r::dp_size]
                per_rank_counts.append(int((bucket_ids[r_indices] == bucket_id).sum()))
            min_count = min(per_rank_counts)
            if min_count < self._mbs:
                lost = sum(per_rank_counts)
                dropped_bucket_ids.append(int(bucket_id))
                total_dropped_samples += lost
                # WARNING per bucket so a large silent drop cannot slip past
                # the operator. Rank 0 alone logs so the message doesn't
                # multiply by world_size.
                label = self._labels.get(int(bucket_id), "")
                label_suffix = f" ({label})" if label else ""
                logger.warning_rank0(
                    f"[BucketBatchSampler] DROP bucket {bucket_id}{label_suffix}: "
                    f"per-rank sample counts across dp_size={dp_size} ranks = {per_rank_counts} "
                    f"(min={min_count} < mbs={self._mbs}); {lost} sample(s) in this bucket will "
                    "not be used."
                )
                continue
            # Bucket survives -- take this rank's shard for training.
            rank_mask = rank_bucket_ids == bucket_id
            self._per_bucket_indices[int(bucket_id)] = rank_indices[rank_mask]

        if not self._per_bucket_indices:
            raise ValueError(
                f"drop_insufficient_bucket dropped every configured bucket ({len(dropped_bucket_ids)} "
                f"buckets, {total_dropped_samples} samples total) -- no bucket has >= mbs={self._mbs} "
                "samples on every rank. Check the resolution policy matches the dataset's actual "
                "resolutions, or reduce mbs / grow the dataset."
            )
        if dropped_bucket_ids:
            logger.warning_rank0(
                f"[BucketBatchSampler] drop_insufficient_bucket summary: dropped "
                f"{len(dropped_bucket_ids)}/{len(policy_bucket_ids)} buckets "
                f"({total_dropped_samples} samples unusable, "
                f"{100.0 * total_dropped_samples / max(int(bucket_ids.shape[0]), 1):.1f}% of dataset); "
                f"active {len(self._per_bucket_indices)} bucket(s)."
            )
        # Expose the drop set so the caller can restrict the scheduler to match;
        # without that, the scheduler would still weighted-pick a dropped
        # bucket_id and ``_resolve_bucket`` would raise mid-run.
        self._dropped_bucket_ids = frozenset(dropped_bucket_ids)

        # Runtime state -- the only mutable pieces. All keyed by bucket_id.
        self._global_step: int = 0
        self._epochs: dict[int, int] = dict.fromkeys(self._per_bucket_indices, 0)
        self._cursors: dict[int, int] = dict.fromkeys(self._per_bucket_indices, 0)
        self._perms: dict[int, np.ndarray] = {bid: self._make_perm(bid) for bid in self._per_bucket_indices}

        # Also record the sampled bucket_ids seen so ``load_state_dict`` can
        # bail out early if a checkpoint's bucket set drifts from the config.
        # This is the ACTIVE set (post-drop), so resume detects both policy
        # changes and dataset changes that alter the drop set -- either
        # invalidates the cursor sequence.
        self._configured_bucket_ids = frozenset(self._per_bucket_indices)

    # ---- iteration ----

    def __iter__(self) -> Iterator[list[int]]:
        """Yield a ``dataloader_batch_size``-long list per optimizer step, indefinitely.

        Structure of each yielded batch: ``num_micro_batch`` contiguous same-
        bucket segments of ``mbs`` indices each. ``MakeMicroBatchCollator``
        slices this along that boundary (``features[i:i+mbs]``) so each
        micro-batch is guaranteed same-bucket by construction.
        """
        while True:
            batch: list[int] = []
            for micro_step in range(self._num_micro_batch):
                bucket_id = self._resolve_bucket(self._global_step, micro_step)
                perm = self._perms[bucket_id]
                cursor = self._cursors[bucket_id]
                if cursor + self._mbs > perm.size:
                    # Epoch boundary -- reshuffle deterministically from
                    # (seed, dp_rank, bucket_id, epoch+1).
                    self._epochs[bucket_id] += 1
                    perm = self._make_perm(bucket_id)
                    self._perms[bucket_id] = perm
                    cursor = 0
                batch.extend(int(x) for x in perm[cursor : cursor + self._mbs])
                self._cursors[bucket_id] = cursor + self._mbs
            self._global_step += 1
            yield batch

    def __len__(self) -> int:
        # Torch's ``BatchSampler`` supports ``__len__`` -- but this sampler yields
        # indefinitely (training is bounded by ``train_steps`` at the trainer,
        # not by dataset exhaustion). Returning a per-bucket-cycle length would
        # be misleading since ``schedule_fn`` picks buckets in weighted order.
        # torch DataLoader tolerates a missing ``__len__``: ``for batch in
        # dataloader`` runs until StopIteration, and we never raise it.
        raise TypeError(
            "BucketBatchSampler has no fixed length -- the trainer controls "
            "step count via train_steps. Do not call len() on it."
        )

    # ---- resume ----

    def state_dict(self) -> dict:
        """Serialize the exact state needed to resume ``__iter__`` at the next optimizer step.

        Cursors / epoch counters / global_step are per-bucket. Together with
        the immutable identity (``seed``, ``dp_rank``, and the
        ``BucketIndexer`` fingerprint recorded elsewhere in the checkpoint
        manifest), the next run reproduces the exact same index sequence.
        """
        return {
            "version": 1,
            "seed": self._seed,
            "dp_rank": self._dp_rank,
            "dp_size": self._dp_size,
            "micro_batch_size": self._mbs,
            "num_micro_batch": self._num_micro_batch,
            "global_step": self._global_step,
            "epochs": {int(k): int(v) for k, v in self._epochs.items()},
            "cursors": {int(k): int(v) for k, v in self._cursors.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore ``global_step`` / per-bucket cursors + epochs and reshuffle perms."""
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError(f"Unsupported BucketBatchSampler state_dict version: {state.get('version')!r}.")
        expected_identity = {
            "seed": self._seed,
            "dp_rank": self._dp_rank,
            "dp_size": self._dp_size,
            "micro_batch_size": self._mbs,
            "num_micro_batch": self._num_micro_batch,
        }
        for key, expected in expected_identity.items():
            actual = state.get(key)
            if actual != expected:
                raise ValueError(
                    f"BucketBatchSampler state mismatch on {key}: state has {actual!r}, sampler "
                    f"was built with {expected!r}. Resume requires identical identity -- check YAML "
                    "config drift."
                )
        state_bucket_ids = set(map(int, state.get("cursors", {}).keys()))
        if state_bucket_ids != self._configured_bucket_ids:
            raise ValueError(
                "BucketBatchSampler state bucket set does not match the active resolution policy. "
                f"State: {sorted(state_bucket_ids)}, config: {sorted(self._configured_bucket_ids)}. "
                "Resume aborted -- likely a policy or bucket-selection change."
            )

        self._global_step = int(state["global_step"])
        self._epochs = {int(k): int(v) for k, v in state["epochs"].items()}
        self._cursors = {int(k): int(v) for k, v in state["cursors"].items()}
        # Rebuild perms so subsequent draws come from the restored (seed, rank,
        # bucket, epoch) permutation -- matches what a fresh sampler at this
        # state would produce.
        for bucket_id in self._per_bucket_indices:
            self._perms[bucket_id] = self._make_perm(bucket_id)

    # ---- introspection (tests + logging) ----

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def dataloader_batch_size(self) -> int:
        return self._mbs * self._num_micro_batch

    def bucket_size(self, bucket_id: int) -> int:
        """Rank-local sample count in ``bucket_id`` (post-shard, pre-permutation)."""
        arr = self._per_bucket_indices.get(int(bucket_id))
        return 0 if arr is None else int(arr.size)

    @property
    def active_bucket_ids(self) -> frozenset[int]:
        """The subset of policy buckets that survived the build-time drop pass.

        Under the default ``drop_insufficient_bucket`` policy this excludes
        any bucket that fell below ``mbs`` on any DP rank. Callers pass this
        to :meth:`BucketScheduler.restrict_to` so the synchronized picker
        never chooses a bucket the sampler has no queue for.
        """
        return frozenset(self._per_bucket_indices)

    @property
    def dropped_bucket_ids(self) -> frozenset[int]:
        """The buckets skipped at build-time (may be empty if the dataset covers every bucket)."""
        return self._dropped_bucket_ids

    # ---- internals ----

    def _resolve_bucket(self, global_step: int, micro_step: int) -> int:
        bucket_id = int(self._schedule(global_step, micro_step))
        if bucket_id not in self._per_bucket_indices:
            raise KeyError(
                f"schedule_fn returned bucket_id={bucket_id} which has no samples on this rank; "
                f"active buckets: {sorted(self._per_bucket_indices)}. Likely a policy / scheduler "
                "misconfiguration."
            )
        return bucket_id

    def _make_perm(self, bucket_id: int) -> np.ndarray:
        """Deterministic per-(seed, dp_rank, bucket_id, epoch) shuffle."""
        # ``np.random.default_rng`` accepts a sequence seed; mix all four
        # identity fields so different ranks / buckets / epochs never collide.
        rng = np.random.default_rng((self._seed, self._dp_rank, int(bucket_id), int(self._epochs[bucket_id])))
        arr = self._per_bucket_indices[int(bucket_id)].copy()
        rng.shuffle(arr)
        return arr


__all__ = ["BucketBatchSampler"]
