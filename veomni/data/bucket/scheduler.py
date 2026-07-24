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

"""Deterministic per-micro-step bucket scheduler.

At the start of every optimizer micro-step a single ``bucket_id`` is chosen from
the active weighted bucket set and consumed by the whole training world: DP
replicas draw distinct samples from that bucket while SP/EP ranks consume the
same logical micro-batch. This module owns only the *selection* -- a pure,
deterministic function of ``(scheduler_seed, global_step, micro_step)`` -- so
every rank computes the same id independently (no collective) and DCP resume
reproduces the sequence exactly from the restored ``global_step``.

Model-specific weight derivation (HunyuanImage 3's per-anchor / per-ratio
weights, etc.) belongs upstream; this scheduler only sees ``bucket_id ->
weight``. Dataloader wiring (bucket-partitioned candidate queues) and the
``drop_insufficient_bucket`` availability check live in the batch sampler; this
scheduler then honours ``restrict_to(active_bucket_ids)`` so its picks stay
inside the sampler's actual queues.

The selection uses a frozen BLAKE2b construction under a dedicated ``bucket``
stream so it never collides with training RNG streams and carries
golden-vector-testable determinism.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping


_BUCKET_STREAM = "bucket"


def _bucket_uniform(scheduler_seed: int, global_step: int, micro_step: int) -> float:
    """A stable uniform in ``[0, 1)`` from the scheduler's step identity."""
    for value in (scheduler_seed, global_step, micro_step):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Bucket scheduler identity fields must be integers.")
    if global_step < 0 or micro_step < 0:
        raise ValueError("Bucket scheduler step fields must be non-negative.")
    payload = json.dumps(
        [scheduler_seed, global_step, micro_step, _BUCKET_STREAM],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # 53-bit mantissa keeps the ratio exactly representable as a float.
    return (int.from_bytes(digest, byteorder="big") >> 11) / float(1 << 53)


class BucketScheduler:
    """Deterministic weighted selection of one bucket per ``(global_step, micro_step)``.

    Args:
        weights: ``bucket_id -> non-negative weight``. Zero-weight buckets are
            legal but never selected; keep the caller free to pass a full
            table without filtering.
        scheduler_seed: Frozen seed mixed into the BLAKE2b input; changes here
            invalidate DCP resume (via ``policy_hash``).

    The scheduler is stateless -- ``select_bucket_id`` is a pure function of
    ``(seed, global_step, micro_step)`` -- so DP replicas agree without a
    collective and resume needs only ``global_step`` (restored by the trainer).
    """

    def __init__(self, *, weights: Mapping[int, float], scheduler_seed: int) -> None:
        if isinstance(scheduler_seed, bool) or not isinstance(scheduler_seed, int):
            raise TypeError("scheduler_seed must be an int.")
        if not isinstance(weights, Mapping):
            raise TypeError("weights must be a Mapping[bucket_id, weight].")
        if not weights:
            raise ValueError("weights must be non-empty.")

        normalized: dict[int, float] = {}
        for raw_id, raw_weight in weights.items():
            bucket_id = int(raw_id)
            weight = float(raw_weight)
            if weight <= 0.0:
                # Non-positive weights would silently disable a bucket; make it explicit.
                raise ValueError(f"Bucket {bucket_id} weight must be positive, got {weight}.")
            normalized[bucket_id] = weight

        self._scheduler_seed = int(scheduler_seed)
        self._weights: dict[int, float] = normalized
        self._total_weight: float = sum(normalized.values())

    def select_bucket_id(self, global_step: int, micro_step: int) -> int:
        """Return the scheduled ``bucket_id`` for this optimizer micro-step."""
        target = _bucket_uniform(self._scheduler_seed, global_step, micro_step) * self._total_weight
        cumulative = 0.0
        # Iterate in insertion order so callers see a stable pick given the same
        # ``weights`` mapping. ``dict`` preserves insertion order since 3.7.
        last_bucket_id = -1
        for bucket_id, weight in self._weights.items():
            cumulative += weight
            if target < cumulative:
                return bucket_id
            last_bucket_id = bucket_id
        return last_bucket_id  # float guard

    def restrict_to(self, allowed_bucket_ids: Iterable[int]) -> None:
        """Restrict future selections to ``allowed_bucket_ids`` (weights renormalize).

        Intended for use with the batch sampler's
        ``drop_insufficient_bucket`` policy: the sampler computes the empty-
        bucket set from its own ``bucket_ids`` array and calls this so the
        scheduler stops picking buckets the sampler has dropped. Mutates in
        place; ``policy_hash`` reflects the shrunk set on subsequent calls, so
        the DCP manifest gate naturally rejects a resume across a drop-set
        change.
        """
        allowed = frozenset(int(bucket_id) for bucket_id in allowed_bucket_ids)
        if not allowed:
            raise ValueError("restrict_to received an empty allowed set.")
        current = set(self._weights)
        unknown = allowed - current
        if unknown:
            raise ValueError(
                f"restrict_to received bucket_ids not in the scheduler: {sorted(unknown)}. Current: {sorted(current)}."
            )
        restricted = {bucket_id: self._weights[bucket_id] for bucket_id in self._weights if bucket_id in allowed}
        total = sum(restricted.values())
        if total <= 0.0:
            raise ValueError("restrict_to left total bucket weight non-positive.")
        self._weights = restricted
        self._total_weight = total

    @property
    def num_buckets(self) -> int:
        return len(self._weights)

    def policy_hash(self) -> str:
        """Stable hash of the active ``(seed, weights)`` set (for the DCP manifest).

        Bucket **geometry** drift (height/width/etc.) is captured separately by
        :meth:`BucketIndexer.fingerprint` via the caller-supplied
        ``policy_fingerprint``; this hash intentionally only covers what the
        scheduler itself owns.
        """
        table = [[int(bucket_id), round(float(weight), 6)] for bucket_id, weight in sorted(self._weights.items())]
        payload = json.dumps(
            {"seed": self._scheduler_seed, "mode": "synchronized_weighted", "table": table},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def state_dict(self) -> dict:
        """Stateless selection -> the manifest only needs identity, not counters."""
        return {
            "scheduler_seed": self._scheduler_seed,
            "selection_mode": "synchronized_weighted",
            "policy_hash": self.policy_hash(),
            "num_buckets": self.num_buckets,
        }


__all__ = ["BucketScheduler"]
