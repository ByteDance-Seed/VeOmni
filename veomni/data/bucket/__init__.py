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

"""Generic aspect-ratio / resolution bucketing primitives (data plane).

This package hosts the model-agnostic mechanics:

* :class:`BucketScheduler` -- deterministic weighted per-``(global_step,
  micro_step)`` bucket selection via a frozen BLAKE2b construction.
* :class:`BucketIndexer` -- startup dataset scan that emits
  ``bucket_ids[N]:int32`` and a stable fingerprint for DCP resume gating.
* :class:`BucketBatchSampler` -- ``torch.utils.data.Sampler`` that yields
  ``num_micro_batch x micro_batch_size`` indices, each ``mbs``-segment drawn
  from one bucket, with cross-rank agreement and DCP resume state.

Model-specific resolution tables (HunyuanImage 3's ``ResolutionGroup`` replica,
etc.) live in the per-model package and wire these primitives via small
factory helpers.
"""

from .batch_sampler import BucketBatchSampler
from .indexer import BucketIndexer, extract_image_size
from .scheduler import BucketScheduler


__all__ = [
    "BucketBatchSampler",
    "BucketIndexer",
    "BucketScheduler",
    "extract_image_size",
]
