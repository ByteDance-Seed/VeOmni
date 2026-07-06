# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""VeOmni data adapter over DeepSpec's on-disk *target cache*.

DeepSpec stores per-token target hidden states in a custom binary cache and
reads it with ``deepspec.data.CacheDataset`` (returns CPU tensors per sample)
and ``deepspec.data.CacheCollator`` (pads a list of samples into a batch).

VeOmni's ``build_dataloader(dyn_bsz=False)`` path expects:

* a ``torch.utils.data.Dataset`` whose ``__getitem__`` returns the *transformed*
  sample. VeOmni's own ``MappingDataset`` wraps raw data as ``[sample]`` (a
  1-element list) and ``MakeMicroBatchCollator`` later does ``features[i][0]``
  to invert that. We mirror that contract exactly by returning ``[sample]``.
* a per-micro-batch collate function. We reuse DeepSpec's ``CacheCollator``
  verbatim (it pads ``input_ids`` / ``loss_mask`` / ``target_hidden_states`` /
  ``target_last_hidden_states`` and builds ``attention_mask``).

The dataloader is then responsible for gradient accumulation:
``MakeMicroBatchCollator`` splits each ``dataloader_batch_size`` block into
``num_micro_batch`` micro batches, so the trainer receives ``list[dict]``.
"""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from ...integrations.deepspec import ensure_deepspec_importable


class TargetCacheMappingDataset(Dataset):
    """Wrap ``deepspec.data.CacheDataset`` for VeOmni's mapping dataloader.

    ``__getitem__`` returns ``[sample_dict]`` to match the 1-to-N contract that
    ``MakeMicroBatchCollator`` inverts (``features[i][0]``).
    """

    def __init__(self, cache_dir: str, max_open_shards: int = 4):
        ensure_deepspec_importable()
        from deepspec.data import CacheDataset

        self._dataset = CacheDataset(cache_dir=cache_dir, max_open_shards=max_open_shards)

    # --- pass-through metadata used by the trainer's cache validation --- #
    @property
    def manifest(self) -> Dict[str, Any]:
        return self._dataset.manifest

    @property
    def hidden_size(self) -> int:
        return self._dataset.hidden_size

    @property
    def target_layer_ids(self) -> List[int]:
        return self._dataset.target_layer_ids

    @property
    def num_target_layers(self) -> int:
        return self._dataset.num_target_layers

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> List[Dict[str, Any]]:
        return [self._dataset[index]]


def build_target_cache_dataset(cache_dir: str, max_open_shards: int = 4) -> TargetCacheMappingDataset:
    return TargetCacheMappingDataset(cache_dir=cache_dir, max_open_shards=max_open_shards)


def build_cache_collator():
    """Return DeepSpec's ``CacheCollator`` (used as the per-micro-batch collate)."""
    ensure_deepspec_importable()
    from deepspec.data import CacheCollator

    return CacheCollator()


__all__ = [
    "TargetCacheMappingDataset",
    "build_target_cache_dataset",
    "build_cache_collator",
]
