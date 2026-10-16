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

"""Marker for SeedOmni modules whose online encoding can be split into a cached half and a per-step half."""

from __future__ import annotations


class OfflineEncodingMixin:
    """Marks a module as cacheable: an ``offline_training`` run builds it on meta.

    The module picks its own method names; the graph YAML of each task names
    them. The module guarantees, for every input ``x``::

        online_method(x) == offline_training_method(offline_embedding_method(x))

    * ``offline_embedding`` — the graph calls the offline-embedding method on
      the fully loaded module, and the trainer writes the ``conversation_list``
      the graph leaves in the batch to ``train.offline_cache_dir``. Edit the
      items in ``pre_forward`` / ``post_forward``, which run on the batch, or
      return ``conversation_list`` from the method: the method body receives
      its kwargs after the module wrap's FSDP2 input cast, which under mixed
      precision copies every ``ConversationItem``, so in-place edits there are
      lost.
    * ``offline_training`` — every module that mixes this in is built on meta
      and never loads its weights, so the offline-training method the graph
      calls on it must read only the config.
    * ``online_training`` — the graph calls the online method as usual.
    """


__all__ = ["OfflineEncodingMixin"]
