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

"""Generic offline-encoding surfaces for SeedOmni modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class OfflineEncodingMixin(ABC):
    """The two graph endpoints of a module whose config has ``support_cache``.

    ``train.training_task`` picks which one a run uses, and the module's
    modules / graph YAML wire it in:

    * ``offline_embedding`` — the graph runs :meth:`offline_encode`, and the
      trainer writes the encoded conversations to ``train.offline_cache_dir``.
    * ``offline_training`` — the graph runs :meth:`online_process` on the
      cached conversations. The module is built on meta and never loads its
      weights, so :meth:`online_process` must read only the config.
    * ``online_training`` — neither; the module encodes through its normal endpoint.
    """

    @abstractmethod
    def offline_encode(self, **kwargs: Any) -> dict[str, Any]:
        """Produce deterministic tensor cache artifacts from tensor inputs."""

    @abstractmethod
    def online_process(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize runtime tensors from offline encoded cache tensors, without touching weights."""


__all__ = ["OfflineEncodingMixin"]
