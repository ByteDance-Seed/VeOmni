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

"""Generic offline-encoding surfaces for SeedOmni V2 modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Self

import torch

from ....utils import logging


ENCODED_CACHE_KIND_META_KEY = "encoded_cache_kind"

logger = logging.get_logger(__name__)


class OfflineEncodedCache(ABC):
    """Typed DTO for module-local offline cache semantics.

    Conversation items still carry plain tensors in ``value``. Concrete cache
    views use this DTO only at module boundaries, converting with
    :meth:`to_tensor` before writing a cache item and :meth:`from_tensor` after
    reading one back.
    """

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """Pack this typed cache view into a tensor carrier value."""

    @classmethod
    @abstractmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """Reconstruct this typed cache view from a tensor carrier value."""


class OfflineEncodingMixin:
    """Offline encoding capability helper for tensor-level cache workflows.

    Concrete modules can use ``cache_mode`` to avoid loading components that a
    mode cannot call: ``encode_only`` needs offline encoding weights,
    ``process_only`` needs online processing weights, and ``full`` needs both.
    """

    DEFAULT_CACHE_MODE = "full"
    VALID_CACHE_MODES = frozenset({"full", "encode_only", "process_only"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = args[0] if args else kwargs.get("config", getattr(self, "config", None))
        if not hasattr(config, "cache_mode"):
            logger.warning_rank0(
                f"{type(self).__name__} config does not define `cache_mode`; falling back to 'full' and "
                "loading all weights for offline_encode."
            )

        super().__init__(*args, **kwargs)

    @property
    def cache_mode(self) -> str:
        config = getattr(self, "config", None)
        return getattr(config, "cache_mode", self.DEFAULT_CACHE_MODE)

    def validated_cache_mode(self) -> str:
        mode = self.cache_mode
        if mode not in self.VALID_CACHE_MODES:
            valid = ", ".join(sorted(self.VALID_CACHE_MODES))
            raise ValueError(f"{type(self).__name__}.cache_mode must be one of {{{valid}}}; got {mode!r}.")
        return mode

    def pre_forward(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Gate offline-cache call-sites before dispatching module hooks."""
        if method == "offline_encode" and not getattr(self, f"_{method}_checked", False):
            self._check_cache_mode(method=method, allowed={"full", "encode_only"})
            setattr(self, f"_{method}_checked", True)
        elif method == "online_process" and not getattr(self, f"_{method}_checked", False):
            self._check_cache_mode(method=method, allowed={"full", "process_only"})
            setattr(self, f"_{method}_checked", True)
        return super().pre_forward(method=method, **kwargs)

    def _check_cache_mode(self, *, method: str, allowed: Collection[str]) -> None:
        mode = self.validated_cache_mode()
        if mode not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(
                f"{type(self).__name__}.{method} requires cache_mode in {{{allowed_text}}}; "
                f"current cache_mode is {mode!r}."
            )

    def load_partial_dcp_checkpoint(self, load_dir: str, *, trainer: Any) -> None:
        """Load runtime DCP state for a non-``full`` offline-cache module.

        The default is a no-op for modules such as a VAE ``process_only`` stage
        that has no online runtime state. Modules with trainable online
        components can override this to restore a partial model/optimizer state.
        """
        del load_dir, trainer

    def save_partial_dcp_checkpoint(self, save_dir: str, *, trainer: Any, state: Any) -> None:
        """Save runtime DCP state for a non-``full`` offline-cache module.

        The default is a no-op for modules with no online runtime state.
        Modules with trainable online components can override this to persist a
        partial model/optimizer state.
        """
        del save_dir, trainer, state

    def save_full_hf_checkpoint(self, output_dir: str, *, source_path: str, trainer: Any, state: Any) -> None:
        """Save a full HuggingFace artifact for a non-``full`` offline-cache module.

        Concrete modules own the merge policy: a no-parameter process-only
        module may copy the frozen source split, while a partial online module
        may combine frozen source weights with trainable runtime weights.
        """
        del output_dir, source_path, trainer, state
        raise NotImplementedError(f"{type(self).__name__}.save_full_hf_checkpoint is not implemented.")

    @abstractmethod
    def offline_encode(self, **kwargs) -> Any:
        """Produce deterministic tensor cache artifacts from tensor inputs."""

    @abstractmethod
    def online_process(self, **kwargs) -> Any:
        """Materialize runtime tensors from offline encoded cache tensors."""


__all__ = [
    "ENCODED_CACHE_KIND_META_KEY",
    "OfflineEncodedCache",
    "OfflineEncodingMixin",
]
