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
from typing import Any


RUNTIME_CONFIG_KEYS = ("support_cache", "train_type")


class OfflineEncodingMixin(ABC):
    """Offline-cache capability for accelerated module mixins.

    ``support_cache`` / ``train_type`` are launcher/runtime fields — not part of
    the HF-native ``config.json``. :meth:`patch_config` applies them onto the
    live config object; :meth:`__init__` does this automatically before the
    native model body runs.

    Modules that expose offline-cache graph endpoints must implement
    :meth:`offline_encode` and :meth:`online_process` on a sibling ``*OfflineMixin``
    placed **before** this mixin in ``VeOmniMixin`` bases so the concrete tensor
    call-sites win MRO lookup.
    """

    DEFAULT_CACHE_MODE = "full"
    VALID_CACHE_MODES = frozenset({"full", "encode_only", "process_only"})

    config: Any

    @classmethod
    def derive_cache_mode(cls, *, support_cache: bool, train_type: str) -> str:
        if not support_cache:
            return cls.DEFAULT_CACHE_MODE
        if train_type == "offline_cache":
            return "encode_only"
        if train_type == "train_with_cache":
            return "process_only"
        return cls.DEFAULT_CACHE_MODE

    @classmethod
    def patch_config(cls, config: Any, **overrides: Any) -> None:
        """Apply launcher/runtime offline-cache fields onto a live config object."""
        support_cache = overrides.get("support_cache", getattr(config, "support_cache", False))
        train_type = overrides.get("train_type", getattr(config, "train_type", "train"))
        config.support_cache = bool(support_cache)
        config.train_type = str(train_type)

    @classmethod
    def validated_cache_mode(cls, config: Any) -> str:
        mode = cls.derive_cache_mode(
            support_cache=bool(getattr(config, "support_cache", False)),
            train_type=str(getattr(config, "train_type", "train")),
        )
        if mode not in cls.VALID_CACHE_MODES:
            valid = ", ".join(sorted(cls.VALID_CACHE_MODES))
            raise ValueError(f"{type(config).__name__}.cache_mode must be one of {{{valid}}}; got {mode!r}.")
        return mode

    @property
    def cache_mode(self) -> str:
        return self.validated_cache_mode(self.config)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        runtime_overrides = {key: kwargs.pop(key) for key in RUNTIME_CONFIG_KEYS if key in kwargs}
        config = kwargs.get("config")
        if config is None and args:
            candidate = args[0]
            if hasattr(candidate, "model_type"):
                config = candidate
        if config is not None:
            self.patch_config(config, **runtime_overrides)
        super().__init__(*args, **kwargs)
        if config is None and hasattr(self, "config"):
            self.patch_config(self.config, **runtime_overrides)

    @abstractmethod
    def offline_encode(self, **kwargs: Any) -> dict[str, Any]:
        """Produce deterministic tensor cache artifacts from tensor inputs."""

    @abstractmethod
    def online_process(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize runtime tensors from offline encoded cache tensors."""

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
        mode = self.cache_mode
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


__all__ = ["OfflineEncodingMixin", "RUNTIME_CONFIG_KEYS"]
