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

from .training_module_mixin import TrainingModuleMixin


_ALLOWED_CACHE_MODES = {
    "offline_encode": frozenset({"full", "encode_only"}),
    "online_process": frozenset({"full", "process_only"}),
}


class OfflineEncodingMixin(ABC):
    """Offline-cache capability for accelerated module mixins.

    ``config.support_cache`` declares that a module *can* run from an offline
    cache; it is a property of the checkpoint and is saved in ``config.json``.
    ``cache_mode`` selects what this run actually does and is a per-run choice,
    so it is a constructor kwarg kept on the instance, never on the config:

    * ``full`` — no cache; the module encodes online.
    * ``encode_only`` — only :meth:`offline_encode` runs, producing the cache.
    * ``process_only`` — only :meth:`online_process` runs, consuming the cache.

    :meth:`__init__` sets :attr:`cache_mode` before the native model body runs,
    so the body can skip building sub-networks the mode never uses.

    Modules that expose offline-cache graph endpoints must implement
    :meth:`offline_encode` and :meth:`online_process` on a sibling ``*OfflineMixin``
    placed **before** this mixin in ``VeOmniMixin`` bases so the concrete tensor
    call-sites win MRO lookup. This mixin itself must sit before
    :class:`TrainingModuleMixin`, whose ``pre_forward`` does not chain to
    ``super()`` and would otherwise skip the cache-mode gate.
    """

    DEFAULT_CACHE_MODE = "full"
    VALID_CACHE_MODES = frozenset({"full", "encode_only", "process_only"})

    config: Any
    cache_mode: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        if TrainingModuleMixin in mro and mro.index(TrainingModuleMixin) < mro.index(OfflineEncodingMixin):
            raise TypeError(
                f"{cls.__name__}: OfflineEncodingMixin must come before TrainingModuleMixin in the bases, "
                "otherwise its cache-mode gate in pre_forward never runs."
            )

    @classmethod
    def validate_cache_mode(cls, cache_mode: str, config: Any | None = None) -> str:
        """Check ``cache_mode`` is known and, when ``config`` is given, that it supports caching."""
        if cache_mode not in cls.VALID_CACHE_MODES:
            valid = ", ".join(sorted(cls.VALID_CACHE_MODES))
            raise ValueError(f"{cls.__name__}.cache_mode must be one of {{{valid}}}; got {cache_mode!r}.")
        if config is None or cache_mode == cls.DEFAULT_CACHE_MODE:
            return cache_mode
        if not getattr(config, "support_cache", False):
            raise ValueError(
                f"{cls.__name__}.cache_mode={cache_mode!r} requires {type(config).__name__}.support_cache=True."
            )
        return cache_mode

    def __init__(self, *args: Any, cache_mode: str = DEFAULT_CACHE_MODE, **kwargs: Any) -> None:
        config = kwargs.get("config")
        if config is None and args and hasattr(args[0], "model_type"):
            config = args[0]
        self.validate_cache_mode(cache_mode, config)
        self.cache_mode = cache_mode
        super().__init__(*args, **kwargs)
        if config is None and cache_mode != self.DEFAULT_CACHE_MODE:
            config = getattr(self, "config", None)
            if config is None:
                raise ValueError(
                    f"{type(self).__name__}: cache_mode={cache_mode!r} needs a config with support_cache."
                )
            self.validate_cache_mode(cache_mode, config)

    @abstractmethod
    def offline_encode(self, **kwargs: Any) -> dict[str, Any]:
        """Produce deterministic tensor cache artifacts from tensor inputs."""

    @abstractmethod
    def online_process(self, **kwargs: Any) -> dict[str, Any]:
        """Materialize runtime tensors from offline encoded cache tensors."""

    def pre_forward(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Gate offline-cache call-sites before dispatching module hooks."""
        allowed = _ALLOWED_CACHE_MODES.get(method)
        if allowed is not None and self.cache_mode not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(
                f"{type(self).__name__}.{method} requires cache_mode in {{{allowed_text}}}; "
                f"current cache_mode is {self.cache_mode!r}."
            )
        return super().pre_forward(method=method, **kwargs)


__all__ = ["OfflineEncodingMixin"]
