"""Reference model contract for parity-suite oracle execution."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from torch import nn


class ParityReferenceModel(nn.Module):
    """Mixin/base for oracle models that expose recipe-independent reference kinds."""

    config_class: type[Any] | None = None
    model_type: str | None = None

    @classmethod
    def register_auto_model(cls, exist_ok: bool = True) -> None:
        """Register this reference wrapper with Transformers auto classes."""

        config_class = cls.config_class
        if config_class is None:
            raise ValueError(f"{cls.__name__}.config_class must be set before auto registration.")
        model_type = cls.model_type or getattr(config_class, "model_type", None)
        if not model_type:
            raise ValueError(f"{cls.__name__} must define model_type or config_class.model_type.")

        from transformers import AutoConfig, AutoModel

        AutoConfig.register(model_type, config_class, exist_ok=exist_ok)
        AutoModel.register(config_class, cls, exist_ok=exist_ok)

    def run_reference_kind(self, kind: str | None, inputs: Mapping[str, Any], context: Any) -> Any:
        """Run one reference entrypoint requested by the parity driver."""

        if kind is None:
            return self(**inputs)
        method_name = f"run_reference_{kind}"
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(f"{type(self).__name__} does not implement reference kind {kind!r}.")
        return method(inputs, context)

    @contextmanager
    def reference_options(self, options: Mapping[str, Any] | None) -> Iterator[None]:
        """Temporarily apply reference config options while one oracle run executes."""

        with reference_options(self, options):
            yield


@contextmanager
def reference_options(model: Any, options: Mapping[str, Any] | None) -> Iterator[None]:
    """Temporarily set config attributes on ``model`` and its wrapped model."""

    if not options:
        with nullcontext():
            yield
        return
    if not isinstance(options, Mapping):
        raise TypeError("reference.options must be a mapping.")

    targets = _config_targets(model)
    if not targets:
        raise AttributeError(f"{type(model).__name__} has no config object for reference options.")

    originals: list[tuple[Any, str, Any]] = []
    try:
        for key, value in options.items():
            if not all(hasattr(target, key) for target in targets):
                raise AttributeError(f"Unknown reference option {key!r} for {type(model).__name__} config.")
            for target in targets:
                originals.append((target, key, getattr(target, key)))
                setattr(target, key, value)
        yield
    finally:
        for target, key, value in reversed(originals):
            setattr(target, key, value)


def _config_targets(model: Any) -> tuple[Any, ...]:
    targets: list[Any] = []
    for candidate in (getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)):
        if candidate is None or any(candidate is existing for existing in targets):
            continue
        targets.append(candidate)
    return tuple(targets)


__all__ = ["ParityReferenceModel", "reference_options"]
