"""Reference model contract and loading for parity-suite oracle execution."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from torch import nn
from transformers.initialization import no_init_weights

from veomni.models.module_utils import init_empty_weights


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


def load_transformers_reference_model(
    *,
    module: str | None,
    checkpoint: Path | None,
    **kwargs: Any,
) -> Any:
    """Load a reference model through Transformers ``AutoModel``."""

    if module is not None:
        _register_reference_model(module)
    model_id = checkpoint or module
    if model_id is None:
        raise ValueError("Transformers reference loader requires reference.module or reference.checkpoint.")
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, **kwargs)


@contextmanager
def empty_init_context() -> Iterator[None]:
    with no_init_weights(), init_empty_weights():
        yield


def load_safetensors_weights(
    model: nn.Module,
    weights_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_prefixes: tuple[str, ...] | None = None,
) -> None:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference weights not found: {path}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if include_prefixes is None or key.startswith(include_prefixes):
                state_dict[key] = handle.get_tensor(key).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if include_prefixes is not None:
        relevant_missing = [key for key in missing if key.startswith(include_prefixes)]
        relevant_unexpected = [key for key in unexpected if key.startswith(include_prefixes)]
    else:
        relevant_missing = list(missing)
        relevant_unexpected = list(unexpected)
    if relevant_missing:
        raise RuntimeError(f"Missing reference weight keys from {path}: {relevant_missing[:20]}")
    if relevant_unexpected:
        raise RuntimeError(f"Unexpected reference weight keys from {path}: {relevant_unexpected[:20]}")


def _register_reference_model(reference_module: str) -> None:
    if ":" not in reference_module:
        raise ValueError("reference.module must use 'module.path:ClassName' for Transformers reference registration.")
    module_name, class_name = reference_module.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError("reference.module must use 'module.path:ClassName' for Transformers reference registration.")

    module = importlib.import_module(module_name)
    reference_class = getattr(module, class_name)
    register_auto_model = getattr(reference_class, "register_auto_model", None)
    if register_auto_model is not None:
        register_auto_model()


__all__ = [
    "ParityReferenceModel",
    "empty_init_context",
    "load_safetensors_weights",
    "load_transformers_reference_model",
    "reference_options",
]
