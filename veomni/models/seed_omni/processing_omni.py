"""OmniProcessor — composed request preprocessing for :class:`OmniModel`.

Mirrors HuggingFace ``AutoProcessor``: collect each module's CPU preprocessor in
``config.module_names`` order, run the preprocessor chain over a batch dict, and
return a generate-ready request.

Every module's :class:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase`
(defined in its own ``processing.py`` and pointed at by ``preprocessor_class`` on
its native model class) builds straight off its checkpoint subfolder via
:meth:`~veomni.models.seed_omni.processing.base.ModulePreprocessorBase.from_pretrained` —
no model instance (weight-free, meta-device, or otherwise) is built or required.
:meth:`OmniProcessor.from_config` reads each module's ``preprocessor_class`` off
the class registered for its ``model_type`` and is the single code path backing
both :meth:`OmniProcessor.from_pretrained` (checkpoint on disk) and callers that
already hold a resolved config in memory.

Usage::

    processor = OmniProcessor.from_pretrained(checkpoint_root)
    model = OmniModel.from_pretrained(checkpoint_root, device_map="auto")
    processor.preprocess_batch(batch, inference=True)
    model.reset()
    generated = model.generate(batch, generation_kwargs={"max_new_tokens": 128})

Or, when the model is built first, reuse ``model.config`` instead of re-reading
``checkpoint_root`` a second time::

    model = OmniModel.from_pretrained(checkpoint_root, config=my_resolved_config, device_map="auto")
    processor = OmniProcessor.from_config(model.config, checkpoint_root=checkpoint_root)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ...utils import logging
from .configuration_omni import OmniConfig
from .modules import OMNI_MODEL_REGISTRY, read_model_type
from .processing import ModulePreprocessorBase


logger = logging.get_logger(__name__)


class OmniProcessor:
    """Composed SeedOmni request preprocessor (HF ``AutoProcessor``-style API)."""

    def __init__(self, preprocessors: dict[str, ModulePreprocessorBase]) -> None:
        self._preprocessors: dict[str, ModulePreprocessorBase] = dict(preprocessors)

    def __len__(self) -> int:
        """Number of worker-side preprocessors contributed by active graph modules."""
        return len(self._preprocessors)

    def bind_dummy_inputs(self, module_configs: Mapping[str, Any], *, dtype: Any = None) -> None:
        """Training-only: attach each preprocessor's dummy tensor(s).

        Computed from each module's own already-resolved config + ``dtype``
        alone (see :meth:`ModulePreprocessorBase.bind_dummy_inputs`) — no disk re-read
        here: ``module_configs`` is ``{module_name: config}`` taken straight
        from the already-built live modules. Call once, right after the training
        model finishes building. Unnecessary for pure inference: the dummy
        branch is never exercised there.
        """
        for name, preprocessor in self._preprocessors.items():
            config = module_configs.get(name)
            if config is not None:
                preprocessor.bind_dummy_inputs(config, dtype)

    @classmethod
    def from_config(
        cls,
        config: OmniConfig,
        *,
        checkpoint_root: str | os.PathLike | None = None,
    ) -> OmniProcessor:
        """Build preprocessors straight off an already-resolved :class:`OmniConfig`.

        No live/real module is built or required — see the module docstring. This is
        the single builder both :meth:`from_pretrained` and any caller holding an
        in-memory config should use.

        Collects CPU preprocessors in ``config.module_names`` order: for each
        module, reads its ``preprocessor_class`` off the class registered for its
        ``model_type`` and calls :meth:`ModulePreprocessorBase.from_pretrained` directly on
        its checkpoint subfolder. ``config.module_model_config(name)`` (the module's
        YAML ``model_config:`` block) is forwarded as ``config_overrides``.
        ``config.module_processor_config(name)`` (YAML ``processor_config:``) is
        splatted as kwargs, matching ``build_processor(path, **processor_config)``.
        """
        root = checkpoint_root if checkpoint_root is not None else getattr(config, "_name_or_path", None)
        root = None if root is None else str(root)
        preprocessors: dict[str, ModulePreprocessorBase] = {}
        for name in config.module_names:
            module_path = config.resolve_module_path(root, name)
            model_type = read_model_type(module_path)
            mod_cls = OMNI_MODEL_REGISTRY[model_type]()
            preprocessor_cls = getattr(mod_cls, "preprocessor_class", None)
            if preprocessor_cls is None:
                continue
            preprocessor = preprocessor_cls.from_pretrained(
                module_path,
                config_overrides=config.module_model_config(name),
                **config.module_processor_config(name),
            )
            if preprocessor is not None:
                preprocessors[name] = preprocessor
                logger.info_rank0(f"OmniProcessor: module '{name}' contributes {type(preprocessor).__name__}.")
        return cls(preprocessors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **config_kwargs: Any,
    ) -> OmniProcessor:
        """Load preprocessors from a split-checkpoint root (no module weights)."""
        config = OmniConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        root = getattr(config, "_name_or_path", None) or str(pretrained_model_name_or_path)
        return cls.from_config(config, checkpoint_root=root)

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        inference: bool = False,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the module preprocessor chain over ``batch`` and return it."""
        self.preprocess_batch(batch, inference=inference, **generation_kwargs)
        return batch

    def preprocess_batch(
        self,
        batch: dict[str, Any],
        *,
        inference: bool = False,
        **generation_kwargs: Any,
    ) -> None:
        """Run the module preprocessor chain over a collated batch.

        Training passes ``inference=False`` (default); inference passes
        ``inference=True``.
        """
        for preprocessor in self._preprocessors.values():
            preprocessor(
                batch,
                inference=inference,
                generation_kwargs=generation_kwargs or None,
            )


__all__ = [
    "OmniProcessor",
]
