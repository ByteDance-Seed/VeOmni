"""OmniProcessor — composed request preprocessing for :class:`OmniModel`.

Mirrors HuggingFace ``AutoProcessor``: collect each module's CPU preprocessor in
``config.module_names`` order, build a ``conversation_list`` from user inputs,
run the preprocessor chain, and return a generate-ready request dict.

Every module's :class:`~veomni.models.seed_omni.mixins.module_processor_mixin.Preprocessor`
(declared on its own ``processing.py`` as ``XxxModuleMixin.preprocessor_class``)
builds straight off its checkpoint subfolder via
:meth:`~veomni.models.seed_omni.mixins.module_processor_mixin.Preprocessor.from_pretrained` —
no model instance (weight-free, meta-device, or otherwise) is built or required.
:meth:`OmniProcessor.from_config` reads each module's ``preprocessor_class`` off
the class registered for its ``model_type`` and is the single code path backing
both :meth:`OmniProcessor.from_pretrained` (checkpoint on disk) and callers that
already hold a resolved config in memory (e.g. ``OmniTrainer`` builds its
dataloader's collator this way, decoupled from ``self.model``). Callers that
need a :class:`~veomni.data.data_collator.SeedOmniCollator` (e.g. ``OmniTrainer``)
build one directly from a processor — that composition is theirs to own, not
this module's.

Usage::

    processor = OmniProcessor.from_pretrained(checkpoint_root)
    model = OmniModel.from_pretrained(checkpoint_root, device_map="auto")
    inputs = processor(text="Describe this image.", images=["/path/to.jpg"])
    model.reset()
    generated = model.generate(inputs, generation_kwargs={"max_new_tokens": 128})

Or, when the model is built first (e.g. from an already-resolved launcher-YAML
config with per-module ``model_config:`` overrides — see ``OmniTrainer`` /
``OmniInferencer``), reuse ``model.config`` instead of re-reading
``checkpoint_root`` a second time — saves one redundant ``config.json`` load,
and keeps the two builds looking at the exact same config::

    model = OmniModel.from_pretrained(checkpoint_root, config=my_resolved_config, device_map="auto")
    processor = OmniProcessor.from_config(model.config, checkpoint_root=checkpoint_root)
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Union

from PIL import Image

from ...data.multimodal.image_utils import load_image
from ...utils import logging
from .configuration_omni import OmniConfig
from .mixins.module_processor_mixin import Preprocessor
from .modules import OMNI_MODEL_REGISTRY, read_model_type
from .utils.conversation import build_conversation


logger = logging.get_logger(__name__)

ImageInput = Union[str, Any]
MediaInput = Union[ImageInput, Sequence[ImageInput]]


def _normalize_images(images: MediaInput) -> list[Any]:
    if isinstance(images, (str, bytes)) or isinstance(images, Image.Image):
        images = [images]
    normalized: list[Any] = []
    for image in images:
        if isinstance(image, Image.Image):
            normalized.append(image)
        else:
            normalized.append(load_image(image))
    return normalized


class OmniProcessor:
    """Composed SeedOmni request preprocessor (HF ``AutoProcessor``-style API)."""

    def __init__(self, preprocessors: dict[str, Preprocessor]) -> None:
        self._preprocessors: dict[str, Preprocessor] = dict(preprocessors)

    def __len__(self) -> int:
        """Number of worker-side preprocessors contributed by active graph modules."""
        return len(self._preprocessors)

    def bind_dummy_inputs(self, module_configs: Mapping[str, Any], *, dtype: Any = None) -> None:
        """Training-only: attach each preprocessor's FSDP-anchor dummy tensor(s).

        Computed from each module's own already-resolved config + ``dtype``
        alone (see :meth:`Preprocessor.bind_dummy_inputs`) — no disk re-read
        here: ``module_configs`` is ``{module_name: config}`` taken straight
        from the already-built live modules (e.g.
        ``{name: module_runtime.model_config for name, module_runtime in
        self.model.module_runtimes.items()}``), so it is already the exact
        config the live model was built with, overrides included. Call once,
        right after the training model finishes building
        (:meth:`~veomni.trainer.omni.omni_trainer.OmniTrainer._build_train_dataloader`
        runs after ``_build_model``). Unnecessary for pure inference
        (``OmniInferencer``): the dummy branch is never exercised there.
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
        in-memory config (e.g. ``OmniTrainer`` / ``OmniInferencer`` launcher configs,
        built before or independently of ``self.model``) should use.

        Collects CPU preprocessors in ``config.module_names`` order: for each
        module, reads its ``preprocessor_class`` off the class registered for its
        ``model_type`` and calls :meth:`Preprocessor.from_pretrained` directly on
        its checkpoint subfolder — no model instance is built at all (not even a
        ``meta``-device one). ``config.module_model_config(name)`` (the module's
        YAML ``model_config:`` block) is forwarded as ``config_overrides`` so a
        preprocessor that reads a behavior-affecting config field (e.g.
        ``enable_image``, ``cache_mode``) agrees with what the live model is
        actually configured with, instead of silently falling back to the
        on-disk ``config.json`` default.
        """
        root = checkpoint_root if checkpoint_root is not None else getattr(config, "_name_or_path", None)
        root = None if root is None else str(root)
        preprocessors: dict[str, Preprocessor] = {}
        for name in config.module_names:
            module_path = config.resolve_module_path(root, name)
            model_type = read_model_type(module_path)
            mod_cls = OMNI_MODEL_REGISTRY[model_type]()
            preprocessor_cls = getattr(mod_cls, "preprocessor_class", None)
            if preprocessor_cls is None:
                continue
            preprocessor = preprocessor_cls.from_pretrained(
                module_path, config_overrides=config.module_model_config(name)
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
        text: str = "",
        *,
        images: MediaInput | None = None,
        videos: MediaInput | None = None,
        inference: bool = True,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Build and preprocess a single inference request.

        Returns a dict suitable for :meth:`OmniModel.generate` — currently
        ``{"conversation_list": [...]}``.
        """
        del videos  # video inputs follow the same path once callers pass PIL/VideoInputs

        image_items = _normalize_images(images) if images is not None else []
        conversation = build_conversation(prompt=text, images=image_items)
        return self.preprocess(conversation, inference=inference, **generation_kwargs)

    def preprocess(
        self,
        conversation: list[Any],
        *,
        inference: bool = True,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the module preprocessor chain on an existing ``conversation_list``."""
        self.preprocess_batch([conversation], inference=inference, **generation_kwargs)
        return {"conversation_list": conversation}

    def preprocess_batch(
        self,
        conversation_batches: Sequence[list[Any]],
        *,
        inference: bool = False,
        **generation_kwargs: Any,
    ) -> None:
        """Run the module preprocessor chain over a batched ``conversation_list``.

        Training passes ``inference=False`` (default); single-request inference uses
        :meth:`preprocess` / :meth:`__call__` with ``inference=True``.
        """
        for preprocessor in self._preprocessors.values():
            preprocessor(conversation_batches, inference=inference, generation_kwargs=generation_kwargs or None)


__all__ = [
    "OmniProcessor",
]
