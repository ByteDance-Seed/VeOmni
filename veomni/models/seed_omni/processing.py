"""OmniProcessor — composed request preprocessing for :class:`OmniModel`.

Mirrors HuggingFace ``AutoProcessor``: collect each module's CPU preprocessor in
``config.module_names`` order, build a ``conversation_list`` from user inputs,
run the preprocessor chain, and return a generate-ready request dict.

Usage::

    processor = OmniProcessor.from_pretrained(checkpoint_root)
    model = OmniModel.from_pretrained(checkpoint_root, device_map="auto")
    inputs = processor(text="Describe this image.", images=["/path/to.jpg"])
    model.reset()
    generated = model.generate(inputs, generation_kwargs={"max_new_tokens": 128})
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Callable, Union

from PIL import Image

from ...data.multimodal.image_utils import load_image
from ...utils import logging
from .collator import ComposedModel, collect_cpu_preprocessors, run_cpu_preprocessors
from .configuration_omni import OmniConfig
from .mixins.modulemixin import CPUPreprocessor, ModuleMixin
from .modeling_omni import OmniModel
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


def _load_module_assets(module: ModuleMixin, module_path: str) -> None:
    """Load processor / tokenizer sidecars without module weights."""
    from ..auto import build_tokenizer

    cls = type(module)
    for attr, class_attr in (
        ("_processor", "processor_class"),
        ("_image_processor", "image_processor_class"),
        ("_video_processor", "video_processor_class"),
    ):
        asset_class = getattr(cls, class_attr, None)
        if asset_class is None:
            continue
        try:
            setattr(module, attr, asset_class.from_pretrained(module_path))
        except Exception:
            setattr(module, attr, None)

    try:
        module.tokenizer = build_tokenizer(module_path)
    except Exception:
        module._tokenizer = None


def build_cpu_preprocessor_from_checkpoint(module_path: str) -> CPUPreprocessor | None:
    """Build one module's inference CPU preprocessor from its checkpoint subfolder."""
    model_type = read_model_type(module_path)
    mod_cls = OMNI_MODEL_REGISTRY[model_type]()
    config = mod_cls.config_class.from_pretrained(module_path)
    module = mod_cls.from_config(config)
    _load_module_assets(module, module_path)
    builder = getattr(module, "build_cpu_preprocessor", None)
    if builder is None:
        return None
    return builder()


def collect_cpu_preprocessors_from_checkpoint(
    config: OmniConfig,
    checkpoint_root: str | os.PathLike,
) -> tuple[Callable[..., None], ...]:
    """Collect CPU preprocessors in ``config.module_names`` order without loading weights."""
    root = str(checkpoint_root)
    preprocessors: list[Callable[..., None]] = []
    for name in config.module_names:
        module_path = config.resolve_module_path(root, name)
        preprocessor = build_cpu_preprocessor_from_checkpoint(module_path)
        if preprocessor is not None:
            preprocessors.append(preprocessor)
            logger.info_rank0(f"OmniProcessor: module '{name}' contributes {type(preprocessor).__name__}.")
    return tuple(preprocessors)


class OmniProcessor:
    """Composed SeedOmni request preprocessor (HF ``AutoProcessor``-style API)."""

    def __init__(
        self,
        config: OmniConfig,
        preprocessors: Sequence[Callable[..., None]],
    ) -> None:
        self.config = config
        self._preprocessors = tuple(preprocessors)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **config_kwargs: Any,
    ) -> OmniProcessor:
        """Load preprocessors from a split-checkpoint root (no module weights)."""
        config = OmniConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        root = getattr(config, "_name_or_path", None) or str(pretrained_model_name_or_path)
        preprocessors = collect_cpu_preprocessors_from_checkpoint(config, root)
        return cls(config, preprocessors)

    @classmethod
    def from_composed_model(cls, model: ComposedModel) -> OmniProcessor:
        """Reuse CPU preprocessors from a loaded composed model (``OmniModel`` or runtime)."""
        return cls(model.config, collect_cpu_preprocessors(model))

    @classmethod
    def from_model(cls, model: OmniModel) -> OmniProcessor:
        """Reuse CPU preprocessors already wired on a loaded :class:`OmniModel`."""
        return cls.from_composed_model(model)

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
        run_cpu_preprocessors(
            self._preprocessors,
            conversation_batches,
            inference=inference,
            generation_kwargs=generation_kwargs or None,
        )


AutoProcessor = OmniProcessor

__all__ = [
    "AutoProcessor",
    "OmniProcessor",
    "build_cpu_preprocessor_from_checkpoint",
    "collect_cpu_preprocessors_from_checkpoint",
]
