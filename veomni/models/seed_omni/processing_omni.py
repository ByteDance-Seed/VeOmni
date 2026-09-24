"""OmniProcessor — composed request preprocessing for :class:`OmniModel`.

Mirrors HuggingFace ``AutoProcessor``: collect each module's CPU preprocessor in
``config.module_names`` order, build a ``conversation_list`` from user inputs,
run the preprocessor chain, and return a generate-ready request dict.

Every module's :class:`~veomni.models.seed_omni.modules.module_processing_base.ModulePreprocessorBase`
(defined in its own ``processing.py`` and pointed at by ``preprocessor_class`` on
its native model class) builds straight off its checkpoint subfolder via
:meth:`~veomni.models.seed_omni.modules.module_processing_base.ModulePreprocessorBase.from_pretrained` —
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

from ...utils import logging  # VeOmni shared logger (rank-0 helpers); not seed_omni-local.
from .configuration_omni import OmniConfig
from .modules import OMNI_MODEL_REGISTRY, read_model_type
from .modules.module_processing_base import ModulePreprocessorBase
from .utils.conversation import build_conversation


logger = logging.get_logger(__name__)

# A ref a fetcher can decode: a path, a URL, or the raw bytes of a container.
# What counts beyond that is the fetcher's business — the image one also takes a
# PIL image, and the audio one a waveform array given its rate in ``mm_configs``.
MediaRef = Union[str, bytes, Any]
MediaInput = Union[MediaRef, Sequence[MediaRef]]

# Ref types that are themselves sequences, and so would otherwise be iterated
# into their elements: one bytes-backed clip becoming a few thousand integer
# "refs". ``bytearray`` is what a parquet corpus storing media inline produces and
# the loaders take it; ``memoryview`` is listed because they do *not*, and one ref
# they reject by name beats a few thousand they reject by type.
_ATOMIC_REFS = (str, bytes, bytearray, memoryview)


def _as_list(media: MediaInput | None) -> list[Any]:
    """One ref or a sequence of them, as a list — decoding is the fetchers' job.

    Non-sequence media (PIL image, tensor, ndarray, ``VideoInputs``) is a single
    item, as is any of :data:`_ATOMIC_REFS`.
    """
    if media is None:
        return []
    if isinstance(media, _ATOMIC_REFS) or not isinstance(media, Sequence):
        return [media]
    return list(media)


class OmniProcessor:
    """Composed SeedOmni request preprocessor (HF ``AutoProcessor``-style API)."""

    def __init__(self, preprocessors: dict[str, ModulePreprocessorBase]) -> None:
        self._preprocessors: dict[str, ModulePreprocessorBase] = dict(preprocessors)

    def __len__(self) -> int:
        """Number of worker-side preprocessors contributed by active graph modules."""
        return len(self._preprocessors)

    def bind_dummy_inputs(self, module_configs: Mapping[str, Any], *, dtype: Any = None) -> None:
        """Training-only: attach each preprocessor's FSDP-anchor dummy tensor(s).

        Computed from each module's own already-resolved config + ``dtype``
        alone (see :meth:`ModulePreprocessorBase.bind_dummy_inputs`) — no disk re-read
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
        ``model_type`` and calls :meth:`ModulePreprocessorBase.from_pretrained` directly on
        its checkpoint subfolder — no model instance is built at all (not even a
        ``meta``-device one). The entry's ``model_config`` is forwarded as
        ``config_overrides`` so a preprocessor that reads a behavior-affecting
        model field (e.g. ``enable_image``) agrees with the live model. The
        entry's ``processor_config`` is splatted as kwargs, matching
        ``build_processor(path, **processor_config)``.
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
            entry = config._module_entries[name]
            preprocessor = preprocessor_cls.from_pretrained(
                module_path,
                config_overrides=dict(entry.get("model_config") or {}),
                **dict(entry.get("processor_config") or {}),
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
        audios: MediaInput | None = None,
        videos: MediaInput | None = None,
        mm_configs: Mapping[str, Any] | None = None,
        inference: bool = True,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Build and preprocess a single inference request.

        Each modality takes one ref or a list of them — a path, a URL, or the raw
        bytes of a container — decoded by the very fetchers the training transform
        uses. That is what makes a request carry the same metadata a training
        sample does: a clip's sampling rate and a video's frame timeline are
        facts only the decode knew, and a module cannot recover them from the
        payload. Whether a modality also accepts already-decoded media is its
        fetcher's business, not this signature's: images take a PIL image,
        waveform arrays need ``mm_configs={"audio_sampling_rate": ...}`` to say
        what rate they are in, and pixel tensors are not accepted at all.

        A clip with sound is **one** entry in ``videos`` with
        ``mm_configs={"use_audio_in_video": True}``, not a video plus a
        parallel entry in ``audios``: the two streams share a timeline the
        backbone interleaves, and splitting them across two items throws away
        the alignment. ``audios`` is for standalone sound.

        ``mm_configs`` carries the decode knobs (``fps`` / ``max_frames`` /
        ``image_max_pixels`` / ``video_max_pixels`` / ``audio_sampling_rate`` /
        ``use_audio_in_video``) — a named dict rather than more ``**`` because
        ``**generation_kwargs`` would otherwise swallow them. It shares its name
        with training's ``data.mm_configs`` because it is the same bag reaching
        the same fetchers: a key set here means what it means there.

        Returns a dict suitable for :meth:`OmniModel.generate` —
        ``{"conversation_list": [...]}`` (a single conversation).
        """
        # Imported in-function: ``veomni.data.seed_omni`` reaches back into
        # ``veomni.models.seed_omni`` for the conversation carrier, so a
        # module-scope import here would close that loop.
        from ...data.seed_omni.utils.media import fetch_media

        media_refs = {"image": _as_list(images), "audio": _as_list(audios), "video": _as_list(videos)}
        media = fetch_media(media_refs, "OmniProcessor request", mm_configs)
        conversation = build_conversation(prompt=text, media=media)
        return self.preprocess(conversation, inference=inference, **generation_kwargs)

    def preprocess(
        self,
        conversation: list[Any],
        *,
        inference: bool = True,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the module preprocessor chain on an existing ``conversation_list``."""
        batch = {"conversation_list": [conversation]}
        self.preprocess_batch(batch, inference=inference, **generation_kwargs)
        return {"conversation_list": batch["conversation_list"][0]}

    def preprocess_batch(
        self,
        batch: dict[str, Any],
        *,
        inference: bool = False,
        **generation_kwargs: Any,
    ) -> None:
        """Run the module preprocessor chain over a collated batch.

        ``batch`` must contain ``conversation_list`` as
        ``list[list[ConversationItem]]``. Training collator passes the full
        feature dict; :meth:`preprocess` builds ``{"conversation_list": [conversation]}``.
        Packed Janus writes extra tensors onto this same dict.

        Training passes ``inference=False`` (default); single-request inference
        uses :meth:`preprocess` / :meth:`__call__` with ``inference=True``.
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
