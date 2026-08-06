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

"""Picklable CPU preprocessors and the mixin hooks that wire them onto modules."""

from __future__ import annotations

from typing import Any


class Preprocessor:
    """Picklable, weight-free CPU input-prep run inside DataLoader workers.

    A module whose ``pre_forward`` does heavy **CPU** input preparation (e.g. a
    text encoder's chat-template + tokenize, a vision tower's image normalize)
    can move that work off the main/GPU process by declaring one of these on its
    ``modules/<family>/<sub>/processing.py`` (as ``XxxModuleMixin.preprocessor_class``).
    The :class:`~veomni.trainer.omni.omni_trainer.OmniTrainer`
    orchestrator collects the active graph-node modules' preprocessors and runs
    them inside :class:`~veomni.data.data_collator.SeedOmniCollator` — which
    executes in the DataLoader worker — so the work overlaps with GPU compute via
    prefetch instead of blocking the main process inside ``pre_forward``.

    Contract:

    * **No model weights, no model instance.** It is pickled / fork-inherited into
      worker processes, so it must hold only CPU-safe, picklable assets (tokenizer /
      image processor / special-token ids / config ints) — never the ``nn.Module``.
      :meth:`from_pretrained` builds it straight from a module's checkpoint
      subfolder — mirroring HuggingFace's ``XxxProcessor.from_pretrained`` — with
      **no dependency on any live/real model** (weight-free or otherwise).
    * **CPU only.** Workers must not touch the training CUDA device; build CPU
      tensors (no ``device=``).  The main process's thin ``pre_forward`` does the
      single ``.to(device)``.
    * **In-place mutation.** ``__call__`` receives the batched
      ``conversation_list`` (``list[list[ConversationItem]]``) and mutates items'
      ``value`` / ``meta`` in place, tagging the module ``source`` so the thin
      ``pre_forward`` / ``generate`` reads the heavy work back uniformly.
    * **Shared by training + inference.** Training runs it inside
      :class:`~veomni.data.data_collator.SeedOmniCollator` (DataLoader worker);
      inference runs it once over the request in
      :meth:`~veomni.trainer.omni.omni_inferencer.OmniInferencer._preprocess_request`,
      before the FSM. The ``inference`` flag flips the train/infer-only behaviour:
      image modules **skip dummy injection** (no FSDP anchor at inference) and
      text encoders **append the assistant generation prompt**. Extra request
      options (e.g. ``generation_kwargs``) arrive via ``**kwargs`` so a module
      *could* vary its input-prep by them (classifier-free guidance duplicating the
      prompt, …); no current module needs them, but the hook is plumbed through.
    * **Dummy inputs are optional and bound after construction.** A module whose
      ``inference=False`` (training) branch injects an FSDP-anchor dummy item
      (image modules only — text encoders never need one) computes that dummy's
      shape from pure ``(config, dtype)`` — the preprocessor itself still never
      touches a live model or the checkpoint disk to get that ``config``. The
      *orchestrator* does, though: :meth:`~veomni.trainer.omni.omni_trainer.OmniTrainer._build_train_dataloader`
      runs after the training model is already built, so it hands
      :meth:`OmniProcessor.bind_dummy_inputs` each module's already-resolved
      ``ModuleRuntime.model_config`` straight from memory (no disk re-read, no
      config-override re-application) — see :meth:`bind_dummy_inputs`.
      Inference never exercises the dummy branch, so an unbound dummy is harmless
      there.
    """

    def __call__(self, conversation_list: list[list[Any]], inference: bool = False, **kwargs: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "__call__(conversation_list, inference=False, **kwargs) and mutate it in place."
        )

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> Preprocessor | None:
        """Build this module's preprocessor from its checkpoint subfolder alone.

        No model instance (weight-free or otherwise) is built or required.
        ``config_overrides`` mirrors the module's YAML ``model_config:`` block
        (the same dict threaded into the live model's ``config_kwargs`` — see
        ``ModuleRuntime._build_module_model``): a subclass that reads its own
        ``config.json`` for a behavior-affecting field (e.g. ``enable_image``,
        ``cache_mode``) must apply these on top of the on-disk defaults —
        ``XxxConfig.from_pretrained(module_path, **(config_overrides or {}))`` —
        so a preprocessor built independently of any model instance still
        agrees with what the live model was actually configured with. Default:
        this module contributes no preprocessor (e.g. a pure backbone with no
        CPU-side input prep). Concrete modules override on their own
        ``processing.py``-defined ``Preprocessor`` subclass.
        """
        del module_path, config_overrides, kwargs
        return None

    def bind_dummy_inputs(self, config: Any, dtype: Any = None) -> None:
        """Attach the FSDP-anchor dummy tensor(s) for training's ``inference=False``
        branch — computed from ``config`` + ``dtype`` alone (no live model).

        Called once by :meth:`~veomni.trainer.omni.omni_trainer.OmniTrainer._build_train_dataloader`
        after the training collator's preprocessors are collected. Default: no-op
        (text-encoder preprocessors and any module without a dummy branch).
        """
        del config, dtype
        return None


class ModuleProcessorMixin:
    """Mixin wiring a module's picklable ``Preprocessor`` onto the live model.

    Concrete modules declare ``preprocessor_class`` on their family mixin and
    implement ``XxxPreprocessor(Preprocessor)`` in ``processing.py``.  Training
    collects preprocessors into :class:`~veomni.models.seed_omni.processing_omni.OmniProcessor`;
    inference runs the same instances in
    :meth:`~veomni.trainer.omni.omni_inferencer.OmniInferencer._preprocess_request`.
    """

    preprocessor_class: type[Preprocessor] | None = None

    def bind_preprocessor(self, preprocessor: Preprocessor | None) -> None:
        """Attach a preprocessor's already-loaded assets onto this module instance.

        ``preprocessor`` was built independently of this model (see
        :meth:`Preprocessor.from_pretrained`) — this only copies its well-known
        asset attributes onto the conventional instance names a module's own
        ``forward`` / ``generate`` code reads directly (``self._image_processor``,
        ``self._video_processor``, ``self.tokenizer``/``self._chat_template``, …),
        so those call sites need no changes. Copied verbatim (no rebuilding): a
        text preprocessor's ``_chat_template`` was already derived from its
        ``_tokenizer`` at construction time. Override on a family mixin only if a
        module needs custom wiring beyond this attribute copy.
        """
        if preprocessor is None:
            return
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer", "_chat_template"):
            if hasattr(preprocessor, attr):
                setattr(self, attr, getattr(preprocessor, attr))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then bind the per-module preprocessor if declared.

        The preprocessor itself is built by :meth:`Preprocessor.from_pretrained`
        straight from the checkpoint dir — no model instance is involved in
        building it; :meth:`bind_preprocessor` only copies its assets onto
        ``model``. On failure the module gets no preprocessor bound (best-effort;
        surfaced lazily when the modality is actually used).

        Forwards this same call's ``kwargs`` on as ``config_overrides`` — HF's
        ``PretrainedConfig.from_pretrained`` only applies keys it recognizes as
        attributes and silently ignores the rest (e.g. ``device_map``), so
        reusing the raw call kwargs is safe and keeps the preprocessor's
        config-derived behavior (e.g. ``enable_image``, ``cache_mode``) in sync
        with the config overrides just applied to ``model.config`` above.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        preprocessor_cls = getattr(cls, "preprocessor_class", None)
        if preprocessor_cls is not None:
            try:
                preprocessor = preprocessor_cls.from_pretrained(pretrained_model_name_or_path, config_overrides=kwargs)
            except Exception:
                preprocessor = None
            model.bind_preprocessor(preprocessor)
        return model


__all__ = ["ModuleProcessorMixin", "Preprocessor"]
