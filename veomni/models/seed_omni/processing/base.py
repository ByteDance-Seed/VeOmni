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

"""Base class for per-module CPU input preparation (DataLoader workers).

Terminology (three different "processor" layers)
------------------------------------------------
* **HF asset — ``XxxProcessor``** (in each ``modules/*/processing.py``):
  HuggingFace-style checkpoint sidecar — image processor, tokenizer, etc.
  Saved/loaded via ``save_pretrained`` / ``from_pretrained`` on the asset itself
  (e.g. ``JanusSiglipProcessor``, ``BagelVAEProcessor``).  Holds resize /
  normalize constants; no ``nn.Module`` weights.

* **Module CPU worker — ``XxxPreprocessor(ModulePreprocessorBase)``** (same file):
  Picklable, weight-free object run inside DataLoader workers (training) or
  once before the generation FSM (inference).  Usually *wraps* the HF asset
  (``self._image_processor``, ``self._tokenizer``, …) and mutates
  ``conversation_list`` in place via ``__call__``.

* **Omni orchestrator — ``OmniProcessor``** (``processing_omni.py``):
  Composes one ``XxxPreprocessor`` per active graph module — the SeedOmni
  analogue of HuggingFace ``AutoProcessor``.

This module defines only the **middle** layer's abstract base.  It is **not**
a graph mixin; each module's native model class declares
``preprocessor_class = XxxPreprocessor``
as a registry pointer and optionally receives copied assets via
:func:`~veomni.models.seed_omni.processing.binding.bind_module_assets`.
"""

from __future__ import annotations

from typing import Any


class ModulePreprocessorBase:
    """Picklable, weight-free CPU input-prep run inside DataLoader workers.

    Subclass as ``XxxPreprocessor`` in ``modules/<family>/<sub>/processing.py``.
    The HF asset (``XxxProcessor`` / tokenizer) lives in the same file but is a
    separate class — do not confuse the two.

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
    ) -> ModulePreprocessorBase | None:
        """Build this module's CPU worker from its checkpoint subfolder alone.

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
        ``processing.py``-defined ``ModulePreprocessorBase`` subclass.
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


__all__ = ["ModulePreprocessorBase"]
