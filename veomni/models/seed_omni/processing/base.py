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
  Saved/loaded via ``save_pretrained`` / ``from_pretrained`` on the asset itself.
  Holds resize / normalize constants; no ``nn.Module`` weights.

* **Module CPU worker — ``XxxPreprocessor(ModulePreprocessorBase)``** (same file):
  Picklable, weight-free object run inside DataLoader workers (training) or
  once before the generation FSM (inference).  Usually *wraps* the HF asset
  (``self._image_processor``, ``self._tokenizer``, …) and mutates the collator
  ``batch`` dict in place via ``__call__``.

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
    * **In-place mutation.** ``__call__`` receives the collator ``batch`` dict
      and mutates it in place. Subclasses decide which keys they read and write.
    * **Shared by training + inference.** Training runs it inside a collator
      (DataLoader worker); inference runs it once over the request before the
      FSM. The ``inference`` flag flips train/infer-only behaviour.
    """

    def __call__(self, batch: dict[str, Any], inference: bool = False, **kwargs: Any) -> None:
        """Run CPU prep on a collated ``batch`` dict (mutates it in place)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__(batch, inference=False, **kwargs) "
            "and mutate the batch in place."
        )

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> ModulePreprocessorBase | None:
        """Build this module's CPU worker from its checkpoint subfolder alone.

        No model instance (weight-free or otherwise) is built or required.
        ``config_overrides`` mirrors the module's YAML ``model_config:`` block
        (the same dict threaded into the live model's ``config_kwargs`` — see
        ``ModuleRuntime.build_model``): a subclass that reads its own
        ``config.json`` for a behavior-affecting field must apply these on top
        of the on-disk defaults so a preprocessor built independently of any
        model instance still agrees with what the live model was actually
        configured with. Default: this module contributes no preprocessor
        (e.g. a pure backbone with no CPU-side input prep). Concrete modules
        override on their own ``processing.py``-defined
        ``ModulePreprocessorBase`` subclass.
        """
        del module_path, config_overrides, kwargs
        return None

    def bind_dummy_inputs(self, config: Any, dtype: Any = None) -> None:
        """Attach optional dummy tensor(s) for training's ``inference=False`` branch.

        Computed from ``config`` + ``dtype`` alone (no live model). Default: no-op.
        """
        del config, dtype
        return None


__all__ = ["ModulePreprocessorBase"]
