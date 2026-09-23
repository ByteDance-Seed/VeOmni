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

"""HF-native base for every SeedOmni sub-module ``modeling.py``."""

from __future__ import annotations

import os
from typing import Any

from transformers import PreTrainedModel


class PretrainedOmniModule(PreTrainedModel):
    """Base for every ``modules/<family>/<sub>/modeling.py`` class.

    Subclasses hold weights, ``forward``, and FSM ``generate`` endpoints only.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then bind module-owned processor / tokenizer sidecars."""
        from .module_processing_base import bind_module_assets

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # ``kwargs`` may carry an HF ``config=<PretrainedConfig>`` (the loaded
        # module config, forwarded by ``OmniModel._load_modules``) — that's a
        # different "config" than the launcher/runtime overrides dict
        # ``config_overrides`` represents. Strip the HF ``config`` key before
        # binding so it cannot collide with a preprocessor helper that also
        # takes a positional ``config``.
        config_overrides = {k: v for k, v in kwargs.items() if k != "config"}
        bind_module_assets(
            model,
            checkpoint_path=str(pretrained_model_name_or_path),
            config_overrides=config_overrides,
        )
        return model

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        *args: Any,
        save_module_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write this module's config and its processor / tokenizer sidecars.

        ``save_module_weights=False`` skips the weight files. The sidecars are
        written either way, so a weights-free export can still preprocess.
        Weights go last so this module's ``config.json`` is the one that remains.
        """
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        for attr in ("_processor", "_image_processor", "_video_processor", "_tokenizer"):
            asset = getattr(self, attr, None)
            if asset is not None and hasattr(asset, "save_pretrained"):
                asset.save_pretrained(save_directory)
        if not save_module_weights:
            self.config.save_pretrained(save_directory)
            return
        super().save_pretrained(save_directory, *args, **kwargs)

    def get_assets(self) -> list[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    def generate(self, generation_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Default FSM endpoint: same as training ``forward``. Override when inference differs.

        :class:`~veomni.models.seed_omni.modeling_omni.OmniModel` calls
        ``generate(**ctx, generation_kwargs=...)`` for every generation-graph
        node. Bare endpoints resolve to this method.
        """
        del generation_kwargs
        return self.forward(**kwargs)

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing generation request."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full request-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Abort-only flush of leftover buffers into a one-shot ``generated`` payload.

        Invoked when generation hits ``max_new_tokens`` before ``done``, not
        after a normal FSM completion.
        """
        del ctx
        return {}


__all__ = ["PretrainedOmniModule"]
