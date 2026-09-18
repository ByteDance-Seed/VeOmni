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

from typing import Any

from transformers import PreTrainedModel


class OmniPreTrainedModel(PreTrainedModel):
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

    def get_assets(self) -> list[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing generation request."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full request-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Flush module-private generation buffers into a one-shot ``generated`` payload."""
        del ctx
        return {}


__all__ = ["OmniPreTrainedModel"]
