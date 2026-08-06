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

"""HF-native SeedOmni sub-model base — extends :class:`PreTrainedModel` for eager inference."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel


class OmniPreTrainedModel(PreTrainedModel):
    """VeOmni HF-native base for every ``modules/*/modeling.py`` class.

    Subclasses hold weights, ``forward``, and FSM ``generate`` endpoints only.
    VeOmni training / distributed hooks live on ``accelerated.py`` mixins composed
    at runtime (``OMNI_ACCELERATED_MODEL_REGISTRY``).
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then bind module-owned processor / tokenizer sidecars."""
        from .processing.binding import bind_module_assets

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        bind_module_assets(
            model,
            checkpoint_path=str(pretrained_model_name_or_path),
            config_overrides=kwargs,
        )
        return model

    def get_assets(self) -> list[Any]:
        """Module-owned auxiliary artefacts to save alongside the weights."""
        return []

    def reset_local_inference_state(self) -> None:
        """Reset per-turn state inside an ongoing conversation."""
        return None

    def reset_global_inference_state(self) -> None:
        """Reset the full conversation-level inference state."""
        self.reset_local_inference_state()

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        """Flush module-private generation buffers into a one-shot ``generated`` payload."""
        del ctx
        return {}


__all__ = ["OmniPreTrainedModel"]
