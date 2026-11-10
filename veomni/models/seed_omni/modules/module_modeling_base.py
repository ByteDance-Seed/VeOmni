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

from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel


class PretrainedOmniModule(PreTrainedModel):
    """Base for every ``modules/<family>/<sub>/modeling.py`` class.

    Subclasses hold weights, ``forward``, and FSM ``generate`` endpoints only.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        """Load weights, then bind module-owned processor / tokenizer sidecars.

        A module may declare :meth:`_allowed_checkpoint_mismatches`, naming the
        keys its own ``_create_checkpoint_tensor_converter`` re-shapes rather than
        loads. That converter only runs through VeOmni's loaders; HF's
        ``from_pretrained`` (what a single-process eager inference run uses) sees
        the same mismatch and refuses to start. So the load is retried with
        ``ignore_mismatched_sizes=True``, but only when *every* mismatched key is
        one the module declared — anything else still raises, so an unintended
        shape change is not swallowed.
        """
        from .module_processing_base import bind_module_assets

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except RuntimeError as error:
            allowed = cls._allowed_checkpoint_mismatches(pretrained_model_name_or_path, args, kwargs)
            if not allowed:
                raise
            # HF reports the mismatch two ways depending on version (a "size mismatch"
            # state-dict error, or the generic "ignore_mismatched_sizes is False"), so
            # retry rather than matching on the message, and let the check below decide.
            model, info = super().from_pretrained(
                pretrained_model_name_or_path, *args, ignore_mismatched_sizes=True, output_loading_info=True, **kwargs
            )
            mismatched = {name for name, _, _ in info["mismatched_keys"]}
            unexpected = mismatched - allowed
            if unexpected or not mismatched:
                raise RuntimeError(
                    f"{cls.__name__}: the load failed, and its size mismatches are not the ones this module "
                    f"re-initialises on purpose (declared {sorted(allowed)}, got {sorted(mismatched)})."
                ) from error
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

    @classmethod
    def _allowed_checkpoint_mismatches(
        cls,
        pretrained_model_name_or_path: Any,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> frozenset[str]:
        """Checkpoint keys whose shape mismatch this module re-initialises on purpose.

        Empty by default. Override when a module's
        :meth:`_create_checkpoint_tensor_converter` reshapes a tensor at load:
        the converter covers VeOmni's loaders, and this covers HF's.
        """
        return frozenset()

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
