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

"""Config class for DeepSpec draft models loaded through VeOmni.

A DeepSpec draft config is a deep copy of the *target* model's HuggingFace
config (so the draft inherits ``hidden_size``, ``num_attention_heads``,
``rope_parameters``, ``rms_norm_eps``, ``layer_types`` …) plus a handful of
draft-specific fields (``target_layer_ids``, ``block_size``, ``num_anchors``,
``markov_rank``, ``ttt_length`` …).

VeOmni looks a model class up by ``config.model_type`` via ``MODELING_REGISTRY``.
If the draft config kept the target's ``model_type`` (``"qwen3"``) it would
shadow VeOmni's own Qwen3. So the prep step rewrites ``model_type`` to
``"deepspec_draft"`` and records the target's original type in
``base_model_type`` so this class can rebuild a *fully-normalized* target config
and layer the draft fields on top. That fidelity matters because the DeepSpec
modeling code reads many target-config attributes directly
(e.g. ``Qwen3RotaryEmbedding`` reads ``config.rope_parameters["rope_type"]``).
"""

from typing import Any, Dict

from transformers import PretrainedConfig


class DeepSpecDraftConfig(PretrainedConfig):
    """A PretrainedConfig that mirrors the target config + draft extras.

    Construction strategy:

    * If ``base_model_type`` is present and resolvable via HF's
      ``CONFIG_MAPPING``, instantiate that target config class from the same
      kwargs (so all target-side normalization / defaults run), then copy its
      attributes onto ``self``. This yields a faithful superset config.
    * Otherwise fall back to storing the kwargs flat (still works as long as the
      serialized json already contains every field the modeling code reads,
      which it does because it was produced by ``target_config.to_dict()``).
    """

    model_type = "deepspec_draft"
    # Sub-configs are not expected, but keep HF happy for is_composition checks.
    is_composition = False

    def __init__(self, base_model_type: str = None, **kwargs: Any):
        self.base_model_type = base_model_type
        # ``model_type`` is a class attribute; HF's ``from_pretrained`` round-trips
        # it as a kwarg. Drop it so it doesn't fight the class attribute.
        kwargs.pop("model_type", None)

        # Rebuild a normalized target config and adopt its attributes, so any
        # field the target config would compute in __init__ is present here too.
        normalized: Dict[str, Any] = {}
        if base_model_type is not None:
            normalized = self._normalized_target_attributes(base_model_type, kwargs)

        # Target attributes first, then explicit kwargs win (draft overrides such
        # as num_hidden_layers, architectures, tie_word_embeddings).
        merged: Dict[str, Any] = {}
        merged.update(normalized)
        merged.update(kwargs)

        # ``architectures`` / ``tie_word_embeddings`` are consumed by
        # PretrainedConfig.__init__; the rest are set as plain attributes.
        super().__init__(**merged)

        # PretrainedConfig only promotes *known* kwargs to attributes reliably;
        # guarantee every merged key is present as an attribute for the modeling
        # code, without clobbering what the base __init__ already set.
        for key, value in merged.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @staticmethod
    def _normalized_target_attributes(base_model_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate the target config class and return its attribute dict."""
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if base_model_type not in CONFIG_MAPPING:
                return {}
            target_cls = CONFIG_MAPPING[base_model_type]
            # Drop keys that would confuse the target config constructor.
            target_kwargs = dict(kwargs)
            target_kwargs.pop("model_type", None)
            target_kwargs.pop("base_model_type", None)
            target_config = target_cls(**target_kwargs)
            return {k: v for k, v in target_config.to_dict().items() if k != "model_type"}
        except Exception:
            # Be forgiving: a missing/renamed target config type must not break
            # loading — the flat kwargs already carry every needed field.
            return {}

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["model_type"] = self.model_type
        output["base_model_type"] = self.base_model_type
        return output


__all__ = ["DeepSpecDraftConfig"]
