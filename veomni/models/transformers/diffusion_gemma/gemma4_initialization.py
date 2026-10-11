# Copyright 2025 ByteDance Ltd. and/or its affiliates
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

"""Initialize DiffusionGemma from an unmodified Google Gemma 4 checkpoint.

The two models use the same Gemma 4 text and vision tensors, but expose them
under different module prefixes. DiffusionGemma additionally owns one encoder
``layer_scalar`` per text layer and four decoder self-conditioning parameters.
This module keeps that conversion explicit and validates the complete source
layout before training can start.
"""

from typing import Dict, List, Optional

import torch

from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


_RAW_TO_DIFFUSION_PREFIXES = (
    ("model.language_model.", "model.decoder."),
    ("model.vision_tower.", "model.encoder.vision_tower."),
    ("model.embed_vision.", "model.encoder.embed_vision."),
)

SELF_CONDITIONING_PARAMETER_NAMES = frozenset(
    {
        "model.decoder.self_conditioning.pre_norm.weight",
        "model.decoder.self_conditioning.gate_proj.weight",
        "model.decoder.self_conditioning.up_proj.weight",
        "model.decoder.self_conditioning.down_proj.weight",
    }
)


def convert_gemma4_checkpoint_key(name: str) -> str:
    """Map one official Gemma 4 tensor name to its DiffusionGemma name."""
    for source_prefix, target_prefix in _RAW_TO_DIFFUSION_PREFIXES:
        if name.startswith(source_prefix):
            return target_prefix + name[len(source_prefix) :]
    raise ValueError(f"Unexpected raw Gemma 4 checkpoint key: {name}")


def _encoder_layer_scalar_name(decoder_name: str) -> str:
    if not decoder_name.startswith("model.decoder.layers.") or not decoder_name.endswith(".layer_scalar"):
        raise ValueError(f"Not a decoder layer_scalar key: {decoder_name}")
    return "model.encoder.language_model." + decoder_name[len("model.decoder.") :]


def convert_gemma4_fqn_to_index_mapping(fqn_to_index_mapping: Dict[str, int]) -> Dict[str, int]:
    """Convert the raw two-shard index into a complete DiffusionGemma save map."""
    raw_keys = [
        name for name in fqn_to_index_mapping if name.startswith(tuple(p[0] for p in _RAW_TO_DIFFUSION_PREFIXES))
    ]
    diffusion_keys = [
        name
        for name in fqn_to_index_mapping
        if name.startswith(("model.decoder.", "model.encoder.")) or name == "lm_head.weight"
    ]
    if raw_keys and diffusion_keys:
        raise RuntimeError("Checkpoint index mixes raw Gemma 4 and DiffusionGemma tensor names.")
    if not raw_keys:
        return dict(fqn_to_index_mapping)
    if len(raw_keys) != len(fqn_to_index_mapping):
        unknown = sorted(set(fqn_to_index_mapping) - set(raw_keys))
        raise RuntimeError(f"Unexpected keys in raw Gemma 4 checkpoint index: {unknown}")

    converted: Dict[str, int] = {}
    for source_name, shard_index in fqn_to_index_mapping.items():
        target_name = convert_gemma4_checkpoint_key(source_name)
        if target_name in converted:
            raise RuntimeError(f"Gemma 4 checkpoint key collision at {target_name}")
        converted[target_name] = shard_index
        if target_name.endswith(".layer_scalar"):
            converted[_encoder_layer_scalar_name(target_name)] = shard_index

    anchor = "model.decoder.norm.weight"
    if anchor not in converted:
        raise RuntimeError(f"Raw Gemma 4 checkpoint is missing shard anchor {anchor}")
    for name in SELF_CONDITIONING_PARAMETER_NAMES:
        converted[name] = converted[anchor]
    return converted


def _expected_raw_gemma4_keys(model) -> set[str]:
    expected: set[str] = set()
    for name in model.state_dict():
        if name in SELF_CONDITIONING_PARAMETER_NAMES or name == "lm_head.weight":
            continue
        if name.startswith("model.decoder."):
            expected.add("model.language_model." + name[len("model.decoder.") :])
        elif name.startswith("model.encoder.vision_tower."):
            expected.add("model.vision_tower." + name[len("model.encoder.vision_tower.") :])
        elif name.startswith("model.encoder.embed_vision."):
            expected.add("model.embed_vision." + name[len("model.encoder.embed_vision.") :])
        elif name.startswith("model.encoder.language_model.") and name.endswith(".layer_scalar"):
            # The one raw language-model scalar initializes both independent buffers.
            continue
    return expected


class Gemma4ToDiffusionGemmaCheckpointTensorConverter:
    """Stream raw Gemma 4 tensors into the DiffusionGemma module layout."""

    def __init__(self, model):
        self._model = model
        self._expected_source_names = _expected_raw_gemma4_keys(model)
        self._seen_source_names: set[str] = set()
        self._encoder_layer_scalars: Dict[str, torch.Tensor] = {}

    def can_handle(self, name: str) -> bool:
        return name.startswith(tuple(source for source, _ in _RAW_TO_DIFFUSION_PREFIXES))

    def convert(self, name: str, tensor: torch.Tensor) -> Optional[ConvertedCheckpointTensor]:
        if name in self._seen_source_names:
            raise RuntimeError(f"Duplicate raw Gemma 4 checkpoint key: {name}")
        self._seen_source_names.add(name)
        target_name = convert_gemma4_checkpoint_key(name)
        if target_name.endswith(".layer_scalar"):
            self._encoder_layer_scalars[_encoder_layer_scalar_name(target_name)] = tensor.clone()
        return ConvertedCheckpointTensor(name=target_name, tensor=tensor)

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if not self._seen_source_names:
            return []

        missing = sorted(self._expected_source_names - self._seen_source_names)
        unexpected = sorted(self._seen_source_names - self._expected_source_names)
        if missing or unexpected:
            raise RuntimeError(
                "Raw Gemma 4 checkpoint layout does not exactly match DiffusionGemma: "
                f"missing={missing}, unexpected={unexpected}"
            )

        expected_scalar_count = int(self._model.config.text_config.num_hidden_layers)
        if len(self._encoder_layer_scalars) != expected_scalar_count:
            raise RuntimeError(
                "Raw Gemma 4 checkpoint has the wrong number of layer_scalar tensors: "
                f"expected={expected_scalar_count}, actual={len(self._encoder_layer_scalars)}"
            )

        self._model._veomni_checkpoint_source_format = "raw_gemma4"
        return [
            ConvertedCheckpointTensor(name=name, tensor=tensor)
            for name, tensor in sorted(self._encoder_layer_scalars.items())
        ]


def create_gemma4_to_diffusion_gemma_checkpoint_tensor_converter(model):
    return Gemma4ToDiffusionGemmaCheckpointTensorConverter(model)


def validate_gemma4_initialization_missing_parameters(model, missing: set[str]) -> None:
    """Validate raw initialization and retain only genuinely new parameters.

    The generic loader reports tied aliases as missing because the raw Gemma 4
    checkpoint stores the language stack and token embedding once. DiffusionGemma
    re-ties those aliases after loading, so initializing them here would waste a
    second model-sized allocation immediately before it is discarded. FSDP changes
    which aliases survive ``named_parameters`` deduplication, so accept any subset of
    the known tied aliases while still requiring the exact self-conditioning set.
    """
    tied_aliases = {
        name for name in missing if name == "lm_head.weight" or name.startswith("model.encoder.language_model.")
    }
    genuinely_missing = missing - tied_aliases
    source_format = getattr(model, "_veomni_checkpoint_source_format", None)
    # Only rank zero reads checkpoint tensors in the broadcast loader and can set
    # the converter marker. Receiving ranks identify the same raw layout from its
    # canonical missing-name signature. Any other unmarked layout belongs to the
    # normal checkpoint path and is left to the generic initializer.
    if source_format != "raw_gemma4" and genuinely_missing != SELF_CONDITIONING_PARAMETER_NAMES:
        return
    if genuinely_missing != SELF_CONDITIONING_PARAMETER_NAMES:
        raise RuntimeError(
            "Raw Gemma 4 initialization left an unexpected parameter set: "
            f"expected_new={sorted(SELF_CONDITIONING_PARAMETER_NAMES)}, "
            f"actual_new={sorted(genuinely_missing)}, tied_aliases={sorted(tied_aliases)}"
        )
    missing.intersection_update(SELF_CONDITIONING_PARAMETER_NAMES)
