# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass
from typing import Literal, Mapping


ComponentState = Literal["absent", "frozen", "trainable"]

COMPONENT_NAMES = (
    "transformer",
    "text_embedding",
    "image_projector",
    "timestep_modules",
    "image_head",
    "vae_encoder",
    "vae_decoder",
    "vision_model",
    "vision_aligner",
    "lm_head",
)
COMPONENT_STATES = frozenset(("absent", "frozen", "trainable"))

_INITIAL_REQUIRED_STATES: dict[str, ComponentState] = {
    "transformer": "trainable",
    "text_embedding": "trainable",
    "image_projector": "trainable",
    "timestep_modules": "trainable",
    "image_head": "trainable",
    "vae_decoder": "absent",
    "vision_model": "absent",
    "vision_aligner": "absent",
    "lm_head": "absent",
}


@dataclass(frozen=True)
class HunyuanImage3ComponentPolicy:
    transformer: ComponentState
    text_embedding: ComponentState
    image_projector: ComponentState
    timestep_modules: ComponentState
    image_head: ComponentState
    vae_encoder: ComponentState
    vae_decoder: ComponentState
    vision_model: ComponentState
    vision_aligner: ComponentState
    lm_head: ComponentState

    @classmethod
    def from_dict(cls, values: Mapping[str, str]) -> "HunyuanImage3ComponentPolicy":
        if not isinstance(values, Mapping):
            raise TypeError("component_policy must be a mapping.")

        missing = sorted(set(COMPONENT_NAMES).difference(values))
        unknown = sorted(set(values).difference(COMPONENT_NAMES))
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError(f"Invalid HunyuanImage 3 component_policy: {', '.join(details)}.")

        invalid_states = {name: values[name] for name in COMPONENT_NAMES if values[name] not in COMPONENT_STATES}
        if invalid_states:
            raise ValueError(
                "Invalid HunyuanImage 3 component states: "
                f"{invalid_states}; expected one of {sorted(COMPONENT_STATES)}."
            )

        policy = cls(**{name: values[name] for name in COMPONENT_NAMES})
        policy.validate_initial_t2i()
        return policy

    def validate_initial_t2i(self) -> None:
        mismatches = {
            name: {"expected": expected, "actual": getattr(self, name)}
            for name, expected in _INITIAL_REQUIRED_STATES.items()
            if getattr(self, name) != expected
        }
        if self.vae_encoder not in ("absent", "frozen"):
            mismatches["vae_encoder"] = {"expected": "absent or frozen", "actual": self.vae_encoder}
        if mismatches:
            raise ValueError(
                f"The initial single_gen_t2i_v1 capability does not support this component_policy: {mismatches}."
            )

    def as_dict(self) -> dict[str, ComponentState]:
        return {name: getattr(self, name) for name in COMPONENT_NAMES}

    def state(self, name: str) -> ComponentState:
        if name not in COMPONENT_NAMES:
            raise KeyError(f"Unknown HunyuanImage 3 component: {name}.")
        return getattr(self, name)

    def checkpoint_prefix_is_absent(self, name: str) -> bool:
        prefix_to_component = {
            "lm_head.": "lm_head",
            "vae.decoder.": "vae_decoder",
            "vision_model.": "vision_model",
            "vision_aligner.": "vision_aligner",
        }
        if self.vae_encoder == "absent" and name.startswith("vae.encoder."):
            return True
        return any(
            name.startswith(prefix) and self.state(component) == "absent"
            for prefix, component in prefix_to_component.items()
        )


__all__ = [
    "COMPONENT_NAMES",
    "COMPONENT_STATES",
    "ComponentState",
    "HunyuanImage3ComponentPolicy",
]
