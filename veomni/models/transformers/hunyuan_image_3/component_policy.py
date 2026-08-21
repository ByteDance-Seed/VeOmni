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

"""Per-component lifecycle recipe for the HunyuanImage 3 model.

Each component is ``absent`` (not constructed, and its official checkpoint
tensors are skipped on import), ``frozen`` (constructed, ``requires_grad=False``)
or ``trainable``. This is a training knob, not part of the official Base config
-- the trainer injects it at ``build_foundation_model`` time, and every field
defaults to the T2I recipe so a config spells out only its deviations.

What a given capability actually requires is enforced where it is used (the
model's ``_validate_reference_components``), not by a whitelist here: adding a
capability should mean adding components, not editing a validator that says no.
"""

from dataclasses import dataclass, fields
from typing import Literal, Mapping


ComponentState = Literal["absent", "frozen", "trainable"]

COMPONENT_STATES = frozenset(("absent", "frozen", "trainable"))

#: Component -> official checkpoint key prefix, for the components the runtime
#: may drop. The importer skips tensors under a prefix whose component is
#: ``absent``; reassembling an official checkpoint means restoring those same
#: prefixes from the pinned Base. Components without an entry are always present
#: and map identically.
CHECKPOINT_PREFIXES: dict[str, str] = {
    "lm_head": "lm_head.",
    "vae_encoder": "vae.encoder.",
    "vae_decoder": "vae.decoder.",
    "vision_model": "vision_model.",
    "vision_aligner": "vision_aligner.",
}


@dataclass(frozen=True)
class HunyuanImage3ComponentPolicy:
    """Lifecycle state per component; defaults are the T2I recipe."""

    transformer: ComponentState = "trainable"
    text_embedding: ComponentState = "trainable"
    image_projector: ComponentState = "trainable"
    timestep_modules: ComponentState = "trainable"
    image_head: ComponentState = "trainable"
    vae_encoder: ComponentState = "frozen"
    vae_decoder: ComponentState = "absent"
    vision_model: ComponentState = "absent"
    vision_aligner: ComponentState = "absent"
    lm_head: ComponentState = "absent"

    @classmethod
    def from_dict(cls, values: Mapping[str, str]) -> "HunyuanImage3ComponentPolicy":
        """Build from a partial mapping; unlisted components take their default."""
        if not isinstance(values, Mapping):
            raise TypeError("component_policy must be a mapping.")

        unknown = sorted(set(values).difference(COMPONENT_NAMES))
        if unknown:
            raise ValueError(f"Unknown HunyuanImage 3 components in component_policy: {unknown}.")

        invalid_states = {name: state for name, state in values.items() if state not in COMPONENT_STATES}
        if invalid_states:
            raise ValueError(
                f"Invalid HunyuanImage 3 component states: {invalid_states}; expected one of {sorted(COMPONENT_STATES)}."
            )
        return cls(**dict(values))

    def as_dict(self) -> dict[str, ComponentState]:
        return {name: getattr(self, name) for name in COMPONENT_NAMES}

    def state(self, name: str) -> ComponentState:
        if name not in COMPONENT_NAMES:
            raise KeyError(f"Unknown HunyuanImage 3 component: {name}.")
        return getattr(self, name)

    def checkpoint_prefix_is_absent(self, name: str) -> bool:
        """True when an official checkpoint key belongs to an absent component."""
        return any(
            name.startswith(prefix) and self.state(component) == "absent"
            for component, prefix in CHECKPOINT_PREFIXES.items()
        )


COMPONENT_NAMES: tuple[str, ...] = tuple(field.name for field in fields(HunyuanImage3ComponentPolicy))

#: The T2I recipe, materialised.
DEFAULT_COMPONENT_POLICY: dict[str, ComponentState] = HunyuanImage3ComponentPolicy().as_dict()


__all__ = [
    "CHECKPOINT_PREFIXES",
    "COMPONENT_NAMES",
    "COMPONENT_STATES",
    "DEFAULT_COMPONENT_POLICY",
    "ComponentState",
    "HunyuanImage3ComponentPolicy",
]
