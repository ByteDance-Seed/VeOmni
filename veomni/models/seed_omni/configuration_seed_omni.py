"""
OmniConfig — central configuration for OmniModel V2.

YAML structure (maps 1-to-1 to this class):

  modules:
    vision_encoder: {model_type: janus_vision_encoder, ...extra kwargs}
    vq_decoder:     {model_type: janus_vq_decoder, ...}
    ar_llm:         {model_type: janus_llm, ...}

  connections:
    run_ar:           {module: ar_llm}
    vision_to_ar:     {from: vision_encoder, output: image_embeds, to: ar_llm, as: und_image_embeds}
    vae_enc_to_ar:    {from: vq_decoder,     output: gen_embeds,   to: ar_llm, as: gen_image_embeds}
    ar_to_vq:         {from: ar_llm,         output: vq_token_id,  to: vq_decoder, as: token_id}
    vq_dec_to_ar:     {from: vq_decoder,     output: embed,        to: ar_llm, as: inputs_embeds}

  training_graph:
    connections: [vision_to_ar, run_ar]

  generation_states:
    initial: text_ar
    states:
      text_ar:
        body: [run_ar]
        token_length: {type: variable}
        transitions:
          - {condition: {type: token_match, token_id: 100577}, next_state: image_vq}
      image_vq:
        body: [run_ar, ar_to_vq, vq_dec_to_ar]
        token_length: {type: fixed, value: 576}
        transitions:
          - {condition: {type: steps_complete}, next_state: text_ar}
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class OmniConfig(PretrainedConfig):
    """Configuration for OmniModel V2.

    All nested dicts are stored as plain Python dicts for JSON serialisability.
    OmniModel's ``__init__`` calls :meth:`module_config` and the graph /
    generation helpers when it needs typed access.
    """

    model_type = "omni"

    def __init__(
        self,
        modules: Optional[Dict[str, Dict]] = None,
        connections: Optional[Dict[str, Dict]] = None,
        training_graph: Optional[Dict] = None,
        generation_states: Optional[Dict] = None,
        **kwargs,
    ):
        # Store all as plain dicts so they serialise cleanly with PretrainedConfig
        self.modules: Dict[str, Dict] = modules or {}
        self.connections: Dict[str, Dict] = connections or {}
        self.training_graph: Dict = training_graph or {"connections": []}
        self.generation_states: Optional[Dict] = generation_states

        super().__init__(**kwargs)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def module_config(self, name: str) -> Dict[str, Any]:
        """Return the raw config dict for a module by name."""
        cfg = self.modules.get(name)
        if cfg is None:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        return deepcopy(cfg)

    @property
    def training_connections(self) -> List[str]:
        return self.training_graph.get("connections", [])

    @property
    def module_names(self) -> List[str]:
        return list(self.modules.keys())

    def has_generation_states(self) -> bool:
        return self.generation_states is not None

    # ── Factory: load from YAML dict ──────────────────────────────────────────

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}, **kwargs)
