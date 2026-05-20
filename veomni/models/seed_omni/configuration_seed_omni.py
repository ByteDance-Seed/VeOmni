"""
OmniConfig — central configuration for OmniModel V2.

YAML structure (maps 1-to-1 to this class):

  # ── Global asset (the ONLY global asset; everything else is per-module).
  tokenizer_path: /path/to/tokenizer

  # ── Modules: one entry per OmniModule.  `model_type` is read from each
  #    module's `config.json` (loaded via HF AutoConfig) and is NOT specified
  #    here — the YAML carries only the location and overrides.
  modules:
    siglip:        {weights_path: ..., config_path: ..., micro_batch_size: 4, ...}
    janus_llama:   {weights_path: ..., micro_batch_size: 2}
    wte_lm_head:   {weights_path: ...}
    vqvae:         {weights_path: ..., freeze: true}

  # ── Graph nodes (call-sites): one entry per `module.method` invocation.
  #    The same module may appear under multiple node names.
  nodes:
    siglip_encode: {module: siglip}
    vae_encode:    {module: vqvae.encode}
    vae_decode:    {module: vqvae.decode}
    tok_encode:    {module: wte_lm_head.encode}
    tok_decode:    {module: wte_lm_head.decode}
    janus_llama:   {module: janus_llama}

  # ── Graph edges (data dependencies). `from`/`to` reference node names; a
  #    module name is accepted when that module has exactly one declared node.
  #    `to: end` declares a leaf node (virtual sink) — every node MUST
  #    appear on at least one edge.
  edges:
    siglip_to_ar:    {from: siglip_encode, output: image_embeds, to: janus_llama, as: und_image_embeds}
    vae_enc_to_ar:   {from: vae_encode,    output: gen_embeds,   to: janus_llama, as: gen_image_embeds}
    tok_enc_to_ar:   {from: tok_encode,    output: inputs_embeds, to: janus_llama, as: inputs_embeds}
    ar_to_tok_dec:   {from: janus_llama,   output: hidden_states, to: tok_decode,  as: hidden_states}
    ar_to_vae_dec:   {from: janus_llama,   output: hidden_states, to: vae_decode,  as: hidden_states}
    vae_token_to_dec:{from: vae_encode,    output: vq_token_ids,  to: vae_decode,  as: gt_token_ids}
    tok_dec_sink:    {from: tok_decode,    to: end}
    vae_dec_sink:    {from: vae_decode,    to: end}

  # ── Active training subset.  ONLY edges — active nodes are derived from
  #    the endpoints of these edges.
  training_graph:
    edges: [siglip_to_ar, vae_enc_to_ar, tok_enc_to_ar, ar_to_tok_dec,
            ar_to_vae_dec, vae_token_to_dec, tok_dec_sink, vae_dec_sink]

  # ── Inference FSM.  Each state.body is a list of EDGE NAMES; node order
  #    is derived (unique endpoints in declaration order, excluding `end`).
  generation_graph:
    initial: text_ar
    done_state: done
    states:
      text_ar:
        body: [tok_enc_to_ar, ar_to_tok_dec, tok_dec_sink]
        token_length: {type: variable}
        transitions:
          - {condition: {type: token_match, token_id: 100578}, next_state: image_vq}
      image_vq:
        body: [ar_to_vae_dec, vae_dec_to_ar]
        token_length: {type: fixed, value: 576}
        transitions:
          - {condition: {type: steps_complete}, next_state: text_ar}
      done:
        body: []
        token_length: {type: fixed, value: 0}
        transitions: []
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class OmniConfig(PretrainedConfig):
    """Configuration for OmniModel V2.

    All nested dicts are stored as plain Python dicts for JSON serialisability.
    Typed accessors (``module_config``, ``training_edges``) provide a stable
    surface for the runtime / visualisation tools.

    The single global asset is :attr:`tokenizer_path`; everything else is
    per-module and saved alongside the module's checkpoint.
    """

    model_type = "omni"

    def __init__(
        self,
        modules: Optional[Dict[str, Dict]] = None,
        nodes: Optional[Dict[str, Dict]] = None,
        edges: Optional[Dict[str, Dict]] = None,
        training_graph: Optional[Dict] = None,
        generation_graph: Optional[Dict] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        self.modules: Dict[str, Dict] = modules or {}
        self.nodes: Dict[str, Dict] = nodes or {}
        self.edges: Dict[str, Dict] = edges or {}
        self.training_graph: Dict = training_graph or {"edges": []}
        self.generation_graph: Optional[Dict] = generation_graph
        self.tokenizer_path: Optional[str] = tokenizer_path

        super().__init__(**kwargs)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def module_config(self, name: str) -> Dict[str, Any]:
        """Return the raw config dict for a module by name (deep-copied)."""
        cfg = self.modules.get(name)
        if cfg is None:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        return deepcopy(cfg)

    @property
    def training_edges(self) -> List[str]:
        """Active training subset (edge names).  Active nodes are derived."""
        return list(self.training_graph.get("edges", []))

    @property
    def module_names(self) -> List[str]:
        return list(self.modules.keys())

    def has_generation_graph(self) -> bool:
        return self.generation_graph is not None

    # ── Factory: load from YAML dict ──────────────────────────────────────────

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**accepted, **kwargs)


__all__ = ["OmniConfig"]
