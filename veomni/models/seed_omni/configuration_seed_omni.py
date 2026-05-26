"""
OmniConfig — central configuration for OmniModel V2.

YAML structure (maps 1-to-1 to this class):

  # ── Global asset (the ONLY global asset; everything else is per-module).
  tokenizer_path: /path/to/tokenizer

  # ── Modules: one entry per OmniModule.  `model_type` is read from each
  #    module's `config.json` (model_type read by OMNI_*_REGISTRY) and is NOT specified
  #    here — the YAML carries only the location and overrides.
  modules:
    siglip:        {weights_path: ..., config_path: ...}
    janus_llama:   {weights_path: ...}
    text_encoder:  {weights_path: ...}
    vqvae:         {weights_path: ..., freeze: true}

  # NOTE: ``micro_batch_size`` / DP / SP / TP / CP are NOT per-module — they
  # live globally under ``train.*`` in the launcher YAML.  All modules share
  # the same global micro_batch_size and parallel mesh; per-module parallel
  # customisation will be revisited after the OmniTrainer build flow is
  # fully working.

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
  #    The `done` state is auto-injected by the framework — never declare
  #    it here, never set `done_state`.  Transitions whose
  #    `next_state: done` land on the built-in terminal state which then
  #    triggers each active module's `finalize` hook (text decode /
  #    image save / etc).
  generation_graph:
    initial: text_ar
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
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    @classmethod
    def from_yamls(cls, *paths: Union[str, Path], **kwargs) -> "OmniConfig":
        """Load + deep-merge multiple YAML configs (later overrides earlier).

        SeedOmni V2 splits configuration into a **training YAML** that
        carries the master vocabulary (``tokenizer_path``, ``modules``,
        ``nodes``, ``edges``, ``training_graph``) and one or more
        **inference YAMLs** that only carry a ``generation_graph`` for a
        specific scenario (interleave / T2I-only / understanding).  See
        ``configs/seed_omni/janus_1.3b/`` for the canonical layout.

        Merge semantics
        ---------------
        Top-level keys are merged depth-first:

        * **Dict-vs-dict**: recursive merge.  ``modules.foo.weights_path``
          in the inference YAML overrides the training YAML's value while
          keeping every other ``modules.foo.*`` field intact.
        * **List-vs-list / scalar-vs-scalar**: later wins outright (no
          element-wise merge — that would be ambiguous for ordered
          ``training_graph.edges``).
        * **None-vs-anything**: anything wins; ``None`` never overwrites
          a real value.

        Examples
        --------
        Load training-only (single file)::

            cfg = OmniConfig.from_yamls("configs/seed_omni/janus_1.3b/train_joint.yaml")

        Load training + an inference scenario (paint a generation_graph
        on top)::

            cfg = OmniConfig.from_yamls(
                "configs/seed_omni/janus_1.3b/train_joint.yaml",
                "configs/seed_omni/janus_1.3b/infer_interleave.yaml",
            )
        """
        import yaml

        if not paths:
            raise ValueError("OmniConfig.from_yamls requires at least one path.")

        merged: Dict[str, Any] = {}
        for p in paths:
            text = Path(p).read_text()
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise TypeError(f"YAML at '{p}' must be a top-level mapping; got {type(data).__name__}.")
            merged = _deep_merge(merged, data)
        return cls.from_dict(merged, **kwargs)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a deep-copy of ``base`` and return it.

    Dict values merge recursively; non-dict values from ``override``
    replace the base wholesale.  ``None`` in ``override`` does NOT clear
    the base — use an explicit empty dict / list / scalar to override.
    """
    out: Dict[str, Any] = deepcopy(base)
    for k, v in override.items():
        if v is None:
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


__all__ = ["OmniConfig"]
