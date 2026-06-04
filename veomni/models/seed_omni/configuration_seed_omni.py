"""
OmniConfig — central configuration for OmniModel V2.

YAML structure (maps 1-to-1 to this class):

  # ── Global asset (the ONLY global asset; everything else is per-module).
  model_path: /path/to/split/checkpoint/root

  # ── Modules: one entry per OmniModule.  `model_type` is read from each
  #    module's `config.json` (model_type read by OMNI_*_REGISTRY) and is NOT specified
  #    here — the YAML carries only the location and overrides.
  modules:
    siglip:        {weights_path: ..., config_path: ...}
    janus_llama:   {weights_path: ...}
    text_encoder:  {weights_path: ...}
    # Per-module config overrides live under `model_config:` (mirrors
    # ModelArguments.model_config) — forwarded to the module's config.
    vqvae:         {weights_path: ..., model_config: {freeze: true}}

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
    siglip_to_ar:    {from: siglip_encode, to: janus_llama}
    vae_enc_to_ar:   {from: vae_encode,    to: janus_llama}
    tok_enc_to_ar:   {from: tok_encode,    to: janus_llama}
    ar_to_tok_dec:   {from: janus_llama,   to: tok_decode}
    ar_to_vae_dec:   {from: janus_llama,   to: vae_decode}
    vae_token_to_dec:{from: vae_encode,    to: vae_decode}
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
  #    States carry no iteration budget — a state body iterates until one of
  #    its transitions fires, and modules decide when via signals (the AR
  #    loop) or after a single pass via `default` (bridge/leaf states).
  generation_graph:
    initial: text_ar
    states:
      text_ar:
        body: [tok_enc_to_ar, ar_to_tok_dec, tok_dec_sink]
        transitions:
          - {condition: {type: module_signal, key: start_image_gen}, next_state: image_vq}
      image_vq:
        body: [ar_to_vae_dec, vae_dec_to_ar]
        transitions:
          - {condition: {type: module_signal, key: image_complete}, next_state: text_ar}
"""

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig


class OmniConfig(PretrainedConfig):
    """Configuration for OmniModel V2.

    All nested dicts are stored as plain Python dicts for JSON serialisability.
    Typed accessors (``module_config``, ``training_edges``) provide a stable
    surface for the runtime / visualisation tools.

    Tokenizers and processors are per-module assets saved alongside each
    module's checkpoint (e.g. ``janus_text_encoder/tokenizer.json``).
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
        """Build an :class:`OmniConfig` from a YAML-shaped dict.

        Pops ``model_path`` (required) and resolves every relative
        ``modules.*.weights_path`` under it; absolute paths pass through
        unchanged.  Unknown top-level keys are dropped silently so the
        same launcher YAML can carry training-only fields without
        polluting the inference config.
        """
        model_path = config_dict.pop("model_path")
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        for mod_cfg in accepted.get("modules", {}).values():
            weights_path = mod_cfg.get("weights_path")
            if weights_path is not None and not os.path.isabs(weights_path):
                mod_cfg["weights_path"] = os.path.join(model_path, weights_path)
        return cls(**accepted, **kwargs)

    @classmethod
    def from_paths(
        cls,
        model_path: Union[str, os.PathLike],
        train_yaml_path: Union[str, os.PathLike],
        infer_yaml_path: Union[str, os.PathLike] = None,
        tokenizer_path: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> "OmniConfig":
        """Load an :class:`OmniConfig` from a training YAML + optional inference YAML.

        SeedOmni V2 splits configuration into a **training YAML** that
        carries the master vocabulary (``modules``, ``nodes``, ``edges``,
        ``training_graph``) and one or more **inference YAMLs** that only
        carry a ``generation_graph`` for a specific scenario (interleave /
        T2I-only / understanding).  See ``configs/seed_omni/janus_1.3b/``
        for the canonical layout.

        The two YAMLs are flat-merged via ``dict.update`` — the inference
        YAML's top-level keys (in practice just ``generation_graph``)
        replace anything with the same name in the training YAML.  Anything
        deeper (``modules.foo.*``) cannot be partially overridden; declare
        the full block in whichever YAML owns it.
        """
        import yaml

        base_cfg = {"model_path": model_path}
        if tokenizer_path is not None:
            base_cfg["tokenizer_path"] = tokenizer_path
        with open(train_yaml_path, encoding="utf-8") as f:
            base_cfg.update(yaml.safe_load(f))
        if infer_yaml_path is not None:
            with open(infer_yaml_path, encoding="utf-8") as f:
                base_cfg.update(yaml.safe_load(f))
        return cls.from_dict(base_cfg, **kwargs)


__all__ = ["OmniConfig"]
