"""
OmniConfig — central configuration for OmniModel V2.

YAML structure (maps 1-to-1 to this class):

  # ── Global asset (the ONLY global asset; everything else is per-module).
  model_path: /path/to/split/checkpoint/root

  # ── Modules: one entry per OmniModule.  Each block mirrors a mini
  #    ``VeOmniOmniArguments`` (``model`` / ``train`` / ``data``).  ``model_type``
  #    is read from each module's ``config.json`` and is NOT specified here.
  modules:
    siglip:
      model: {weights_path: ...}
    janus_llama:
      model: {weights_path: ...}
    vqvae:
      model:
        weights_path: ...
        model_config: {freeze: true}
      train:
        init_device: meta
        accelerator:
          fsdp_config: {fsdp_mode: fsdp2}

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
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig

from ...arguments.arguments_types import VeOmniArguments
from ...arguments.parser import _deep_update, _instantiate_recursive


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
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.modules: Dict[str, Dict] = modules or {}
        self.nodes: Dict[str, Dict] = nodes or {}
        self.edges: Dict[str, Dict] = edges or {}
        self.training_graph: Dict = training_graph or {"edges": []}

        self.generation_graph: Optional[Dict] = generation_graph
        self.generation_kwargs: Optional[Dict] = generation_kwargs

        super().__init__(**kwargs)

    def module_config(self, name: str) -> VeOmniArguments:
        """Return the raw config dict for a module by name (deep-copied)."""
        cfg = self.modules.get(name, None)
        if cfg is None:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        return _instantiate_recursive(VeOmniArguments, cfg)

    @property
    def training_edges(self) -> List[str]:
        """Active training subset (edge names).  Active nodes are derived."""
        return list(self.training_graph.get("edges", []))

    @property
    def module_names(self) -> List[str]:
        return list(self.modules.keys())

    def has_generation_graph(self) -> bool:
        return self.generation_graph is not None

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        """Build an :class:`OmniConfig` from a YAML-shaped dict.

        Pops ``model_path`` (the split-checkpoint root from the launcher YAML —
        consumed by :func:`build_module_args`, not stored here).  Module blocks
        are kept verbatim; path resolution happens when merging each module onto
        the global :class:`VeOmniOmniArguments`.  Unknown top-level keys are
        dropped silently so the launcher YAML can carry training-only fields.
        """
        config_dict = dict(config_dict)
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**accepted, **kwargs)

    @classmethod
    def _init(
        cls,
        global_args: VeOmniArguments,
        model_path: Union[str, os.PathLike],
        train_yaml_path: Union[str, os.PathLike],
        infer_yaml_path: Union[str, os.PathLike] = None,
    ) -> "OmniConfig":
        """Load an :class:`OmniConfig` from a training YAML + optional inference YAML.

        SeedOmni V2 splits configuration into a **training YAML** that
        carries the master vocabulary (``modules``, ``nodes``, ``edges``,
        ``training_graph``) and one or more **inference YAMLs** that only
        carry a ``generation_graph`` for a specific scenario (interleave /
        T2I-only / understanding).  See ``configs/seed_omni/janus_1.3b/``
        for the canonical layout.

        Per-module args are built from the launcher ``global_args``
        (:class:`VeOmniArguments` parsed from ``veomni_janus.yaml``).  Train
        and infer YAML ``modules:`` blocks are deep-merged into one override
        dict per module name, then merged onto ``global_args`` (same path as
        :func:`parse_args` via ``_deep_update`` + ``_instantiate_recursive``).

        Relative ``model.weights_path`` subfolders are joined with ``model_path``
        before the merge sets ``model.model_path`` / ``config_path``.
        """
        from dataclasses import asdict

        import yaml

        with open(train_yaml_path, encoding="utf-8") as f:
            omni_train_config = yaml.safe_load(f)

        modules_overrides: Dict[str, Any] = omni_train_config.pop("modules")
        modules_overrides = cls._resolve_model_path(model_path, modules_overrides)
        base_cfg = {
            "nodes": omni_train_config.pop("nodes"),
            "edges": omni_train_config.pop("edges"),
            "training_graph": omni_train_config.pop("training_graph"),
        }
        if infer_yaml_path is not None:
            with open(infer_yaml_path, encoding="utf-8") as f:
                omni_infer_config = yaml.safe_load(f)
            infer_modules_overrides = omni_infer_config.pop("modules", {})
            if infer_modules_overrides:
                infer_modules_overrides = cls._resolve_model_path(model_path, infer_modules_overrides)
            infer_modules_overrides = cls._resolve_default_accelerator(modules_overrides, infer_modules_overrides)
            modules_overrides = _deep_update(modules_overrides, infer_modules_overrides)
            _deep_update(
                base_cfg,
                {
                    "generation_graph": omni_infer_config.pop("generation_graph"),
                    "generation_kwargs": omni_infer_config.pop("generation_kwargs"),
                },
            )
        base_cfg["modules"] = {
            name: _deep_update(asdict(global_args), override) for name, override in modules_overrides.items()
        }
        return cls.from_dict(base_cfg)

    @staticmethod
    def _resolve_model_path(
        model_path: Union[str, os.PathLike],
        modules_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Join relative ``model.model_path`` under ``model_path``."""
        if not modules_config:
            return {}
        checkpoint_root = str(model_path)
        for mod_cfg in modules_config.values():
            model = mod_cfg.setdefault("model", {})
            resolved = model.get("model_path") or model.get("weights_path")
            if resolved is None:
                continue
            if not os.path.isabs(resolved):
                resolved = os.path.join(checkpoint_root, resolved)
            _deep_update(
                model,
                {
                    "model_path": resolved,
                    "config_path": resolved,
                    "tokenizer_path": resolved,
                },
            )
        return modules_config

    @staticmethod
    def _resolve_default_accelerator(
        train_modules_config: Dict[str, Any],
        infer_modules_overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Default inference load: eager (``fsdp_mode: eager``) for every train module.

        Infer YAMLs usually omit ``modules:`` entirely.  Build one override per
        module name from ``train_modules_config``, each setting
        ``train.accelerator.fsdp_config.fsdp_mode`` to ``eager``.  Non-empty
        ``infer_modules_overrides`` deep-merge on top (e.g. opt-in FSDP2 for one
        module).
        """
        eager_by_module = {
            name: {
                "train": {
                    "accelerator": {
                        "fsdp_config": {
                            "fsdp_mode": "eager",
                        }
                    }
                }
            }
            for name in train_modules_config
        }
        return _deep_update(eager_by_module, infer_modules_overrides)


__all__ = ["OmniConfig"]
