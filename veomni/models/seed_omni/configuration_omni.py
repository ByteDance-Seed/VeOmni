"""
OmniConfig — central configuration for OmniModel V2.

Built from the **omni split-file layout** via :meth:`OmniConfig.from_omni_args` (see
:class:`~veomni.arguments.arguments_types_omni.OmniArguments`):

  * ``model.model_path``   — split-checkpoint root folder (one subdir per module).
  * ``model.modules``      — per-module training overrides (``modules_train.yaml``).
  * ``model.train_graph``  — the training DAG (``graph_train.yaml``).
  * ``infer.modules``      — per-module inference overrides (``modules_infer.yaml``).
  * ``infer.infer_graph``  — scenario -> generation-graph file map.

Per-module override block (one entry per graph sub-model).  ``model_type`` is
read from each module's ``config.json`` and is NOT specified here.  Note that
``accelerator`` sits at the block top level (omni convention) and is lifted under
``train.accelerator`` by :meth:`OmniConfig._lift_accelerator` before merging:

  janus_siglip:
    model: {model_path: janus_siglip}
  janus_vqvae:
    model:
      model_path: janus_vqvae
      model_config: {freeze: true}
  janus_llama:
    model: {model_path: janus_llama}
    accelerator:
      fsdp_config: {fsdp_mode: fsdp2}

Internally OmniConfig still stores fully-merged per-module dicts + a training
graph (flat edge list) + an optional generation graph.

  # ── Active training subset.  A flat list of edges; each endpoint is a
  #    self-describing `module[.method]` string (bare module → `.forward`).
  #    Active nodes are derived from the endpoints.  `to: end` declares a leaf
  #    node (virtual sink) — every node MUST appear on at least one edge.
  training_graph:
    - {from: siglip,            to: janus_llama}
    - {from: vqvae.encode,      to: janus_llama}
    - {from: wte_lm_head.encode, to: janus_llama}
    - {from: janus_llama,       to: wte_lm_head.decode}
    - {from: janus_llama,       to: vqvae.decode}
    - {from: wte_lm_head.decode, to: end}
    - {from: vqvae.decode,      to: end}

  # ── Inference FSM.  Each state.body is a list of inline `{from, to}` edges
  #    (endpoints as `module[.method]` strings, bare module → `.generate`);
  #    node order is derived (unique endpoints in declaration order, excluding
  #    `end`).  The `done` state is auto-injected by the framework — never
  #    declare it here, never set `done_state`.  Transitions whose
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
        body:
          - {from: wte_lm_head, to: janus_llama}
          - {from: janus_llama, to: wte_lm_head}
          - {from: wte_lm_head, to: end}
        transitions:
          - {condition: {type: module_signal, key: start_image_gen}, next_state: image_vq}
      image_vq:
        body:
          - {from: janus_llama, to: vqvae}
          - {from: vqvae,       to: janus_llama}
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
    Typed accessors (``module_config``, ``training_graph``) provide a stable
    surface for the runtime / visualisation tools.

    Tokenizers and processors are per-module assets saved alongside each
    module's checkpoint (e.g. ``janus_text_encoder/tokenizer.json``).
    """

    model_type = "omni"

    def __init__(
        self,
        modules: Optional[Dict[str, Dict]] = None,
        training_graph: Optional[List[Dict]] = None,
        generation_graph: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.modules: Dict[str, Dict] = modules or {}
        # Flat list of edges; each endpoint is a `module[.method]` string.
        self.training_graph: List[Dict] = training_graph or []

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
    def training_edges(self) -> List[Dict]:
        """Active training subset — the flat list of edge dicts."""
        return list(self.training_graph)

    @property
    def module_names(self) -> List[str]:
        return list(self.modules.keys())

    def has_training_graph(self) -> bool:
        return bool(self.training_graph)

    def has_generation_graph(self) -> bool:
        return self.generation_graph is not None

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        """Build an :class:`OmniConfig` from a YAML-shaped dict.

        Pops ``model_path`` (the split-checkpoint root from the launcher YAML —
        consumed by :func:`build_module_args`, not stored here).          Module blocks
        are kept verbatim; path resolution happens when merging each module onto
        the projected :class:`~veomni.arguments.arguments_types.VeOmniArguments`
        base (see :meth:`from_omni_args`).  Unknown top-level keys are dropped
        silently so the launcher YAML can carry training-only fields.
        """
        config_dict = dict(config_dict)
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**accepted, **kwargs)

    @classmethod
    def from_omni_args(
        cls,
        global_args: VeOmniArguments,
        model_path: Union[str, os.PathLike],
        modules: Union[str, os.PathLike, Dict[str, Any]],
        train_graph: Union[str, os.PathLike, List, None] = None,
        infer_modules: Union[str, os.PathLike, Dict[str, Any], None] = None,
        infer_graph: Union[str, os.PathLike, Dict, None] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "OmniConfig":
        """Load an :class:`OmniConfig` from the **omni split-file layout**.

        SeedOmni V2 keeps the per-module overrides and the graphs in separate
        files, referenced from
        :class:`~veomni.arguments.arguments_types_omni.OmniArguments`:

        * ``modules`` (``model.modules``) — per-module training overrides.
        * ``train_graph`` (``model.train_graph``) — the training DAG.
        * ``infer_modules`` (``infer.modules``) — per-module inference overrides
          (deep-merged onto the training modules; modules default to eager
          load).
        * ``infer_graph`` — the selected scenario's generation-graph file.
        * ``generation_kwargs`` — free-form generation knobs (``infer.generation_kwargs``).

        Each argument may be a path (loaded here) or an already-loaded object
        (e.g. a dict produced by CLI ``--model.modules.* ...`` overrides).
        Per-module blocks carry ``accelerator`` at the block top level (omni
        convention); it is lifted under ``train.accelerator`` before merging so
        the reused per-module ``BaseTrainer`` finds it where it expects.

        Per-module args are built by deep-merging each override onto
        ``asdict(global_args)`` (the projected base) — same mechanics as
        :func:`~veomni.arguments.parser.parse_args`.
        """
        from dataclasses import asdict

        modules_overrides = cls._load_modules(modules)
        modules_overrides = cls._lift_accelerator(modules_overrides)
        modules_overrides = cls._resolve_model_path(model_path, modules_overrides)

        base_cfg: Dict[str, Any] = {}
        if train_graph is not None:
            base_cfg["training_graph"] = cls._load_graph(train_graph, "training_graph") or []

        if infer_graph is not None:
            infer_overrides = cls._load_modules(infer_modules) if infer_modules else {}
            infer_overrides = cls._lift_accelerator(infer_overrides)
            if infer_overrides:
                infer_overrides = cls._resolve_model_path(model_path, infer_overrides)
            infer_overrides = cls._resolve_default_accelerator(modules_overrides, infer_overrides)
            modules_overrides = _deep_update(modules_overrides, infer_overrides)
            base_cfg["generation_graph"] = cls._load_graph(infer_graph, "generation_graph")
            base_cfg["generation_kwargs"] = dict(generation_kwargs or {})

        base_cfg["modules"] = {
            name: _deep_update(asdict(global_args), override) for name, override in modules_overrides.items()
        }
        return cls.from_dict(base_cfg)

    @staticmethod
    def _load_modules(modules: Union[str, os.PathLike, Dict[str, Any], None]) -> Dict[str, Any]:
        """Return a deep-copyable module-override dict from a path or inline dict."""
        from copy import deepcopy

        if modules is None:
            return {}
        if isinstance(modules, (str, os.PathLike)):
            import yaml

            with open(modules, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            return loaded
        return deepcopy(modules)

    @staticmethod
    def _load_graph(graph: Union[str, os.PathLike, Dict, List, None], key: str):
        """Load a graph file (or pass through an inline object).

        Graph files may wrap the payload under ``key`` (e.g. ``training_graph:``
        / ``generation_graph:``) or declare it bare at the top level; both are
        accepted.
        """
        if graph is None:
            return None
        if isinstance(graph, (str, os.PathLike)):
            import yaml

            with open(graph, encoding="utf-8") as f:
                graph = yaml.safe_load(f)
        if isinstance(graph, dict) and key in graph:
            return graph[key]
        return graph

    @staticmethod
    def _lift_accelerator(modules_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Move each module block's top-level ``accelerator`` under ``train``.

        v2 declares ``accelerator`` as a peer of ``model`` / ``train`` (both in
        the launcher and per-module blocks), but the reused per-module
        ``BaseTrainer`` reads ``args.train.accelerator``.  Lift it so the merge
        lands where the trainer expects.
        """
        if not modules_config:
            return {}
        for mod_cfg in modules_config.values():
            if isinstance(mod_cfg, dict) and "accelerator" in mod_cfg:
                acc = mod_cfg.pop("accelerator")
                train = mod_cfg.setdefault("train", {})
                train["accelerator"] = _deep_update(train.get("accelerator", {}) or {}, acc or {})
        return modules_config

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
