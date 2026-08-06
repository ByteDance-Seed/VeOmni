"""
OmniConfig — central configuration for OmniModel V2.

:class:`OmniConfig` describes **only** the composed model: module checkpoint layout,
training DAG, and every generation FSM.  It is a plain ``PretrainedConfig`` — it reads
and writes an on-disk checkpoint and knows nothing about the VeOmni launcher.

A checkpoint carries **all** the generation scenarios it was exported with, keyed by
``infer_type`` (``infer_gen`` / ``infer_edit`` / ``infer_und`` / …).  ``infer_type``
selects which one is active; :attr:`OmniConfig.generation_graph` returns it.  Scenario
choice is therefore a property of the config, not something baked in at YAML-load time.

Building one from the launcher split-file layout is the runtime layer's job:
:func:`~veomni.omni_arguments.arguments_types.resolve_omni_model` /
:func:`~veomni.omni_arguments.arguments_types.build_omni_model_runtime`
return :class:`~veomni.omni_arguments.arguments_types.OmniModelRuntimeArguments`
(the full launcher view), and its ``.to_hf_config()`` projects that onto this
checkpoint-shaped config.
Per-module **runtime** args (FSDP, optimizer, dataloader, …) live on
:class:`~veomni.omni_arguments.arguments_types.OmniModelRuntimeArguments.modules`.

Launcher inputs consumed by that builder (see :class:`~veomni.arguments.OmniArguments`):

  * ``model.model_path``   — split-checkpoint root folder (one subdir per module).
  * ``model.model_config.modules`` — per-module YAML overrides (``modules_train.yaml``).
  * ``model.model_config.train_graph`` — the training DAG (``graph_train.yaml``).
  * ``model.model_config.modules`` — per-module YAML (``modules_train.yaml`` for
    training; override with ``--model.model_config.modules …/modules_infer.yaml``
    at inference time).
  * ``model.model_config.infer_graph`` — scenario -> generation-graph file map.

Stored ``modules`` entries look like:

  janus_llama:
    subfolder: janus_llama
    model:
      ops_implementation:
        attn_implementation: flash_attention_2
      model_config: {freeze: false}

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

  # ── Inference FSMs, one per scenario.  Each state.body is a list of inline
  #    `{from, to}` edges (endpoints as `module[.method]` strings, bare module →
  #    `.generate`); node order is derived (unique endpoints in declaration
  #    order, excluding `end`).  The `done` state is auto-injected by the
  #    framework — never declare it here, never set `done_state`.  Transitions
  #    whose `next_state: done` land on the built-in terminal state which then
  #    triggers each active module's `finalize` hook (text decode /
  #    image save / etc).
  #    States carry no iteration budget — a state body iterates until one of
  #    its transitions fires, and modules decide when via signals (the AR
  #    loop) or after a single pass via `default` (bridge/leaf states).
  infer_type: infer_interleave
  generation_graphs:
    infer_interleave:
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
    infer_und:
      initial: text_ar
      states: {...}
"""

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import yaml
from transformers import PretrainedConfig


DEFAULT_TRAINING_GRAPH_FILE = "training_graph.yaml"
DEFAULT_GENERATION_GRAPH_FILE = "generation_graph.yaml"


def select_graph(
    graphs: Dict[str, Any],
    scenario: Optional[str],
    *,
    empty_hint: str = "",
    unknown_hint: str = "scenario",
) -> Any:
    """Pick the active graph out of a scenario map; unset ``scenario`` takes the first.

    Shared by :class:`OmniConfig` and
    :class:`~veomni.omni_arguments.arguments_types.OmniModelRuntimeArguments`
    so both resolve training and generation scenarios identically.
    """
    if not graphs:
        raise ValueError(f"No graph scenarios are declared. {empty_hint}".strip())
    if scenario is None:
        return next(iter(graphs.values()))
    if scenario not in graphs:
        known = ", ".join(graphs)
        raise KeyError(f"Unknown {unknown_hint} {scenario!r}; expected one of: {known}.")
    return graphs[scenario]


class OmniConfig(PretrainedConfig):
    """Configuration for OmniModel V2.

    All nested dicts are stored as plain Python dicts for JSON serialisability.
    Typed accessors (``module_model_config``, ``module_subfolder``,
    ``training_edges``) provide a stable surface for the runtime / visualisation
    tools.

    Tokenizers and processors are per-module assets saved alongside each
    module's checkpoint (e.g. ``janus_text_encoder/tokenizer.json``).
    """

    model_type = "omni"
    # ``modules`` / ``training_graph`` / ``generation_graphs`` are required, so
    # transformers must not probe defaults via a bare ``OmniConfig()`` — it does
    # that in ``to_diff_dict`` (and therefore ``__repr__``) unless told otherwise.
    has_no_defaults_at_init = True

    def __init__(
        self,
        modules: Dict[str, Dict],
        training_graph: List[Dict],
        generation_graphs: Dict[str, Dict],
        *,
        infer_type: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.modules = modules
        self.training_graph = training_graph
        self.generation_graphs = generation_graphs
        self.infer_type = infer_type
        self.generation_kwargs = generation_kwargs

        super().__init__(**kwargs)

    @property
    def training_edges(self) -> List[Dict]:
        """Active training subset — the flat list of edge dicts."""
        return list(self.training_graph)

    @property
    def infer_types(self) -> List[str]:
        """Declared generation scenarios, in declaration order."""
        return list(self.generation_graphs)

    @property
    def generation_graph(self) -> Dict:
        """The generation FSM selected by :attr:`infer_type`.

        A checkpoint carries every scenario it was exported with; ``infer_type``
        picks which one :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`
        binds. Unset means the first declared scenario.
        """
        return select_graph(
            self.generation_graphs,
            self.infer_type,
            empty_hint=(
                "Populate `generation_graphs` (via `model.model_config.infer_graph`, or by loading a "
                f"checkpoint whose `{DEFAULT_GENERATION_GRAPH_FILE}` sidecar has them)."
            ),
            unknown_hint="infer_type",
        )

    @generation_graph.setter
    def generation_graph(self, value: Dict) -> None:
        raise AttributeError(
            "`generation_graph` is read-only — it is whichever entry of `generation_graphs` "
            "`infer_type` names. Assign `generation_graphs` / `infer_type` instead."
        )

    @property
    def module_names(self) -> List[str]:
        # Config ``modules:`` declaration order (dict insertion order). This is the
        # canonical, FIXED order that drives serial CPU-preprocessor execution in
        # both training (SeedOmniCollator) and inference (OmniInferencer); declare
        # modules so any order-dependent prep (e.g. vision patchify before the text
        # chat-template) runs in the right sequence.
        return list(self.modules.keys())

    def module_subfolder(self, name: str) -> str:
        """Return the resolved on-disk path segment for ``name`` (may be absolute)."""
        entry = self.modules.get(name)
        if entry is None:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        if isinstance(entry, PretrainedConfig):
            return self.module_checkpoint_subfolder(name)
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            model_block = entry.get("model")
            if isinstance(model_block, dict):
                path = model_block.get("model_path") or model_block.get("weights_path")
                if path:
                    return path
            subfolder = entry.get("subfolder")
            if subfolder:
                return str(subfolder)
        return name

    def module_checkpoint_subfolder(self, name: str) -> str:
        """Relative subfolder under an omni checkpoint root for ``name``."""
        if name not in self.modules:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        return name

    def normalize_modules_for_hf_export(self) -> Dict[str, Dict[str, Any]]:
        """Slim ``modules`` block for HF ``config.json`` (subfolder + load options)."""
        from transformers import PretrainedConfig

        normalized: Dict[str, Dict[str, Any]] = {}
        for name in self.module_names:
            entry = self.modules.get(name)
            if isinstance(entry, PretrainedConfig):
                normalized[name] = {"subfolder": self.module_checkpoint_subfolder(name)}
                ops = self.module_ops_implementation(name)
                if ops:
                    normalized[name]["model"] = {"ops_implementation": deepcopy(ops)}
                continue
            slim: Dict[str, Any] = {"subfolder": self.module_checkpoint_subfolder(name)}
            model_block = self._module_export_model_block(name)
            if model_block:
                slim["model"] = model_block
            normalized[name] = slim
        return normalized

    def _module_export_model_block(self, name: str) -> Dict[str, Any]:
        model_block: Dict[str, Any] = {}
        ops = self.module_ops_implementation(name)
        if ops:
            model_block["ops_implementation"] = deepcopy(ops)
        model_config = self.module_model_config(name)
        if model_config:
            model_block["model_config"] = model_config
        return model_block

    def copy_for_hf_export(
        self,
        *,
        training_graph: Optional[List[Dict]] = None,
        generation_graphs: Optional[Dict[str, Dict]] = None,
    ) -> "OmniConfig":
        """Return a checkpoint-serializable copy with model-only module entries and graph sidecars."""
        export_dict = self.to_dict()
        export_dict["modules"] = self.normalize_modules_for_hf_export()
        export_dict["training_graph"] = list(training_graph if training_graph is not None else self.training_graph)
        export_dict["generation_graphs"] = deepcopy(
            generation_graphs if generation_graphs is not None else self.generation_graphs
        )
        export_dict["infer_type"] = self.infer_type
        accepted = {k: v for k, v in export_dict.items() if k in OmniConfig.__init__.__code__.co_varnames}
        return OmniConfig.from_dict(accepted)

    def module_model_config(self, name: str) -> Dict[str, Any]:
        """Per-module ``from_pretrained`` overrides stored in the config."""
        from transformers import PretrainedConfig

        entry = self.modules.get(name)
        if isinstance(entry, PretrainedConfig):
            return {}
        if not isinstance(entry, dict):
            return {}
        model_block = entry.get("model")
        if not isinstance(model_block, dict):
            return {}
        overrides = model_block.get("model_config")
        return dict(overrides or {})

    def module_ops_implementation(self, name: str) -> Dict[str, Any]:
        """Per-module VeOmni kernel options persisted in the checkpoint."""
        cached = getattr(self, "_module_load_options", {}).get(name, {})
        ops = cached.get("ops_implementation")
        if ops:
            return dict(ops)

        entry = self.modules.get(name)
        if not isinstance(entry, dict):
            return {}
        model_block = entry.get("model")
        if not isinstance(model_block, dict):
            return {}
        ops = model_block.get("ops_implementation")
        return dict(ops or {})

    def _stash_module_load_options(self) -> None:
        """Preserve ``model.ops_implementation`` before module configs are hydrated."""
        options: Dict[str, Dict[str, Any]] = {}
        for name in self.module_names:
            ops = self.module_ops_implementation(name)
            if ops:
                options[name] = {"ops_implementation": deepcopy(ops)}
        self._module_load_options = options

    def resolve_module_path(self, checkpoint_root: Optional[Union[str, os.PathLike]], name: str) -> str:
        """Resolve the on-disk path for module ``name`` under ``checkpoint_root``."""
        subfolder = self.module_subfolder(name)
        if os.path.isabs(subfolder):
            return subfolder
        if checkpoint_root is None:
            return subfolder
        return os.path.join(str(checkpoint_root), subfolder)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict; ``modules`` is slimmed to subfolder stubs."""
        modules = self.modules
        self.modules = self.normalize_modules_for_hf_export()
        try:
            return super().to_dict()
        finally:
            self.modules = modules

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load ``config.json``, graph sidecars, and per-module ``PretrainedConfig`` objects."""
        config = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        root = getattr(config, "_name_or_path", None) or str(pretrained_model_name_or_path)
        config._stash_module_load_options()
        config._hydrate_graphs_from_checkpoint(root)
        config._hydrate_modules_from_checkpoint(root)
        return config

    def _hydrate_modules_from_checkpoint(self, checkpoint_root: Union[str, os.PathLike]) -> None:
        """Load each module's ``config.json`` into :class:`PretrainedConfig` instances."""
        root = str(checkpoint_root)
        hydrated: Dict[str, Any] = {}
        for name in self.module_names:
            entry = self.modules.get(name)
            if isinstance(entry, PretrainedConfig):
                hydrated[name] = entry
                continue
            overrides = self.module_model_config(name)
            subfolder = self.module_checkpoint_subfolder(name)
            module_dir = os.path.join(root, subfolder)
            config_path = os.path.join(module_dir, "config.json")
            if not os.path.isfile(config_path):
                hydrated[name] = entry if entry is not None else {"subfolder": subfolder}
                continue
            from .modules import OMNI_MODEL_REGISTRY, read_hf_model_type

            model_type = read_hf_model_type(module_dir)
            if model_type in OMNI_MODEL_REGISTRY.valid_keys():
                mod_cls = OMNI_MODEL_REGISTRY[model_type]()
                hf_config = mod_cls.config_class.from_pretrained(module_dir)
            else:
                with open(config_path, encoding="utf-8") as f:
                    hf_config = PretrainedConfig.from_dict(json.load(f))
            if overrides:
                hf_config.update(deepcopy(overrides))
            hydrated[name] = hf_config
        self.modules = hydrated

    def _hydrate_graphs_from_checkpoint(self, checkpoint_root: Union[str, os.PathLike]) -> None:
        """Load ``training_graph`` and ``generation_graphs`` from their fixed-name YAML sidecars."""
        root = str(checkpoint_root)

        training_path = os.path.join(root, DEFAULT_TRAINING_GRAPH_FILE)
        if not os.path.isfile(training_path):
            raise FileNotFoundError(f"Omni checkpoint missing required graph sidecar: {training_path}")
        self.training_graph = self._read_graph_file(training_path, "training_graph")

        generation_path = os.path.join(root, DEFAULT_GENERATION_GRAPH_FILE)
        if not os.path.isfile(generation_path):
            raise FileNotFoundError(f"Omni checkpoint missing required graph sidecar: {generation_path}")
        self.generation_graphs = self._read_generation_graphs(generation_path)

    @staticmethod
    def _read_generation_graphs(path: str) -> Dict[str, Dict]:
        """Read the ``generation_graphs`` sidecar: ``{scenario_name: fsm_spec}``."""
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Malformed generation-graph sidecar {path}: expected a mapping, got {type(payload)}.")
        if "generation_graphs" not in payload:
            if "generation_graph" in payload:
                raise ValueError(
                    f"{path} uses the single-graph `generation_graph:` layout, which is no longer supported. "
                    "A checkpoint now stores every scenario under `generation_graphs: {<infer_type>: <fsm>}` "
                    "and selects one via `infer_type`. Re-export the checkpoint with "
                    "scripts/seed_omni/export_omni_checkpoint.py."
                )
            raise ValueError(f"Malformed generation-graph sidecar {path}: missing top-level `generation_graphs:` key.")
        graphs = payload["generation_graphs"]
        if not isinstance(graphs, dict):
            raise ValueError(f"Malformed generation-graph sidecar {path}: `generation_graphs` must be a mapping.")
        return graphs

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """Write ``config.json`` plus graph YAML sidecars for HF-style reload."""
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        for name, entry in self.modules.items():
            if isinstance(entry, PretrainedConfig):
                module_dir = os.path.join(save_directory, self.module_checkpoint_subfolder(name))
                os.makedirs(module_dir, exist_ok=True)
                entry.save_pretrained(module_dir)

        export_config = self.copy_for_hf_export()

        self._write_graph_file(
            os.path.join(save_directory, DEFAULT_TRAINING_GRAPH_FILE),
            "training_graph",
            export_config.training_graph,
        )
        self._write_graph_file(
            os.path.join(save_directory, DEFAULT_GENERATION_GRAPH_FILE),
            "generation_graphs",
            export_config.generation_graphs,
        )

        # The sidecar files above are the sole source of truth for both graphs —
        # `config.json` never carries them inline, so `from_pretrained` always
        # re-hydrates via `_hydrate_graphs_from_checkpoint` (fixed filenames).
        config_dict = export_config.to_dict()
        config_dict.pop("training_graph", None)
        config_dict.pop("generation_graphs", None)

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            # No sort_keys: ``modules`` declaration order is load-bearing (it drives
            # serial CPU-preprocessor execution — see :meth:`module_names`), and JSON
            # key sorting is recursive, so it would silently reorder the modules.
            json.dump(config_dict, f, indent=2)
            f.write("\n")

        from .utils.visualize import save_graph_mermaid_diagrams

        save_graph_mermaid_diagrams(export_config, save_directory)

        if push_to_hub:
            raise NotImplementedError("OmniConfig push_to_hub is not implemented yet.")

    @staticmethod
    def _write_graph_file(path: str, key: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({key: payload}, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _read_graph_file(path: str, key: str):
        """Read a graph sidecar written by :meth:`_write_graph_file`.

        Checkpoint sidecars are plain YAML documents; the launcher's ``inherit:``
        composition is a build-time feature and never appears here.
        """
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        return payload

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        """Build an :class:`OmniConfig` from a YAML-shaped dict.

        Module blocks are kept verbatim; the split-checkpoint root and per-module
        path resolution belong to the runtime layer
        (:class:`~veomni.omni_arguments.arguments_types.OmniModelRuntimeArguments`)
        and are not stored here. Unknown top-level keys are dropped silently so the
        launcher YAML can carry training-only fields.

        HF ``config.json`` stores graph payloads in YAML sidecars and leaves
        placeholder ``training_graph`` / omits ``generation_graphs`` until
        :meth:`from_pretrained` hydrates them.
        """
        config_dict = dict(config_dict)
        # Unknown keys are dropped below, so the retired singular field would go
        # silently inert — leaving a config whose FSM vanished. Reject it instead.
        if "generation_graph" in config_dict:
            raise ValueError(
                "`generation_graph` (a single FSM) is no longer a config field. A config now "
                "declares every scenario in `generation_graphs: {<infer_type>: <fsm>}` and "
                "selects one via `infer_type`."
            )
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        if "generation_graphs" not in accepted:
            accepted["generation_graphs"] = {}
        if "training_graph" not in accepted:
            accepted["training_graph"] = []
        if "modules" not in accepted:
            accepted["modules"] = {}
        return cls(**{**accepted, **kwargs})


__all__ = ["OmniConfig", "select_graph"]
