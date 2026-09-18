"""OmniConfig — HF ``PretrainedConfig`` for a composed :class:`OmniModel`.

This is the checkpoint-shaped composite only: a map of per-module descriptors,
the training DAG, and every generation FSM. Per-module subfolder / path /
``model_config`` conversion lives on
:class:`~veomni.models.seed_omni.modules.module_configuration_base.OmniModuleConfig`.
This class does not know about VeOmni runtime (ops, FSDP, freeze, launcher YAML).

A checkpoint stores every generation scenario under ``generation_graphs``, keyed
by ``infer_type``. :attr:`OmniConfig.generation_graph` is the active one.

VeOmni runtime fields live on
:class:`~veomni.arguments.omni_arguments_types.OmniModelRuntimeArguments` and
are projected onto this config with ``to_hf_config()``.
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
    """Pick the active graph out of a scenario map; unset ``scenario`` takes the first."""
    if not graphs:
        raise ValueError(f"No graph scenarios are declared. {empty_hint}".strip())
    if scenario is None:
        return next(iter(graphs.values()))
    if scenario not in graphs:
        known = ", ".join(graphs)
        raise KeyError(f"Unknown {unknown_hint} {scenario!r}; expected one of: {known}.")
    return graphs[scenario]


class OmniConfig(PretrainedConfig):
    """Checkpoint config for :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

    Nested dicts stay as plain Python dicts for JSON serialisability. Tokenizers
    and processors are per-module assets next to each module's weights.
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
    def infer_types(self) -> List[str]:
        """Declared generation scenarios, in declaration order."""
        return list(self.generation_graphs)

    @property
    def generation_graph(self) -> Dict:
        """The generation FSM selected by :attr:`infer_type` (first scenario if unset)."""
        return select_graph(
            self.generation_graphs,
            self.infer_type,
            empty_hint=f"Populate `generation_graphs` (or load a checkpoint with `{DEFAULT_GENERATION_GRAPH_FILE}`).",
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
        # canonical order for serial CPU-preprocessor execution.
        return list(self.modules.keys())

    def module(self, name: str):
        """HF descriptor for ``name`` (dict, hydrated family config, or path string)."""
        from .modules.module_configuration_base import OmniModuleConfig

        if name not in self.modules:
            raise KeyError(f"Module '{name}' not found in OmniConfig.modules")
        return OmniModuleConfig(name, self.modules[name])

    def module_subfolder(self, name: str) -> str:
        """On-disk path segment for ``name`` (relative subfolder, or an absolute load path)."""
        return self.module(name).subfolder

    def module_checkpoint_subfolder(self, name: str) -> str:
        """Relative subfolder under an omni checkpoint root for ``name``."""
        return self.module(name).checkpoint_subfolder

    def normalize_modules_for_hf_export(self) -> Dict[str, Dict[str, Any]]:
        """Slim ``modules`` block for HF ``config.json`` (subfolder + optional config overrides)."""
        return {name: self.module(name).to_export_dict() for name in self.module_names}

    def copy_for_hf_export(
        self,
        *,
        training_graph: Optional[List[Dict]] = None,
        generation_graphs: Optional[Dict[str, Dict]] = None,
    ) -> "OmniConfig":
        """Return a checkpoint-serializable copy (no in-memory load paths)."""
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
        """Per-module ``from_pretrained`` overrides stored on the omni config entry."""
        return self.module(name).model_config_overrides()

    def module_processor_config(self, name: str) -> Dict[str, Any]:
        """Per-module preprocessor ``from_pretrained`` kwargs."""
        return self.module(name).processor_config()

    def resolve_module_path(self, checkpoint_root: Optional[Union[str, os.PathLike]], name: str) -> str:
        """Resolve the on-disk path for module ``name`` under ``checkpoint_root``."""
        return self.module(name).resolve_path(checkpoint_root)

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
        """Load ``config.json``, graph YAML sidecars, and each module's typed config.

        Per-module ``config.json`` is resolved through ``OMNI_MODEL_REGISTRY`` —
        only registered omni modules can live in an :class:`OmniConfig`.
        """
        config = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        root = getattr(config, "_name_or_path", None) or str(pretrained_model_name_or_path)
        config._hydrate_graphs_from_checkpoint(root)
        config._hydrate_modules_from_checkpoint(root)
        return config

    def _hydrate_modules_from_checkpoint(self, checkpoint_root: Union[str, os.PathLike]) -> None:
        """Load each in-root module's ``config.json`` via the omni module registry."""
        self.modules = {name: self.module(name).hydrate(checkpoint_root) for name in self.module_names}

    def _hydrate_graphs_from_checkpoint(self, checkpoint_root: Union[str, os.PathLike]) -> None:
        """Load ``training_graph`` and ``generation_graphs`` from their YAML sidecars."""
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

        config_dict = export_config.to_dict()
        config_dict.pop("training_graph", None)
        config_dict.pop("generation_graphs", None)

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
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
        """Read a graph sidecar written by :meth:`_write_graph_file`."""
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        return payload

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        """Build an :class:`OmniConfig` from a dict.

        Unknown top-level keys are dropped. Graph payloads in an HF checkpoint
        live in YAML sidecars; :meth:`from_pretrained` hydrates them.
        """
        config_dict = dict(config_dict)
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        if "generation_graphs" not in accepted:
            accepted["generation_graphs"] = {}
        if "training_graph" not in accepted:
            accepted["training_graph"] = []
        if "modules" not in accepted:
            accepted["modules"] = {}
        return cls(**{**accepted, **kwargs})


__all__ = ["OmniConfig", "select_graph"]
