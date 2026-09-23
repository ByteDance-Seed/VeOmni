"""OmniConfig — HF ``PretrainedConfig`` for a composed :class:`OmniModel`.

This is the checkpoint-shaped composite only: a map of per-module entries, the
training DAG, and every generation FSM. ``_module_entries[name]`` is a dict
(``model_path``, ``ops_implementation``, ``model_config``, ``processor_config``).
:attr:`OmniConfig._module_configs` is each module's own loaded ``config.json``,
an :class:`OmniModuleConfig`.

Kernels are stated in two places, because a module and the model composing it
are different authorities: this config's root ``ops_implementation`` is the
composed model's default for every module, the entry's ``ops_implementation``
is its say over one module, and the module's own ``config.json`` carries what
it was exported with. Loading a module config merges them per field onto that
config, lowest to highest: the file, then this root, then the entry. A caller's
kwargs are not a fourth place: :meth:`OmniConfig.apply_omni_kwargs` merges
each into the field it overrides — root ``ops_implementation``, or a module
entry's ``model_path`` / ``ops_implementation`` / ``model_config`` /
``processor_config`` — before anything loads.

A checkpoint stores every training DAG under ``training_graphs`` (keyed by
``train_type``) and every generation FSM under ``generation_graphs`` (keyed by
``infer_type``). :attr:`OmniConfig.training_graph` / :attr:`OmniConfig.generation_graph`
are the active entries. A single-scenario file uses the name ``default``.
"""

import json
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Tuple, Union

import yaml
from transformers import PretrainedConfig


if TYPE_CHECKING:
    from .modules.module_configuration_base import OmniModuleConfig


DEFAULT_TRAINING_GRAPH_FILE = "training_graph.yaml"
DEFAULT_GENERATION_GRAPH_FILE = "generation_graph.yaml"
DEFAULT_GRAPH_SCENARIO = "default"


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


# Fields a caller may overwrite on one ``_module_entries[name]`` entry. Root overwrites
# are only kernels (``ops_implementation`` and each ``xxx_implementation``).
_MODULE_OVERRIDE_FIELDS = frozenset({"model_path", "ops_implementation", "model_config", "processor_config"})
_MODULE_DICT_FIELDS = frozenset({"model_config", "processor_config"})


def _ops_field_names() -> frozenset:
    """Field names selectable on :class:`~veomni.arguments.OpsImplementationConfig`."""
    from dataclasses import fields

    from ...arguments import OpsImplementationConfig

    return frozenset(f.name for f in fields(OpsImplementationConfig))


def pop_omni_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Lift :class:`OmniConfig` overwrites out of a load's kwargs.

    Transformers turns an unrecognized kwarg into a config attribute, and a
    dotted one into an attribute nothing will ever look up. These keys are this
    config's own overwrites — the root ``ops_implementation`` (and each
    ``xxx_implementation``) and every ``<module>.<field>`` — so they come out
    before ``from_pretrained`` runs. Whether a dotted key is legal is decided
    later, by :meth:`OmniConfig.apply_omni_kwargs`, once the checkpoint's
    module names exist.
    """
    ops_fields = _ops_field_names()
    return {
        key: kwargs.pop(key) for key in list(kwargs) if key in ops_fields or key == "ops_implementation" or "." in key
    }


def _split_omni_kwargs(
    kwargs: Dict[str, Any],
    module_names: Collection[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Split popped overwrites into root kernels, per-module entry fields, and the rest.

    Root, merged into :attr:`OmniConfig.ops_implementation`:
    ``xxx_implementation=`` / ``ops_implementation={...}``.

    Per module, merged into that entry:

    * ``<module>.xxx_implementation=`` / ``<module>.ops_implementation={...}``
    * ``<module>.model_path=`` — replaces the path
    * ``<module>.model_config={...}`` / ``<module>.processor_config={...}`` —
      dicts, merged per field

    Returns ``(root_ops, per_module, remaining)``. ``remaining`` is what the
    caller meant for HF (``torch_dtype``, ``device_map``, …). A block states
    choices rather than replacing the layer it lands on.
    """
    ops_fields = _ops_field_names()
    root_ops: Dict[str, Any] = {}
    per_module: Dict[str, Dict[str, Any]] = {}
    remaining: Dict[str, Any] = {}

    def ops_selection(key: str, field: str, value: Any) -> Dict[str, Any]:
        if field == "ops_implementation":
            if not isinstance(value, dict):
                raise ValueError(
                    f"Load kwarg {key!r} must be a dict of kernel selections, got {type(value).__name__}."
                )
            unknown = sorted(set(value) - ops_fields)
            if unknown:
                raise ValueError(f"Load kwarg {key!r} has no such kernel fields: {', '.join(unknown)}.")
            return dict(value)
        return {field: value}

    for key, value in kwargs.items():
        if key in ops_fields or key == "ops_implementation":
            root_ops.update(ops_selection(key, key, value))
            continue
        if "." not in key:
            remaining[key] = value
            continue
        name, _, field = key.partition(".")
        if name not in module_names:
            known = ", ".join(sorted(module_names)) or "none"
            raise ValueError(
                f"Load kwarg {key!r} names module {name!r}, which this checkpoint does not "
                f"declare. Known modules: {known}."
            )
        overrides = per_module.setdefault(name, {})
        if field in ops_fields or field == "ops_implementation":
            overrides.setdefault("ops_implementation", {}).update(ops_selection(key, field, value))
        elif field == "model_path":
            if not isinstance(value, str) or not value:
                raise ValueError(f"Load kwarg {key!r} must be a non-empty path string, got {value!r}.")
            overrides["model_path"] = value
        elif field in _MODULE_DICT_FIELDS:
            if not isinstance(value, dict):
                raise ValueError(f"Load kwarg {key!r} must be a dict, got {type(value).__name__}.")
            overrides.setdefault(field, {}).update(value)
        else:
            overridable = ", ".join(sorted(_MODULE_OVERRIDE_FIELDS | ops_fields))
            raise ValueError(
                f"Load kwarg {key!r} cannot override {field!r}. Overridable module fields: {overridable}."
            )

    return root_ops, per_module, remaining


class OmniConfig(PretrainedConfig):
    """Checkpoint config for :class:`~veomni.models.seed_omni.modeling_omni.OmniModel`.

    ``_module_entries`` is the dict of checkpoint entries. A loaded module config is
    :attr:`_module_configs`. Tokenizers and processors are per-module assets
    next to each module's weights.
    """

    model_type = "omni"
    # ``_module_entries`` / ``training_graphs`` / ``generation_graphs`` are required, so
    # transformers must not probe defaults via a bare ``OmniConfig()`` — it does
    # that in ``to_diff_dict`` (and therefore ``__repr__``) unless told otherwise.
    has_no_defaults_at_init = True

    def __init__(
        self,
        _module_entries: Dict[str, Dict[str, Any]],
        training_graphs: Dict[str, List[Dict]],
        generation_graphs: Dict[str, Dict],
        *,
        train_type: Optional[str] = None,
        infer_type: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        ops_implementation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._module_entries = _module_entries
        self.training_graphs = training_graphs
        self.generation_graphs = generation_graphs
        self.train_type = train_type
        self.infer_type = infer_type
        self.generation_kwargs = generation_kwargs
        # Model-level kernels: this composed model's defaults for every module
        # it holds. Sparse — only what was explicitly selected. A loaded module
        # config ranks its own file under this, then under the module entry.
        self.ops_implementation: Dict[str, Any] = dict(ops_implementation or {})
        # Each module's own typed config, built by :meth:`from_pretrained`.
        # In-memory only: the modules are described by their own files, and a
        # copy here would be a second place to keep in sync. Dropped by
        # :meth:`to_dict`.
        self._module_configs: Dict[str, OmniModuleConfig] = {}

        super().__init__(**kwargs)

    @property
    def train_types(self) -> List[str]:
        """Declared training scenarios, in declaration order."""
        return list(self.training_graphs)

    @property
    def infer_types(self) -> List[str]:
        """Declared generation scenarios, in declaration order."""
        return list(self.generation_graphs)

    @property
    def training_graph(self) -> List[Dict]:
        """The training DAG selected by :attr:`train_type` (first scenario if unset)."""
        if not self.training_graphs:
            return []
        return select_graph(
            self.training_graphs,
            self.train_type,
            empty_hint=f"Populate `training_graphs` (or load a checkpoint with `{DEFAULT_TRAINING_GRAPH_FILE}`).",
            unknown_hint="train_type",
        )

    @training_graph.setter
    def training_graph(self, value: List[Dict]) -> None:
        raise AttributeError(
            "`training_graph` is read-only — it is whichever entry of `training_graphs` "
            "`train_type` names. Assign `training_graphs` / `train_type` instead."
        )

    @property
    def generation_graph(self) -> Dict:
        """The generation FSM selected by :attr:`infer_type` (first scenario if unset).

        An empty ``generation_graphs`` returns ``{}``.
        """
        if not self.generation_graphs:
            return {}
        return select_graph(
            self.generation_graphs,
            self.infer_type,
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
        # ``_module_entries`` declaration order (dict insertion order). This is the
        # canonical order for serial CPU-preprocessor execution.
        return list(self._module_entries.keys())

    def copy_for_hf_export(
        self,
        *,
        training_graphs: Optional[Dict[str, List[Dict]]] = None,
        generation_graphs: Optional[Dict[str, Dict]] = None,
    ) -> "OmniConfig":
        """Return a checkpoint-serializable copy (no in-memory load paths)."""
        export_dict = self.to_dict()
        export_dict["training_graphs"] = deepcopy(
            training_graphs if training_graphs is not None else self.training_graphs
        )
        export_dict["generation_graphs"] = deepcopy(
            generation_graphs if generation_graphs is not None else self.generation_graphs
        )
        export_dict["train_type"] = self.train_type
        export_dict["infer_type"] = self.infer_type
        accepted = {k: v for k, v in export_dict.items() if k in OmniConfig.__init__.__code__.co_varnames}
        return OmniConfig.from_dict(accepted)

    def apply_omni_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Absorb a caller's overwrites into this config.

        Returns the kwargs that were not overwrites, for the caller to forward
        to HF. Root ``xxx_implementation=`` / ``ops_implementation={...}`` merge
        into :attr:`ops_implementation`. A ``<module>.<field>=`` merges into
        that module's entry: ``model_path``, ``ops_implementation``,
        ``model_config``, ``processor_config``. Dict fields merge per key, so a
        caller states choices rather than replacing the whole block;
        ``model_path`` is a single path and replaces the entry's.

        Resolution happens here, once, rather than being threaded down to the
        loader: afterwards this config alone describes the run, and a save
        reproduces it — the same treatment ``infer_type`` and
        ``training_graphs`` already get as load kwargs.

        A ``model_path`` or ``model_config`` overwrite drops that module's
        cached config: both change which file a later load reads. A kernel
        overwrite is written onto a cached module config when one is already
        loaded, in the same order as load — the config's current ops, then
        this root, then the module entry.
        """
        root_ops, per_module, remaining = _split_omni_kwargs(kwargs, self.module_names)
        self.ops_implementation = {**self.ops_implementation, **root_ops}
        for name, overrides in per_module.items():
            entry = self._module_entries[name]
            if "ops_implementation" in overrides:
                entry["ops_implementation"] = {
                    **dict(entry.get("ops_implementation") or {}),
                    **overrides["ops_implementation"],
                }
            if "model_config" in overrides:
                entry["model_config"] = {**dict(entry.get("model_config") or {}), **overrides["model_config"]}
            if "processor_config" in overrides:
                entry["processor_config"] = {
                    **dict(entry.get("processor_config") or {}),
                    **overrides["processor_config"],
                }
            if "model_path" in overrides:
                entry["model_path"] = overrides["model_path"]
            if "model_path" in overrides or "model_config" in overrides:
                self._module_configs.pop(name, None)
        # Root kernels apply to every loaded module. An entry's kernels apply to that one.
        ops_names = (
            self.module_names
            if root_ops
            else [name for name, overrides in per_module.items() if "ops_implementation" in overrides]
        )
        for name in ops_names:
            loaded = self._module_configs.get(name)
            if loaded is None:
                continue
            loaded.ops_implementation = {
                **dict(loaded.ops_implementation or {}),
                **self.ops_implementation,
                **dict(self._module_entries[name].get("ops_implementation") or {}),
            }
            attn_implementation = loaded.ops_implementation.get("attn_implementation")
            if attn_implementation is not None:
                loaded._attn_implementation = attn_implementation
        return remaining

    @property
    def checkpoint_root(self) -> Optional[str]:
        """Directory this config was loaded from, when it was loaded from one."""
        return getattr(self, "_name_or_path", None) or None

    def resolve_module_path(self, checkpoint_root: Optional[Union[str, os.PathLike]], name: str) -> str:
        """Resolve the on-disk path for module ``name`` under ``checkpoint_root``."""
        from .modules.module_configuration_base import OmniModuleConfig

        return OmniModuleConfig.resolve_path(checkpoint_root, name, self._module_entries[name].get("model_path"))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict. ``model_path`` is the module name."""
        for name in self.module_names:
            self._module_entries[name]["model_path"] = name
        output = super().to_dict()
        output.pop("_module_configs", None)
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load ``config.json``, graph YAML sidecars, and each module's typed config.

        Per-module ``config.json`` is resolved through ``OMNI_MODEL_REGISTRY`` —
        only registered omni modules can live in an :class:`OmniConfig`.

        Caller overwrites may be passed here too, and mean the same as they do
        at :meth:`~veomni.models.seed_omni.OmniModel.from_pretrained`: root
        ``ops_implementation`` and a module's ``model_path``,
        ``ops_implementation``, ``model_config``, ``processor_config``. The
        config that comes back is the resolved one, and handing it to
        ``OmniModel.from_pretrained(path, config=...)`` loads the same model.
        """
        from .modules.module_configuration_base import OmniModuleConfig

        # Out before transformers sees them: it would turn each one into a
        # config attribute. Judged against this config's entries once it exists.
        omni_kwargs = pop_omni_kwargs(kwargs)
        # A caller may pass a graph explicitly to run a checkpoint under a graph
        # it was not exported with (a launcher YAML overriding the sidecar), so
        # the sidecar read skips whatever the caller already supplied.
        config = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        root = config.checkpoint_root or str(pretrained_model_name_or_path)
        config.apply_omni_kwargs(omni_kwargs)
        config._load_graphs_from_pretrained(root)
        for name in config.module_names:
            entry = config._module_entries[name]
            module_name_or_path = OmniModuleConfig.resolve_path(root, name, entry.get("model_path"))
            # Ops rank: the file's own, then this config's base, then this entry.
            config._module_configs[name] = OmniModuleConfig.from_pretrained(
                module_name_or_path,
                model_config=entry.get("model_config"),
                processor_config=entry.get("processor_config"),
                ops_implementation=entry.get("ops_implementation"),
                base_ops_implementation=config.ops_implementation,
            )
        return config

    def _load_graphs_from_pretrained(self, checkpoint_root: Union[str, os.PathLike]) -> None:
        """Load ``training_graphs`` and ``generation_graphs`` from YAML sidecars when present.

        Attributes named in ``skip`` are left alone, because a sidecar is the
        checkpoint's default rather than an override of an explicit caller.

        A module-only split checkpoint has neither file yet; train / generate
        then fail until a sidecar is added or the caller passes an override.
        """
        root = str(checkpoint_root)
        if not self.training_graphs:
            self.training_graphs = self._read_graph_sidecar(root, DEFAULT_TRAINING_GRAPH_FILE, list)
        if not self.generation_graphs:
            self.generation_graphs = self._read_graph_sidecar(root, DEFAULT_GENERATION_GRAPH_FILE, dict)

    @classmethod
    def _read_graph_sidecar(cls, root: str, filename: str, payload_type: type) -> Dict:
        """A sidecar's scenario map, or ``{}`` when the checkpoint has no such file."""
        path = os.path.join(root, filename)
        return cls._read_graph_file(path, payload_type) if os.path.isfile(path) else {}

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """Write ``config.json`` plus graph YAML sidecars for HF-style reload."""
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        for name in self.module_names:
            hf_config = self._module_configs[name]
            module_dir = os.path.join(save_directory, name)
            os.makedirs(module_dir, exist_ok=True)
            hf_config.save_pretrained(module_dir)

        export_config = self.copy_for_hf_export()

        if export_config.training_graphs:
            self._write_graph_file(
                os.path.join(save_directory, DEFAULT_TRAINING_GRAPH_FILE),
                export_config.training_graphs,
            )
        if export_config.generation_graphs:
            self._write_graph_file(
                os.path.join(save_directory, DEFAULT_GENERATION_GRAPH_FILE),
                export_config.generation_graphs,
            )

        config_dict = export_config.to_dict()
        config_dict.pop("training_graphs", None)
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
    def _write_graph_file(path: str, payload: Any) -> None:
        """Dump ``payload`` as the YAML document. Filename identifies train vs generation."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _read_graph_file(path: str, value_type: type) -> Dict[str, Any]:
        """Load a graph sidecar: ``{scenario_name: graph}``.

        ``value_type`` is the per-scenario payload: ``list`` (training DAG) or
        ``dict`` (generation FSM). A bare payload (one DAG / one FSM) is stored
        under :data:`DEFAULT_GRAPH_SCENARIO`.
        """
        with open(path, encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, list) and value_type is list:
            payload = {DEFAULT_GRAPH_SCENARIO: payload}
        elif isinstance(payload, dict) and value_type is dict and {"initial", "states"} <= set(payload):
            payload = {DEFAULT_GRAPH_SCENARIO: payload}
        if not isinstance(payload, dict):
            raise ValueError(
                f"Malformed graph sidecar {path}: expected a mapping of scenario name to graph, "
                f"got {type(payload).__name__}."
            )
        for name, graph in payload.items():
            if not isinstance(graph, value_type):
                raise ValueError(
                    f"Malformed graph sidecar {path}: scenario {name!r} expected "
                    f"{value_type.__name__}, got {type(graph).__name__}."
                )
        return payload

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "OmniConfig":
        """Build an :class:`OmniConfig` from a dict.

        Unknown top-level keys are dropped. Graph payloads in an HF checkpoint
        live in YAML sidecars; :meth:`from_pretrained` loads them.
        """
        config_dict = dict(config_dict)
        accepted = {k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames}
        if "generation_graphs" not in accepted:
            accepted["generation_graphs"] = {}
        if "training_graphs" not in accepted:
            accepted["training_graphs"] = {}
        if "_module_entries" not in accepted:
            accepted["_module_entries"] = {}
        return cls(**{**accepted, **kwargs})


__all__ = ["OmniConfig"]
