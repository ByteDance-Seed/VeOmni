# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Run a SeedOmni model straight off its upstream HuggingFace checkpoint.

A family registers an :class:`OmniHFLayout` under its upstream ``model_type``
(``OMNI_HF_LAYOUT_REGISTRY["qwen3"]``). The layout says how the monolithic
checkpoint splits into modules: each module's config built from the HF
config, its assets (tokenizer / processor), and a reversible key-prefix map
from HF weight names to the module's own.

Pointing ``model.model_path`` (or :meth:`OmniModel.from_pretrained`) at such a
checkpoint then works without an offline convert:

* :func:`resolve_omni_checkpoint_root` writes a **weight-free view** of the split
  checkpoint into a per-process temp directory — module configs, assets, root
  ``config.json``, graph sidecars, and an :data:`HF_SOURCE_FILE` naming the
  source. Everything that reads a split checkpoint's configs or assets reads
  the view unchanged.
* Weights are read from the source by :class:`HFSourceKeyConverter`, which
  renames each HF key onto the module that owns it (and lets keys of sibling
  modules pass unread), then hands the result to the module's own converter.
* :func:`save_hf_source_checkpoint` reverses the rename and writes one
  checkpoint in the source's own layout: same keys, dtypes and shards, plus the
  source's non-weight files.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from ....utils import helper
from ....utils.registry import Registry
from ...checkpoint_tensor_loading import (
    ConvertedCheckpointTensor,
    checkpoint_converter_fused_expert_target,
    checkpoint_converter_is_dim0_zero_pad,
    checkpoint_converter_record_skip_without_loading,
    checkpoint_converter_should_skip_without_loading,
    maybe_convert_checkpoint_tensor,
    shard_index_from_filename,
)
from .convert_registry import GraphFiles, load_family_graphs


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...checkpoint_tensor_loading import CheckpointTensorConverter
    from ..modules.module_configuration_base import OmniModuleConfig


logger = helper.create_logger(__name__)

OMNI_HF_LAYOUT_REGISTRY = Registry("OmniHFLayout")

HF_SOURCE_FILE = "hf_source.json"
"""Sidecar in a weight-free view root naming the HF checkpoint its weights come from."""

_SAFE_WEIGHTS_NAME = "model.safetensors"
# Per-rank files ``HuggingFaceStorageWriter`` stages before consolidating into the save root.
_DCP_STAGING_DIRNAME = "sharded"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
_WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")


@dataclass(frozen=True)
class OmniHFModuleLayout:
    """How one module is cut out of the upstream checkpoint.

    ``key_prefixes`` maps HF weight-name prefixes onto this module's own, e.g.
    ``(("model.embed_tokens.", "embed_tokens."),)``. A source key belongs to
    the module whose source prefix is the **longest** match across the whole
    layout, so a backbone can claim ``"model."`` while a sibling claims
    ``"model.embed_tokens."``.

    ``build_config`` turns the upstream HF config into this module's
    :class:`OmniModuleConfig`. ``build_assets`` (optional) loads the module's
    HF assets from the checkpoint directory, keyed like
    :func:`~veomni.models.seed_omni.utils.convert_registry.attach_module_assets`
    (``tokenizer`` / ``processor`` / ``image_processor`` / ``video_processor``).
    """

    key_prefixes: tuple[tuple[str, str], ...]
    build_config: Callable[[PretrainedConfig], OmniModuleConfig]
    build_assets: Callable[[str], dict[str, Any]] | None = None


@dataclass(frozen=True)
class OmniHFLayout:
    """A family's split of its upstream HF checkpoint into SeedOmni modules.

    ``graph_dir`` / ``training_graphs`` / ``generation_graphs`` name the
    family's default graph YAML (see :func:`load_family_graphs`); a launcher
    graph overrides them. ``infer_type`` selects the default generation scenario.

    ``tied_source_keys`` maps a source key to the source key it duplicates
    when the model ties them (a checkpoint may store ``lm_head.weight`` beside
    ``model.embed_tokens.weight`` although the config ties the two). When no
    live module produces the duplicate, the export writes the original's
    current value under it instead of the stale source copy.
    """

    modules: Mapping[str, OmniHFModuleLayout]
    tied_source_keys: Mapping[str, str] = field(default_factory=dict)
    graph_dir: str | None = None
    training_graphs: GraphFiles | None = None
    generation_graphs: GraphFiles | None = None
    infer_type: str | None = None
    load_hf_config: Callable[[str], PretrainedConfig] | None = field(default=None, compare=False)

    @cached_property
    def _rules(self) -> list[tuple[str, str, str]]:
        """``(source_prefix, module, module_prefix)``, longest source prefix first."""
        rules = [
            (source, name, target) for name, module in self.modules.items() for source, target in module.key_prefixes
        ]
        sources = [source for source, _, _ in rules]
        duplicated = sorted({source for source in sources if sources.count(source) > 1})
        if duplicated:
            raise ValueError(f"OmniHFLayout claims source prefixes {duplicated} for more than one module.")
        return sorted(rules, key=lambda rule: len(rule[0]), reverse=True)

    def route(self, source_key: str) -> tuple[str, str] | None:
        """``(module, module_key)`` owning ``source_key``, or ``None`` when no module claims it."""
        for source, name, target in self._rules:
            if source_key.startswith(source):
                return name, target + source_key[len(source) :]
        return None

    def source_key(self, module: str, module_key: str) -> str | None:
        """The HF key ``module_key`` of ``module`` was read from, or ``None`` when it maps to none.

        The reverse of :meth:`route`: the result must route back to the same
        module and key, which rejects a module key that only looks like it
        belongs to a prefix another module claims more specifically.
        """
        candidates = [(source, target) for source, name, target in self._rules if name == module]
        for source, target in sorted(candidates, key=lambda rule: len(rule[1]), reverse=True):
            if module_key.startswith(target):
                key = source + module_key[len(target) :]
                return key if self.route(key) == (module, module_key) else None
        return None

    def read_hf_config(self, source_path: str) -> PretrainedConfig:
        if self.load_hf_config is not None:
            return self.load_hf_config(source_path)
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(source_path)

    def build_module_configs(self, source_path: str) -> dict[str, OmniModuleConfig]:
        hf_config = self.read_hf_config(source_path)
        return {name: module.build_config(hf_config) for name, module in self.modules.items()}

    def build_module_assets(self, source_path: str) -> dict[str, dict[str, Any]]:
        return {
            name: module.build_assets(source_path) if module.build_assets is not None else {}
            for name, module in self.modules.items()
        }

    def load_graphs(
        self,
        *,
        training_graph: str | None = None,
        generation_graph: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The family's default graphs, or ``({}, {})`` when none are declared or found.

        Run outside a VeOmni checkout, the default YAML is not on disk; the view
        then carries no graphs and the launcher has to supply them.
        """
        if self.graph_dir is None and training_graph is None and generation_graph is None:
            return {}, {}
        try:
            return load_family_graphs(
                self.graph_dir or "",
                training=self.training_graphs or {},
                generation=self.generation_graphs or {},
                training_graph=training_graph,
                generation_graph=generation_graph,
            )
        except FileNotFoundError as e:
            logger.warning_rank0(f"OmniHFLayout: no default graphs ({e}); the launcher must supply them.")
            return {}, {}


@dataclass(frozen=True)
class HFSource:
    """The upstream HF checkpoint a weight-free view reads its weights from."""

    path: str
    model_type: str

    @property
    def layout(self) -> OmniHFLayout:
        return get_hf_layout(self.model_type)


def get_hf_layout(model_type: str) -> OmniHFLayout:
    import veomni.models.seed_omni.modules  # noqa: F401  (registers every family's layout)

    return OMNI_HF_LAYOUT_REGISTRY[model_type]()


def has_hf_layout(model_type: str | None) -> bool:
    import veomni.models.seed_omni.modules  # noqa: F401

    return model_type is not None and model_type in OMNI_HF_LAYOUT_REGISTRY.valid_keys()


def read_hf_source(checkpoint_root: str | os.PathLike | None) -> HFSource | None:
    """The :class:`HFSource` a view root records, or ``None`` for a regular split checkpoint."""
    if not checkpoint_root:
        return None
    path = os.path.join(str(checkpoint_root), HF_SOURCE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return HFSource(path=payload["path"], model_type=payload["model_type"])


def _read_root_model_type(path: str) -> str | None:
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f).get("model_type")


_VIEW_CACHE: dict[str, str] = {}


def resolve_omni_checkpoint_root(path: str | os.PathLike | None) -> str | os.PathLike | None:
    """Return ``path`` itself for a split checkpoint, or a weight-free view of an HF one.

    Detection is by the root ``config.json``'s ``model_type``: ``omni`` (or no
    ``config.json``) is a split checkpoint and passes through; a type with a
    registered :class:`OmniHFLayout` gets a view; any other type is an error,
    since reading it as an omni root would only fail later and more obscurely.
    The view is built once per process and source.
    """
    if not path:
        return path
    if os.path.isfile(os.path.join(path, HF_SOURCE_FILE)):
        return path
    model_type = _read_root_model_type(str(path))
    if model_type is None or model_type == "omni":
        return path
    if not has_hf_layout(model_type):
        raise ValueError(
            f"{path} is a HuggingFace `{model_type}` checkpoint, and no SeedOmni HF layout is registered "
            f"for `{model_type}`. Convert it with scripts/seed_omni/convert_model.py, or register an "
            "OmniHFLayout for the family."
        )
    key = os.path.realpath(path)
    view = _VIEW_CACHE.get(key)
    if view is None or not os.path.isfile(os.path.join(view, HF_SOURCE_FILE)):
        view = _write_hf_view(HFSource(path=key, model_type=model_type))
        _VIEW_CACHE[key] = view
    return view


def _write_hf_view(source: HFSource) -> str:
    """Write the weight-free split view of ``source`` into a fresh temp directory."""
    from ..configuration_omni import OmniConfig

    layout = source.layout
    view = tempfile.mkdtemp(prefix=f"veomni_omni_{source.model_type}_")
    atexit.register(shutil.rmtree, view, True)

    module_configs = layout.build_module_configs(source.path)
    training_graphs, generation_graphs = layout.load_graphs()
    infer_type = layout.infer_type if layout.infer_type in generation_graphs else None
    config = OmniConfig(
        _module_entries={name: {"model_path": name} for name in module_configs},
        training_graphs=training_graphs,
        generation_graphs=generation_graphs,
        infer_type=infer_type,
    )
    config._module_configs = dict(module_configs)
    config.save_pretrained(view)
    for name, assets in layout.build_module_assets(source.path).items():
        for asset in assets.values():
            if asset is not None and hasattr(asset, "save_pretrained"):
                asset.save_pretrained(os.path.join(view, name))
    with open(os.path.join(view, HF_SOURCE_FILE), "w", encoding="utf-8") as f:
        json.dump({"path": source.path, "model_type": source.model_type}, f, indent=2)
        f.write("\n")
    logger.info_rank0(f"OmniHFLayout: `{source.model_type}` checkpoint {source.path} → weight-free view {view}.")
    return view


class HFSourceKeyConverter:
    """Read one module's weights straight from the upstream HF checkpoint.

    Claims every source key: a key the layout routes to this module is renamed
    and handed to the module's own converter (``inner``); any other key belongs
    to a sibling module (or to none) and is skipped without being read. So is a
    ``tied_source_keys`` duplicate whose target the module does not have (it
    ties that weight instead of storing it).
    """

    def __init__(
        self,
        layout: OmniHFLayout,
        module_name: str,
        inner: CheckpointTensorConverter | None = None,
        absent_tied_keys: frozenset[str] = frozenset(),
    ) -> None:
        self.layout = layout
        self.module_name = module_name
        self.inner = inner
        self.absent_tied_keys = absent_tied_keys
        self.skipped = 0

    def _target(self, name: str) -> str | None:
        if name in self.absent_tied_keys:
            return None
        routed = self.layout.route(name)
        if routed is None or routed[0] != self.module_name:
            return None
        return routed[1]

    def can_handle(self, name: str) -> bool:
        return True

    def should_skip_without_loading(self, name: str) -> bool:
        target = self._target(name)
        return target is None or checkpoint_converter_should_skip_without_loading(self.inner, target)

    def record_skip_without_loading(self, name: str) -> None:
        target = self._target(name)
        if target is None:
            self.skipped += 1
        else:
            checkpoint_converter_record_skip_without_loading(self.inner, target)

    def convert(self, name: str, tensor: torch.Tensor) -> ConvertedCheckpointTensor | None:
        target = self._target(name)
        if target is None:
            self.skipped += 1
            return None
        return maybe_convert_checkpoint_tensor(target, tensor, self.inner)

    def finalize(self) -> list[ConvertedCheckpointTensor]:
        return list(self.inner.finalize()) if self.inner is not None else []

    def is_dim0_zero_pad(self, name: str) -> bool:
        target = self._target(name)
        return target is not None and checkpoint_converter_is_dim0_zero_pad(self.inner, target)

    def fused_expert_target(self, name: str) -> tuple[str, int] | None:
        target = self._target(name)
        return None if target is None else checkpoint_converter_fused_expert_target(self.inner, target)

    def for_expert_range(self, start: int, num_local: int) -> HFSourceKeyConverter:
        return HFSourceKeyConverter(
            self.layout,
            self.module_name,
            self.inner.for_expert_range(start, num_local),
            self.absent_tied_keys,
        )


def attach_hf_source_converter(model: torch.nn.Module, source: HFSource, module_name: str) -> None:
    """Make every VeOmni weight loader read ``model`` (module ``module_name``) from ``source``.

    Installed on the instance, ahead of the class's own converter factory,
    which then sees the module's key names exactly as it would reading a split
    checkpoint.
    """
    layout = source.layout
    if module_name not in layout.modules:
        raise KeyError(
            f"Module {module_name!r} is not part of the `{source.model_type}` HF layout "
            f"(modules: {sorted(layout.modules)}). Under an HF model_path every module must come from the layout."
        )
    inner_factory = getattr(type(model), "_create_checkpoint_tensor_converter", None)
    # Read off the bare module now: the factory may later be handed a LoRA
    # wrapper, whose names carry a ``base_model.model.`` prefix.
    module_keys = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    module_keys.update(name for name, _ in model.named_buffers(remove_duplicate=False))
    absent_tied = frozenset(
        key
        for key in layout.tied_source_keys
        if (routed := layout.route(key)) is not None and routed[0] == module_name and routed[1] not in module_keys
    )

    def factory(live_model: torch.nn.Module) -> HFSourceKeyConverter:
        inner = inner_factory(live_model) if inner_factory is not None else None
        return HFSourceKeyConverter(layout, module_name, inner, absent_tied)

    model._create_checkpoint_tensor_converter = factory


def load_module_from_hf_source(
    module_cls: type,
    config: OmniModuleConfig,
    source: HFSource,
    module_name: str,
    *,
    device: str | torch.device,
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
) -> torch.nn.Module:
    """Build module ``module_name`` from ``config`` and load its weights from ``source`` onto ``device``.

    The unwrapped counterpart of ``from_pretrained`` for a module whose weights
    live in an upstream HF checkpoint (eager inference, bare
    :meth:`OmniModel.from_pretrained`). Assets are not bound here.
    """
    from ...module_utils import init_empty_weights, load_model_weights

    init_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        init_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        init_kwargs["attn_implementation"] = attn_implementation
    with init_empty_weights():
        model = module_cls._from_config(config, **init_kwargs)
    attach_hf_source_converter(model, source, module_name)
    load_model_weights(model, source.path, init_device=str(device))
    return model


def convert_with_hf_layout(
    model_path: str,
    *,
    training_graph: str | None = None,
    generation_graph: str | None = None,
) -> dict[str, Any]:
    """Offline split through the family's :class:`OmniHFLayout` — the same key map the runtime load uses.

    Returns the kwargs :func:`~veomni.models.seed_omni.utils.convert_registry.convert_checkpoint`
    writes out. Modules load on CPU in the checkpoint's own ``dtype``.
    """
    from ..modules import OMNI_MODEL_REGISTRY
    from .convert_registry import attach_module_assets

    model_type = _read_root_model_type(model_path)
    source = HFSource(path=os.path.realpath(model_path), model_type=model_type)
    layout = source.layout
    hf_config = layout.read_hf_config(source.path)
    dtype = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    module_configs = {name: module.build_config(hf_config) for name, module in layout.modules.items()}
    assets = layout.build_module_assets(source.path)
    modules = {}
    for name, module_config in module_configs.items():
        logger.info_rank0(f"convert: loading module '{name}' from {source.path}")
        module_cls = OMNI_MODEL_REGISTRY[module_config.model_type]()
        module = load_module_from_hf_source(module_cls, module_config, source, name, device="cpu", torch_dtype=dtype)
        modules[name] = attach_module_assets(module, **assets[name])
    training_graphs, generation_graphs = layout.load_graphs(
        training_graph=training_graph, generation_graph=generation_graph
    )
    return {
        "modules": modules,
        "training_graphs": training_graphs,
        "generation_graphs": generation_graphs,
        "infer_type": layout.infer_type if layout.infer_type in generation_graphs else None,
    }


def source_weight_files(source_path: str) -> dict[str, str]:
    """``{weight key: shard filename}`` of a safetensors HF checkpoint."""
    index_path = os.path.join(source_path, _SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    single = os.path.join(source_path, _SAFE_WEIGHTS_NAME)
    if os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as f:
            return dict.fromkeys(f.keys(), _SAFE_WEIGHTS_NAME)
    raise FileNotFoundError(f"{source_path} has neither {_SAFE_WEIGHTS_INDEX_NAME} nor {_SAFE_WEIGHTS_NAME}.")


def _source_dtypes(source_path: str, weight_files: Mapping[str, str]) -> dict[str, torch.dtype]:
    from safetensors import safe_open

    dtypes: dict[str, torch.dtype] = {}
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_files.items():
        by_file.setdefault(filename, []).append(key)
    for filename, keys in by_file.items():
        with safe_open(os.path.join(source_path, filename), framework="pt", device="cpu") as f:
            for key in keys:
                dtypes[key] = f.get_slice(key).get_dtype()
    return {key: _SAFETENSORS_DTYPES[value] for key, value in dtypes.items()}


_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _read_source_tensors(
    source_path: str, weight_files: Mapping[str, str], keys: list[str]
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_files[key], []).append(key)
    for filename, file_keys in by_file.items():
        with safe_open(os.path.join(source_path, filename), framework="pt", device="cpu") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def copy_source_non_weight_files(source_path: str, save_path: str) -> None:
    """Copy the source's top-level non-weight files (config, tokenizer, generation config, …)."""
    os.makedirs(save_path, exist_ok=True)
    for entry in sorted(os.listdir(source_path)):
        src = os.path.join(source_path, entry)
        if not os.path.isfile(src) or entry.endswith(_WEIGHT_FILE_SUFFIXES + _WEIGHT_INDEX_SUFFIXES):
            continue
        shutil.copyfile(src, os.path.join(save_path, entry))


@torch.no_grad()
def save_hf_source_checkpoint(
    source: HFSource,
    modules: Mapping[str, tuple[torch.nn.Module, Any]],
    save_path: str,
) -> None:
    """Write ``modules``' live weights back as one checkpoint in ``source``'s own layout.

    ``modules`` maps a layout module name to ``(model, parallel_state)``; all
    ranks must call this. Each module's flat state dict has its own weight
    conversions reverted, is renamed back to source keys, and is cast to the
    source dtype. Source keys no live module produced — modules that stayed
    frozen, a sibling built on meta, or weights no module claims — are copied
    from the source verbatim, except a declared tied duplicate
    (:attr:`OmniHFLayout.tied_source_keys`), which takes its original's value.
    Keys the source does not hold are dropped. The result keeps the source's shard files
    and non-weight files, so it loads wherever the source does.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    from ....checkpoint.dcp_checkpointer import ModelState
    from ....checkpoint.dcp_consolidation import apply_dcp_consolidation_patch
    from ....utils.save_safetensor_utils import _revert_model_weight_conversions_for_hf

    apply_dcp_consolidation_patch()
    layout = source.layout
    weight_files = source_weight_files(source.path)
    source_dtypes = _source_dtypes(source.path, weight_files)

    save_state: dict[str, torch.Tensor] = {}
    for name, (model, parallel_state) in modules.items():
        module_state = ModelState(model, parallel_state=parallel_state).state_dict()
        module_state = _revert_model_weight_conversions_for_hf(model, module_state)
        dropped = []
        for key, tensor in module_state.items():
            source_key = layout.source_key(name, key)
            if source_key is None or source_key not in weight_files:
                dropped.append(key)
                continue
            if source_key in save_state:
                raise ValueError(f"Source key {source_key!r} is produced by more than one module.")
            dtype = source_dtypes[source_key]
            save_state[source_key] = tensor if tensor.dtype == dtype else tensor.to(dtype)
        if dropped:
            logger.info_rank0(f"save_hf_source_checkpoint: module '{name}' keys not in the source, dropped: {dropped}")

    for key, original in layout.tied_source_keys.items():
        if key in weight_files and key not in save_state and original in save_state:
            save_state[key] = save_state[original].clone()
    carried = [key for key in weight_files if key not in save_state]
    if carried:
        logger.info_rank0(
            f"save_hf_source_checkpoint: copying {len(carried)} source tensor(s) no live module produced."
        )
        rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if rank0:
            save_state.update(_read_source_tensors(source.path, weight_files, carried))

    single_file = set(weight_files.values()) == {_SAFE_WEIGHTS_NAME}
    fqn_to_index_mapping = {
        key: 1 if single_file else shard_index_from_filename(filename) for key, filename in weight_files.items()
    }
    storage_writer = HuggingFaceStorageWriter(
        path=save_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    if dist.is_initialized():
        dist.barrier()
    start = time.time()
    dcp.save(state_dict=save_state, storage_writer=storage_writer)
    del save_state
    if dist.is_initialized():
        dist.barrier()

    if not dist.is_initialized() or dist.get_rank() == 0:
        shutil.rmtree(os.path.join(save_path, _DCP_STAGING_DIRNAME), ignore_errors=True)
        if single_file:
            _restore_single_file_name(save_path)
        copy_source_non_weight_files(source.path, save_path)
    if dist.is_initialized():
        dist.barrier()
    helper.empty_cache()
    logger.info_rank0(
        f"save_hf_source_checkpoint: `{source.model_type}` checkpoint saved at {save_path} "
        f"in {time.time() - start:.2f}s."
    )


def _restore_single_file_name(save_path: str) -> None:
    """Rename the writer's lone ``model-00001-of-00001.safetensors`` back to ``model.safetensors``."""
    shards = [name for name in os.listdir(save_path) if name.endswith(".safetensors")]
    if len(shards) != 1:
        raise RuntimeError(f"Expected one safetensors file in {save_path}, found {shards}.")
    os.replace(os.path.join(save_path, shards[0]), os.path.join(save_path, _SAFE_WEIGHTS_NAME))
    index = os.path.join(save_path, _SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index):
        os.remove(index)


__all__ = [
    "HF_SOURCE_FILE",
    "OMNI_HF_LAYOUT_REGISTRY",
    "HFSource",
    "HFSourceKeyConverter",
    "OmniHFLayout",
    "OmniHFModuleLayout",
    "attach_hf_source_converter",
    "convert_with_hf_layout",
    "copy_source_non_weight_files",
    "get_hf_layout",
    "has_hf_layout",
    "load_module_from_hf_source",
    "read_hf_source",
    "resolve_omni_checkpoint_root",
    "save_hf_source_checkpoint",
    "source_weight_files",
]
