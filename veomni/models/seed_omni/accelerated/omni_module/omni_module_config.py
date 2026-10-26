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

"""VeOmni per-module runtime config — the accelerated counterpart of ``OmniModuleConfig``.

:class:`OmniModuleRuntimeArguments` is one module's slice of a composed Omni model:
the inherited :class:`~veomni.arguments.arguments_types.ModelArguments` fields
(``model_path``, ``ops_implementation``, ``accelerator``, ``optimizer``, freeze,
LoRA) plus the projection onto an HF :class:`OmniModuleConfig` descriptor.

It does **not** inherit :class:`~veomni.models.seed_omni.configuration_omni.OmniConfig`
or :class:`~veomni.models.seed_omni.modules.module_configuration_base.OmniModuleConfig`:
those are HuggingFace checkpoint shapes. This dataclass is the launcher/runtime
view; :meth:`to_hf_config` is the one-way projection.

It is re-exported from :mod:`veomni.arguments.omni_arguments_types`. YAML resolution (``build_omni_model_runtime_args``, ``build_omni_module_runtime_args``)
stays in ``arguments/`` to avoid an arguments ↔ accelerated import cycle.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from .....arguments.arguments_types import ModelArguments


LAUNCHER_CONFIG_KEYS = frozenset({"modules", "train_graph", "train_type", "infer_graph", "infer_type"})


def hf_module_model_config(model_config: dict | None) -> dict:
    """Drop launcher layout keys before merging or exporting per-module ``model_config``."""
    if not model_config:
        return {}
    return {key: value for key, value in model_config.items() if key not in LAUNCHER_CONFIG_KEYS}


@dataclass
class OmniModuleRuntimeArguments(ModelArguments):
    """Per-module runtime — one module's slice of a composed Omni model.

    ``ModelArguments`` already is a complete training unit: the model
    fields plus this module's own ``accelerator`` and ``optimizer``, with
    ``model_path`` localized and ``fqn_to_index_mapping`` parsed lazily and cached
    per index path (several modules routinely share one checkpoint). All this adds
    is the projection onto an HF module descriptor.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.accelerator.fsdp_config.fsdp_mode == "eager" and self.needs_distributed_build:
            self.accelerator.fsdp_config.fsdp_mode = "fsdp2"
            self.accelerator.fsdp_config.extra_parallel_only = True
            self.accelerator._validate_init_device()

    @property
    def needs_distributed_build(self) -> bool:
        """True when this module must be built under a process group, not loaded whole.

        ``eager`` is the inference default and means one rank holds the module
        entire, loaded through ``device_map`` with no collectives. Anything else
        -- FSDP2, DDP, expert parallelism, vocab parallelism -- needs a
        :class:`ParallelState` and a sharded build.

        The two parallelisms that are *not* ``fsdp_mode`` are the reason this is
        a property rather than a comparison written out at each call site. A
        module can ask for ``ep`` or ``emb`` while leaving ``fsdp_mode`` alone,
        and reading only ``fsdp_mode`` then sends it down the single-process
        path -- where the sharding quietly does not happen and the whole module
        is loaded on every rank. DeepSeek-V4.1 is both cases: its n-gram tables
        shard over ``emb`` and nothing else, and one of them is 91.6 GiB.
        """
        fsdp_mode = self.accelerator.fsdp_config.fsdp_mode
        if fsdp_mode and str(fsdp_mode).lower() not in ("eager",):
            return True
        return self.accelerator.ep_size > 1 or any(size > 1 for size in self.accelerator.extra_parallel_sizes or ())

    def to_hf_config(self) -> dict:
        """Project onto this module's ``OmniConfig._module_entries`` entry.

        The module's name is the key that entry is filed under, so it is not
        repeated inside it.

        ``model_path`` is carried through explicitly: by the time this runs,
        ``build_omni_module_runtime_args`` / ``_resolve_model_path`` has already
        resolved it to an absolute path — usually ``<checkpoint_root>/<name>``,
        but a launcher YAML module override may point it at a wholly different
        checkpoint (e.g. Qwen3 visual-instruction-tuning composing
        ``qwen3_llm``/``qwen3_text_encoder`` from one HF model with
        ``qwen3vl_vision`` from another). Dropping it and re-deriving
        ``checkpoint_root/name`` downstream (as
        :meth:`~veomni.models.seed_omni.modules.module_configuration_base.OmniModuleConfig.resolve_path`
        does for an entry without one) would silently resolve to the wrong path
        for that module.

        Kernels are projected too, so a checkpoint remembers what each module
        was trained with: ``build_omni_module_runtime_args`` layers the entry under
        the launcher's per-module YAML, and a bare ``OmniModel.from_pretrained``
        applies it to the module config.
        """
        entry: dict = {}
        if self.model_path:
            entry["model_path"] = self.model_path
        if self.ops_implementation is not None:
            entry["ops_implementation"] = {k: v for k, v in asdict(self.ops_implementation).items() if v is not None}
        if model_config := hf_module_model_config(self.model_config):
            entry["model_config"] = deepcopy(model_config)
        if self.processor_config:
            entry["processor_config"] = deepcopy(self.processor_config)
        return entry


__all__ = [
    "OmniModuleRuntimeArguments",
    "hf_module_model_config",
]
