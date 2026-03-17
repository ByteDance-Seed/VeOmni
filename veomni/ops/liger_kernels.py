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

import importlib
import pathlib
from typing import Dict

from transformers import PretrainedConfig, PreTrainedModel

from ..models.loader import LIGER_KERNEL_MAPPING_REGISTRY
from ..utils import logging


logger = logging.get_logger(__name__)


def _build_liger_kernel_mapping(kernel_mapping: Dict[str, Dict[str, str]]) -> Dict:
    """
    Build a kernels-library compatible mapping from the KERNEL_MAPPING registry entries.

    Each entry specifies a ``type`` ("layer" or "func"), a ``module`` (dotted Python module path),
    and a ``name`` (class or function name within that module).

    For "layer" entries, a ``LocalLayerRepository`` is created.
    For "func" entries, a ``LocalFuncRepository`` is created.

    Returns a dict suitable for ``use_kernel_mapping``, e.g.::

        {
            "RMSNorm": {"cuda": LocalLayerRepository(...)},
            "rotary_pos_emb": {"cuda": LocalFuncRepository(...)},
        }
    """
    from kernels import LocalFuncRepository, LocalLayerRepository

    resolved = {}
    for entry_name, entry in kernel_mapping.items():
        entry_type = entry["type"]
        module_path = entry["module"]
        name = entry["name"]

        mod = importlib.import_module(module_path)
        mod_file = pathlib.Path(mod.__file__)
        repo_path = mod_file.parent
        # Packages (__init__.py) use the directory name; plain modules use the file stem.
        package_name = mod_file.parent.name if mod_file.name == "__init__.py" else mod_file.stem

        if entry_type == "layer":
            repo = LocalLayerRepository(repo_path=repo_path, package_name=package_name, layer_name=name)
        elif entry_type == "func":
            repo = LocalFuncRepository(repo_path=repo_path, package_name=package_name, func_name=name)
        else:
            raise ValueError(f"Unknown kernel mapping type '{entry_type}' for entry '{entry_name}'")

        logger.info_rank0(f"Resolved kernel mapping '{entry_name}' ({entry_type}): {module_path}.{name} -> {repo}")
        resolved[entry_name] = {"cuda": repo}

    return resolved


def apply_liger_kernels(model: PreTrainedModel, config: PretrainedConfig) -> None:
    """Resolve LIGER_KERNEL_MAPPING_REGISTRY and kernelize the model."""
    logger.info_rank0("use_liger is enabled, attempting to build Liger kernel mapping...")

    model_type = config.model_type
    if model_type in LIGER_KERNEL_MAPPING_REGISTRY.valid_keys():
        kernel_mapping = LIGER_KERNEL_MAPPING_REGISTRY[model_type]()
        resolved_mapping = _build_liger_kernel_mapping(kernel_mapping)
        logger.info_rank0(f"Liger kernel mapping for {model_type}: {resolved_mapping}")

        from kernels import Mode, kernelize, use_kernel_mapping

        mode = Mode.TRAINING if model.training else Mode.INFERENCE
        with use_kernel_mapping(resolved_mapping, inherit_mapping=False):
            kernelize(model, mode=mode, device="cuda")
        model._use_kernels = True

        logger.info_rank0("Setup Liger kernels completed.")
    else:
        logger.warning_rank0(
            f"use_liger=True but no liger kernel mapping registered for model type '{model_type}'. "
            f"Available: {LIGER_KERNEL_MAPPING_REGISTRY.valid_keys()}"
        )
