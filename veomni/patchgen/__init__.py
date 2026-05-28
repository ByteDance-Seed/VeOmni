# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""VeOmni patch generation utilities.

This package is designed to be reused as a library by projects that depend
on VeOmni so they can patch their own models in their own repos. Callers
wire up their own discovery root + CLI via :func:`run_codegen.build_cli`
and :func:`check_patchgen.build_cli`. See
``docs/transformers_v5/patchgen.md`` section "Using patchgen from a
dependent project" for the full recipe.

The patchgen layer itself is transformers-version-agnostic: it only reads
the source ``.py`` file off ``sys.path`` (no ``import transformers``
required) and rewrites its AST. Dependent projects that pin transformers
v4 can use this library the same way v5 callers do — provided their patch
config targets a module file that actually exists on their installed
transformers.
"""

from ._normalize import ruff_fix_and_format
from .check_patchgen import build_cli as build_check_cli
from .check_patchgen import check_config, run_check
from .codegen import (
    CodegenError,
    ModelingCodeGenerator,
    generate_from_config,
    get_module_source,
    load_patch_config_module,
)
from .patch_spec import (
    ImportSpec,
    Patch,
    PatchConfig,
    PatchType,
    PositionedHelper,
    create_patch_from_external,
)
from .run_codegen import (
    DiscoveryConfig,
    build_unified_diff,
    default_diff_path,
    default_output_dir_for_module,
    list_patch_configs,
    normalize_patch_module,
    run_codegen,
)
from .run_codegen import (
    build_cli as build_run_codegen_cli,
)


__all__ = [
    # Patch spec
    "Patch",
    "PatchConfig",
    "PatchType",
    "ImportSpec",
    "PositionedHelper",
    "create_patch_from_external",
    # Codegen
    "CodegenError",
    "ModelingCodeGenerator",
    "generate_from_config",
    "get_module_source",
    "load_patch_config_module",
    # Normalization
    "ruff_fix_and_format",
    # Discovery + run_codegen
    "DiscoveryConfig",
    "build_run_codegen_cli",
    "build_unified_diff",
    "default_diff_path",
    "default_output_dir_for_module",
    "list_patch_configs",
    "normalize_patch_module",
    "run_codegen",
    # Drift check
    "build_check_cli",
    "check_config",
    "run_check",
]
