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
"""AST-based patchgen for HuggingFace modeling files.

Discovers ``*_patch_gen_config.py`` files under the caller's
:class:`DiscoveryConfig`, rewrites the targeted ``transformers`` modeling
module, and writes the result to a ``generated/`` sibling dir.

The unified ``patchgen`` console script (``patchgen.cli:main``) is the
intended entrypoint. Direct downstream callers can also build their own
CLI via :func:`run_codegen.build_cli` and :func:`check_patchgen.build_cli`.

The patchgen layer itself is transformers-version-agnostic: it only reads
the source ``.py`` file off ``sys.path`` (no ``import transformers``
required) and rewrites its AST.
"""

from typing import TYPE_CHECKING

from ._normalize import ruff_fix_and_format
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


# Lazy re-exports for the CLI-bearing submodules. Mapping is
# ``<public name in this package>: (<submodule>, <attr in submodule>)``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # check_patchgen
    "build_check_cli": ("check_patchgen", "build_cli"),
    "check_config": ("check_patchgen", "check_config"),
    "run_check": ("check_patchgen", "run_check"),
    # run_codegen
    "DiscoveryConfig": ("run_codegen", "DiscoveryConfig"),
    "build_run_codegen_cli": ("run_codegen", "build_cli"),
    "build_unified_diff": ("run_codegen", "build_unified_diff"),
    "default_diff_path": ("run_codegen", "default_diff_path"),
    "default_output_dir_for_module": ("run_codegen", "default_output_dir_for_module"),
    "list_patch_configs": ("run_codegen", "list_patch_configs"),
    "normalize_patch_module": ("run_codegen", "normalize_patch_module"),
    "run_codegen": ("run_codegen", "run_codegen"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute lookup.

    Resolves names in ``_LAZY_EXPORTS`` by importing the backing submodule
    on first access, then caches the resolved value in ``globals()`` so
    subsequent accesses skip ``__getattr__`` entirely.

    Subtlety: ``run_codegen`` is both a public function AND the name of
    the submodule that defines it. Python's import machinery executes
    ``setattr(parent_pkg, submodule_name, submodule)`` after loading a
    submodule, which would leave ``patchgen.run_codegen`` pointing at the
    submodule (overriding any prior function binding) whenever *any* lazy
    lookup imports that submodule for *any* symbol. We work around it by
    pre-binding **all** re-exports from the just-imported submodule, so
    the function shadows the submodule entry before control returns to
    the caller.
    """
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        submod_name = _LAZY_EXPORTS[name][0]
        submod = import_module(f".{submod_name}", __name__)
        for export_name, (sm, sm_attr) in _LAZY_EXPORTS.items():
            if sm == submod_name:
                globals()[export_name] = getattr(submod, sm_attr)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


# Re-import the lazy names under TYPE_CHECKING so static analysers, IDE
# autocomplete, and Sphinx see them as ordinary re-exports. These imports
# are skipped at runtime — the only runtime path is __getattr__ above.
if TYPE_CHECKING:
    from .check_patchgen import build_cli as build_check_cli
    from .check_patchgen import check_config, run_check
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
    # Discovery + run_codegen (lazy)
    "DiscoveryConfig",
    "build_run_codegen_cli",
    "build_unified_diff",
    "default_diff_path",
    "default_output_dir_for_module",
    "list_patch_configs",
    "normalize_patch_module",
    "run_codegen",
    # Drift check (lazy)
    "build_check_cli",
    "check_config",
    "run_check",
]
