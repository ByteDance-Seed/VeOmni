#!/usr/bin/env python3
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
"""Drift gate for patchgen-generated files.

Discovers all patch gen configs under the caller's
:class:`DiscoveryConfig`, regenerates each, normalizes through the shared
ruff pipeline, and compares against the checked-in ``.py`` / ``.diff``.

The unified ``patchgen --check`` console script (``patchgen.cli``) is the
user entrypoint. This module exposes :func:`build_cli` for any direct
downstream caller that wants to mount its own drift gate.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from ._normalize import ruff_fix_and_format
from .codegen import ModelingCodeGenerator, load_patch_config_module
from .run_codegen import (
    DiscoveryConfig,
    build_unified_diff,
    default_diff_path,
    default_output_dir_for_module,
    list_patch_configs,
    strip_diff_trailing_ws,
)


def check_config(
    module_name: str,
    *,
    fix: bool = False,
    ruff_extra_ignore: tuple[str, ...] = (),
    ruff_isolated: bool = False,
    search_roots: Optional[list[Path]] = None,
) -> bool:
    """Check a single config for drift.

    Returns True when the checked-in files are up to date (or were fixed).

    Args:
        search_roots: optional list of filesystem roots to prepend to the
            loader's file-walk. Pass ``[discovery.package_root]`` from a
            caller that drives discovery via ``DiscoveryConfig`` so the
            loader still resolves the config file when the project isn't
            installed on ``sys.path``.
    """
    # Use the patchgen-aware loader so parent ``__init__.py`` side effects
    # (heavy 3rdparty imports) don't gate the drift check.
    module = load_patch_config_module(module_name, search_roots=search_roots)
    config = module.config

    output_dir = default_output_dir_for_module(module)
    checked_in_py = output_dir / config.target_file
    checked_in_diff = default_diff_path(output_dir, config.target_file)

    # -- generate to a temp file ------------------------------------------------
    generator = ModelingCodeGenerator(config)
    generator.load_source()
    generated = generator.generate()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(generated)
        tmp_path = Path(tmp.name)

    try:
        ruff_fix_and_format(tmp_path, extra_ignore=ruff_extra_ignore or None, isolated=ruff_isolated)
        normalized_py = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    # -- generate diff ----------------------------------------------------------
    normalized_diff = strip_diff_trailing_ws(
        build_unified_diff(
            original_source=generator.source_code,
            generated_source=normalized_py,
            source_module=config.source_module,
            target_file=config.target_file,
        )
    )

    # -- compare ----------------------------------------------------------------
    ok = True

    existing_py = checked_in_py.read_text(encoding="utf-8") if checked_in_py.exists() else ""
    if existing_py != normalized_py:
        if fix:
            checked_in_py.parent.mkdir(parents=True, exist_ok=True)
            checked_in_py.write_text(normalized_py, encoding="utf-8")
            print(f"  FIXED {checked_in_py}")
        else:
            ok = False
            diff = difflib.unified_diff(
                existing_py.splitlines(keepends=True),
                normalized_py.splitlines(keepends=True),
                fromfile=f"a/{checked_in_py}",
                tofile=f"b/{checked_in_py}",
                n=3,
            )
            print(f"  DRIFT {checked_in_py}")
            sys.stdout.writelines(diff)

    existing_diff = (
        strip_diff_trailing_ws(checked_in_diff.read_text(encoding="utf-8")) if checked_in_diff.exists() else ""
    )
    if existing_diff != normalized_diff:
        if fix:
            checked_in_diff.parent.mkdir(parents=True, exist_ok=True)
            checked_in_diff.write_text(normalized_diff, encoding="utf-8")
            print(f"  FIXED {checked_in_diff}")
        else:
            ok = False
            diff = difflib.unified_diff(
                existing_diff.splitlines(keepends=True),
                normalized_diff.splitlines(keepends=True),
                fromfile=f"a/{checked_in_diff}",
                tofile=f"b/{checked_in_diff}",
                n=3,
            )
            print(f"  DRIFT {checked_in_diff}")
            sys.stdout.writelines(diff)

    if ok and not fix:
        print(f"  OK    {checked_in_py}")
    return ok


def run_check(
    discovery: DiscoveryConfig,
    *,
    fix: bool = False,
    configs: Optional[list[str]] = None,
) -> int:
    """Run a drift check across ``configs`` (or auto-discover via ``discovery``).

    Returns a process exit code (0 = clean / fixed, 1 = drift in check mode,
    0 if no configs found at all).
    """
    if configs is None:
        configs = list_patch_configs(discovery)

    if not configs:
        print("No patch configs found.")
        return 0

    print(f"Found {len(configs)} patch config(s):\n")
    all_ok = True
    for cfg in configs:
        print(f"[{cfg}]")
        ok = check_config(
            cfg,
            fix=fix,
            ruff_extra_ignore=discovery.ruff_extra_ignore,
            ruff_isolated=discovery.ruff_isolated,
            search_roots=[discovery.package_root],
        )
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All generated files are up to date.")
        return 0
    print("Generated files are out of date. Re-run with --fix.")
    return 1


def _build_parser(prog_name: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Check patchgen generated files for drift",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Overwrite checked-in files with regenerated output",
    )
    return parser


def build_cli(
    discovery: DiscoveryConfig,
    prog_name: Optional[str] = None,
) -> Callable[[Optional[list[str]]], int]:
    """Return a ``main()``-shaped callable that runs a drift check rooted at
    ``discovery``.

    Downstream projects mount their own ``python -m <pkg>.patchgen.check``
    CLI without copy-pasting argparse.
    """
    parser = _build_parser(prog_name=prog_name)

    def _main(argv: Optional[list[str]] = None) -> int:
        args = parser.parse_args(argv)
        return run_check(discovery, fix=args.fix)

    return _main


if __name__ == "__main__":
    # Reject `python -m patchgen.check_patchgen ...` loudly. Without this
    # guard the module imports cleanly and exits 0 without parsing argv — a
    # silent no-op that's easy for automation to misread as a passing drift
    # check.
    import sys

    sys.stderr.write(
        "patchgen.check_patchgen is library code, not an executable entry point.\n"
        "Use the `patchgen` console script (or `python -m patchgen`) instead:\n"
        "  patchgen --check [--fix]\n"
    )
    sys.exit(2)
