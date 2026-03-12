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
"""
CI check script for patchgen determinism.

Discovers all patch gen configs, regenerates each one, runs ruff check --fix
and ruff format on the output, then compares against the checked-in .py and
.diff files.  Exits 1 if any drift is detected.

Usage:
    # Check mode (CI) - exits 1 on drift
    python -m veomni.patchgen.check_patchgen

    # Fix mode - overwrite checked-in files with regenerated output
    python -m veomni.patchgen.check_patchgen --fix
"""

import argparse
import difflib
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

from .codegen import ModelingCodeGenerator
from .run_codegen import (
    build_unified_diff,
    default_diff_path,
    default_output_dir_for_module,
    list_patch_configs,
)


def _ruff_fix_and_format(path: Path) -> None:
    """Run ``ruff check --fix`` and ``ruff format`` on *path*."""
    subprocess.run(
        ["ruff", "check", "--fix", "--quiet", str(path)],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["ruff", "format", "--quiet", str(path)],
        check=False,
        capture_output=True,
    )


def _strip_trailing_ws(text: str) -> str:
    """Strip trailing whitespace from every line.

    Unified diffs produced by :func:`difflib.unified_diff` leave trailing
    spaces on context lines which editors and pre-commit hooks may strip,
    causing spurious drift.
    """
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n" if text else text


def check_config(module_name: str, *, fix: bool = False) -> bool:
    """Check a single config for drift.

    Returns True when the checked-in files are up to date (or were fixed).
    """
    module = importlib.import_module(module_name)
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
        # -- run ruff on temp file to normalize style ---------------------------
        _ruff_fix_and_format(tmp_path)
        normalized_py = tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)

    # -- generate diff ----------------------------------------------------------
    normalized_diff = _strip_trailing_ws(
        build_unified_diff(
            original_source=generator.source_code,
            generated_source=normalized_py,
            source_module=config.source_module,
            target_file=config.target_file,
        )
    )

    # -- compare ----------------------------------------------------------------
    ok = True

    if checked_in_py.exists():
        existing_py = checked_in_py.read_text()
    else:
        existing_py = ""

    if existing_py != normalized_py:
        if fix:
            checked_in_py.parent.mkdir(parents=True, exist_ok=True)
            checked_in_py.write_text(normalized_py)
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

    if checked_in_diff.exists():
        existing_diff = _strip_trailing_ws(checked_in_diff.read_text())
    else:
        existing_diff = ""

    if existing_diff != normalized_diff:
        if fix:
            checked_in_diff.parent.mkdir(parents=True, exist_ok=True)
            checked_in_diff.write_text(normalized_diff)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check patchgen generated files for drift")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Overwrite checked-in files with regenerated output",
    )
    args = parser.parse_args()

    configs = list_patch_configs()
    if not configs:
        print("No patch configs found.")
        return 0

    print(f"Found {len(configs)} patch config(s):\n")
    all_ok = True
    for cfg in configs:
        print(f"[{cfg}]")
        ok = check_config(cfg, fix=args.fix)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All generated files are up to date.")
        return 0
    else:
        print("Generated files are out of date. Run: python -m veomni.patchgen.check_patchgen --fix")
        return 1


if __name__ == "__main__":
    sys.exit(main())
