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
CI check script to verify that generated patchgen files are in sync with their configs.

Follows the HuggingFace transformers `check_modular_conversion.py` approach:
regenerate code, apply the same formatting/fixup tools that pre-commit hooks use
(ruff, pyupgrade, trailing-whitespace), then diff against checked-in files.

Usage:
    # Check mode: fail if drift detected
    python -m veomni.patchgen.check_patchgen

    # Fix mode: regenerate & overwrite all files
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
from .patch_spec import PatchConfig
from .run_codegen import build_unified_diff, default_diff_path, default_output_dir_for_module, list_patch_configs


def run_formatters(file_path: str) -> None:
    """Run ruff format and fix on a file, modifying it in place."""
    subprocess.run(["ruff", "check", "--fix", file_path], capture_output=True)
    subprocess.run(["ruff", "format", file_path], capture_output=True)


def strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line, matching the trailing-whitespace hook."""
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n" if text else ""


def check_config(config_module_name: str, fix: bool = False) -> bool:
    """
    Check a single patch config for drift.

    Returns True if the generated file matches the checked-in file (or was fixed).
    """
    # Import the config module
    module = importlib.import_module(config_module_name)
    config = module.config
    if not isinstance(config, PatchConfig):
        print(f"  [SKIP] {config_module_name} — config is not a PatchConfig")
        return True

    # Determine output path
    output_dir = default_output_dir_for_module(module)
    checked_in_path = output_dir / config.target_file

    # Generate into memory
    generator = ModelingCodeGenerator(config)
    generator.load_source()
    generated_content = generator.generate()

    # Write to a temp file and run formatters on it
    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{config.target_file}", delete=False, encoding="utf-8") as tmp:
        tmp.write(generated_content)
        tmp_path = tmp.name

    try:
        run_formatters(tmp_path)
        formatted_content = Path(tmp_path).read_text(encoding="utf-8")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Derive a short label from the module name for display
    # e.g. "veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config" -> "qwen3/qwen3_gpu_patch_gen_config"
    parts = config_module_name.split(".")
    if len(parts) >= 4:
        label = "/".join(parts[3:])
    else:
        label = config_module_name

    # Generate the expected diff content (source vs generated) and strip trailing whitespace
    # (unified diffs naturally produce context lines with trailing spaces)
    diff_path = default_diff_path(output_dir, config.target_file)
    expected_diff = build_unified_diff(
        original_source=generator.source_code,
        generated_source=formatted_content,
        source_module=config.source_module,
        target_file=config.target_file,
    )
    expected_diff = strip_trailing_whitespace(expected_diff)

    # Fix mode: write both the generated file and diff to disk
    if fix:
        checked_in_path.parent.mkdir(parents=True, exist_ok=True)
        checked_in_path.write_text(formatted_content, encoding="utf-8")
        diff_path.write_text(expected_diff, encoding="utf-8")
        print(f"  [FIXED] {label}")
        return True

    # Check mode: compare against checked-in files
    ok = True

    # Check the generated .py file
    if not checked_in_path.exists():
        print(f"  [MISSING] {label}")
        print(f"    File not found: {checked_in_path}")
        ok = False
    else:
        checked_in_content = checked_in_path.read_text(encoding="utf-8")
        if formatted_content != checked_in_content:
            print(f"  [DIFF] {label}")
            py_diff = difflib.unified_diff(
                formatted_content.splitlines(keepends=True),
                checked_in_content.splitlines(keepends=True),
                fromfile="expected (regenerated)",
                tofile="actual (checked-in)",
            )
            for line in py_diff:
                print(f"    {line}", end="")
            print()
            ok = False

    # Check the .diff file
    if diff_path.exists():
        checked_in_diff = diff_path.read_text(encoding="utf-8")
        if expected_diff != checked_in_diff:
            if ok:
                print(f"  [DIFF] {label} (.diff file)")
            else:
                print("    (.diff file also drifted)")
            ok = False

    if ok:
        print(f"  [OK] {label}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Check that generated patchgen files are in sync with their configs.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Regenerate and overwrite all generated files",
    )

    args = parser.parse_args()

    print("Checking patchgen consistency...")

    configs = list_patch_configs()
    if not configs:
        print("  No patch configurations found.")
        return 0

    ok_count = 0
    drift_count = 0

    for config_module in configs:
        try:
            passed = check_config(config_module, fix=args.fix)
        except Exception as e:
            print(f"  [ERROR] {config_module}: {e}")
            passed = False

        if passed:
            ok_count += 1
        else:
            drift_count += 1

    total = ok_count + drift_count
    print(f"\nResult: {ok_count}/{total} OK, {drift_count} drifted.")

    if drift_count > 0 and not args.fix:
        print("To fix: python -m veomni.patchgen.check_patchgen --fix")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
