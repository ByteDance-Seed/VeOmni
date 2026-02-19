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
Runner script for the Modeling Code Generator.

This script provides a convenient way to run the code generator with
common configurations. It can be used as a CLI tool or imported.

Usage:
    # Generate from a specific patch configuration
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config

    # Generate to a specific output directory
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -o /path/to/output

    # List available patch configurations
    python -m veomni.patchgen.run_codegen --list

    # Dry run (show what would be generated without writing)
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --dry-run

    # Generate modeling code and save unified diff alongside it
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --diff
"""

import argparse
import difflib
import importlib
import sys
from pathlib import Path
from typing import Optional

from .codegen import CodegenError, ModelingCodeGenerator
from .patch_spec import PatchConfig


MODULE_DIR = Path(__file__).parent
VEOMNI_DIR = MODULE_DIR.parent
MODELS_DIR = VEOMNI_DIR / "models" / "transformers"
PACKAGE_NAME = __package__ or "veomni.patchgen"
PATCHES_PACKAGE = "veomni.models.transformers.qwen3.patches"


def build_unified_diff(
    original_source: str,
    generated_source: str,
    source_module: str,
    target_file: str,
    context_lines: int = 3,
) -> str:
    """Build unified diff text between source module code and generated code."""
    module_path = source_module.replace(".", "/") + ".py"
    original_lines = original_source.splitlines(keepends=True)
    generated_lines = generated_source.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        generated_lines,
        fromfile=f"a/{module_path}",
        tofile=f"b/{target_file}",
        n=context_lines,
    )
    return "".join(diff)


def default_diff_path(output_dir: Path, target_file: str) -> Path:
    """Return default .diff path in the output directory for a generated target file."""
    return output_dir / Path(target_file).with_suffix(".diff").name


def list_patch_configs(models_dir: Path = MODELS_DIR) -> list[str]:
    """List all available patch configurations under veomni/models/transformers."""
    configs = []
    if not models_dir.exists():
        return configs

    for patches_dir in models_dir.rglob("patches"):
        if not patches_dir.is_dir():
            continue
        for py_file in patches_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_path = py_file.relative_to(VEOMNI_DIR).with_suffix("")
            module_name = ".".join(("veomni",) + module_path.parts)
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "config") and isinstance(module.config, PatchConfig):
                    configs.append(module_name)
            except ImportError:
                continue

    return configs


def normalize_patch_module(patch_module: str) -> str:
    if patch_module.startswith(f"{PACKAGE_NAME}."):
        return patch_module
    if patch_module.startswith("patches."):
        return f"{PATCHES_PACKAGE}.{patch_module.removeprefix('patches.')}"
    return patch_module


def default_output_dir_for_module(module: object) -> Path:
    module_path = Path(module.__file__).resolve()
    if module_path.parent.name == "patches":
        return module_path.parent.parent / "generated"
    return MODULE_DIR / "generated"


def print_config_summary(config: PatchConfig) -> None:
    """Print a summary of a patch configuration."""
    print("\n" + "=" * 70)
    print("PATCH CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nSource: {config.source_module}")
    print(f"Target: {config.target_file}")
    if config.description:
        print(f"Description: {config.description}")

    print(f"\nPatches ({len(config.patches)}):")
    for patch in config.patches:
        print(f"  • [{patch.patch_type.value}] {patch.target}")
        if patch.description:
            print(f"    └─ {patch.description}")

    if config.exclude:
        print(f"\nExcluded: {', '.join(config.exclude)}")

    if config.additional_imports:
        print(f"\nAdditional imports: {len(config.additional_imports)}")

    print("=" * 70)


def run_codegen(
    patch_module: str,
    output_dir: Optional[Path],
    config_name: str = "config",
    dry_run: bool = False,
    save_diff: bool = False,
    verbose: bool = False,
) -> Optional[str]:
    """
    Run code generation for a patch configuration.

    Args:
        patch_module: Module path containing the PatchConfig
        output_dir: Directory to write generated files (defaults to sibling generated/ next to patch module)
        config_name: Name of the config variable in the module
        dry_run: If True, don't write files
        save_diff: If True, save a unified diff alongside the generated modeling file
        verbose: If True, print detailed progress

    Returns:
        The generated source code, or None on error
    """
    try:
        # Import the patch module
        if verbose:
            print(f"Loading patch module: {patch_module}")
        module = importlib.import_module(normalize_patch_module(patch_module))
        config = getattr(module, config_name)

        if output_dir is None:
            output_dir = default_output_dir_for_module(module)

        if not isinstance(config, PatchConfig):
            print(f"Error: {config_name} in {patch_module} is not a PatchConfig", file=sys.stderr)
            return None

        if verbose:
            print_config_summary(config)

        # Generate
        if verbose:
            print("\nGenerating code...")

        generator = ModelingCodeGenerator(config)
        generator.load_source()

        if dry_run:
            print("\n[DRY RUN] Would generate:")
            print(f"  Output: {output_dir / config.target_file}")
            if save_diff:
                print(f"  Diff:   {default_diff_path(output_dir, config.target_file)}")
            print(f"  Source lines: ~{len(generator.source_code.splitlines())}")
            print(f"  Patches to apply: {len(config.patches)}")
            return generator.source_code

        # Actually generate
        output_path = output_dir / config.target_file
        output = generator.generate(output_path)

        print(f"\n✓ Generated: {output_path}")
        print(f"  Lines: {len(output.splitlines())}")

        if save_diff:
            diff_output = build_unified_diff(
                original_source=generator.source_code,
                generated_source=output,
                source_module=config.source_module,
                target_file=config.target_file,
            )
            diff_path = default_diff_path(output_dir, config.target_file)
            diff_path.write_text(diff_output)
            print(f"✓ Diff: {diff_path}")
            print(f"  Lines: {len(diff_output.splitlines())}")

        return output

    except ImportError as e:
        print(f"Error importing {patch_module}: {e}", file=sys.stderr)
        return None
    except CodegenError as e:
        print(f"Code generation error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Modeling Code Generator - Generate patched HuggingFace modeling code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config
  %(prog)s veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -o /path/to/output
  %(prog)s veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --dry-run
  %(prog)s veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --diff
  %(prog)s --list
        """,
    )

    parser.add_argument(
        "patch_module",
        nargs="?",
        help="Patch module to use (e.g., 'veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling generated/ next to patch module)",
    )
    parser.add_argument(
        "-c",
        "--config-name",
        default="config",
        help="Config variable name in the patch module (default: config)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available patch configurations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Save a unified .diff file alongside the generated modeling code",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available patch configurations:")
        configs = list_patch_configs()
        if configs:
            for config in configs:
                print(f"  • {config}")
        else:
            print("  (none found)")
        return 0

    # Require patch_module for generation
    if not args.patch_module:
        parser.error("patch_module is required unless using --list")

    # Run generation
    result = run_codegen(
        patch_module=normalize_patch_module(args.patch_module),
        output_dir=args.output_dir,
        config_name=args.config_name,
        dry_run=args.dry_run,
        save_diff=args.diff,
        verbose=args.verbose,
    )

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
