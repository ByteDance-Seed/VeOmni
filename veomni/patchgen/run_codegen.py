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
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.patches.qwen3_gpu_patches

    # Generate to a specific output directory
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.patches.qwen3_gpu_patches -o /path/to/output

    # List available patch configurations
    python -m veomni.patchgen.run_codegen --list

    # Dry run (show what would be generated without writing)
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --dry-run

    # Diff generated file against original HuggingFace code
    python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --diff
"""

import argparse
import difflib
import importlib
import shutil
import subprocess
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
    verbose: bool = False,
) -> Optional[str]:
    """
    Run code generation for a patch configuration.

    Args:
        patch_module: Module path containing the PatchConfig
        output_dir: Directory to write generated files (defaults to sibling generated/ next to patch module)
        config_name: Name of the config variable in the module
        dry_run: If True, don't write files
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
            print(f"  Diff:   {default_diff_path(output_dir, config.target_file)}")
            print(f"  Source lines: ~{len(generator.source_code.splitlines())}")
            print(f"  Patches to apply: {len(config.patches)}")
            return generator.source_code

        # Actually generate
        output_path = output_dir / config.target_file
        output = generator.generate(output_path)

        print(f"\n✓ Generated: {output_path}")
        print(f"  Lines: {len(output.splitlines())}")

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


def run_diff(
    patch_module: str,
    output_dir: Optional[Path],
    config_name: str = "config",
    use_external_diff: bool = True,
    context_lines: int = 3,
    save_patch: Optional[Path] = None,
    open_in_vscode: bool = False,
) -> int:
    """
    Show diff between generated code and original HuggingFace code.

    Args:
        patch_module: Module path containing the PatchConfig
        output_dir: Directory where generated files are stored
        config_name: Name of the config variable in the module
        use_external_diff: If True, try to use external diff tool (delta, diff)
        context_lines: Number of context lines for unified diff
        save_patch: If provided, save the diff to this file path
        open_in_vscode: If True, open diff in VS Code

    Returns:
        0 on success, 1 on error
    """
    try:
        # Import the patch module to get config
        module = importlib.import_module(normalize_patch_module(patch_module))
        config = getattr(module, config_name)

        if output_dir is None:
            output_dir = default_output_dir_for_module(module)

        if not isinstance(config, PatchConfig):
            print(f"Error: {config_name} in {patch_module} is not a PatchConfig", file=sys.stderr)
            return 1

        # Get the generated file path
        generated_path = output_dir / config.target_file
        if not generated_path.exists():
            print(f"Error: Generated file not found: {generated_path}", file=sys.stderr)
            print(f"Run 'python run_codegen.py {patch_module}' first to generate it.")
            return 1

        # Get original HF source
        from .codegen import get_module_source

        original_source = get_module_source(config.source_module)

        # Read generated file
        generated_source = generated_path.read_text()

        # Create original file path for reference
        module_path = config.source_module.replace(".", "/") + ".py"

        # Write original to a persistent temp file (needed for VS Code)
        original_tmp_path = output_dir / f"_original_{config.target_file}"
        original_tmp_path.write_text(original_source)

        # Open in VS Code
        if open_in_vscode:
            if shutil.which("code"):
                print("Opening diff in VS Code:")
                print(f"  Left (Original):  {original_tmp_path}")
                print(f"  Right (Generated): {generated_path}")
                subprocess.run(["code", "--diff", str(original_tmp_path), str(generated_path)])
                return 0
            else:
                print("Error: 'code' command not found. Install VS Code and add to PATH.", file=sys.stderr)
                return 1

        # Generate unified diff
        diff_output = build_unified_diff(
            original_source=original_source,
            generated_source=generated_source,
            source_module=config.source_module,
            target_file=config.target_file,
            context_lines=context_lines,
        )

        # Save to patch file
        if save_patch:
            save_patch.parent.mkdir(parents=True, exist_ok=True)
            save_patch.write_text(diff_output)
            print(f"Saved patch file: {save_patch}")
            print(f"  Open in VS Code: code {save_patch}")
            return 0

        # Try external diff tools for terminal output
        if use_external_diff:
            diff_tools = ["delta", "diff"]
            diff_tool = None
            for tool in diff_tools:
                if shutil.which(tool):
                    diff_tool = tool
                    break

            if diff_tool:
                try:
                    if diff_tool == "delta":
                        cmd = [
                            "delta",
                            "--side-by-side",
                            "--file-modified-label",
                            str(generated_path),
                            str(original_tmp_path),
                            str(generated_path),
                        ]
                    else:
                        cmd = [
                            "diff",
                            "-u",
                            f"--label=Original: {module_path}",
                            f"--label=Generated: {generated_path}",
                            str(original_tmp_path),
                            str(generated_path),
                        ]

                    print("Comparing:")
                    print(f"  Original:  {config.source_module}")
                    print(f"  Generated: {generated_path}")
                    print()

                    result = subprocess.run(cmd)
                    return 0 if result.returncode in (0, 1) else result.returncode
                finally:
                    # Clean up temp file for terminal diff
                    original_tmp_path.unlink(missing_ok=True)

        # Fallback to Python's difflib
        print("Comparing:")
        print(f"  Original:  {config.source_module}")
        print(f"  Generated: {generated_path}")
        print()

        if diff_output:
            print(diff_output)
        else:
            print("No differences found.")

        # Clean up
        original_tmp_path.unlink(missing_ok=True)
        return 0

    except ImportError as e:
        print(f"Error importing {patch_module}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Modeling Code Generator - Generate patched HuggingFace modeling code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches -o /path/to/output
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --dry-run
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --diff
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --diff --vscode
  %(prog)s veomni.models.transformers.qwen3.patches.qwen3_gpu_patches --diff --save-patch changes.patch
  %(prog)s --list
        """,
    )

    parser.add_argument(
        "patch_module",
        nargs="?",
        help="Patch module to use (e.g., 'veomni.models.transformers.qwen3.patches.qwen3_gpu_patches')",
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
        help="Show diff between generated file and original HuggingFace code",
    )
    parser.add_argument(
        "--no-external-diff",
        action="store_true",
        help="Use Python difflib instead of external diff tools (delta, diff)",
    )
    parser.add_argument(
        "--vscode",
        action="store_true",
        help="Open diff in VS Code's built-in diff viewer (requires --diff)",
    )
    parser.add_argument(
        "--save-patch",
        type=Path,
        metavar="FILE",
        help="Save diff to a .patch file for viewing in VS Code (requires --diff)",
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

    # Require patch_module for generation or diff
    if not args.patch_module:
        parser.error("patch_module is required unless using --list")

    # Diff mode
    if args.diff:
        return run_diff(
            patch_module=normalize_patch_module(args.patch_module),
            output_dir=args.output_dir,
            config_name=args.config_name,
            use_external_diff=not args.no_external_diff,
            save_patch=args.save_patch,
            open_in_vscode=args.vscode,
        )

    # Run generation
    result = run_codegen(
        patch_module=normalize_patch_module(args.patch_module),
        output_dir=args.output_dir,
        config_name=args.config_name,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
