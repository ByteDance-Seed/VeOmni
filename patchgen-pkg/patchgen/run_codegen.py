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
"""Runner for the Modeling Code Generator.

The unified ``patchgen`` console script (``patchgen.cli``) is the user
entrypoint; this module exposes the :func:`build_cli` factory that
``patchgen.cli`` and any direct downstream caller can use to drive
codegen against their own :class:`DiscoveryConfig`.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ._normalize import ruff_fix_and_format
from .codegen import CodegenError, ModelingCodeGenerator, load_patch_config_module
from .patch_spec import PatchConfig


@dataclass(frozen=True)
class DiscoveryConfig:
    """Where to find ``*_patch_gen_config.py`` files and how to name them.

    Args:
        search_root: directory to walk for ``*_patch_gen_config.py`` files.
        package_prefix: dotted package prefix prepended to each file's path
            relative to ``search_root`` to form an importable module name.
            E.g. ``search_root=<project>/models/transformers`` +
            ``package_prefix="<project>.models.transformers"``.
        legacy_patches_prefix: optional shortcut prefix that callers can use
            on the CLI: ``patches.<name>`` is expanded to
            ``<legacy_patches_prefix>.<name>``. Most projects leave this
            ``None``; it exists to support trees that historically stored
            patch bodies under a ``patches/`` subdirectory.
        ruff_extra_ignore: ruff codes to add on top of the default
            ``E402,B007`` when normalizing generated files. Callers whose
            ``pyproject.toml`` does NOT globally ignore ``E501`` should set
            ``("E501",)`` here.
    """

    search_root: Path
    package_prefix: str
    legacy_patches_prefix: Optional[str] = None
    ruff_extra_ignore: tuple[str, ...] = ()
    ruff_isolated: bool = False  # pass --isolated to ruff (line-length 88, no project config)

    @property
    def package_root(self) -> Path:
        """Filesystem directory that holds the **first segment** of
        ``package_prefix`` — i.e. the directory that must be on ``sys.path``
        for the package's modules to import normally.

        Used to seed ``load_patch_config_module(search_roots=...)`` so the
        file-walk works even when the project is not installed on
        ``sys.path`` (e.g. fresh clone, sandboxed CI). For
        ``DiscoveryConfig(search_root=Path('X/myproj/models'),
        package_prefix='myproj.models')`` this returns ``Path('X')``.

        ``search_root`` is ``.resolve()``-d first so a relative input like
        ``Path('models')`` whose ``parents`` chain is shorter than
        ``len(package_prefix.split('.'))`` does not raise ``IndexError``.
        Resolving against CWD also matches how a user would normally
        construct a relative path here (e.g. running the CLI from the
        project root).
        """
        depth = len(self.package_prefix.split("."))
        resolved = self.search_root.resolve()
        return resolved.parents[depth - 1] if depth > 0 else resolved


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


def strip_diff_trailing_ws(text: str) -> str:
    """Strip trailing whitespace from every line.

    Unified diffs produced by :func:`difflib.unified_diff` leave a single
    trailing space on context lines whose source line was empty. Editors
    and pre-commit hooks routinely strip that, so without this normalization
    the freshly-written ``.diff`` would drift the moment any editor touches
    it.
    """
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n" if text else text


def list_patch_configs(discovery: DiscoveryConfig) -> list[str]:
    """Discover ``*_patch_gen_config.py`` files under ``discovery.search_root``.

    Each file's path relative to ``search_root`` is joined with
    ``discovery.package_prefix`` to form the importable module name. The
    module is then imported (via :func:`load_patch_config_module`, so parent
    ``__init__.py`` side effects are bypassed) and kept only if it exposes a
    ``config`` attribute of type :class:`PatchConfig`.
    """
    configs: list[str] = []
    if not discovery.search_root.exists():
        return configs

    search_roots = [discovery.package_root]
    for py_file in sorted(discovery.search_root.rglob("*_patch_gen_config.py")):
        if py_file.name.startswith("_"):
            continue
        rel_path = py_file.relative_to(discovery.search_root).with_suffix("")
        module_name = ".".join((discovery.package_prefix, *rel_path.parts))
        try:
            module = load_patch_config_module(module_name, search_roots=search_roots)
        except ImportError:
            continue
        if hasattr(module, "config") and isinstance(module.config, PatchConfig):
            configs.append(module_name)

    return configs


def normalize_patch_module(patch_module: str, discovery: DiscoveryConfig) -> str:
    """Expand the ``patches.<X>`` shorthand into a fully-qualified module name.

    If ``discovery.legacy_patches_prefix`` is set and ``patch_module`` starts
    with ``patches.``, the prefix is replaced with the discovery's
    ``legacy_patches_prefix``. Anything else is returned unchanged.
    """
    if discovery.legacy_patches_prefix and patch_module.startswith("patches."):
        return f"{discovery.legacy_patches_prefix}.{patch_module.removeprefix('patches.')}"
    return patch_module


def default_output_dir_for_module(module: object) -> Path:
    """``<patch_module_dir>/generated/``, with ``patches/`` parents collapsed.

    If the patch module lives in a ``patches/`` subdirectory (the legacy
    layout — see ``legacy_patches_prefix`` on :class:`DiscoveryConfig`),
    the generated/ dir sits next to that subdir. Otherwise it sits
    directly next to the config file.
    """
    module_path = Path(module.__file__).resolve()
    if module_path.parent.name == "patches":
        return module_path.parent.parent / "generated"
    return module_path.parent / "generated"


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
    ruff_extra_ignore: tuple[str, ...] = (),
    ruff_isolated: bool = False,
    discovery: Optional[DiscoveryConfig] = None,
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
        ruff_extra_ignore: extra ruff codes passed through to
            :func:`ruff_fix_and_format`. Use this when the caller's
            ``pyproject.toml`` does not globally ignore those codes.
        discovery: optional discovery config used to expand the legacy
            ``patches.<name>`` shorthand and seed ``sys.path`` for the
            patch-config import. If ``None``, the legacy shorthand is
            disabled and the import relies on whatever is already on
            ``sys.path``.

    Returns:
        The generated source code (post-normalization), or None on error.

    Notes:
        The written file is normalized via ``ruff check --fix`` + ``ruff
        format`` before this function returns, and the ``.diff`` (when
        ``save_diff=True``) is built from the normalized form. This guarantees
        ``run_codegen`` output matches what :mod:`check_patchgen` validates
        against — so a fresh regen never produces immediate drift.
    """
    try:
        if discovery is not None:
            patch_module = normalize_patch_module(patch_module, discovery)
        if verbose:
            print(f"Loading patch module: {patch_module}")
        # Use spec_from_file_location so heavy parent __init__.py side effects
        # (model registries, torch kernels, ...) are not triggered just to read
        # the config object. ``discovery.package_root`` seeds the loader's
        # search path so projects that aren't installed on ``sys.path`` still
        # resolve the patch config file.
        search_roots = [discovery.package_root] if discovery is not None else None
        module = load_patch_config_module(patch_module, search_roots=search_roots)
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
        generator.generate(output_path)

        # Normalize via the same ruff pipeline that check_patchgen validates
        # against. Without this, raw codegen output can drift from the
        # committed form whenever ruff would rewrite a single line, and a
        # contributor running run_codegen would commit immediate drift.
        ruff_fix_and_format(
            output_path,
            extra_ignore=ruff_extra_ignore or None,
            isolated=ruff_isolated,
        )
        output = output_path.read_text(encoding="utf-8")

        print(f"\n✓ Generated: {output_path}")
        print(f"  Lines: {len(output.splitlines())}")

        if save_diff:
            diff_output = strip_diff_trailing_ws(
                build_unified_diff(
                    original_source=generator.source_code,
                    generated_source=output,
                    source_module=config.source_module,
                    target_file=config.target_file,
                )
            )
            diff_path = default_diff_path(output_dir, config.target_file)
            diff_path.write_text(diff_output, encoding="utf-8")
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


def _build_parser(prog_name: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Modeling Code Generator - Generate patched HuggingFace modeling code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s <patch_module>
  %(prog)s <patch_module> -o /path/to/output
  %(prog)s <patch_module> --dry-run
  %(prog)s <patch_module> --diff
  %(prog)s --list

Drift check:
  %(prog)s --check          # exit non-zero if generated files are stale
  %(prog)s --check --fix    # overwrite checked-in files to match regen
  See `%(prog)s --check --help` for drift-check-mode options.
        """,
    )

    parser.add_argument(
        "patch_module",
        nargs="?",
        help="Dotted patch-config module name (e.g., '<project>.models.qwen3.qwen3_patch_gen_config')",
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
        "--all",
        action="store_true",
        help="Regenerate all discovered patch configs at once",
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
    return parser


def _run_with_discovery(
    args: argparse.Namespace,
    discovery: DiscoveryConfig,
    parser: argparse.ArgumentParser,
) -> int:
    if args.list:
        print("Available patch configurations:")
        configs = list_patch_configs(discovery)
        if configs:
            for cfg in configs:
                print(f"  • {cfg}")
        else:
            print("  (none found)")
        return 0

    if args.all:
        configs = list_patch_configs(discovery)
        if not configs:
            print("No patch configs found.", file=sys.stderr)
            return 1
        failed: list[str] = []
        for cfg in configs:
            print(f"\n{'=' * 70}")
            print(f"Generating: {cfg}")
            print("=" * 70)
            result = run_codegen(
                patch_module=cfg,
                output_dir=None,
                config_name=args.config_name,
                dry_run=args.dry_run,
                save_diff=args.diff,
                verbose=args.verbose,
                ruff_extra_ignore=discovery.ruff_extra_ignore,
                ruff_isolated=discovery.ruff_isolated,
                discovery=discovery,
            )
            if result is None:
                failed.append(cfg)
        if failed:
            print(f"\nFailed configs: {failed}", file=sys.stderr)
            return 1
        print(f"\nAll {len(configs)} configs generated successfully.")
        return 0

    if not args.patch_module:
        # ``parser.error`` writes the standard ``usage:`` line and exits 2,
        # matching argparse's own missing-argument behavior so the CLI UX
        # stays consistent across the various error paths.
        parser.error("patch_module is required unless using --list or --all")

    result = run_codegen(
        patch_module=args.patch_module,
        output_dir=args.output_dir,
        config_name=args.config_name,
        dry_run=args.dry_run,
        save_diff=args.diff,
        verbose=args.verbose,
        ruff_extra_ignore=discovery.ruff_extra_ignore,
        ruff_isolated=discovery.ruff_isolated,
        discovery=discovery,
    )

    return 0 if result else 1


def build_cli(
    discovery: DiscoveryConfig,
    prog_name: Optional[str] = None,
) -> Callable[[Optional[list[str]]], int]:
    """Return a ``main()``-shaped callable that runs the run_codegen CLI
    rooted at ``discovery``.

    The unified ``patchgen`` console script uses this to mount its own
    CLI; downstream projects that want a differently-named entry (e.g.
    ``python -m <pkg>.patchgen``) can do the same without copy-pasting
    argparse.
    """
    parser = _build_parser(prog_name=prog_name)

    def _main(argv: Optional[list[str]] = None) -> int:
        args = parser.parse_args(argv)
        return _run_with_discovery(args, discovery, parser)

    return _main


if __name__ == "__main__":
    # Reject `python -m patchgen.run_codegen ...` loudly. Without this guard
    # the module imports cleanly and exits 0 without parsing argv — a silent
    # no-op that's easy for automation to misread as success.
    import sys

    sys.stderr.write(
        "patchgen.run_codegen is library code, not an executable entry point.\n"
        "Use the `patchgen` console script (or `python -m patchgen`) instead:\n"
        "  patchgen <module> -o <output_dir> [--diff]\n"
        "  patchgen --all --diff\n"
    )
    sys.exit(2)
