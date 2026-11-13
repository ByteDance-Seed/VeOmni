#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Lightweight sync, diff, and 3-way upgrade helper for HF-derived modeling files.

VeOmni maintains explicit, self-contained modeling modules under
``veomni/models/transformers/<model>/modeling_<model>.py``. Each file carries two
structured header lines near the top of the file::

    # Upstream-Module: transformers.models.qwen3.modeling_qwen3
    # Base-Transformers-Version: 5.16.1

Subcommands:
    status  List all HF-derived modeling files, their base version, and diff stats.
    check   Verify header metadata and upstream module resolution.
    diff    Show unified diff between normalized upstream HF source and VeOmni's file.
    init    Bootstrap a fresh modeling_<m>.py from the installed transformers package.
    merge   Run a 3-way ``git merge-file`` upgrade from old HF source to new HF source
            and print an AST signature-change summary for agent review.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_MODELS_ROOT = REPO_ROOT / "veomni" / "models" / "transformers"

UPSTREAM_MODULE_RE = re.compile(r"^#\s*Upstream-Module:\s*(\S+)\s*$", re.MULTILINE)
BASE_VERSION_RE = re.compile(r"^#\s*Base-Transformers-Version:\s*(\S+)\s*$", re.MULTILINE)
NON_HF_FAMILIES = frozenset({"flux", "movqgan", "wan"})


@dataclass(frozen=True)
class ModelingMetadata:
    path: Path
    upstream_module: str
    base_version: str


def installed_transformers_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def transformers_version_from_root(hf_root: Path) -> str | None:
    """Extract ``__version__`` from an uninstalled ``transformers`` source tree."""
    for candidate in (
        hf_root / "transformers" / "__init__.py",
        hf_root / "src" / "transformers" / "__init__.py",
    ):
        if not candidate.is_file():
            continue
        match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']', candidate.read_text(encoding="utf-8"), re.MULTILINE
        )
        if match:
            return match.group(1)
    return None


def parse_modeling_metadata(path: Path, header_lines: int = 60) -> ModelingMetadata | None:
    """Parse ``Upstream-Module`` and ``Base-Transformers-Version`` from file header."""
    head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:header_lines])
    mod_match = UPSTREAM_MODULE_RE.search(head)
    ver_match = BASE_VERSION_RE.search(head)
    if mod_match is None and ver_match is None:
        return None
    if mod_match is None or ver_match is None:
        raise ValueError(
            f"{path}: incomplete upstream metadata header; both '# Upstream-Module:' "
            "and '# Base-Transformers-Version:' are required."
        )
    return ModelingMetadata(
        path=path,
        upstream_module=mod_match.group(1),
        base_version=ver_match.group(1),
    )


def discover_modeling_files(root: Path = TRANSFORMERS_MODELS_ROOT) -> list[ModelingMetadata]:
    """Return metadata for every HF-derived ``modeling_*.py`` under ``root``."""
    found: list[ModelingMetadata] = []
    for path in sorted(root.glob("*/modeling_*.py")):
        meta = parse_modeling_metadata(path)
        if meta is not None:
            found.append(meta)
    return found


def find_module_file(module_name: str, search_roots: list[Path] | None = None) -> Path:
    """Locate ``module_name`` on disk without importing it."""
    rel = Path(*module_name.split(".")).with_suffix(".py")
    roots = list(search_roots) if search_roots is not None else [Path(p) for p in sys.path if p]
    for base in roots:
        candidate = base / rel
        if candidate.is_file():
            return candidate
        src_candidate = base / "src" / rel
        if src_candidate.is_file():
            return src_candidate
    raise FileNotFoundError(f"Cannot locate source file for module '{module_name}' in search roots.")


def rewrite_relative_imports(source: str, upstream_module: str) -> str:
    """Convert relative imports (both top-level and nested) to absolute ``transformers.*`` imports."""
    tree = ast.parse(source)
    lines = source.splitlines()
    base_parts = upstream_module.split(".")

    import_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.level <= len(base_parts)
    ]
    import_nodes.sort(key=lambda n: (n.lineno, n.col_offset))

    replacements: list[tuple[int, int, str]] = []
    for node in import_nodes:
        prefix = ".".join(base_parts[: -node.level])
        abs_module = f"{prefix}.{node.module}" if node.module else prefix
        abs_stmt = ast.unparse(ast.ImportFrom(module=abs_module, names=node.names, level=0))
        indent = " " * node.col_offset
        indented_stmt = "\n".join(indent + l if l.strip() else l for l in abs_stmt.splitlines())
        end_lineno = node.end_lineno or node.lineno
        replacements.append((node.lineno - 1, end_lineno, indented_stmt))

    for start, end, new_stmt in reversed(replacements):
        lines[start:end] = new_stmt.splitlines()

    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def ruff_format_text(source: str, repo_root: Path = REPO_ROOT) -> str:
    """Sort imports and format Python source with ruff using repo configuration when available."""
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return source
    isort_proc = subprocess.run(
        [ruff_bin, "check", "--select", "I", "--fix", "--stdin-filename", "modeling.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        cwd=repo_root,
        check=False,
    )
    if isort_proc.returncode == 0 and isort_proc.stdout:
        source = isort_proc.stdout
    proc = subprocess.run(
        [ruff_bin, "format", "--stdin-filename", "modeling.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        cwd=repo_root,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout
    return source


def normalize_upstream_source(source: str, upstream_module: str, *, format_with_ruff: bool = True) -> str:
    rewritten = rewrite_relative_imports(source, upstream_module)
    if format_with_ruff:
        return ruff_format_text(rewritten)
    return rewritten


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    positional = args.posonlyargs + args.args
    first_default = len(positional) - len(args.defaults)
    for idx, arg in enumerate(positional):
        token = f"{arg.arg}=..." if idx >= first_default else arg.arg
        parts.append(token)
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(f"{arg.arg}=..." if default is not None else arg.arg)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return f"({', '.join(parts)})"


def collect_signatures(source: str) -> dict[str, str]:
    """Index top-level functions and ``Class.method`` signatures from ``source``."""
    tree = ast.parse(source)
    sigs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs[node.name] = format_signature(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sigs[f"{node.name}.{item.name}"] = format_signature(item)
    return sigs


def diff_signatures(old_source: str, new_source: str) -> list[str]:
    """Return human-readable signature drift lines between two upstream versions."""
    old_sigs = collect_signatures(old_source)
    new_sigs = collect_signatures(new_source)
    notes: list[str] = []
    for name in sorted(old_sigs.keys() | new_sigs.keys()):
        if name not in new_sigs:
            notes.append(f"  REMOVED  {name}{old_sigs[name]}")
        elif name not in old_sigs:
            notes.append(f"  ADDED    {name}{new_sigs[name]}")
        elif old_sigs[name] != new_sigs[name]:
            notes.append(f"  CHANGED  {name}: {old_sigs[name]} -> {new_sigs[name]}")
    return notes


def resolve_targets(
    target: str | None,
    root: Path = TRANSFORMERS_MODELS_ROOT,
) -> list[ModelingMetadata]:
    all_files = discover_modeling_files(root)
    if target is None:
        return all_files
    target_path = Path(target)
    if target_path.is_file():
        meta = parse_modeling_metadata(target_path.resolve())
        if meta is None:
            raise ValueError(f"{target_path} does not have Upstream-Module header metadata.")
        return [meta]
    matches = [
        m
        for m in all_files
        if m.path.parent.name == target or m.path.stem == target or m.upstream_module.endswith(f".{target}")
    ]
    if not matches:
        raise ValueError(f"No HF-derived modeling file matched {target!r}.")
    return matches


def compute_unified_diff(meta: ModelingMetadata, search_roots: list[Path] | None = None) -> list[str]:
    upstream_file = find_module_file(meta.upstream_module, search_roots=search_roots)
    upstream_norm = normalize_upstream_source(upstream_file.read_text(encoding="utf-8"), meta.upstream_module)
    ours = meta.path.read_text(encoding="utf-8")
    rel_ours = meta.path.relative_to(REPO_ROOT) if meta.path.is_relative_to(REPO_ROOT) else meta.path
    return list(
        difflib.unified_diff(
            upstream_norm.splitlines(keepends=True),
            ours.splitlines(keepends=True),
            fromfile=f"upstream/{meta.upstream_module}@{installed_transformers_version()}",
            tofile=str(rel_ours),
            n=3,
        )
    )


def update_base_version_header(text: str, new_version: str) -> str:
    return BASE_VERSION_RE.sub(f"# Base-Transformers-Version: {new_version}", text, count=1)


def three_way_merge(
    ours_text: str,
    base_upstream_text: str,
    theirs_upstream_text: str,
    upstream_module: str,
    new_version: str,
) -> tuple[str, bool]:
    """Run ``git merge-file`` across (ours, normalized base, normalized theirs).

    Returns ``(merged_text, had_conflicts)``.
    """
    base_norm = normalize_upstream_source(base_upstream_text, upstream_module)
    theirs_norm = normalize_upstream_source(theirs_upstream_text, upstream_module)
    ours_updated = update_base_version_header(ours_text, new_version)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ours_p = tmp_dir / "ours.py"
        base_p = tmp_dir / "base.py"
        theirs_p = tmp_dir / "theirs.py"
        ours_p.write_text(ours_updated, encoding="utf-8")
        base_p.write_text(base_norm, encoding="utf-8")
        theirs_p.write_text(theirs_norm, encoding="utf-8")

        proc = subprocess.run(
            [
                "git",
                "merge-file",
                "-L",
                "veomni",
                "-L",
                "hf-base",
                "-L",
                f"hf-{new_version}",
                str(ours_p),
                str(base_p),
                str(theirs_p),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode < 0 or proc.returncode > 127:
            raise RuntimeError(f"git merge-file failed: {proc.stderr}")
        merged = ours_p.read_text(encoding="utf-8")
        return merged, proc.returncode > 0


def cmd_status(args: argparse.Namespace) -> int:
    installed_ver = installed_transformers_version()
    metas = discover_modeling_files(args.root)
    if not metas:
        print(f"No HF-derived modeling files with Upstream-Module headers found under {args.root}.")
        return 0

    print(f"Installed transformers: {installed_ver}")
    print(f"{'Model':20s} {'Base-Version':14s} {'+Lines':>8s} {'-Lines':>8s}  {'File'}")
    for meta in metas:
        diff_lines = compute_unified_diff(meta)
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        rel = meta.path.relative_to(REPO_ROOT) if meta.path.is_relative_to(REPO_ROOT) else meta.path
        print(f"{meta.path.parent.name:20s} {meta.base_version:14s} {f'+{added}':>8s} {f'-{removed}':>8s}  {rel}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    installed_ver = installed_transformers_version()
    errors: list[str] = []
    metas: list[ModelingMetadata] = []
    for path in sorted(args.root.glob("*/modeling_*.py")):
        try:
            meta = parse_modeling_metadata(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if meta is None:
            if path.parent.name not in NON_HF_FAMILIES:
                errors.append(
                    f"{path}: missing '# Upstream-Module:' and '# Base-Transformers-Version:' headers "
                    f"(add headers or register '{path.parent.name}' in NON_HF_FAMILIES)."
                )
            continue
        metas.append(meta)

    for meta in metas:
        if not meta.upstream_module.startswith("transformers.models."):
            errors.append(
                f"{meta.path}: Upstream-Module {meta.upstream_module!r} must start with 'transformers.models.'."
            )
        if installed_ver is not None:
            try:
                find_module_file(meta.upstream_module)
            except FileNotFoundError as exc:
                errors.append(f"{meta.path}: {exc}")
        if args.strict_version:
            if installed_ver is None:
                errors.append("transformers is not installed, cannot verify --strict-version.")
            elif meta.base_version != installed_ver:
                errors.append(
                    f"{meta.path}: Base-Transformers-Version is {meta.base_version}, expected installed {installed_ver}."
                )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    ver_note = (
        f"transformers=={installed_ver}" if installed_ver is not None else "header syntax (transformers not installed)"
    )
    print(f"Verified {len(metas)} HF-derived modeling file(s) against {ver_note}.")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    metas = resolve_targets(args.target, root=args.root)
    for meta in metas:
        sys.stdout.writelines(compute_unified_diff(meta))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    installed_ver = installed_transformers_version()
    if not installed_ver:
        raise RuntimeError("Cannot determine installed `transformers` version for Base-Transformers-Version header.")
    upstream_file = find_module_file(args.upstream_module)
    normalized = normalize_upstream_source(upstream_file.read_text(encoding="utf-8"), args.upstream_module)
    header = f"# Upstream-Module: {args.upstream_module}\n# Base-Transformers-Version: {installed_ver}\n\n"
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ruff_format_text(header + normalized), encoding="utf-8")
    print(f"Initialized {output_path} from {args.upstream_module} (transformers=={installed_ver}).")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    if args.new_version:
        new_ver = args.new_version
    elif args.new_hf_root is not None:
        new_ver = transformers_version_from_root(args.new_hf_root)
        if not new_ver:
            raise RuntimeError(
                f"Could not detect __version__ under --new-hf-root={args.new_hf_root}; pass --new-version explicitly."
            )
    else:
        new_ver = installed_transformers_version()
        if not new_ver:
            raise RuntimeError("Cannot determine installed `transformers` version; pass --new-version explicitly.")
    new_roots = [args.new_hf_root] if args.new_hf_root else None
    metas = resolve_targets(args.target, root=args.root)
    any_conflicts = False

    for meta in metas:
        old_file = find_module_file(meta.upstream_module, search_roots=[args.old_hf_root])
        new_file = find_module_file(meta.upstream_module, search_roots=new_roots)
        old_src = old_file.read_text(encoding="utf-8")
        new_src = new_file.read_text(encoding="utf-8")
        ours_src = meta.path.read_text(encoding="utf-8")

        merged, had_conflicts = three_way_merge(
            ours_text=ours_src,
            base_upstream_text=old_src,
            theirs_upstream_text=new_src,
            upstream_module=meta.upstream_module,
            new_version=new_ver,
        )
        if not args.dry_run:
            meta.path.write_text(merged, encoding="utf-8")

        status = "CONFLICT" if had_conflicts else "CLEAN"
        print(f"[{status}] {meta.path} ({meta.base_version} -> {new_ver})")
        sig_changes = diff_signatures(old_src, new_src)
        if sig_changes:
            print("  Upstream signature changes:")
            for line in sig_changes:
                print(f"  {line}")
        if had_conflicts:
            any_conflicts = True

    return 1 if any_conflicts else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=TRANSFORMERS_MODELS_ROOT,
        help="Root directory containing transformers model subdirectories.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show base version and diff line counts for all HF-derived models.")
    p_status.set_defaults(func=cmd_status)

    p_check = sub.add_parser("check", help="Validate Upstream-Module and Base-Transformers-Version headers.")
    p_check.add_argument(
        "--strict-version",
        action="store_true",
        help="Require Base-Transformers-Version to match installed transformers.__version__.",
    )
    p_check.set_defaults(func=cmd_check)

    p_diff = sub.add_parser("diff", help="Print unified diff against normalized upstream HF source.")
    p_diff.add_argument("target", nargs="?", help="Model directory name, file stem, or path (default: all).")
    p_diff.set_defaults(func=cmd_diff)

    p_init = sub.add_parser("init", help="Bootstrap a new modeling_<m>.py from installed transformers.")
    p_init.add_argument("upstream_module", help="e.g. transformers.models.qwen3.modeling_qwen3")
    p_init.add_argument("-o", "--output", type=Path, required=True, help="Destination modeling_<m>.py path.")
    p_init.set_defaults(func=cmd_init)

    p_merge = sub.add_parser("merge", help="3-way merge HF upgrade (old HF -> new HF) onto modeling_<m>.py.")
    p_merge.add_argument(
        "--old-hf-root",
        type=Path,
        required=True,
        help="Checkout or site-packages root of the previous transformers version.",
    )
    p_merge.add_argument(
        "--new-hf-root",
        type=Path,
        default=None,
        help="Checkout or site-packages root of the new transformers version (default: installed).",
    )
    p_merge.add_argument("--new-version", default=None, help="Override new Base-Transformers-Version string.")
    p_merge.add_argument(
        "--dry-run", action="store_true", help="Report conflicts and signature drift without writing."
    )
    p_merge.add_argument("target", nargs="?", help="Model directory name, file stem, or path (default: all).")
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
