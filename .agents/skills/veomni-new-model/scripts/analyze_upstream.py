#!/usr/bin/env python3
"""Inventory local source for VeOmni onboarding without executing upstream code.

This is an optional evidence collector, not a model classifier or a migration
planner. Framework selection and task scope remain decisions in SKILL.md.
No network or third-party Python dependency is required.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
MAX_SOURCE_BYTES = 2_000_000
CONFIG_KEYS = {
    "model_type",
    "architectures",
    "_class_name",
    "processor_class",
    "num_experts",
    "num_local_experts",
}


def git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    return result.stdout.strip()


def public_remote(remote: str) -> str:
    """Remove URL credentials/query/fragment from recorded source locations."""
    if "://" not in remote:
        return remote
    parts = urlsplit(remote)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


def iter_sources(root: Path, output: Path):
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in IGNORED_DIRS
            and not (Path(current) / name).is_symlink()
            and (Path(current) / name).resolve() != output
        )
        for name in sorted(names):
            path = Path(current) / name
            if not path.is_symlink():
                yield path


def config_signals(value, location: str = "$") -> list[dict]:
    records = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in CONFIG_KEYS:
                records.append({"location": child_location, "value": child})
            records.extend(config_signals(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(config_signals(child, f"{location}[{index}]"))
    return records


def inventory(root: Path, output: Path) -> dict:
    status = git_value(root, "status", "--porcelain", "--untracked-files=normal", "--", ".")
    result = {
        "schema_version": 2,
        "source": {
            "path": str(root),
            "repository_root": git_value(root, "rev-parse", "--show-toplevel"),
            "revision": git_value(root, "rev-parse", "HEAD"),
            "remote": public_remote(git_value(root, "remote", "get-url", "origin")),
            "dirty": None if status == "UNKNOWN" else bool(status),
        },
        "classes": [],
        "functions": [],
        "imports": [],
        "configs": [],
        "licenses": [],
        "skipped_or_unreadable": [],
        "limitations": [
            "Imports/config keys are observations, not proof of the target model's framework.",
            "Inspect aliases, indirect inheritance, construction and mixed-component boundaries manually.",
            "Python AST and selected JSON configs only; dynamic code and other source/config formats need review.",
            "No checkpoint inspection, code execution, automatic route, or validation requirement is produced.",
        ],
    }
    for path in iter_sources(root, output):
        name = path.relative_to(root).as_posix()
        if path.name.lower().startswith(("license", "copying", "notice")):
            result["licenses"].append(name)
        is_config = path.suffix.lower() == ".json" and (
            "config" in path.name.lower() or path.name.lower() in {"model_index.json", "params.json"}
        )
        if path.suffix != ".py" and not is_config:
            continue
        try:
            if path.stat().st_size > MAX_SOURCE_BYTES:
                result["skipped_or_unreadable"].append({"file": name, "reason": "size limit"})
                continue
            source = path.read_text(encoding="utf-8")
            if is_config:
                result["configs"].append({"file": name, "signals": config_signals(json.loads(source))})
                continue
            tree = ast.parse(source, filename=name)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result["functions"].append({"file": name, "line": node.lineno, "name": node.name})
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    result["classes"].append(
                        {
                            "file": name,
                            "line": node.lineno,
                            "name": node.name,
                            "bases": [ast.unparse(base) for base in node.bases],
                            "methods": [
                                child.name
                                for child in node.body
                                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            ],
                        }
                    )
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    result["imports"].append(
                        {
                            "file": name,
                            "line": node.lineno,
                            "statement": ast.unparse(node),
                        }
                    )
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
            result["skipped_or_unreadable"].append({"file": name, "reason": f"{type(exc).__name__}: {exc}"})
    return result


def render_summary(data: dict) -> str:
    source = data["source"]
    lines = [
        "# Local source inventory",
        "",
        f"Source: {source['path']}",
        f"Revision: {source['revision']}",
        f"Dirty source: {source['dirty']}",
        "",
        "Use the JSON inventory for complete imports, config signals, and locations.",
        "Select routes per actual component; decide conversion/training/resume needs from task scope.",
        "",
        "## Classes (first 80)",
        "",
    ]
    for item in data["classes"][:80]:
        lines.append(f"- {item['file']}:{item['line']} — {item['name']}({', '.join(item['bases'])})")
    if not data["classes"]:
        lines.append("- No statically declared Python class found; inspect construction manually.")
    lines.extend(["", "## Limits and incomplete reads", ""])
    lines.extend(f"- {note}" for note in data["limitations"])
    lines.extend(f"- {item['file']}: {item['reason']}" for item in data["skipped_or_unreadable"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="Local source directory")
    parser.add_argument("--output", required=True, type=Path, help="Inventory directory; existing files are preserved")
    args = parser.parse_args(argv)
    root, output = args.upstream.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"source directory does not exist: {root}")
    # Prevent inventory files from becoming part of the source or containing it.
    if output == root or output in root.parents:
        parser.error("output must not equal or contain the source directory")
    names = ("source-inventory.json", "source-inventory.md")
    try:
        if any((output / name).exists() or (output / name).is_symlink() for name in names):
            raise FileExistsError("inventory output already exists; choose a new output directory")
        data = inventory(root, output)
        output.mkdir(parents=True, exist_ok=True)
        payloads = (json.dumps(data, indent=2, ensure_ascii=False) + "\n", render_summary(data))
        for name, payload in zip(names, payloads):
            with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote source inventory to {output}")
    print("No framework route or implementation requirements were inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
