#!/usr/bin/env python3
"""Verify that task script paths referenced in docs shell blocks exist on disk."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Fenced code blocks used for copy-paste training commands
SHELL_FENCE = re.compile(r"```(?:shell|bash|sh)\s*\n(.*?)```", re.DOTALL | re.MULTILINE)
# Paths like tasks/train_vlm.py or tasks/deprecated_task/train_wan.py
TASK_SCRIPT = re.compile(r"tasks/[A-Za-z0-9_./]+\.py")

# Placeholder examples in docs (not real files)
ALLOWLIST_BASENAMES = frozenset({"your_train_script.py"})


def collect_paths_from_docs(docs_dir: Path) -> list[tuple[Path, str]]:
    """Return (markdown_file, referenced_path) pairs."""
    pairs: list[tuple[Path, str]] = []
    for md_path in sorted(docs_dir.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for block in SHELL_FENCE.findall(text):
            for match in TASK_SCRIPT.findall(block):
                pairs.append((md_path, match))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="VeOmni repository root (default: two levels above this file, i.e. repo root from scripts/ci/)",
    )
    args = parser.parse_args()
    repo_root: Path = args.repo_root
    docs_dir = repo_root / "docs"
    if not docs_dir.is_dir():
        print(f"error: docs directory not found: {docs_dir}", file=sys.stderr)
        return 2

    errors: list[str] = []
    for md_path, rel in collect_paths_from_docs(docs_dir):
        basename = Path(rel).name
        if basename in ALLOWLIST_BASENAMES:
            continue
        full = repo_root / rel
        if not full.is_file():
            rel_md = md_path.relative_to(repo_root)
            errors.append(f"{rel_md}: missing {rel}")

    if errors:
        print("Documentation references task scripts that do not exist:\n", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        return 1

    print("All task script paths in docs shell blocks exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
