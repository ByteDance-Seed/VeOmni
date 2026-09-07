#!/usr/bin/env python3
"""Verify that repo paths and skill names referenced by agent docs exist.

The ``.agents/`` skills and knowledge docs describe the repository to coding
agents: directory trees, workflow filenames, test paths. Those references
silently rot as the tree moves, and a stale one sends an agent to edit the wrong
file. This check keeps the mechanical part honest.

What is checked, in ``AGENTS.md``, ``.agents/**/*.md`` and ``.cursor/rules/*``:

* Inline-code spans and Markdown link targets that look repo-root-relative --
  i.e. whose first segment matches an entry that exists at the repo root -- must
  resolve to a real file or directory. A ``path.py::symbol`` reference is
  checked as ``path.py``.
* ``/skill-name`` references must resolve to ``.agents/skills/<name>/SKILL.md``.
* So must a backticked name used as a skill, as in "see ``some-skill`` skill".

Anything containing a placeholder (``<m>``, ``*``, ``{a,b}``, ``$VAR``, ...) is
skipped, as are relative fragments whose first segment is not a repo-root entry
(``ops/config/registry.py`` is a legitimate shorthand for a path under
``veomni/``) and paths under ephemeral or gitignored roots such as
``.pr-drafts/`` and ``.agents_workspace/``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SCAN_GLOBS = ("AGENTS.md", ".agents/**/*.md", ".cursor/rules/*")

# Ephemeral or gitignored roots: agent docs legitimately describe files that only
# exist at runtime, or that git itself owns.
SKIP_ROOTS = frozenset({".git", ".venv", ".pr-drafts", ".agents_workspace"})

# `inline code`
INLINE_CODE = re.compile(r"`([^`\n]+)`")
# [label](target)
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
# /skill-name, as used in the dispatch tables and cross-references
SKILL_REF = re.compile(r"(?<![\w/])/(veomni-[a-z0-9-]+|create-pr)\b")
# "`some-name` skill" / "skill `some-name`" -- prose references to a skill
SKILL_PROSE = re.compile(r"`([a-z][a-z0-9-]{2,})`\s+skill\b|\bskill\s+`([a-z][a-z0-9-]{2,})`")

# Placeholders and shell syntax: not literal paths.
PLACEHOLDER_CHARS = set("<>*?{}$()|\"' \t")


def looks_like_path(candidate: str) -> bool:
    if "/" not in candidate:
        return False
    if candidate.startswith(("http://", "https://", "//")):
        return False
    if any(char in PLACEHOLDER_CHARS for char in candidate):
        return False
    if ".." in candidate:
        return False
    return True


def normalize(candidate: str) -> str:
    """Strip a trailing symbol or line-number suffix, punctuation and slashes."""
    candidate = re.sub(r"::[\w.]+$", "", candidate)
    candidate = re.sub(r":\d+(-\d+)?$", "", candidate)
    return candidate.rstrip("/.,;:")


def repo_root_entries(repo_root: Path) -> set[str]:
    return {entry.name for entry in repo_root.iterdir()}


def collect(text: str) -> set[str]:
    return set(INLINE_CODE.findall(text)) | set(LINK_TARGET.findall(text))


def scan(repo_root: Path) -> list[str]:
    roots = repo_root_entries(repo_root)
    errors: list[str] = []

    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(sorted(repo_root.glob(pattern)))

    for path in files:
        if not path.is_file():
            continue
        where = path.relative_to(repo_root)
        text = path.read_text(encoding="utf-8")

        for raw in sorted(collect(text)):
            candidate = normalize(raw.strip())
            if not looks_like_path(candidate):
                continue
            if candidate.split("/", 1)[0] not in roots:
                continue
            if candidate.split("/", 1)[0] in SKIP_ROOTS:
                continue
            if not (repo_root / candidate).exists():
                errors.append(f"{where}: references missing path `{candidate}`")

        named = set(SKILL_REF.findall(text))
        for before, after in SKILL_PROSE.findall(text):
            named.add(before or after)

        for skill in sorted(named):
            if not (repo_root / ".agents" / "skills" / skill / "SKILL.md").is_file():
                errors.append(f"{where}: references missing skill `/{skill}`")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="VeOmni repository root (default: two levels above this file, i.e. repo root from scripts/ci/)",
    )
    args = parser.parse_args()
    repo_root: Path = args.repo_root.resolve()

    if not (repo_root / ".agents").is_dir():
        print(f"error: .agents directory not found under {repo_root}", file=sys.stderr)
        return 2

    errors = scan(repo_root)
    if errors:
        print("Agent docs reference paths or skills that do not exist:\n", file=sys.stderr)
        for message in errors:
            print(f"  {message}", file=sys.stderr)
        print(
            "\nFix the reference, or if the target moved, update the doc. "
            "Placeholders should use <angle brackets> so they are skipped.",
            file=sys.stderr,
        )
        return 1

    print("All repo paths and skill references in agent docs resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
