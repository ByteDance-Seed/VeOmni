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
* A bare ``*.yml`` / ``*.yaml`` filename must match a tracked file's basename.
  Docs name workflows this way far more often than by full path, and a deleted
  workflow is exactly the rot this check exists to catch. Other slash-less
  references are skipped: docs legitimately name upstream transformers modules
  and illustrative filenames that do not exist here.
* ``/skill-name`` references must resolve to ``.agents/skills/<name>/SKILL.md``.
* So must a backticked name used in prose skill form, as in "see the
  ``<name>`` skill".

Skill references are additionally checked across every tracked text file, not
just the doc globs: a renamed skill leaves dangling ``/skill-name`` mentions in
code comments and workflow files too.

Anything containing a placeholder (``<m>``, ``*``, ``{a,b}``, ``$VAR``, ...) is
skipped, as are relative fragments whose first segment is not a repo-root entry
(``ops/config/registry.py`` is a legitimate shorthand for a path under
``veomni/``) and paths under ephemeral or gitignored roots such as
``.pr-drafts/`` and ``.agents_workspace/``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
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
# /skill-name, as used in the dispatch tables and cross-references. The
# lookbehind also rejects `.`, so a relative path to a sibling checkout
# (`../veomni-a`) is not mistaken for a skill -- the repo is called VeOmni, so
# those paths turn up in docs.
SKILL_REF = re.compile(r"(?<![\w/.])/(veomni-[a-z0-9-]+|create-pr)\b")
# "`<name>` skill" / "skill `<name>`" -- prose references to a skill
SKILL_PROSE = re.compile(r"`([a-z][a-z0-9-]{2,})`\s+skill\b|\bskill\s+`([a-z][a-z0-9-]{2,})`")

# Placeholders and shell syntax: not literal paths.
PLACEHOLDER_CHARS = set("<>*?{}$()|\"' \t")

# Slash-less references worth resolving by basename. Deliberately narrow: a bare
# `.py` name is usually an upstream transformers module or an example filename.
BARE_NAME_SUFFIXES = frozenset({".yml", ".yaml"})

# Repo-wide skill-reference scan: skip anything that is not plausibly text.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".mdc", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".txt", ".json", ".j2", ""}
)
MAX_SCAN_BYTES = 2 * 1024 * 1024


def looks_like_path(candidate: str) -> bool:
    if candidate.startswith(("http://", "https://", "//")):
        return False
    if any(char in PLACEHOLDER_CHARS for char in candidate):
        return False
    if ".." in candidate:
        return False
    if "/" not in candidate:
        return Path(candidate).suffix in BARE_NAME_SUFFIXES
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
    basenames = {path.name for path in tracked_files(repo_root)}
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
            if "/" not in candidate:
                if candidate not in basenames:
                    errors.append(f"{where}: references missing file `{candidate}`")
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


def tracked_files(repo_root: Path) -> list[Path]:
    """Every git-tracked file, as absolute paths."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [repo_root / name for name in listing.split("\0") if name]


def tracked_text_files(repo_root: Path) -> list[Path]:
    """Every git-tracked file that is plausibly text, for the skill-name scan."""
    return [
        path
        for path in tracked_files(repo_root)
        if path.suffix in TEXT_SUFFIXES and path.is_file() and path.stat().st_size <= MAX_SCAN_BYTES
    ]


def scan_skill_refs(repo_root: Path) -> list[str]:
    """Flag `/skill-name` references anywhere in the tree that resolve to nothing."""
    errors: list[str] = []
    for path in tracked_text_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # ``name`` is the reST convention in Python docstrings; fold it so the
        # prose form ("see the ``<name>`` skill") matches the same way as in Markdown.
        text = text.replace("``", "`")
        named = set(SKILL_REF.findall(text))
        for before, after in SKILL_PROSE.findall(text):
            named.add(before or after)
        for skill in sorted(named):
            if not (repo_root / ".agents" / "skills" / skill / "SKILL.md").is_file():
                errors.append(f"{path.relative_to(repo_root)}: references missing skill `/{skill}`")
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

    # scan() already checks skill refs inside the doc globs, so the repo-wide
    # pass re-reports those. Deduplicate on the whole message: keying on the
    # file alone would drop a genuine skill error just because that file
    # happened to have an unrelated path error.
    errors = scan(repo_root)
    seen = set(errors)
    errors += [error for error in scan_skill_refs(repo_root) if error not in seen]
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
