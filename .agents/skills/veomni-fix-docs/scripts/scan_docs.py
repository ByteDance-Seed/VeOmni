#!/usr/bin/env python3
"""Scan repository Markdown for deterministic documentation defects."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
BAD_HEADING_RE = re.compile(r"^#{1,6}[^#\s]")
DUPLICATE_WORD_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]+)\s+\1\b", re.IGNORECASE)
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(.+?)\s*$")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)(\1)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_SLUG_RE = re.compile(r"[^\w\- ]", re.UNICODE)
SCHEMES_TO_SKIP = {"data", "ftp", "http", "https", "irc", "mailto", "ssh", "tel"}


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    severity: str
    rule: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Markdown files or directories; defaults to tracked and new unignored Markdown",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format")
    parser.add_argument(
        "--fail-on",
        choices=("error", "warning", "never"),
        default="error",
        help="Minimum severity that produces exit code 1 (default: error)",
    )
    return parser.parse_args()


def repository_markdown(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
            "*.mdx",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        paths = (root / Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item)
        return sorted(path for path in paths if path.is_file())
    return sorted(path for pattern in ("*.md", "*.mdx") for path in root.rglob(pattern) if ".git" not in path.parts)


def selected_markdown(root: Path, raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return repository_markdown(root)

    selected: set[Path] = set()
    for raw_path in raw_paths:
        path = (root / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path is outside the repository root: {raw_path}") from exc
        if not path.exists():
            raise ValueError(f"path does not exist: {raw_path}")
        if path.is_dir():
            selected.update(child for pattern in ("*.md", "*.mdx") for child in path.rglob(pattern))
        elif path.suffix.lower() in {".md", ".mdx"}:
            selected.add(path)
        else:
            raise ValueError(f"path is not a Markdown file or directory: {raw_path}")
    if not selected:
        raise ValueError("the selected paths contain no Markdown files")
    return sorted(selected)


def strip_inline_code(text: str) -> str:
    return INLINE_CODE_RE.sub(lambda match: f" CODE{match.start()} ", text)


def github_slug(text: str) -> str:
    text = INLINE_CODE_RE.sub(lambda match: match.group(2), text)
    text = HTML_TAG_RE.sub("", text).strip().lower()
    return re.sub(r"\s+", "-", NON_SLUG_RE.sub("", text))


def markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence: tuple[str, int] | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return anchors

    for line in lines:
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if in_fence is None:
                in_fence = (marker[0], len(marker))
            elif marker[0] == in_fence[0] and len(marker) >= in_fence[1]:
                in_fence = None
            continue
        if in_fence is not None:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            base = github_slug(heading.group(2))
            count = counts.get(base, 0)
            anchors.add(base if count == 0 else f"{base}-{count}")
            counts[base] = count + 1
        for explicit in re.findall(r"<a\s+(?:name|id)=[\"']([^\"']+)[\"']", line, flags=re.IGNORECASE):
            anchors.add(explicit)
        myst_target = re.match(r"^\s*\(([^)]+)\)=\s*$", line)
        if myst_target:
            anchors.add(myst_target.group(1))
    return anchors


def extract_destination(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1 : raw.index(">")]
    return raw.split(maxsplit=1)[0]


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def inline_link_destinations(line: str) -> list[str]:
    """Return inline Markdown link destinations, preserving balanced parentheses."""
    destinations: list[str] = []
    cursor = 0
    while True:
        opener = line.find("](", cursor)
        if opener < 0:
            break
        cursor = opener + 2
        if is_escaped(line, opener) or line.rfind("[", 0, opener) < 0:
            continue

        start = cursor
        depth = 1
        index = start
        while index < len(line):
            char = line[index]
            if char == "\\":
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    destinations.append(line[start:index])
                    cursor = index + 1
                    break
            index += 1
        else:
            cursor = len(line)
    return destinations


def validate_link(root: Path, source: Path, destination: str, line_number: int) -> Finding | None:
    destination = destination.strip()
    if not destination or any(char in destination for char in "{}$*"):
        return None

    if destination.startswith("#"):
        fragment = unquote(destination[1:])
        if fragment not in markdown_anchors(source):
            return Finding(
                source.relative_to(root).as_posix(),
                line_number,
                "warning",
                "missing-anchor",
                f"heading anchor was not found in this document: #{fragment}",
            )
        return None

    parsed = urlsplit(destination)
    if parsed.scheme.lower() in SCHEMES_TO_SKIP or parsed.netloc:
        return None

    link_path = unquote(parsed.path)
    if not link_path:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", link_path):
        return None

    target = (root / link_path.lstrip("/")) if link_path.startswith("/") else (source.parent / link_path)
    target = target.resolve()
    try:
        display = target.relative_to(root).as_posix()
    except ValueError:
        display = str(target)

    if not target.exists():
        return Finding(
            source.relative_to(root).as_posix(),
            line_number,
            "error",
            "broken-local-link",
            f"local target does not exist: {destination} (resolved to {display})",
        )

    fragment = unquote(parsed.fragment)
    is_github_line_anchor = re.fullmatch(r"L\d+(?:-L\d+)?", fragment, flags=re.IGNORECASE)
    if fragment and target.suffix.lower() in {".md", ".mdx"} and not is_github_line_anchor:
        anchors = markdown_anchors(target)
        if fragment not in anchors:
            return Finding(
                source.relative_to(root).as_posix(),
                line_number,
                "warning",
                "missing-anchor",
                f"heading anchor was not found: #{fragment} in {display}",
            )
    return None


def scan_file(root: Path, path: Path) -> list[Finding]:
    relative = path.relative_to(root).as_posix()
    findings: list[Finding] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [Finding(relative, 1, "error", "encoding", "file is not valid UTF-8")]
    except OSError as exc:
        return [Finding(relative, 1, "error", "read-error", str(exc))]

    in_fence: tuple[str, int, int] | None = None
    for line_number, line in enumerate(lines, start=1):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if in_fence is None:
                in_fence = (marker[0], len(marker), line_number)
            elif marker[0] == in_fence[0] and len(marker) >= in_fence[1]:
                in_fence = None
            continue
        if in_fence is not None:
            continue

        if BAD_HEADING_RE.match(line):
            findings.append(
                Finding(relative, line_number, "error", "heading-space", "add a space after the heading marker")
            )

        prose = strip_inline_code(line)
        duplicate = DUPLICATE_WORD_RE.search(prose)
        if duplicate and not duplicate.group(1).isupper():
            findings.append(
                Finding(
                    relative, line_number, "warning", "duplicate-word", f"verify repeated word: {duplicate.group(0)!r}"
                )
            )

        destinations = inline_link_destinations(strip_inline_code(line))
        reference = REFERENCE_LINK_RE.match(line)
        if reference:
            destinations.append(reference.group(1))
        for raw_destination in destinations:
            finding = validate_link(root, path, extract_destination(raw_destination), line_number)
            if finding:
                findings.append(finding)

    if in_fence is not None:
        findings.append(
            Finding(
                relative,
                in_fence[2],
                "error",
                "unclosed-fence",
                f"fenced code block opened here is not closed ({in_fence[0] * in_fence[1]})",
            )
        )
    return findings


def emit(findings: list[Finding], output_format: str, files_scanned: int) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {"files_scanned": files_scanned, "findings": [asdict(finding) for finding in findings]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.severity}: {finding.rule}: {finding.message}")
    errors = sum(finding.severity == "error" for finding in findings)
    warnings = sum(finding.severity == "warning" for finding in findings)
    print(f"Scanned {files_scanned} Markdown file(s): {errors} error(s), {warnings} warning(s)")


def emit_usage_error(message: str, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps({"files_scanned": 0, "findings": [], "error": message}, ensure_ascii=False, indent=2))
    else:
        print(f"error: {message}", file=sys.stderr)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        emit_usage_error(f"repository root does not exist: {root}", args.format)
        return 2

    try:
        files = selected_markdown(root, args.paths)
    except ValueError as exc:
        emit_usage_error(str(exc), args.format)
        return 2
    findings = sorted(finding for path in files for finding in scan_file(root, path))
    emit(findings, args.format, len(files))

    if args.fail_on == "never":
        return 0
    if args.fail_on == "warning" and findings:
        return 1
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
