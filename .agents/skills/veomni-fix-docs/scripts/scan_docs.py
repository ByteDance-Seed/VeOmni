#!/usr/bin/env python3
"""Check Markdown links, anchors and prose without modifying files or using the network."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


BAD_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[^#\s]")
DUPLICATE_WORD_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]+)\s+\1\b", re.IGNORECASE)
FALLBACK_EXCLUDES = {".git", ".venv", "venv", "node_modules", "__pycache__", "_build"}
NOT_CHECKED = [
    "External URL availability and remote anchors (no network requests).",
    "MyST directive bodies, generated targets, MDX/JSX semantics and site-specific routing.",
    "Spelling, grammar and technical meaning beyond repeated words; these require a separate review.",
]


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    severity: str
    rule: str
    message: str


@dataclass
class Document:
    anchors: set[str] = field(default_factory=set)
    myst_targets: set[str] = field(default_factory=set)
    links: list[tuple[int, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files/directories, relative to --root")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--fail-on", choices=("error", "warning", "never"), default="error")
    return parser.parse_args()


def repository_markdown(root: Path) -> list[Path]:
    try:
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
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0:
        candidates = [root / os.fsdecode(item) for item in result.stdout.split(b"\0") if item]
    else:
        candidates = []
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [name for name in children if name not in FALLBACK_EXCLUDES]
            candidates.extend(Path(directory) / name for name in files if Path(name).suffix.lower() in {".md", ".mdx"})
    return sorted({path for path in candidates if path.is_file() and path.resolve().is_relative_to(root)})


def selected_markdown(root: Path, raw_paths: list[str]) -> list[Path]:
    candidates = None
    selected: set[Path] = set()
    if not raw_paths:
        selected.update(repository_markdown(root))
    for raw_path in raw_paths:
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path is outside the repository root: {raw_path}")
        if not path.exists():
            raise ValueError(f"path does not exist: {raw_path}")
        if path.is_dir():
            if candidates is None:
                candidates = repository_markdown(root)
            selected.update(child for child in candidates if child.is_relative_to(path))
        elif path.suffix.lower() in {".md", ".mdx"}:
            selected.add(path)
        else:
            raise ValueError(f"path is not a Markdown file or directory: {raw_path}")
    if not selected:
        raise ValueError("the selected scope contains no Markdown files")
    return sorted(selected)


def make_parser():
    from markdown_it import MarkdownIt
    from mdit_py_plugins.footnote import footnote_plugin

    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"]).use(footnote_plugin)
    # Preserve URLs for diagnostics: normalization can hide malformed host brackets.
    parser.normalizeLink = lambda value: value
    return parser


def plain_text(tokens, include_code: bool = True) -> str:
    parts = []
    for token in tokens:
        if token.type == "text" or (token.type == "code_inline" and include_code):
            parts.append(token.content)
        elif token.type == "image":
            parts.append(plain_text(token.children or [], include_code))
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif token.type == "code_inline":
            parts.append("\0")
    return "".join(parts)


def github_slug(title: str, assigned: set[str]) -> str:
    """Slug rendered heading text, retaining Unicode letters/numbers/marks and unique IDs."""
    characters = []
    for char in title.strip().lower():
        if "\ufe00" <= char <= "\ufe0f" or "\U000e0100" <= char <= "\U000e01ef":
            continue  # Emoji presentation selectors do not become part of the anchor.
        if char in " -_" or unicodedata.category(char)[0] in "LNM":
            characters.append(char)
    base = "".join(characters).replace(" ", "-")
    result = base
    count = 0
    while result in assigned:
        count += 1
        result = f"{base}-{count}"
    assigned.add(result)
    return result


class HTMLReferences(HTMLParser):
    """Read visible HTML references, including multiline attributes and explicit IDs."""

    def __init__(self, document: Document):
        super().__init__(convert_charrefs=True)
        self.document = document
        self.source_line = 1
        self.literal_depth = 0

    def consume(self, content: str, line: int) -> None:
        # Reset positions while retaining literal element state across inline tokens.
        self.reset()
        self.source_line = line
        self.feed(content)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        literal = tag in {"pre", "code", "script", "style"}
        if self.literal_depth:
            self.literal_depth += int(literal)
            return
        values = dict(attrs)
        explicit = values.get("id") or (values.get("name") if tag == "a" else None)
        if explicit:
            self.document.anchors.add(explicit)
        if literal:
            self.literal_depth += 1
            return
        attribute = "href" if tag == "a" else "src" if tag == "img" else None
        if attribute and values.get(attribute) is not None:
            self.document.links.append((self.source_line + self.getpos()[0] - 1, values[attribute]))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"pre", "code", "script", "style"} and self.literal_depth:
            self.literal_depth -= 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


class Scanner:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.parser = make_parser()
        self.documents: dict[Path, Document] = {}
        self.skipped: Counter = Counter()

    def finding(self, source: Path, line: int, severity: str, rule: str, message: str) -> Finding:
        return Finding(source.relative_to(self.root).as_posix(), line, severity, rule, message)

    def document(self, path: Path) -> Document:
        if path in self.documents:
            return self.documents[path]
        document = Document()
        self.documents[path] = document
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            document.findings.append(self.finding(path, 1, "error", "encoding", "file is not valid UTF-8"))
            return document
        except OSError as exc:
            document.findings.append(self.finding(path, 1, "error", "read-error", str(exc)))
            return document

        tokens = self.parser.parse(text)
        heading_ids: set[str] = set()
        html = HTMLReferences(document)
        if path.suffix.lower() == ".mdx":
            document.skipped["mdx-semantics"] += 1
        for index, token in enumerate(tokens):
            line = token.map[0] + 1 if token.map else 1
            if token.type == "heading_open":
                title = plain_text(tokens[index + 1].children or [])
                document.anchors.add(github_slug(title, heading_ids))
            elif token.type == "html_block":
                html.consume(token.content, line)
            elif token.type == "fence":
                if token.info.lstrip().startswith("{"):
                    document.skipped["myst-directive"] += 1
                # The token map includes the closing line only when a closer was consumed.
                content_lines = token.content.count("\n") + int(
                    bool(token.content) and not token.content.endswith("\n")
                )
                if token.map[1] - token.map[0] <= 1 + content_lines:
                    document.findings.append(
                        self.finding(
                            path, line, "warning", "unclosed-fence", "verify code fence: no closing line found"
                        )
                    )
            elif token.type == "inline":
                for content in token.content.splitlines():
                    target = re.fullmatch(r"\(([^)]+)\)=\s*", content)
                    if target:
                        document.myst_targets.add(target.group(1))
                children = token.children or []
                child_line = line
                prose = []
                for child in children:
                    was_literal = bool(html.literal_depth)
                    if child.type in {"link_open", "image"} and not html.literal_depth:
                        attr = "src" if child.type == "image" else "href"
                        document.links.append((child_line, child.attrGet(attr) or ""))
                    elif child.type == "html_inline":
                        html.consume(child.content, child_line)
                    newlines = 1 if child.type in {"softbreak", "hardbreak"} else child.content.count("\n")
                    if was_literal or html.literal_depth:
                        prose.append("\0" + "\n" * newlines)
                    elif child.type == "html_inline":
                        prose.append("\n" * newlines)
                    else:
                        prose.append(plain_text([child], include_code=False))
                    child_line += newlines
                for offset, content in enumerate("".join(prose).split("\n")):
                    if BAD_HEADING_RE.match(content):
                        document.findings.append(
                            self.finding(
                                path, line + offset, "warning", "heading-space", "verify intended heading spacing"
                            )
                        )
                    for duplicate in DUPLICATE_WORD_RE.finditer(content):
                        if not duplicate.group(1).isupper():
                            document.findings.append(
                                self.finding(
                                    path,
                                    line + offset,
                                    "warning",
                                    "duplicate-word",
                                    f"verify repeated word: {duplicate.group(0)!r}",
                                )
                            )
        return document

    def validate_link(self, source: Path, line: int, destination: str) -> Finding | None:
        destination = destination.strip()
        if not destination or destination == "#":
            return None
        if any(char in unquote(destination) for char in "{}$*"):
            self.skipped["template-target"] += 1
            return None
        try:
            parsed = urlsplit(destination)
        except ValueError as exc:
            return self.finding(source, line, "error", "invalid-url", f"invalid URL {destination!r}: {exc}")
        if parsed.scheme or parsed.netloc:
            self.skipped["external-url"] += 1
            return None
        link_path = unquote(parsed.path)
        target = source
        if link_path:
            requested = self.root / link_path.lstrip("/") if link_path.startswith("/") else source.parent / link_path
            requested = Path(os.path.abspath(requested))
            if not requested.is_relative_to(self.root):
                self.skipped["outside-root"] += 1
                return None
            target = self.root
            wrong_case = False
            try:
                for part in requested.relative_to(self.root).parts:
                    entries = {child.name: child for child in target.iterdir()}
                    if part in entries:
                        target = entries[part]
                    else:
                        matches = [child for name, child in entries.items() if name.casefold() == part.casefold()]
                        if len(matches) != 1:
                            return self.finding(
                                source,
                                line,
                                "error",
                                "broken-local-link",
                                f"local target does not exist: {destination}",
                            )
                        target = matches[0]
                        wrong_case = True
                    if not target.resolve().is_relative_to(self.root):
                        self.skipped["outside-root"] += 1
                        return None
                if not target.exists():
                    return self.finding(source, line, "error", "broken-local-link", f"missing target: {destination}")
            except OSError as exc:
                return self.finding(
                    source, line, "error", "read-error", f"cannot inspect target {destination!r}: {exc}"
                )
            if wrong_case:
                actual = target.relative_to(self.root).as_posix()
                return self.finding(
                    source,
                    line,
                    "error",
                    "path-case-mismatch",
                    f"path case differs: {destination}; actual path: {actual}",
                )

        fragment = unquote(parsed.fragment)
        if fragment and target.is_file() and target.suffix.lower() in {".md", ".mdx"}:
            if re.fullmatch(r"L\d+(?:-L\d+)?", fragment):
                self.skipped["github-source-line-anchor"] += 1
                return None
            document = self.document(target)
            if any(finding.rule in {"encoding", "read-error"} for finding in document.findings):
                return self.finding(
                    source, line, "warning", "unverified-anchor", f"cannot read anchor target: {destination}"
                )
            if fragment in document.myst_targets:
                self.skipped["myst-target"] += 1
            elif fragment not in document.anchors:
                return self.finding(
                    source,
                    line,
                    "warning",
                    "missing-anchor",
                    f"anchor not found by the GitHub rules: #{fragment}; verify rendered target before editing",
                )
        return None

    def scan(self, paths: list[Path]) -> list[Finding]:
        self.skipped.clear()
        findings = []
        for path in paths:
            document = self.document(path)
            self.skipped.update(document.skipped)
            findings.extend(document.findings)
            for line, destination in document.links:
                finding = self.validate_link(path, line, destination)
                if finding:
                    findings.append(finding)
        return sorted(set(findings))


def emit(findings: list[Finding], output_format: str, files_scanned: int, skipped: Counter) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "files_scanned": files_scanned,
                    "findings": [asdict(finding) for finding in findings],
                    "coverage": {"skipped": dict(sorted(skipped.items())), "not_checked": NOT_CHECKED},
                },
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
    print(f"Skipped targets/constructs: {dict(sorted(skipped.items()))}")
    for limitation in NOT_CHECKED:
        print(f"Not checked: {limitation}")


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
    try:
        if not root.is_dir():
            raise ValueError(f"repository root does not exist: {root}")
        scanner = Scanner(root)
        files = selected_markdown(root, args.paths)
    except ImportError as exc:
        requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
        emit_usage_error(f'{exc}. Install dependencies: python -m pip install -r "{requirements}"', args.format)
        return 2
    except (ValueError, OSError) as exc:
        emit_usage_error(str(exc), args.format)
        return 2
    findings = scanner.scan(files)
    emit(findings, args.format, len(files), scanner.skipped)
    if args.fail_on == "never":
        return 0
    if args.fail_on == "warning" and findings:
        return 1
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
