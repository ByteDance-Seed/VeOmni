#!/usr/bin/env python3
"""Check local documentation links and navigation after a strict Sphinx build.

Uses the locked documentation environment. No network requests or training imports.
The Sphinx environment pickle must come from this checkout's own trusted build.
Root Markdown links are checked for file existence; HTML fragments are checked
against the rendered site. External URLs and source-only heading fragments are
outside this check's scope.
"""

from __future__ import annotations

import argparse
import pickle
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


class PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if not value:
                continue
            if key == "id" or (tag == "a" and key == "name"):
                self.targets.add(value)
            if key in ("href", "src", "poster"):
                self.links.append(value)


def parse_html(text: str) -> PageLinks:
    page = PageLinks()
    page.feed(text)
    page.close()
    return page


def check_links(
    path: Path,
    page: PageLinks,
    root: Path,
    *,
    check_fragments: bool,
    cache: dict[Path, PageLinks],
) -> list[str]:
    errors = []
    for link in page.links:
        try:
            url = urlsplit(link)
        except ValueError:
            errors.append(f"{path}: malformed URL {link!r}")
            continue
        if url.scheme or url.netloc:
            continue
        relative = unquote(url.path)
        if relative.startswith("/"):
            target = root / relative.lstrip("/")
        elif relative:
            target = path.parent / relative
        else:
            target = path
        target = target.resolve()
        if not target.is_relative_to(root):
            errors.append(f"{path}: local link escapes checked tree: {link}")
            continue
        if target.is_dir() and check_fragments:
            target /= "index.html"
        if not target.exists():
            errors.append(f"{path}: missing local target: {link}")
            continue
        if check_fragments and url.fragment and target.suffix == ".html":
            if target not in cache:
                cache[target] = parse_html(target.read_text(encoding="utf-8"))
            if unquote(url.fragment) not in cache[target].targets:
                errors.append(f"{path}: missing HTML anchor: {link}")
    return errors


def check_navigation(found: set[str], root_doc: str, includes: dict[str, list[str]], orphans: set[str]) -> list[str]:
    reachable: set[str] = set()
    pending = [root_doc]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(includes.get(name, []))
    return [f"{name}: page is not reachable from {root_doc}'s toctree" for name in sorted(found - reachable - orphans)]


def check(repo: Path, html_dir: Path, environment: Path) -> list[str]:
    errors = []
    parser = MarkdownIt("commonmark", {"html": True})
    for name in ("README.md", "CONTRIBUTING.md", "docs/README.md"):
        path = repo / name
        if not path.is_file():
            errors.append(f"Missing documentation entry point: {path}")
            continue
        page = parse_html(parser.render(path.read_text(encoding="utf-8")))
        errors.extend(check_links(path, page, repo, check_fragments=False, cache={}))

    if not (html_dir / "index.html").is_file() or not environment.is_file():
        return errors + ["Build the documentation first: make -C docs html SPHINXOPTS=-W"]

    # Read only the artifact produced by our own Sphinx build, never a download.
    with environment.open("rb") as stream:
        env = pickle.load(stream)
    if Path(env.srcdir).resolve() != repo / "docs":
        return errors + ["Sphinx environment belongs to another checkout; rebuild the documentation."]
    orphans = {name for name, metadata in env.metadata.items() if "orphan" in metadata}
    errors.extend(check_navigation(set(env.found_docs), env.config.root_doc, env.toctree_includes, orphans))

    # Themes also copy unrendered Jinja macros with an .html suffix into
    # _static. Downloaded source files are attachments, not site pages.
    pages = sorted(
        path
        for path in html_dir.rglob("*.html")
        if path.relative_to(html_dir).parts[0] not in {"_static", "_sources", "_downloads"}
    )
    cache = {path: parse_html(path.read_text(encoding="utf-8")) for path in pages}
    for path in pages:
        errors.extend(check_links(path, cache[path], html_dir, check_fragments=True, cache=cache))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    errors = check(repo, repo / "docs/_build/html", repo / "docs/_build/doctrees/environment.pickle")
    if errors:
        print("Documentation integrity errors:")
        for error in sorted(set(errors)):
            print(f"  {error}")
        return 1
    print("Documentation entry-point links, rendered local links/anchors, and navigation resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
