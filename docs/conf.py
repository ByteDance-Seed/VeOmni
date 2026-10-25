import os
import re
import sys

from pygments.lexers.special import TextLexer


sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
version_file = "../veomni/_version.py"
with open(version_file, encoding="utf-8") as f:
    try:
        version_line = next(line for line in f if line.startswith("__version__"))
        __version__ = version_line.split("=")[1].strip().strip("'\"")
    except (StopIteration, IndexError) as e:
        raise RuntimeError("Unable to find version string.") from e

project = "VeOmni"
copyright = "2025 ByteDance Seed Foundation MLSys Team"
author = "Qianli Ma, Yaowei Zheng, Zhongkai Zhao, Bin jia, Ziyue Huang, Zhelun Shi, Zhi Zhang"
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

language = "en"

exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
myst_heading_anchors = 4

html_theme = "sphinx_book_theme"

html_static_path = []
html_logo = "./assets/logo.png"
html_favicon = "./assets/icon.ico"
REPOSITORY_URL = "https://github.com/ByteDance-Seed/VeOmni"
html_theme_options = {
    "repository_url": REPOSITORY_URL,
    "use_repository_button": True,
}

# Docs link source files relatively (`../../veomni/x.py#L42`) so they open from
# the IDE and on GitHub. Sphinx would turn those into `_downloads/` copies and
# drop the line anchor, so the HTML build points them at GitHub instead. Read
# the Docs pins the built commit, which keeps `#L<n>` on the intended line.
SOURCE_REF = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main")
_RELATIVE_LINK = re.compile(r"\]\((\.\./[^)\s#]+)(#[^)\s]*)?\)")
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")


def _link_sources_to_github(app, docname, source):
    docs_root = os.path.abspath(app.srcdir)
    repo_root = os.path.dirname(docs_root)
    doc_dir = os.path.dirname(os.path.join(docs_root, docname))

    def rewrite(match):
        path, fragment = match.group(1), match.group(2) or ""
        target = os.path.normpath(os.path.join(doc_dir, path))
        inside_repo = target.startswith(repo_root + os.sep)
        if not inside_repo or target.startswith(docs_root + os.sep) or not os.path.exists(target):
            return match.group(0)
        kind = "tree" if os.path.isdir(target) else "blob"
        repo_path = os.path.relpath(target, repo_root).replace(os.sep, "/")
        return f"]({REPOSITORY_URL}/{kind}/{SOURCE_REF}/{repo_path}{fragment})"

    lines, in_fence = [], False
    for line in source[0].splitlines(keepends=True):
        if _CODE_FENCE.match(line):
            in_fence = not in_fence
        lines.append(line if in_fence else _RELATIVE_LINK.sub(rewrite, line))
    source[0] = "".join(lines)


def setup(app):
    app.connect("source-read", _link_sources_to_github)
    # GitHub renders ```mermaid fences as diagrams; Sphinx has no mermaid extension here, so show the source.
    app.add_lexer("mermaid", TextLexer)
