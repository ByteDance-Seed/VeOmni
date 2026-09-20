import os
import sys


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

html_theme = "pydata_sphinx_theme"

templates_path = ["_templates"]
html_static_path = ["assets/css"]
html_css_files = ["veomni.css"]
html_logo = "./assets/logo.png"
html_favicon = "./assets/icon.ico"
html_theme_options = {
    "github_url": "https://github.com/ByteDance-Seed/VeOmni",
    "show_prev_next": True,
    "navigation_depth": 4,
    "show_nav_level": 1,
    "collapse_navigation": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "article_header_start": ["section-label.html"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "search_bar_text": "Search documentation…",
}
html_sidebars = {"**": ["section-sidebar.html"]}
html_context = {"default_mode": "light"}

# Top-level documents own their sidebar subtree; article URLs remain independent
# of the section they belong to. Keep this list aligned with index.md's toctree.
DOC_SECTIONS = [
    ("guide/index", "User Guide"),
    ("models/index", "Models"),
    ("key_features/index", "Features"),
    ("developer/index", "Developer Guide"),
    ("design/index", "Design"),
    ("hardware_support/index", "Hardware"),
]


def page_context(app, pagename, templatename, context, doctree):
    includes = app.env.toctree_includes
    section = DOC_SECTIONS[0][0]
    for root, _ in DOC_SECTIONS:
        pending, visited = [root], set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            pending.extend(includes.get(name, []))
        if pagename in visited:
            section = root
            break
    if pagename.startswith("examples/"):
        section = "models/index"
    context["section_entries"] = [
        (name, app.env.titles[name].astext()) for name in includes.get(section, []) if name in app.env.titles
    ]
    context["doc_sections"] = DOC_SECTIONS
    context["doc_section"] = section
    context["doc_section_title"] = dict(DOC_SECTIONS)[section]


def setup(app):
    app.connect("html-page-context", page_context)
