import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

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
templates_path = ["_templates"]

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

# Keep automatically generated section labels unique across documents and
# avoid registering repeated low-level headings such as "Config".
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
myst_heading_anchors = 6

html_theme = "sphinx_book_theme"

html_static_path = []
html_logo = "./assets/logo.png"
html_favicon = "./assets/icon.ico"
html_theme_options = {
    "repository_url": "https://github.com/ByteDance-Seed/VeOmni",
    "use_repository_button": True,
}
