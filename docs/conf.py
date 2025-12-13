import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
version_file = "../veomni/__init__.py"
with open(version_file) as f:
    __version__ = next(line for line in f if line.startswith("__version__"))
    __version__ = __version__.split("=")[1].strip().strip('"')

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

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
html_logo = "./assets/logo.png"
html_theme_options = {
    "repository_url": "https://github.com/ByteDance-Seed/VeOmni",
    "use_repository_button": True,
}
