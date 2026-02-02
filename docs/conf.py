# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
version_file = "../veomni/__init__.py"
with open(version_file, encoding="utf-8") as f:
    try:
        version_line = next(line for line in f if line.startswith("__version__"))
        __version__ = version_line.split("=")[1].strip().strip("'\"")
    except (StopIteration, IndexError):
        raise RuntimeError("Unable to find version string.")

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
html_favicon = "./assets/icon.ico"
html_theme_options = {
    "repository_url": "https://github.com/ByteDance-Seed/VeOmni",
    "use_repository_button": True,
}
