# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the nevo package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "NEVO"
copyright = "2025, Jorge Mario Cruz-Duarte"
author = "Jorge Mario Cruz-Duarte"
release = "0.1.0"
version = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # generate docs from docstrings
    "sphinx.ext.napoleon",          # NumPy-style docstring support
    "sphinx.ext.viewcode",          # add [source] links
    "sphinx.ext.autosummary",       # summary tables
    "sphinx.ext.intersphinx",       # cross-reference external projects
    "sphinx_autodoc_typehints",     # render type annotations cleanly
    "myst_parser",                  # parse existing Markdown guides
]

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "SETUP.md",         # installation prose; not needed in rendered docs
]

# ---------------------------------------------------------------------------
# Napoleon (NumPy docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autosummary_generate = True
add_module_names = False            # drop the package prefix in signatures

# ---------------------------------------------------------------------------
# Type hints
# ---------------------------------------------------------------------------
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx — link to external docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nengo": ("https://www.nengo.ai/nengo/", None),
}

# ---------------------------------------------------------------------------
# MyST parser options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "NEVO"
pygments_dark_style = "monokai"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/jcrvz/nevo",
    "source_branch": "main",
    "source_directory": "docs/",
}

