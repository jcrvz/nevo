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
copyright = "2026, Jorge Mario Cruz-Duarte & El-Ghazali Talbi"
author = "Jorge Mario Cruz-Duarte & El-Ghazali Talbi"
release = "0.1.1"
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
    "sphinx.ext.mathjax",           # render LaTeX math
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
    "dollarmath",                   # parse $...$ and $$...$$ math blocks
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "NEVO"
html_logo = "_static/nevo-logo.png"
html_favicon = "_static/favicon.ico"
#html_show_sphinx = False
pygments_style = "friendly"
pygments_dark_style = "monokai"

# ---------------------------------------------------------------------------
# Brand colours extracted from docs/images/nevo-logo.png
# Primary palette
#   orange-red  #f06010   – vibrant brand primary
#   amber/gold  #d09030   – warm accent
#   navy dark   #103050   – deep navy secondary
#   navy mid    #102040   – background dark
#   cool gray   #b0b0c0   – neutral light element
# ---------------------------------------------------------------------------
_brand_primary   = "#f06010"   # orange-red
_brand_amber     = "#d09030"   # amber/gold
_brand_navy      = "#103050"   # deep navy secondary
_brand_navy_dark = "#102040"   # darkest navy (dark-mode backgrounds)
_brand_gray      = "#b0b0c0"   # cool gray

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/jcrvz/nevo",
    "source_branch": "main",
    "source_directory": "docs/",
    # ---- light mode CSS variables ----------------------------------------
    "light_css_variables": {
        # brand
        "--color-brand-primary":           _brand_primary,
        "--color-brand-content":           _brand_navy,
        # sidebar
        "--color-sidebar-background":      "#f7f3ee",
        "--color-sidebar-background-border": "#e8ddd0",
        "--color-sidebar-brand-text":      _brand_navy,
        "--color-sidebar-caption-text":    _brand_amber,
        "--color-sidebar-link-text":       _brand_navy,
        "--color-sidebar-link-text--top-level": _brand_navy,
        "--color-sidebar-item-background--current": "#fde8d8",
        "--color-sidebar-item-background--hover":   "#f5e0cc",
        "--color-sidebar-item-expander-background--hover": "#f0d8c0",
        # topbar / announcement
        "--color-announcement-background": _brand_navy,
        "--color-announcement-text":       "#ffffff",
        # inline code
        "--color-inline-code-background":  "#f3ede6",
        # highlight
        "--color-highlight-on-target":     "#fff3e0",
        # admonitions – reuse brand colors
        "--color-admonition-title-background--tip":  "#fdf0e0",
        "--color-admonition-title--tip":             _brand_amber,
        "--color-admonition-title-background--note": "#e8eef5",
        "--color-admonition-title--note":            _brand_navy,
    },
    # ---- dark mode CSS variables -----------------------------------------
    "dark_css_variables": {
        # brand
        "--color-brand-primary":           "#d86818",
        "--color-brand-content":           _brand_amber,
        # sidebar
        "--color-sidebar-background":      "#0e1e30",
        "--color-sidebar-background-border": "#1a2e45",
        "--color-sidebar-brand-text":      "#e8c87a",
        "--color-sidebar-caption-text":    _brand_amber,
        "--color-sidebar-link-text":       "#c8d8e8",
        "--color-sidebar-link-text--top-level": "#e8c87a",
        "--color-sidebar-item-background--current": "#1e3a58",
        "--color-sidebar-item-background--hover":   "#183050",
        # inline code
        "--color-inline-code-background":  "#1a2e40",
        # highlight
        "--color-highlight-on-target":     "#1e3520",
    },
}
