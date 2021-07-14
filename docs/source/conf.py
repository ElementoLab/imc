import os
import sys

import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../../"))


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "imc"
copyright = "2021, Andre Rendeiro"
author = "Andre Rendeiro"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    # "numpydoc",  # numpy-style docs
    "sphinx.ext.napoleon",  # numpy-style docs
    "sphinx_issues",
    "myst_parser",  # to use markdown
    "sphinxarg.ext",  # for CLI parsing of arguments
    "sphinx_autodoc_typehints"  #  <- this would be handy when whole codebase has typehinting
    # "sphinxcontrib.jupyter", <- this could be useful to make jupyter NBs
]
autodoc_typehints = "signature"  # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_typehints

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for type of input -----------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme = "sphinx_material"
# html_theme_options = {
#     "color_primary": "#ff4500",
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

issues_github_path = "ElementoLab/imc"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "urllib3": ("http://urllib3.readthedocs.org/en/latest", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.3.0/reference/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
}
