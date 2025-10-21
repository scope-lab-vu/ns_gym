# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

import inspect
print(f"--- DEBUG: conf.py path location: {os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))}")
print(f"--- DEBUG: sys.path[0] added: {os.path.abspath('../../')}")

sys.path.insert(0, os.path.abspath('../../'))

project = 'NS-Gym'
copyright = '2025, Nathaniel S. Keplinger, Baiting Luo, Yunuo Zhang, Kyle Hollins Wray, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay'
author = 'Nathaniel S. Keplinger, Baiting Luo, Yunuo Zhang, Kyle Hollins Wray, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx_math_dollar',
    ]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Notebook generation options
nbsphinx_execute = 'never'  # Change to 'always' to execute notebooks on build
nbsphinx_allow_errors = True

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
# html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "accent_color": "jade",
    "github_url": "https://github.com/scope-lab-vu/ns_gym",
    "color_mode": "light",
}


autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_special_with_doc = True


root_doc = 'index'

html_static_path = ['_static']

# Add this line to load your new CSS file
html_css_files = [
    'css/custom.css',
]