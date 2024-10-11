# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'NS-Gym'
copyright = '2024, Nathaniel Keplinger, Baiting Luo, Iliyas Bektas, Yunuo Zhang, Kyle Hollins Wray, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay'
author = 'Nathaniel Keplinger, Baiting Luo, Iliyas Bektas, Yunuo Zhang, Kyle Hollins Wray, Aron Laszka, Abhishek Dubey, Ayan Mukhopadhyay'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = ['ns_gym/benchmark_algorithms/*',
                    ]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
}

# To mock imports that are not installed yet
autodoc_mock_imports = ["some_dependency"]