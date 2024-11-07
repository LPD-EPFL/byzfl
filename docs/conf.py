# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from sphinx.util.docfields import Field
from sphinx.ext.autodoc import between


sys.path.insert(0, os.path.abspath(".."))

project = 'ByzFL'
copyright = '2024, EPFL'
author = 'Geovani Rizk, John Stephan, Marc Gonzalez'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc",
              "sphinx.ext.autosummary", 
              "sphinx.ext.mathjax",
              "sphinx.ext.napoleon",
              "sphinx_copybutton", 
              "sphinx.ext.autosectionlabel", 
              "sphinx.ext.intersphinx"]


# Use MathJax to render math in HTML
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate=True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ['custom.css']

napoleon_custom_sections = [
	("Initialization parameters", "params_style"),
	("Input parameters", "params_style"), 
	("Calling the instance", "rubric_style"),
    ("Returns", "params_style")
]

latex_elements = {
    'preamble': r'''
        \usepackage{amsmath}  % Load amsmath for advanced math commands
        \newcommand{\argmin}{\mathop{\mathrm{arg\,min}}
    '''
}

mathjax_config = {
    'TeX': {
        'Macros': {
            'argmin': r'\mathop{\mathrm{arg\,min}}',
        }
    }
}


