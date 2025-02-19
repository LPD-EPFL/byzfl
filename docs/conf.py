# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'ByzFL'
copyright = '2024, EPFL'
author = 'Geovani Rizk, John Stephan, Marc Gonzalez'


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
              "sphinx.ext.intersphinx",
              "sphinx_favicon",
              "sphinx_togglebutton"]


# Use MathJax to render math in HTML
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate=True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_sidebars = {
    "team/index": [],
}

html_theme = "pydata_sphinx_theme"
html_title = "ByzFL"
html_static_path = ['_static']
html_css_files = ['custom.css','https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css']
html_show_sourcelink = False
html_theme_options = {
    "header_links_before_dropdown": 5,  # Adjust based on the layout
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LPD-EPFL/byzfl",  # Replace with your repo URL
            "icon": "fab fa-github",  # Font Awesome GitHub icon
            "type": "fontawesome",  # Ensures the correct icon rendering
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/byzfl/",  # Replace with your repo URL
            "icon": "fa-custom fa-pypi",  # Font Awesome GitHub icon
            "type": "fontawesome",  # Ensures the correct icon rendering
        },

    ],
}

html_logo = "_static/byzfl_logo.png"
html_js_files = ["custom-icon.js"]
html_favicon = "_static/favicon.ico"

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