# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'andfn'
copyright = '2025, Erik A. L. Toller'
author = 'Erik A. L. Toller'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
                "sphinx.ext.autosummary",
                "sphinx.ext.viewcode",
                "sphinx.ext.todo",
                "sphinx.ext.napoleon",
              "sphinx.ext.githubpages"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx-book-theme"
html_theme_options = {
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/eriktoller/andfn",  # required
            "icon": "https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg",
            "type": "url",
        }],
}
