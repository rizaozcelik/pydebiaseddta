# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

sys.path.insert(0, os.path.abspath('../../pydebiaseddta'))


project = 'pydebiaseddta'
copyright = '2022, Rıza Özçelik'
author = 'Rıza Özçelik'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    # 'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    # 'sphinx-autodoc-typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True

# # -- Autodoc configuration -----------------------------------------------------
# autoclass_content = 'class'

# autodoc_member_order = 'bysource'

# autodoc_default_flags = ['members']
