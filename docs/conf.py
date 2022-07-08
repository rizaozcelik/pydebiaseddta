# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

sys.path.insert(0, os.path.abspath("../pydebiaseddta"))
# sys.path.insert(0, os.path.abspath('../../pydebiaseddta/pydebiaseddta'))
# sys.path.insert(0, os.path.abspath('../../pydebiaseddta/pydebiaseddta/predictors'))
# sys.path.insert(0, os.path.abspath('../../pydebiaseddta/pydebiaseddta/guides'))
# sys.path.insert(0, os.path.abspath('../../pydebiaseddta/pydebiaseddta/utils'))


project = "Pain"
copyright = "2022, Rıza Özçelik"
author = "Rıza Özçelik"

release = "0.1"
version = "0.1.0"

# -- General configuration

# extensions = [
#     # 'sphinx.ext.napoleon',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     # 'sphinx-autodoc-typehints',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
# ]

# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
# ]

extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    # 'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures



# intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

# html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
# epub_show_urls = "footnote"

# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True

# # -- Autodoc configuration -----------------------------------------------------
# autoclass_content = 'class'

# autodoc_member_order = 'bysource'

# autodoc_default_flags = ['members']
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_css_files = ["readthedocs-custom.css"]  # Override some CSS settings

html_static_path = ["_static"]
